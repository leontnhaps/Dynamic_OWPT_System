"""Count optical command receptions with the Rx held at one specified angle."""

import argparse
import csv
import json
import select
import signal
import sys
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from common.arc_protocol import (ANGLE_TEST_BIT_MS, ANGLE_TEST_FRAMES, ANGLE_TEST_PERIOD_S,
                                 FrameDecoder, MIN_CONTRAST_V, VERSION, stimulus_bits, timing)
from Rx.arc_bench import PAN_START, ROOT, SAMPLE_MS, TILT_VERTICAL, PanTilt, i2c_bus, write_manifest
from Rx.ina219_ook_stream import BUS_VOLTAGE, INA219, Sampler


def settings(angle, pair, bit_ms, period_s):
    timing(bit_ms, period_s)
    if type(angle) is not int or not 0 <= angle <= 180:
        raise ValueError('정면에서 0..180° 사이의 정수 각도를 입력하세요.')
    if not pair.strip() or not pair.strip().isprintable() or len(pair.strip()) > 80:
        raise ValueError('PC 송신 기록과 같은 비교 ID를 입력하세요(최대 80자).')
    # One marker at t=0, then thirty command packets at period_s intervals.
    last_frame = stimulus_bits(ANGLE_TEST_FRAMES, bit_ms)
    block_s = ANGLE_TEST_FRAMES * period_s + len(last_frame) * bit_ms / 1000
    return dict(schema_version=1, kind='static_angle_reception', profile=VERSION, pair=pair.strip(),
                angle_deg=angle, pan_deg=PAN_START-angle, tilt_deg=TILT_VERTICAL, face_on_pan_deg=PAN_START,
                bit_ms=bit_ms, period_s=period_s, steps=ANGLE_TEST_FRAMES, sample_ms=SAMPLE_MS,
                min_contrast_v=MIN_CONTRAST_V, minimum_capture_s=block_s,
                ina219_bus=3, pca9685_bus=1, i2c_address='0x40', operator_bracketed=False)


class Reception:
    def __init__(self, config, samples, frames):
        self.config = config
        self.samples, self.frames = samples, frames
        self.decoder = FrameDecoder(config['bit_ms'])
        self.count = self.missed = 0
        self.first = self.last = self.last_index = None
        self.max_gap = 0.

    def observe(self, index, timestamp_ns, raw):
        now = timestamp_ns / 1e9
        if self.last is not None and (now <= self.last or index <= self.last_index):
            raise RuntimeError('Rx sample numbers and timestamps must increase')
        missing = 0 if self.last_index is None else index-self.last_index-1
        if self.first is None:
            self.first = now
        if self.last is not None:
            self.max_gap = max(self.max_gap, now-self.last)
        self.missed += missing
        self.last, self.last_index = now, index
        self.count += 1
        volts = raw * .004
        self.samples.writerow((index, f'{now:.9f}', f'{volts:.3f}', missing))
        if not 0 <= raw < 8191:
            raise RuntimeError('INA219 clipped or out-of-range sample')
        decoded = self.decoder.feed(now, volts)
        if decoded is not None:
            seq, start = decoded
            self.frames.writerow((f'{now:.9f}', f'{start:.9f}', seq))

    def stats(self):
        return dict(first_sample_s=self.first, last_sample_s=self.last, sample_count=self.count,
                    missing_samples=self.missed, max_gap_s=self.max_gap,
                    sampling_contiguous=bool(self.count > 1 and self.missed == 0 and
                                             self.max_gap <= 0.85 * self.config['bit_ms'] / 1000),
                    acquisition_duration_s=0. if self.first is None else self.last-self.first)

    def finish(self):
        if self.stats()['acquisition_duration_s'] < self.config['minimum_capture_s']:
            raise RuntimeError('송신 30회를 담을 최소 기록 시간보다 일찍 종료했습니다.')
        if self.max_gap >= self.config['period_s']:
            raise RuntimeError('한 단계 이상 전압 샘플이 누락되어 전체 수신율을 확인할 수 없습니다.')
        self.config['operator_bracketed'] = True


def collect(sensor, reception, flush):
    def sink(**event):
        if event['kind'] == 'sample':
            reception.observe(event['seq'], event['rx_monotonic_ns'], event['bus_raw'])

    worker = Sampler(sensor, 3, 0x40, sink)
    try:
        worker.command(dict(cmd='start', token='static-angle', interval_ms=SAMPLE_MS))
        # Tell the operator to start Tx only after the first actual sensor read.
        while not reception.count:
            time.sleep(max(0, (worker.deadline - time.monotonic_ns()) / 1e9))
            worker.sample_due()
        print('Rx 기록 시작. 이제 PC에서 송신을 누르세요. PC에 completed 및 GPIO OFF 확인이 표시된 뒤 DONE을 입력하세요.', flush=True)
        flush_at = time.monotonic() + 2
        while True:
            worker.sample_due()
            wait = max(0, (worker.deadline - time.monotonic_ns()) / 1e9)
            readable, _, _ = select.select([sys.stdin], [], [], wait)
            if readable:
                line = sys.stdin.readline()
                if not line:
                    raise EOFError('SSH 입력이 종료됐습니다.')
                if line.strip() == 'DONE':
                    reception.finish()
                    return
                print('PC 송신 완료 후 DONE을 입력하세요. 중단은 Ctrl+C입니다.', flush=True)
            if time.monotonic() >= flush_at:
                flush()
                flush_at = time.monotonic() + 2
    finally:
        worker.stop()


def csv_file(stack, folder, name, columns, streams):
    stream = stack.enter_context((folder / name).open('x', encoding='utf-8', newline=''))
    streams.append(stream)
    writer = csv.writer(stream)
    writer.writerow(columns.split())
    return writer


def run(config, root=None):
    root = ROOT if root is None else Path(root)
    instant = datetime.now(timezone.utc)
    folder = root/'captures'/'M6-4'/f"angle_{instant:%Y%m%dT%H%M%S_%fZ}_{uuid4().hex[:8]}"
    folder.mkdir(parents=True)
    config.update(created_utc=instant.isoformat(), status='running', reason=None)
    write_manifest(folder, config)
    print(f'저장 폴더: {folder}', flush=True)
    reception = servos = None
    try:
        with ExitStack() as stack:
            streams = []
            samples = csv_file(stack, folder, 'samples.csv', 'sample_index monotonic_s voltage_v missing_before', streams)
            frames = csv_file(stack, folder, 'frames.csv', 'decoded_at_s frame_start_s sequence', streams)
            commands = csv_file(stack, folder, 'commands.csv', 'write_start_s write_complete_s channel servo_deg step action', streams)
            reception = Reception(config, samples, frames)
            sensor = INA219(stack.enter_context(i2c_bus(3)).fileno())
            sensor.transfer(BUS_VOLTAGE)
            servos = PanTilt(stack.enter_context(i2c_bus(1)), commands)
            pan = config['pan_deg']
            if input(f'발판 기준 표시와 케이블 확인. Pan {pan}°, Tilt {TILT_VERTICAL}°로 이동하려면 START: ').strip() != 'START':
                raise RuntimeError('Operator cancelled before positioning')
            stack.callback(servos.release)
            servos.initialise()
            servos.set_angle(1, TILT_VERTICAL, '', 'hold_angle')
            servos.set_angle(0, pan, '', 'hold_angle')
            input('요청한 자세에서 움직임이 멈추면 Enter. PC 송신은 아직 시작하지 마세요: ')
            def flush():
                for stream in streams:
                    stream.flush()
            collect(sensor, reception, flush)
        config.update(status='completed', reason='operator_confirmed_tx_finished')
    except BaseException as exc:
        config.update(status='aborted', reason=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if reception is not None:
            config.update(reception.stats())
        config.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                      pwm_off_writes_completed=servos.released if servos else None)
        write_manifest(folder, config)
        print(f"{config['status']} · {folder}", flush=True)
    return folder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--angle', type=int, required=True, help='nominal angle from face-on at Pan 180°')
    parser.add_argument('--pair', required=True, help='unique ID entered in the PC sender')
    parser.add_argument('--bit-ms', type=float, default=ANGLE_TEST_BIT_MS)
    parser.add_argument('--period-s', type=float, default=ANGLE_TEST_PERIOD_S)
    parser.add_argument('--plan', action='store_true', help='show settings without opening I2C')
    args = parser.parse_args(argv)
    try:
        config = settings(args.angle, args.pair, args.bit_ms, args.period_s)
    except ValueError as exc:
        parser.error(str(exc))
    if args.plan:
        print(json.dumps(config, indent=2))
        return 0
    def interrupted(signum, _):
        raise KeyboardInterrupt(signal.Signals(signum).name)
    for name in ('SIGTERM', 'SIGHUP'):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), interrupted)
    try:
        run(config)
    except (OSError, EOFError, ValueError, RuntimeError, KeyboardInterrupt) as exc:
        print(f'Rx 중단: {exc}. PC 송신도 중단하세요.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
