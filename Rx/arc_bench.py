"""Stationary, one-axis PV incidence bench. Run from the checkout via python3 -m Rx.arc_bench."""

import argparse
import csv
import json
import math
import signal
import sys
import time
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
from pathlib import Path

from common.arc_protocol import (DEFAULT_BIT_MS, DEFAULT_PERIOD_S, DEFAULT_RADIUS_M, MIN_CONTRAST_V,
                                 VERSION, FrameDecoder, measure_window, timing)
from Rx.ina219_ook_stream import BUS_VOLTAGE, INA219, Sampler


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_MS = 5
TOLERANCE_S = 0.15
PAN_START = 180
TILT_VERTICAL = 20


@contextmanager
def i2c_bus(number):
    import fcntl

    with open(f'/dev/i2c-{number}', 'r+b', buffering=0) as bus:
        try:
            fcntl.flock(bus, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise OSError(f'I2C {number} 사용 중. OOK의 Pi Rx 연결 및 다른 측정을 종료하세요.') from exc
        fcntl.ioctl(bus, 0x0703, 0x40)
        yield bus


class PanTilt:
    def __init__(self, bus, commands):
        self.bus = bus
        self.commands = commands
        self.released = None

    def write(self, reg, *values):
        data = bytes((reg, *values))
        if self.bus.write(data) != len(data):
            raise OSError('Incomplete PCA9685 write.')

    def disable(self, channel):
        before = time.monotonic()
        self.write(0x09 + 4 * channel, 0x10)  # PCA9685 channel FULL_OFF
        self.record(before, channel, '', '', 'pwm_off')

    def record(self, before, channel, degrees, step, action):
        self.commands.writerow([f'{before:.6f}', f'{time.monotonic():.6f}', channel, degrees, step, action])

    def release(self):
        self.released = False
        try:
            self.disable(0)
        finally:
            self.disable(1)
        self.released = True

    def initialise(self):
        self.release()
        self.write(0x00, 0x30)
        self.write(0xFE, 121)  # 25 MHz oscillator, ~50 Hz
        self.write(0x01, 0x04)
        self.write(0x00, 0x20)
        time.sleep(0.005)
        self.write(0x00, 0xA0)

    def set_angle(self, channel, degrees, step, action):
        limits = {0: (0, 180), 1: (0, 35)}
        lo, hi = limits[channel]
        if not lo <= degrees <= hi:
            raise ValueError(f'Servo S{channel} command {degrees} outside {lo}..{hi}')
        ticks = round((500 + 2000 * degrees / 180) * 4096 / 20000)
        before = time.monotonic()
        self.released = False
        self.write(0x06 + channel * 4, 0, 0, ticks & 255, ticks >> 8)
        self.record(before, channel, degrees, step, action)


def write_manifest(folder, metadata):
    target = folder / 'manifest.json'
    temporary = folder / 'manifest.tmp'
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(target)


class Trial:
    def __init__(self, params, csv_files, decoder, move_pan, flush=None, clock=None):
        self.params = params
        self.steps, self.samples, self.frames = csv_files
        self.decoder = decoder
        self.move_pan = move_pan
        self.flush = flush
        self.clock = clock
        self.origin = None
        self.first_sample = None
        self.previous_time = None
        self.previous_index = None
        self.slot = 0
        self.pan = PAN_START
        self.frame_start = None
        self.volts = []
        self.gaps = []
        self.completed = False
        self.rows_written = 0

    def set_pan(self, target, deadline):
        if abs(target - self.pan) != 1:
            raise RuntimeError('Refusing a skipped or compounded 1° motion.')
        if self.clock is not None and self.clock() > deadline:
            raise RuntimeError('Missed servo write deadline; stopping without replay.')
        self.move_pan(target)
        self.pan = target

    def finish_slot(self, partial=False):
        if self.origin is None or self.rows_written > self.slot:
            return
        planned = self.origin + self.slot * self.params['period_s']
        first_bit = self.frame_start if self.frame_start is not None else planned
        lo, hi = (first_bit + phase for phase in self.params['window_s'])
        observed = [(t, value) for t, value in self.volts if lo <= t < hi]
        missing = sum(1 for t in self.gaps if lo <= t < hi)
        # Check both ends as well as the interior; a scheduler stall can cross a window boundary.
        times = [lo, *(t for t, _ in observed), hi]
        largest_gap = max(b - a for a, b in zip(times, times[1:]))
        insufficient = largest_gap > 1.8 * SAMPLE_MS / 1000
        if partial:
            status = 'partial'
        elif not observed:
            status = 'no_samples'
        elif missing or insufficient:
            status = 'sample_gap'
        else:
            status = 'ok'
        mean_v = sum(v for _, v in observed) / len(observed) if observed else None
        self.steps.writerow([self.slot, PAN_START-self.slot, self.slot*self.params['period_s'],
                             self.pan, TILT_VERTICAL, int(self.frame_start is not None),
                             'detected' if self.frame_start is not None else 'scheduled',
                             len(observed), missing, f'{mean_v:.6f}' if observed else '', status,
                             f'{lo:.6f}', f'{hi:.6f}', f'{largest_gap:.6f}'])
        self.rows_written += 1
        if self.flush is not None:
            self.flush()
        if self.slot % 10 == 0 or self.slot == self.params['steps']:
            print(f"{self.slot}/{self.params['steps']} · Pan {self.pan}° · {status}", flush=True)

    def frame(self, seq, start, now):
        if self.origin is None:
            if seq == 0:
                self.origin = self.frame_start = start
                result = 'sync'
            else:
                result = 'waiting_for_sync'
        elif seq != self.slot or abs(start - (self.origin + seq*self.params['period_s'])) > TOLERANCE_S:
            result = 'outside_slot'
        elif self.frame_start is not None:
            result = 'duplicate'
        else:
            self.frame_start = start
            result = 'accepted'
        self.frames.writerow([f'{now:.6f}', f'{start:.6f}', seq, result])
        if result == 'accepted' and self.params['mode'] == 'corrected':
            self.set_pan(self.pan + 1, now + TOLERANCE_S)

    def sample(self, index, now, raw):
        if self.previous_time is not None and (now <= self.previous_time or index <= self.previous_index):
            raise RuntimeError('Non-monotonic sensor data')
        missed = 0 if self.previous_index is None else index - self.previous_index - 1
        volts = raw * 0.004
        self.samples.writerow([index, f'{now:.6f}', f'{volts:.3f}', missed])
        previous_time = self.previous_time
        self.previous_time, self.previous_index = now, index
        if raw == 8191:
            raise RuntimeError('INA219 bus voltage clipped')
        if self.first_sample is None:
            self.first_sample = now
        if self.origin is None and now - self.first_sample > self.params['sync_timeout_s']:
            raise RuntimeError('Optical sync frame 0 timeout')
        if self.origin is not None:
            if previous_time is not None and now - previous_time >= self.params['period_s']:
                raise RuntimeError('Sensor did not sample for a complete slot')
            next_tick = self.origin + (self.slot + 1)*self.params['period_s']
            if now >= next_tick:
                if now - next_tick > TOLERANCE_S:
                    raise RuntimeError('Missed motion deadline; stopped before a catch-up move')
                self.finish_slot()
                if self.slot == self.params['steps']:
                    self.completed = True
                    return
                self.slot += 1
                self.frame_start = None
                self.volts.clear()
                self.gaps.clear()
                self.set_pan(self.pan - 1, next_tick + TOLERANCE_S)
            self.volts.append((now, volts))
            # Nominal times for missing indexes, used only as gap flags, never as observations.
            if missed:
                self.gaps.extend(previous_time + (now - previous_time)*k/(missed+1)
                                 for k in range(1, missed+1))
        event = self.decoder.feed(now, volts)
        if event is not None:
            self.frame(*event, now)


def csv_output(stack, path, fields, streams):
    handle = stack.enter_context(path.open('x', encoding='utf-8', newline=''))
    streams.append(handle)
    writer = csv.writer(handle)
    writer.writerow(fields.split())
    return writer


def acquire(sensor, trial):
    def sink(**event):
        if event['kind'] == 'sample':
            trial.sample(event['seq'], event['rx_monotonic_ns']/1e9, event['bus_raw'])

    worker = Sampler(sensor, 3, 0x40, sink)
    try:
        worker.command(dict(cmd='start', token='arc', interval_ms=SAMPLE_MS))
        while not trial.completed:
            delay = (worker.deadline - time.monotonic_ns())/1e9
            if delay > 0:
                time.sleep(delay)
            worker.sample_due()
    finally:
        worker.stop()


def run(params):
    tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    folder = ROOT / 'captures' / 'M6-4' / f"arc_{tag}_{params['mode']}"
    folder.mkdir(parents=True)
    print(f'저장 폴더: {folder}', flush=True)
    write_manifest(folder, params)
    trial = None
    servos = None
    try:
        with ExitStack() as log_stack, ExitStack() as hardware:
            streams = []
            def output(name, columns):
                return csv_output(log_stack, folder / name, columns, streams)

            def flush():
                for stream in streams:
                    stream.flush()

            files = [output('steps.csv', 'step theta_deg elapsed_s pan_deg tilt_deg frame_accepted window_source '
                            'sample_count missing_samples mean_v status window_start_s window_end_s max_gap_s'),
                     output('samples.csv', 'sample_index monotonic_s voltage_v missing_before'),
                     output('frames.csv', 'decoded_at_s frame_start_s sequence result')]
            commands = output('commands.csv', 'write_start_s write_complete_s channel servo_deg step action')
            sensor = INA219(hardware.enter_context(i2c_bus(3)).fileno())
            sensor.transfer(BUS_VOLTAGE)
            servos = PanTilt(hardware.enter_context(i2c_bus(1)), commands)
            answer = input('케이블 여유·PV 지지 확인. Pan 180°, Tilt 20°로 이동하려면 START 입력: ')
            if answer.strip() != 'START':
                raise RuntimeError('Operator cancelled before positioning')
            hardware.callback(servos.release)
            servos.initialise()
            servos.set_angle(1, TILT_VERTICAL, '', 'initial_pose')
            servos.set_angle(0, PAN_START, '', 'initial_pose')
            input('발판 방향을 조정해 셀 정면을 고정 광원에 맞춘 후 Enter: ')
            def move_pan(target):
                action = 'optical_correction' if target > trial.pan else 'virtual_motion'
                servos.set_angle(0, target, trial.slot, action)

            trial = Trial(params, files, FrameDecoder(params['bit_ms']), move_pan, flush, time.monotonic)
            params['armed_utc'] = datetime.now(timezone.utc).isoformat()
            try:
                print('광학 동기 대기. 이제 Tx 측 시험 패턴을 시작하세요.', flush=True)
                acquire(sensor, trial)
            finally:
                if not trial.completed:
                    trial.finish_slot(partial=True)
        params.update(status='completed', reason='completed')
    except BaseException as exc:
        params.update(status='aborted', reason=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        params.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                      rows_written=trial.rows_written if trial else 0,
                      sync_monotonic_s=trial.origin if trial else None,
                      pwm_off_writes_completed=servos.released if servos else None)
        write_manifest(folder, params)
        print(f"{params['status']} · {folder}", flush=True)
        if params['status'] == 'aborted':
            pwm = '두 축 PWM 해제 명령 완료.' if params['pwm_off_writes_completed'] else '두 축 PWM 해제 미확인.'
            print(f'{pwm} 외부 Tx도 종료하세요.', file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['plan', 'baseline', 'corrected'])
    parser.add_argument('--period-s', type=float, default=DEFAULT_PERIOD_S)
    parser.add_argument('--bit-ms', type=float, default=DEFAULT_BIT_MS)
    parser.add_argument('--radius-m', type=float, default=DEFAULT_RADIUS_M)
    parser.add_argument('--steps', type=int, default=180)
    parser.add_argument('--pair', default='pair01')
    parser.add_argument('--sync-timeout-s', type=float, default=120.0)
    args = parser.parse_args()
    try:
        timing(args.bit_ms, args.period_s)
        args.pair = args.pair.strip()
        if not args.pair or len(args.pair) > 80 or not args.pair.isprintable():
            raise ValueError('Use a nonempty printable pair ID of at most 80 characters')
        if not math.isfinite(args.period_s) or not math.isfinite(args.radius_m) or args.radius_m <= 0:
            raise ValueError('Period and radius must be finite and positive')
        if not math.isfinite(args.sync_timeout_s) or args.sync_timeout_s <= 0 or not 1 <= args.steps <= 180:
            raise ValueError('Use a positive sync timeout and 1..180 steps')
    except ValueError as exc:
        parser.error(str(exc))
    params = dict(vars(args), schema_version=1, sample_ms=SAMPLE_MS, initial_pan_deg=PAN_START,
                  tilt_deg=TILT_VERTICAL, simulated_motion=True, profile=VERSION,
                  window_s=list(measure_window(args.bit_ms)), status='running', reason=None,
                  pca9685_bus=1, ina219_bus=3, pca9685_address='0x40', ina219_address='0x40',
                  min_contrast_v=MIN_CONTRAST_V, frame_tolerance_s=TOLERANCE_S,
                  speed_m_s=math.radians(1)*args.radius_m/args.period_s,
                  setup='Stationary fixture; incidence change emulated by commanded pan. '
                        'Manual initial alignment, nominal radius and servo angles are unverified.',
                  created_utc=datetime.now(timezone.utc).isoformat())
    if args.mode == 'plan':
        params['status'] = 'plan'
        print(json.dumps(params, indent=2, ensure_ascii=False))
        print(f'{args.steps + 1} positions · {args.steps*args.period_s:.1f} s first to last step')
        return 0

    def handle_signal(signum, _):
        raise KeyboardInterrupt(signal.Signals(signum).name)

    for name in ('SIGTERM', 'SIGHUP'):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), handle_signal)
    try:
        run(params)
    except (OSError, ValueError, RuntimeError, KeyboardInterrupt, EOFError) as exc:
        print(f'Rx stopped: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
