"""Decode one manually armed optical angle command on the Rx Pi."""

import json
import os
import select
import signal
import sys
import time
from collections import deque
from contextlib import ExitStack, contextmanager

from common.angle_demo import (BIT_MS, DECODE_SAMPLE_MS, FRAME_LEN, HOME_PAN, HOME_TILT, EXPECTED_SAMPLES,
                               MIN_VOLTAGE_SAMPLES, MAX_VOLTAGE_GAP_NS, VOLTAGE_SAMPLE_MS,
                               VOLTAGE_SETTLE_NS, VOLTAGE_WINDOW_NS, decode_angles, validate_angles,
                               validate_angle_bit_ms, validate_jog)
from common.ook_frame_decode import PayloadDecoder, decode_captured_frame
from Rx.arc_bench import PanTilt, i2c_bus
from Rx.ina219_ook_stream import INA219, JsonPipe, NS_PER_MS, Sampler


LISTEN_NS = 7_000_000_000
ANGLE_BATCH_ROWS = 48


@contextmanager
def open_sensor():
    with i2c_bus(3) as bus:
        yield INA219(bus.fileno())


class ServoWrites(deque):
    """Keep both per-axis I²C write records for the acknowledgement to the PC."""

    def writerow(self, row):
        self.append(row)


class CaptureConflict(ValueError):
    """Reject a competing command without interrupting the active capture."""


class Receiver:
    def __init__(self, servos, emit, sensor_context=open_sensor, clock=time.monotonic_ns):
        self.servos = servos
        self.emit = emit
        self.sensor_context = sensor_context
        self.clock = clock
        self.resources = None
        self.sampler = None
        self.decoder = None
        self.pending_payload = None
        self.angle_rows = None
        self.finish_after_ns = None
        self.token = None
        self.armed = False
        self.expires = None
        self.window_start_ns = None
        self.servo_initialised = False
        self.servo_touched = False
        self.last_pose = None

    def disarm(self):
        resources, self.resources = self.resources, None
        token, rows = self.token, self.angle_rows
        self.angle_rows = None
        self.finish_after_ns = None
        self.sampler = self.decoder = self.token = None
        self.pending_payload = None
        self.armed = False
        self.expires = None
        self.window_start_ns = None
        try:
            if resources is not None:
                resources.close()
        finally:
            if rows is not None:
                remainder = len(rows) % ANGLE_BATCH_ROWS
                if remainder:
                    self.emit(kind='angle_samples', token=token, rows=rows[-remainder:])
                self.emit(kind='angle_samples_end', token=token, count=len(rows))

    def command(self, cmd):
        if self.sampler is not None and self.expires is None:
            if cmd != {'cmd': 'stop', 'token': self.token}:
                raise CaptureConflict('속도 측정 중입니다. 해당 측정 ID의 stop 명령이 필요합니다.')
        if not isinstance(cmd, dict):
            raise ValueError('Expected a command object over SSH.')
        op = cmd.get('cmd')
        fields = {'cmd', 'token'}
        bit_ms = BIT_MS
        if op == 'start':
            fields.add('interval_ms')
        elif op == 'arm' and 'bit_ms' in cmd:
            fields.add('bit_ms')
            bit_ms = validate_angle_bit_ms(cmd['bit_ms'])
        elif op == 'jog':
            if self.token is not None:
                raise CaptureConflict('Rx 수신 또는 전압 측정을 마친 뒤 서보를 점검하세요.')
            fields.update(('pan_step', 'tilt_step'))
        if set(cmd) != fields:
            raise ValueError('Unexpected Rx command fields.')
        token = cmd['token']
        if not isinstance(token, str) or not 1 <= len(token) <= 128:
            raise ValueError('Expected a request token.')
        if op in ('arm', 'measure', 'start'):
            if self.token is not None:
                raise CaptureConflict('Rx가 이미 수신 또는 전압 측정 중입니다.')
            with ExitStack() as stack:
                sensor = stack.enter_context(self.sensor_context())
                callback = self.emit if op == 'start' else self.on_sample
                sampler = Sampler(sensor, 3, 0x40, callback, self.clock)
                stack.callback(sampler.stop)
                interval_ms = cmd.get('interval_ms', DECODE_SAMPLE_MS if op == 'arm' else VOLTAGE_SAMPLE_MS)
                sampler.command(dict(cmd='start', token=token, interval_ms=interval_ms))
                now = self.clock()
                if op == 'arm':
                    self.angle_bit_ms = bit_ms
                    if 'bit_ms' in cmd:
                        self.angle_rows = []
                    else:
                        # Older controllers wait for an unsolicited decode and never send finish_angle.
                        self.decoder = PayloadDecoder(bit_ms)
                    self.expires = now + max(LISTEN_NS, (FRAME_LEN * bit_ms + 2000) * NS_PER_MS)
                elif op == 'measure':
                    self.window_start_ns = now + VOLTAGE_SETTLE_NS
                    self.expires = self.window_start_ns + VOLTAGE_WINDOW_NS
                    self.last_voltage_ns = self.window_start_ns
                    self.max_gap_ns = 0
                    self.voltage_count = 0
                self.sampler, self.token = sampler, token
                self.resources = stack.pop_all()
        elif op == 'finish_angle':
            if token != self.token or self.angle_rows is None or not self.armed:
                raise CaptureConflict('종료할 각도 OOK 수신 요청이 없습니다.')
            now = self.clock()
            if now >= self.expires:
                self.timeout()
                return
            earliest_end = self.angle_rows[0][1] + FRAME_LEN * self.angle_bit_ms * NS_PER_MS
            if now < earliest_end:
                raise CaptureConflict('각도 프레임 길이보다 일찍 송신 종료 요청이 도착했습니다.')
            if self.finish_after_ns is None:
                # Tx has reported final OFF. Keep reading until an actual post-frame
                # sample is available; no PC/Rx clock synchronization is assumed.
                self.finish_after_ns = now + 2 * DECODE_SAMPLE_MS * NS_PER_MS
        elif op == 'commit':
            if token != self.token or self.pending_payload is None:
                raise CaptureConflict('대응하는 광복호 결과가 없습니다.')
            if self.clock() >= self.expires:
                self.timeout()
                return
            payload = self.pending_payload
            self.disarm()  # Consume it before I/O. A repeated commit can never reapply the pose.
            pan, tilt = decode_angles(payload)
            self.move(pan, tilt, 'applied', token, rx_payload_bits=payload)
        elif op == 'stop':
            if self.token is not None and self.expires is not None:
                raise CaptureConflict('속도 측정 stop 명령으로 각도·전압 측정을 중지할 수 없습니다.')
            self.disarm()
            self.emit(kind='stopped', token=token)
        elif op == 'home':
            self.disarm()
            self.move(HOME_PAN, HOME_TILT, 'home', token)
        elif op == 'jog':
            pan_step, tilt_step = cmd['pan_step'], cmd['tilt_step']
            validate_jog(pan_step, tilt_step)
            if self.last_pose is None:
                raise ValueError('현재 Rx 명령각을 알 수 없습니다. 원위치 (SSH)를 먼저 누르세요.')
            pan, tilt = self.last_pose
            self.move(pan + pan_step, tilt + tilt_step, 'jogged', token)
        elif op == 'cancel':
            if (self.angle_rows is not None or self.pending_payload is not None) and self.token != token:
                raise CaptureConflict('취소 요청 ID가 현재 각도 수신과 다릅니다.')
            self.disarm()
            self.emit(kind='cancelled', token=token)
        else:
            raise ValueError('Unknown Rx demo command.')

    def move(self, pan, tilt, kind, token, **evidence):
        validate_angles(pan, tilt)
        self.last_pose = None
        try:
            if not self.servo_initialised:
                self.servo_touched = True
                self.servos.initialise()
                self.servo_initialised = True
            self.servos.set_angle(0, pan, '', kind)
            self.servos.set_angle(1, tilt, '', kind)
        except OSError:
            self.release_servos()
            raise
        self.last_pose = (pan, tilt)
        self.emit(kind=kind, token=token, pan=pan, tilt=tilt,
                  servo_writes=list(self.servos.commands), **evidence)

    def release_servos(self):
        self.last_pose = None
        if self.servo_touched:
            self.servo_initialised = False
            self.servos.release()
            self.servo_touched = False  # A failed release can still be retried on shutdown.

    def on_sample(self, **sample):
        if sample['kind'] == 'started':
            self.sampler_settings = {key: sample[key] for key in ('bus', 'address', 'config', 'interval_ms')}
            return
        if sample['kind'] != 'sample' or self.token is None:
            return
        if self.window_start_ns is not None:
            t = sample['rx_monotonic_ns']
            if self.window_start_ns <= t < self.expires:
                self.max_gap_ns = max(self.max_gap_ns, t - self.last_voltage_ns)
                self.voltage_count += 1
                self.last_voltage_ns = t
                self.emit(kind='voltage', token=self.token, seq=sample['seq'],
                          rx_monotonic_ns=t, bus_raw=sample['bus_raw'])
                if sample['bus_raw'] == 8191:
                    raise ValueError('INA219 전압이 포화됐습니다.')
            return
        code = sample['bus_raw']
        if self.angle_rows is not None:
            self.angle_rows.append([sample['seq'], sample['rx_monotonic_ns'], code])
            if len(self.angle_rows) % ANGLE_BATCH_ROWS == 0:
                self.emit(kind='angle_samples', token=self.token, rows=self.angle_rows[-ANGLE_BATCH_ROWS:])
        if self.clock() >= self.expires:
            self.timeout()
            return
        if code == 8191:
            raise ValueError('INA219 전압이 포화됐습니다.')
        if not self.armed:
            self.armed = True
            settings = (dict(self.sampler_settings, bit_ms=self.angle_bit_ms, capture_mode='post_tx',
                             sample_ms=self.sampler_settings['interval_ms'])
                        if self.angle_rows is not None else {})
            self.emit(kind='armed', token=self.token, **settings)
        if self.decoder is not None:  # Legacy control-only stream.
            payload = self.decoder.feed(sample['rx_monotonic_ns'] / 1e9, code * 0.004)
            if payload is not None:
                token = self.token
                self.disarm()
                self.report_decoded(token, payload)

    def finish_angle(self):
        token, rows, bit_ms = self.token, self.angle_rows, self.angle_bit_ms
        self.disarm()
        result = decode_captured_frame([(ns, code) for _, ns, code in rows], bit_ms)
        if result['verified']:
            self.report_decoded(token, result['payload_bits'], frame_check=result)
        else:
            self.emit(kind='error', token=token, message=result['reason'], frame_check=result)

    def report_decoded(self, token, payload, **evidence):
        self.token = token
        self.pending_payload = payload
        self.expires = self.clock() + LISTEN_NS
        pan, tilt = decode_angles(payload)
        self.emit(kind='decoded', token=token, rx_payload_bits=payload, pan=pan, tilt=tilt, **evidence)

    def timeout(self):
        token = self.token
        if self.window_start_ns is not None:
            start, end, count = self.window_start_ns, self.expires, self.voltage_count
            max_gap = max(self.max_gap_ns, end - self.last_voltage_ns)
            self.disarm()
            stats = dict(count=count, window_start_ns=start, window_end_ns=end, max_gap_ns=max_gap)
            if count < MIN_VOLTAGE_SAMPLES or max_gap > MAX_VOLTAGE_GAP_NS:
                self.emit(kind='error', token=token, **stats,
                          message=f'전압 표본 부족: {count}/{EXPECTED_SAMPLES}개, '
                                  f'최대 공백 {max_gap / NS_PER_MS:.1f} ms.')
            else:
                self.emit(kind='measured', token=token, **stats)
            return
        self.disarm()
        self.emit(kind='timeout', token=token)

    def wait_seconds(self):
        deadline = self.expires
        if self.sampler is not None:
            deadline = self.sampler.deadline if deadline is None else min(deadline, self.sampler.deadline)
        return None if deadline is None else max(0, deadline - self.clock()) / 1e9

    def tick(self):
        if self.expires is not None and self.clock() >= self.expires:
            self.timeout()
        elif self.sampler is not None:
            self.sampler.sample_due()
            if self.finish_after_ns is not None and self.angle_rows[-1][1] >= self.finish_after_ns:
                self.finish_angle()


def serve(receiver, input_fd, output):
    pending = bytearray()
    output.emit(kind='ready')
    try:
        while True:
            output.flush()
            writable = [output.fd] if output.pending else []
            readable, _, _ = select.select([input_fd], writable, [], receiver.wait_seconds())
            if readable:
                chunk = os.read(input_fd, 4096)
                if not chunk:
                    return
                pending.extend(chunk)
                while b'\n' in pending:
                    line, _, pending = pending.partition(b'\n')
                    if len(line) > 4096:
                        raise ValueError('Oversized Rx command.')
                    token = None
                    try:
                        cmd = json.loads(line)
                        token = cmd.get('token') if isinstance(cmd, dict) else None
                        receiver.command(cmd)
                    except CaptureConflict as exc:
                        output.emit(kind='error', token=token, message=str(exc))
                    except (ValueError, OSError) as exc:
                        receiver.disarm()
                        output.emit(kind='error', token=token, message=str(exc))
                if len(pending) > 4096:
                    raise ValueError('Oversized Rx command.')
            token = receiver.token
            try:
                receiver.tick()
            except (ValueError, OSError) as exc:
                receiver.disarm()
                output.emit(kind='error', token=token, message=str(exc))
    finally:
        try:
            receiver.disarm()
        finally:
            receiver.release_servos()


def main():
    def shutdown(signum, _frame):
        raise SystemExit(128 + signum)

    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, shutdown)
    output = JsonPipe(sys.stdout.fileno())
    try:
        with i2c_bus(1) as bus:
            servos = PanTilt(bus, ServoWrites(maxlen=2))
            serve(Receiver(servos, output.emit), sys.stdin.fileno(), output)
        return 0
    except (OSError, ValueError, BufferError) as exc:
        output.emit(kind='error', message=str(exc))
        return 1
    finally:
        output.drain()


if __name__ == '__main__':
    sys.exit(main())
