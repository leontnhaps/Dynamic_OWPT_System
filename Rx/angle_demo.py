"""Decode one manually armed optical angle command on the Rx Pi."""

import json
import os
import select
import signal
import sys
import time
from collections import deque
from contextlib import ExitStack, contextmanager

from common.angle_demo import AngleDecoder, HOME_PAN, HOME_TILT
from Rx.arc_bench import PanTilt, SAMPLE_MS, i2c_bus
from Rx.ina219_ook_stream import INA219, JsonPipe, Sampler


LISTEN_NS = 7_000_000_000


@contextmanager
def open_sensor():
    with i2c_bus(3) as bus:
        yield INA219(bus.fileno())


class ServoWrites(deque):
    """Keep both per-axis I²C write records for the acknowledgement to the PC."""

    def writerow(self, row):
        self.append(row)


class Receiver:
    def __init__(self, servos, emit, sensor_context=open_sensor, clock=time.monotonic_ns):
        self.servos = servos
        self.emit = emit
        self.sensor_context = sensor_context
        self.clock = clock
        self.resources = None
        self.sampler = None
        self.decoder = None
        self.token = None
        self.armed = False
        self.expires = 0
        self.servo_initialised = False
        self.servo_touched = False

    def disarm(self):
        resources, self.resources = self.resources, None
        self.sampler = self.decoder = self.token = None
        self.armed = False
        if resources is not None:
            resources.close()

    def command(self, cmd):
        if not isinstance(cmd, dict) or set(cmd) != {'cmd', 'token'}:
            raise ValueError('Expected only cmd and token over SSH.')
        op, token = cmd['cmd'], cmd['token']
        if not isinstance(token, str) or not 1 <= len(token) <= 128:
            raise ValueError('Expected a request token.')
        if op == 'arm':
            if self.token is not None:
                raise ValueError('Already waiting for an optical frame.')
            with ExitStack() as stack:
                sensor = stack.enter_context(self.sensor_context())
                sampler = Sampler(sensor, 3, 0x40, self.on_sample, self.clock)
                stack.callback(sampler.stop)
                sampler.command(dict(cmd='start', token=token, interval_ms=SAMPLE_MS))
                self.decoder = AngleDecoder()
                self.expires = self.clock() + LISTEN_NS
                self.sampler, self.token = sampler, token
                self.resources = stack.pop_all()
        elif op == 'home':
            self.disarm()
            self.move(HOME_PAN, HOME_TILT, 'home', token)
        elif op == 'cancel':
            self.disarm()
            self.emit(kind='cancelled', token=token)
        else:
            raise ValueError('Unknown Rx demo command.')

    def move(self, pan, tilt, kind, token):
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
        self.emit(kind=kind, token=token, pan=pan, tilt=tilt,
                  servo_writes=list(self.servos.commands))

    def release_servos(self):
        if self.servo_touched:
            self.servo_initialised = False
            self.servos.release()
            self.servo_touched = False  # A failed release can still be retried on shutdown.

    def on_sample(self, **sample):
        if sample['kind'] != 'sample' or self.token is None:
            return
        if self.clock() >= self.expires:
            self.timeout()
            return
        code = sample['bus_raw']
        if code == 8191:
            raise ValueError('INA219 전압이 포화됐습니다.')
        if not self.armed:
            self.armed = True
            self.emit(kind='armed', token=self.token)
        angles = self.decoder.feed(sample['rx_monotonic_ns'] / 1e9, code * 0.004)
        if angles is not None:
            token = self.token
            self.disarm()
            self.move(*angles, 'applied', token)

    def timeout(self):
        token = self.token
        self.disarm()
        self.emit(kind='timeout', token=token)

    def wait_seconds(self):
        if self.sampler is None:
            return None
        return max(0, min(self.sampler.deadline, self.expires) - self.clock()) / 1e9

    def tick(self):
        if self.sampler is not None:
            if self.clock() >= self.expires:
                self.timeout()
            else:
                self.sampler.sample_due()


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
