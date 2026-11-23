"""Pi-side INA219 voltage sampler. Commands and responses are JSON lines over SSH."""

import argparse
import ctypes
import json
import os
import select
import signal
import sys
import time
from contextlib import contextmanager


CONFIG = 0x00
BUS_VOLTAGE = 0x02
# 32 V range, 12-bit ADC (no averaging), continuous bus voltage only.
SAMPLE_CONFIG = 0x219E
NS_PER_MS = 1_000_000
# TI INA219 electrical characteristics, 12-bit unaveraged bus conversion.
MAX_CONVERSION_NS = 586_000


class SMBusData(ctypes.Union):
    _fields_ = [('word', ctypes.c_uint16), ('block', ctypes.c_uint8 * 34)]


class SMBusRequest(ctypes.Structure):
    # Linux i2c-dev's native i2c_smbus_ioctl_data, including native pointer alignment.
    _fields_ = [('direction', ctypes.c_uint8), ('register', ctypes.c_uint8),
                ('size', ctypes.c_uint32), ('data', ctypes.POINTER(SMBusData))]


def swap_word(value):
    return (value >> 8) | ((value & 255) << 8)


class INA219:
    def __init__(self, fd):
        self.fd = fd

    def transfer(self, register, value=None):
        import fcntl

        data = SMBusData()
        if value is not None:
            data.word = swap_word(value)
        # Word-data read is one combined SMBus transaction. INA219 sends MSB first.
        request = SMBusRequest(1 if value is None else 0, register, 3, ctypes.pointer(data))
        fcntl.ioctl(self.fd, 0x0720, request)
        return swap_word(data.word)

    @contextmanager
    def configured(self):
        original = self.transfer(CONFIG)
        if original & 0xC000:
            raise OSError('unexpected INA219 configuration at this bus/address')
        try:
            self.transfer(CONFIG, SAMPLE_CONFIG)
            if self.transfer(CONFIG) != SAMPLE_CONFIG:
                raise OSError('INA219 sampling configuration readback mismatch')
            yield
        finally:
            self.transfer(CONFIG, original)


class Sampler:
    def __init__(self, device, bus, address, emit, clock=time.monotonic_ns):
        self.device = device
        self.bus = bus
        self.address = address
        self.emit = emit
        self.clock = clock
        self.active = None
        self.token = None
        self.deadline = 0

    def command(self, cmd):
        if not isinstance(cmd, dict):
            raise ValueError('expected a JSON command object')
        op = cmd.get('cmd')
        if op == 'quit':
            return False
        token = cmd.get('token')
        if not isinstance(token, str) or not 1 <= len(token) <= 128:
            raise ValueError('a trial token is required')
        if op == 'start':
            interval = cmd.get('interval_ms')
            if type(interval) is not int or not 1 <= interval <= 1000:
                raise ValueError('interval_ms must be an integer from 1 to 1000')
            if self.active is not None:
                raise ValueError('another trial is already sampling')
            config = self.device.configured()
            config.__enter__()
            self.active = config
            self.token = token
            self.period = interval * NS_PER_MS
            self.seq = 0
            # One full period (at least 1 ms) allows a fresh bus conversion.
            self.deadline = self.clock() + self.period
            self.emit(kind='started', token=token, interval_ms=interval, bus=self.bus,
                      address=self.address, config=SAMPLE_CONFIG)
        elif op == 'stop':
            if self.active is not None and token != self.token:
                raise ValueError('stop token does not match active trial')
            self.stop()
            self.emit(kind='stopped', token=token)
        else:
            raise ValueError('unknown sampler command')
        return True

    def stop(self):
        if self.active is not None:
            context, self.active = self.active, None
            context.__exit__(None, None, None)
            self.token = None

    def sample_due(self):
        before = self.clock()
        if self.active is None or before < self.deadline:
            return
        missed = (before - self.deadline) // self.period
        self.seq += missed
        raw = self.device.transfer(BUS_VOLTAGE)
        after = self.clock()
        # Local time of the register read, not a Tx timestamp or a hardware ADC clock.
        self.emit(kind='sample', token=self.token, seq=self.seq,
                  rx_monotonic_ns=(before + after) // 2, bus_raw=raw >> 3)
        self.seq += 1
        self.deadline += (missed + 1) * self.period
        # A late read can land just before the next tick. Do not read the same
        # conversion on that tick or burst reads to catch a missed deadline.
        earliest = after + MAX_CONVERSION_NS
        if self.deadline < earliest:
            skipped = 1 + (earliest - self.deadline - 1) // self.period
            self.seq += skipped
            self.deadline += skipped * self.period


class JsonPipe:
    """Nonblocking output keeps a slow SSH reader from changing the ADC cadence."""
    def __init__(self, output_fd):
        self.fd = output_fd
        os.set_blocking(output_fd, False)
        self.pending = bytearray()

    def emit(self, **event):
        record = json.dumps(event, separators=(',', ':')).encode('ascii') + b'\n'
        if len(self.pending) + len(record) > 131072:
            raise BufferError('SSH output backlog exceeded; trial interrupted')
        self.pending.extend(record)

    def flush(self):
        if self.pending:
            try:
                count = os.write(self.fd, self.pending)
            except BlockingIOError:
                return
            del self.pending[:count]

    def drain(self):
        end = time.monotonic() + 1
        while self.pending:
            remaining = end - time.monotonic()
            if remaining <= 0 or not select.select([], [self.fd], [], remaining)[1]:
                raise OSError('SSH output was not drained')
            self.flush()


def stream(device, bus, address, input_fd, output):
    worker = Sampler(device, bus, address, output.emit)
    commands = bytearray()
    token = None
    try:
        output.emit(kind='ready', bus=bus, address=address)
        while True:
            output.flush()
            timeout = None if worker.active is None else max(0, worker.deadline - worker.clock()) / 1e9
            readable, _, _ = select.select([input_fd], [output.fd] if output.pending else [], [], timeout)
            if readable:
                chunk = os.read(input_fd, 4096)
                if not chunk:
                    return 0
                commands.extend(chunk)
                while b'\n' in commands:
                    end = commands.index(10)
                    line = bytes(commands[:end])
                    del commands[:end + 1]
                    if len(line) > 4096:
                        raise ValueError('oversized sampler command')
                    token = None
                    try:
                        cmd = json.loads(line)
                        token = cmd.get('token') if isinstance(cmd, dict) else None
                        if not worker.command(cmd):
                            return 0
                    except (UnicodeError, ValueError) as exc:
                        output.emit(kind='error', token=token, message=str(exc))
                if len(commands) > 4096:
                    raise ValueError('oversized sampler command')
            worker.sample_due()
    except (OSError, ValueError) as exc:
        output.emit(kind='error', token=worker.token or token, message=str(exc))
        return 1
    finally:
        try:
            worker.stop()
        finally:
            output.drain()


def main():
    import fcntl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bus', type=int, default=3)
    parser.add_argument('--address', type=lambda text: int(text, 0), default=0x40)
    args = parser.parse_args()
    if args.bus < 2 or not 0x40 <= args.address <= 0x4F:
        parser.error('use the dedicated INA219 bus (>=2) and a 7-bit INA219 address 0x40..0x4F')

    def shutdown(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGHUP, shutdown)
    output = JsonPipe(sys.stdout.fileno())
    try:
        with open(f'/dev/i2c-{args.bus}', 'r+b', buffering=0) as handle:
            # Exclude another instance on the same dedicated Rx bus.
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.ioctl(handle, 0x0703, args.address)  # I2C_SLAVE
            device = INA219(handle.fileno())
            device.transfer(CONFIG)  # Confirm address responds before announcing readiness.
            return stream(device, args.bus, args.address, sys.stdin.fileno(), output)
    except (OSError, BufferError, ValueError) as exc:
        print(f'INA219 sampler: {exc}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
