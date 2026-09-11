"""M1-2 command validation; angles are commands, never measured positions."""
import math


def number(value, name):
    if isinstance(value, bool):
        raise ValueError(f'{name}: numeric value required')
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{name}: numeric value required') from None
    if not math.isfinite(value):
        raise ValueError(f'{name}: finite value required')
    return value


def limits_from(data):
    result = {}
    for axis, bounds in [('pan', (-180, 180)), ('tilt', (-45, 90))]:
        low = number(data.get(axis + '_min'), axis + '_min')
        high = number(data.get(axis + '_max'), axis + '_max')
        if not bounds[0] <= low < high <= bounds[1]:
            raise ValueError(f'{axis}: {bounds[0]} <= min < max <= {bounds[1]} required')
        result[axis + '_min'] = low
        result[axis + '_max'] = high
    return result


def move_from(data, limits):
    if limits is None:
        raise ValueError('Apply operating limits before moving')
    result = {}
    for axis in ('pan', 'tilt'):
        value = number(data.get(axis), axis)
        if not limits[axis + '_min'] <= value <= limits[axis + '_max']:
            raise ValueError(f'{axis}: outside applied operating limits')
        result[axis] = value
    speed = number(data.get('speed'), 'speed')
    acc = number(data.get('acc'), 'acc')
    if speed <= 0 or not speed.is_integer() or acc <= 0:
        raise ValueError('SPD: positive integer; ACC: positive number required')
    result.update(speed=int(speed), acc=acc)
    return result


class Servo:
    """DLC Raspberrypi/Rasp_main.py move -> ESP32 T=133 JSON adapter.

    A successful write only confirms host serial transmission. No firmware
    acknowledgement, encoder angle or arrival detection is available here.
    """
    def __init__(self, port=None, simulate=False, transport=None):
        self.port = port
        self.simulate = simulate
        self.serial = transport
        self.limits = None
        self.last = None

    def status(self):
        return dict(available=bool(self.simulate or self.port or self.serial),
                    simulated=self.simulate, port=self.port, baud=115200,
                    limits=self.limits, commanded=self.last,
                    actual_angle=None, arrival_verified=False)

    def configure(self, data):
        self.limits = limits_from(data)
        return self.status()

    def move(self, data):
        move = move_from(data, self.limits)
        packet = dict(T=133, X=move['pan'], Y=move['tilt'],
                      SPD=move['speed'], ACC=move['acc'])
        if not self.simulate:
            if self.serial is None:
                if not self.port:
                    raise ValueError('Servo disabled: start Pi with --servo-port DEVICE')
                import serial
                self.serial = serial.Serial(self.port, 115200, timeout=.2, write_timeout=1)
            import json
            raw = (json.dumps(packet) + '\n').encode()
            try:
                if self.serial.write(raw) != len(raw):
                    raise OSError('Incomplete serial write; commanded position is unknown')
            except Exception:
                self.last = None
                self.close()
                raise
        self.last = dict(move, command_id=data.get('request_id'))
        return dict(self.status(), packet=packet,
                    state='simulated' if self.simulate else 'serial_written')

    def close(self):
        if self.serial is not None:
            try:
                self.serial.close()
            finally:
                self.serial = None
