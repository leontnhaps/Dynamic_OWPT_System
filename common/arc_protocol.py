"""Versioned, fixed-light PV bench stimulus. Sequence numbers apply only to this test."""

import math
from bisect import bisect_left
from collections import deque
from statistics import median

from common.laser_pattern import pattern_settings


VERSION = 'pv-arc-bench-v1'
SYNC = '1010101100110101'
FRAME_LEN = 32
DEFAULT_BIT_MS = 50
DEFAULT_PERIOD_S = 4.0
DEFAULT_RADIUS_M = 5.0
ANGLE_TEST_BIT_MS = 12
ANGLE_TEST_PERIOD_S = 2.5
ANGLE_TEST_FRAMES = 30
MIN_CONTRAST_V = 0.024  # Proposed bench criterion: six INA219 bus-voltage counts.
SAMPLE_S = 0.005


def crc8(*octets):
    """CRC-8/ATM, initial value zero, most significant bit first."""
    crc = 0
    for byte in octets:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ (0x07 if crc & 0x80 else 0)) & 0xff
    return crc


def frame_bits(seq):
    if type(seq) is not int or not 0 <= seq <= 180:
        raise ValueError('arc sequence must be an integer from 0 through 180')
    return f'{SYNC}{seq:08b}{crc8(0xb2, seq):08b}'


def _bit_seconds(bit_ms, sample_s=SAMPLE_S):
    if not math.isfinite(bit_ms) or not 2 * sample_s * 1000 <= bit_ms <= 1000:
        raise ValueError(f'bit duration must support two {sample_s * 1000:g} ms samples and be at most 1000 ms')
    return bit_ms / 1000


def stimulus_bits(seq, bit_ms):
    bit_s = _bit_seconds(bit_ms)
    return frame_bits(seq) + '1' * math.ceil(1.30 / bit_s) + '0'


def measure_window(bit_ms):
    frame_end = FRAME_LEN * _bit_seconds(bit_ms)
    return frame_end + 0.50, frame_end + 1.10


def timing(bit_ms, period_s):
    bits = stimulus_bits(0, bit_ms)
    pattern_settings(dict(bits=bits, bit_ms=bit_ms, repeats=1))
    minimum = len(bits) * bit_ms / 1000 + 0.45
    if not math.isfinite(period_s) or period_s < minimum:
        raise ValueError(f'arc step period must be at least {minimum:.3f} s for this bit duration')


class VoltageFrameDecoder:
    """Sample bit centres using levels from a known sync word, then validate the payload."""

    def __init__(self, bit_ms, sync, frame_len, *, sample_s=SAMPLE_S):
        self.sync = sync
        self.frame_len = frame_len
        self.bit_s = _bit_seconds(bit_ms, sample_s)
        self.history = deque()
        self.last = None
        self.next_search = -math.inf

    def feed(self, t_seconds, volts):
        if not math.isfinite(t_seconds) or not math.isfinite(volts):
            raise ValueError('non-finite INA219 measurement or time')
        if self.last is not None:
            if t_seconds <= self.last:
                raise ValueError('Rx timestamps must increase')
            if t_seconds - self.last > 0.85 * self.bit_s:
                self.history.clear()  # Never decode through an unsampled bit centre.
        self.last = t_seconds
        self.history.append((t_seconds, volts))
        bit = self.bit_s
        # Check a grid of positions as new samples arrive. The report is an estimated
        # optical onset in the receiver clock, not a timestamp from the transmitter.
        start = t_seconds - (self.frame_len - 0.10) * bit
        while self.history and self.history[0][0] < start - bit:
            self.history.popleft()
        if t_seconds < self.next_search or self.history[0][0] > start + 0.25 * bit:
            return None

        times, values = zip(*self.history)
        def centre_value(index):
            left = bisect_left(times, start + (index + 0.25) * bit)
            right = bisect_left(times, start + (index + 0.75) * bit)
            if right == left:
                return None
            return sum(values[left:right]) / (right - left)

        header = [centre_value(i) for i in range(len(self.sync))]
        if None in header:
            return None
        low = median(v for s, v in zip(self.sync, header) if s == '0')
        high = median(v for s, v in zip(self.sync, header) if s == '1')
        if high - low < MIN_CONTRAST_V:
            return None
        threshold = (low + high) / 2
        margin = 0.10 * (high - low)
        if any((v-threshold if s == '1' else threshold-v) < margin for s, v in zip(self.sync, header)):
            return None
        data = [centre_value(i) for i in range(len(self.sync), self.frame_len)]
        if any(v is None or abs(v - threshold) < margin for v in data):
            return None
        word = int(''.join('1' if v >= threshold else '0' for v in data), 2)
        decoded = self.decode_word(word, start)
        if decoded is not None:
            self.next_search = t_seconds + self.frame_len * bit * 0.90
        return decoded

    def decode_word(self, word, start):
        raise NotImplementedError


class FrameDecoder(VoltageFrameDecoder):
    """Decode the original arc bench sequence and return its estimated onset."""

    def __init__(self, bit_ms):
        super().__init__(bit_ms, SYNC, FRAME_LEN)

    def decode_word(self, word, start):
        seq, crc = divmod(word, 256)
        if seq > 180 or crc != crc8(0xb2, seq):
            return None
        return seq, start
