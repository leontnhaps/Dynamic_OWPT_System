"""Reproducible OOK test frames, deliberately separate from angle commands."""
import math
import random
from common.laser_pattern import pattern_settings

VERSION = 'ook-test-v1'
HEADER = '00000000' + '11111111' + '1111100110101'
PAYLOAD_BITS = 32
RULES = dict(version=VERSION, header=HEADER, payload_bits=PAYLOAD_BITS,
             decision_phase=0.5, nearest_sample_window_bits=0.25,
             min_contrast_adc=6, max_header_errors=1,
             threshold='midpoint of median OFF/ON training samples',
             sync='minimum normalized header error; payload excluded',
             missing='no interpolation; missing decision sample is an erasure')


def interval_value(value):
    text = str(value).strip()
    if not text.isascii() or not text.isdigit() or not 1 <= int(text) <= 1000:
        raise ValueError('측정 간격은 1~1000 ms 정수로 입력하세요.')
    return int(text)


def number_list(text, integer=False):
    parts = text.replace(' ', '').split(',')
    if not parts or any(not x for x in parts):
        raise ValueError('쉼표로 구분한 값을 입력하세요. 예: 100,50,30,20,15,10,5')
    values = [interval_value(x) if integer else float(x) for x in parts]
    if any(not math.isfinite(x) or x <= 0 or x > 5000 for x in values):
        raise ValueError('비트 시간은 0 초과~5000 ms입니다.')
    if len(set(values)) != len(values):
        raise ValueError('중복된 조건을 제거하세요.')
    return values


def make_plan(bit_times, sample_times, blocks, seed, pattern='random', mode='sweep', note=''):
    if not 1 <= blocks <= 1000:
        raise ValueError('조건별 반복은 1~1000회입니다.')
    if pattern not in ('random', 'alternating', 'runs') or mode not in ('sweep', 'step'):
        raise ValueError('Unknown experiment mode/pattern')
    rng = random.Random(seed)
    trials = []
    for sample_ms in sample_times:
        interval_value(sample_ms)
        if mode == 'step':
            # Each independently recorded trial contains OFF / ON / OFF for 1s each.
            conditions = [(1000, n) for n in range(blocks)]
        else:
            conditions = [(bit_ms, n) for bit_ms in bit_times for n in range(blocks)]
        for bit_ms, n in conditions:
            payload = (''.join(str(rng.randrange(2)) for _ in range(PAYLOAD_BITS)) if pattern == 'random'
                       else ('10' * 16 if pattern == 'alternating' else '00001111' * 4))
            bits = '010' if mode == 'step' else HEADER + payload + '0000'
            settings = pattern_settings(dict(bits=bits, bit_ms=bit_ms, repeats=1))
            trials.append(dict(settings=settings, sample_ms=sample_ms, mode=mode, pattern=pattern,
                               payload=None if mode == 'step' else payload, repeat_index=n,
                               # Randomize launch relative to sampling. This is a requested PC delay,
                               # not a measured or synchronized phase of Arduino/Pi clocks.
                               launch_delay_ms=100 + rng.random() * max(50, sample_ms)))
    rng.shuffle(trials)
    if len(trials) > 10000:
        raise ValueError('한 실험은 최대 10000회입니다. 조건을 나누어 실행하세요.')
    return dict(schema_version=1, rules=RULES, seed=seed, note=note, mode=mode, pattern=pattern,
                timing_note='Random launch delay does not establish a known Tx/Rx phase.', trials=trials)
