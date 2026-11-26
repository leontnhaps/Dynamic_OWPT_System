import math
import random
import unittest

import numpy as np

from common.ook_experiment import HEADER, PAYLOAD_BITS, SYNC_WORD, THRESHOLD_BITS, TRAILING_ZEROS, make_frame
from common.ook_frame_decode import PayloadDecoder, decode_captured_frame, fit_header
from Tx.Controller.ook_pi_analysis import decode, find_header


def samples(bits, *, phase=53.4, contrast=.4):
    # Millisecond clock with independent sample phase, jitter and quantized voltage.
    rng = random.Random(522)
    rows = []
    for i in range(math.ceil((phase+len(bits)*8+70)/2)):
        t = 2*i+rng.uniform(-.24, .24)
        index = math.floor((t-phase)/8)
        high = 0 <= index < len(bits) and bits[index] == '1'
        v = round((.62+contrast*high+rng.uniform(-.003, .003))/.004)*.004
        rows.append((t,v))
    return rows


def received(rows):
    rx = PayloadDecoder(8)
    return [got for t, v in rows if (got := rx.feed(t/1000, v)) is not None]


class PiFrameDecoderTests(unittest.TestCase):
    def test_envelope_and_whole_cell_decisions_match_offline(self):
        self.assertEqual((THRESHOLD_BITS, SYNC_WORD, TRAILING_ZEROS),
                         ('0000000011111111', '1111100110101', '0000'))
        for payload in ['01'*16, '0'*PAYLOAD_BITS, '1'*PAYLOAD_BITS, '10011110'*4]:
            frame = make_frame(payload)
            self.assertEqual(frame, HEADER+payload+TRAILING_ZEROS)
            for phase in [35.3, 53.4, 103.8]:
                rows = samples(frame, phase=phase)
                times, values = map(np.array, zip(*rows))
                best, _ = find_header(times, values, 8)
                portable = fit_header(list(times), list(values), 8, best-17., best+17.)
                self.assertAlmostEqual(best, portable, places=7)
                result = decode(times, values, 8)
                truth = ''.join(d['decision'] for d in result['bits'])
                self.assertEqual(truth, payload)
                self.assertEqual(received(rows), [truth])
        for invalid in ('', '0'*31, '1'*33, '0'*31+'?', 123):
            with self.assertRaises(ValueError):
                make_frame(invalid)

    def test_unobserved_corrupt_header_or_trailer_cannot_supply_a_frame(self):
        frame = make_frame('10011001'*4)
        rows = samples(frame)
        for i in (4,18,28,61,64):
            corrupt = frame[:i]+str(1-int(frame[i]))+frame[i+1:]
            self.assertEqual(received(samples(corrupt)), [], i)
        for i in (8,42,63):
            gap_start = 53.4+i*8
            self.assertEqual(received([(t,v) for t,v in rows if not gap_start <= t < gap_start+15]), [])
        self.assertEqual(received(rows[:-50]), [])
        self.assertEqual(received(samples(frame, contrast=.009)), [])
        self.assertEqual(received(samples('1010101011010011'+'0'*24+'1')), [])

    def test_separate_frames_emit_once_and_recover_from_a_failed_header(self):
        a, b = '00011101'*4, '01001010'*4
        damaged = make_frame(a)
        damaged = damaged[:22]+str(1-int(damaged[22]))+damaged[23:]
        self.assertEqual(received(samples(damaged+'0'*3+make_frame(a)+make_frame(b))), [a,b])


class CapturedFrameTests(unittest.TestCase):
    payload = f'{103:016b}{24:016b}'
    phase_ms = 20.5

    def capture(self, bit_ms, missing=None, bits=None):
        frame = make_frame(self.payload) if bits is None else bits
        samples = []
        for ms in range(math.ceil(self.phase_ms + len(frame)*bit_ms + 25)):
            relative = ms - self.phase_ms
            if missing is not None and missing[0] <= relative < missing[1]:
                continue
            index = math.floor(relative / bit_ms)
            on = 0 <= index < len(frame) and frame[index] == '1'
            samples.append((1_000_000_000_000 + ms*1_000_000, 1050 if on else 350))
        return samples

    def test_gap_crossing_boundary_is_valid_if_each_bit_was_measured(self):
        for bit_ms in (1, 8, 10, 100):
            with self.subTest(bit_ms=bit_ms):
                gap = None if bit_ms == 1 else (63.6*bit_ms, 64.6*bit_ms)
                result = decode_captured_frame(self.capture(bit_ms, missing=gap), bit_ms)
                self.assertTrue(result['verified'], result)
                self.assertEqual(result['payload_bits'], self.payload)
                self.assertEqual(result['tail_bits'], '0000')

    def test_confirmed_payload_preserved_but_absent_or_wrong_last_stop_bit_rejects(self):
        result = decode_captured_frame(self.capture(8, missing=(511,521)), 8)
        self.assertTrue(result['header_verified'])
        self.assertFalse(result['verified'])
        self.assertEqual((result['payload_bits'], result['tail_bits']), (self.payload, '000?'))
        self.assertIn('4번째', result['reason'])
        result = decode_captured_frame(self.capture(8, bits=make_frame(self.payload)[:-1]+'1'), 8)
        self.assertFalse(result['verified'])
        self.assertEqual((result['payload_bits'], result['tail_bits']), (self.payload, '0001'))

    def test_payload_gap_or_unobserved_header_never_produces_a_valid_command(self):
        gap_start = (len(HEADER)+3)*8
        result = decode_captured_frame(self.capture(8, missing=(gap_start-1, gap_start+9)), 8)
        self.assertTrue(result['header_verified'])
        self.assertFalse(result['verified'])
        self.assertIsNone(result['payload_bits'])
        flat = [(ns, 200) for ns,_ in self.capture(8)]
        for rows in ([], flat, self.capture(8)[:50]):
            result = decode_captured_frame(rows, 8)
            self.assertFalse(result['header_verified'])
            self.assertFalse(result['verified'])
            self.assertIsNone(result['payload_bits'])
