import random
import unittest

from common.arc_protocol import FrameDecoder, frame_bits, measure_window, stimulus_bits, timing


class ArcProtocolTest(unittest.TestCase):
    def sample(self, bits, bit_ms=50, omit=None, contrast=0.20):
        bit = bit_ms / 1000
        end = 0.333 + len(bits) * bit + 0.2
        rng = random.Random(47)
        receiver = FrameDecoder(bit_ms)
        found = []
        i = 0
        while i * 0.005 < end:
            t = i * 0.005 + rng.uniform(-0.00035, 0.00035)
            i += 1
            if omit and omit[0] <= t <= omit[1]:
                continue
            index = int((t - 0.333) // bit)
            high = 0 <= index < len(bits) and bits[index] == '1'
            v = round((0.27 + contrast * high + rng.uniform(-0.004, 0.004)) / 0.004) * 0.004
            event = receiver.feed(t, v)
            if event is not None:
                found.append(event)
        return found

    def test_checked_frames_and_alignment_with_jitter(self):
        for bit_ms, seq in ((20, 0), (30, 59), (50, 180), (100, 1)):
            with self.subTest(bit_ms=bit_ms, seq=seq):
                found = self.sample(stimulus_bits(seq, bit_ms), bit_ms)
                self.assertEqual([event[0] for event in found], [seq])
                self.assertLess(abs(found[0][1] - 0.333), bit_ms / 1000)

    def test_bad_crc_missing_samples_and_background_do_not_move(self):
        frame = frame_bits(41)
        corrupt = frame[:-1] + str(1 - int(frame[-1]))
        self.assertEqual(self.sample(corrupt), [])
        self.assertEqual(self.sample(frame, omit=(0.99, 1.08)), [])
        self.assertEqual(self.sample(frame, contrast=0.01), [])
        self.assertEqual(self.sample('1' * 80), [])

    def test_protocol_bounds_and_light_window(self):
        self.assertEqual(len(frame_bits(180)), 32)
        for invalid in (-1, 181, 1.5):
            with self.assertRaises(ValueError):
                frame_bits(invalid)
        timing(50, 4)
        with self.assertRaises(ValueError):
            timing(50, 2)
        self.assertEqual(measure_window(50), (2.1, 2.7))
        bits = stimulus_bits(5, 50)
        self.assertTrue(all(bits[int(t / 0.05)] == '1' for t in (2.10, 2.30, 2.70)))
        self.assertEqual(bits[-1], '0')


if __name__ == '__main__':
    unittest.main()
