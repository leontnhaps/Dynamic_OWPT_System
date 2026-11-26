import unittest

from common.angle_demo import (BIT_MS, FRAME_LEN, HOME_PAN, HOME_TILT, decode_angles,
                               encode_angles, payload_from_frame, validate_angles)
from common.ook_experiment import make_plan


class AngleDemoProtocolTest(unittest.TestCase):
    def test_command_replaces_only_the_speed_test_payload(self):
        reference = make_plan([BIT_MS], [1], 1, 123)['trials'][0]['settings']['bits']
        for pair in ((0, 0), (180, 35), (HOME_PAN, HOME_TILT), (102, 21)):
            with self.subTest(pair=pair):
                frame = encode_angles(*pair)
                self.assertEqual((BIT_MS, FRAME_LEN, len(frame)), (8, 65, 65))
                self.assertEqual(frame[:29], reference[:29])
                self.assertEqual(frame[-4:], reference[-4:])
                self.assertEqual(decode_angles(payload_from_frame(frame)), pair)
        self.assertEqual(payload_from_frame(encode_angles(103, 20)),
                         '0000000001100111' '0000000000010100')

    def test_ranges_checked_before_transmission_or_motion(self):
        for pair in ((-1, 0), (181, 2), (0, 36), (0, -1), (1, 1.5), (True, 0)):
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                encode_angles(*pair)
        received = decode_angles('1' * 32)
        self.assertEqual(received, (65535, 65535))
        with self.assertRaises(ValueError):
            validate_angles(*received)

    def test_invalid_frame_or_erased_bit_is_never_a_command(self):
        frame = encode_angles(10, 4)
        for damaged in ('1' + frame[1:], frame[:-1], frame[:-1] + '1'):
            with self.subTest(damaged=damaged), self.assertRaises(ValueError):
                payload_from_frame(damaged)
        for payload in ('', '0' * 31 + '?', '0' * 33):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                decode_angles(payload)


if __name__ == '__main__':
    unittest.main()
