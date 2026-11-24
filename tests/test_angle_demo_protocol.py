import random
import unittest

from common.angle_demo import (AngleDecoder, BIT_MS, DISCRIMINATOR, FRAME_LEN,
                               HOME_PAN, HOME_TILT, SYNC, encode_angles)
from common.arc_protocol import FrameDecoder, crc8, stimulus_bits


def receive(bits, *, omit=None, contrast=0.20, receiver=None):
    receiver = receiver or AngleDecoder()
    rng = random.Random(13)
    found = []
    period = BIT_MS / 1000
    start = 0.274
    for i in range(int((start + len(bits) * period + 0.12) / 0.005)):
        t = i * 0.005 + rng.uniform(-0.00035, 0.00035)
        if omit is not None and omit[0] < t < omit[1]:
            continue
        index = int((t - start) // period)
        high = 0 <= index < len(bits) and bits[index] == '1'
        # 4 mV INA219 bus-voltage increments with some ambient noise.
        volts = round((0.4 + high * contrast + rng.uniform(-0.006, 0.006)) / 0.004) * 0.004
        command = receiver.feed(t, volts)
        if command is not None:
            found.append(command)
    return found


class AngleDemoProtocolTest(unittest.TestCase):
    def test_one_command_per_frame_including_both_axis_boundaries(self):
        for pair in ((0, 0), (180, 35), (HOME_PAN, HOME_TILT), (HOME_PAN+1, HOME_TILT-1)):
            with self.subTest(pair=pair):
                frame = encode_angles(*pair)
                self.assertEqual(len(frame), FRAME_LEN)
                self.assertEqual(receive(frame), [pair])

    def test_refuse_invalid_inputs_and_received_angles_even_with_valid_crc(self):
        for pair in ((-1, 0), (181, 2), (0, 36), (0, -1), (1, 1.5), (True, 0)):
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                encode_angles(*pair)
        for pan, tilt in ((181, 20), (103, 36)):
            invalid = f'{SYNC}{pan:08b}{tilt:08b}{crc8(DISCRIMINATOR, pan, tilt):08b}1'
            self.assertEqual(receive(invalid), [])

    def test_damage_dropout_and_ambient_light_do_not_issue_commands(self):
        frame = encode_angles(112, 25)
        for index in (4, 17, 25, 34, 40):
            corrupt = frame[:index] + str(1 - int(frame[index])) + frame[index+1:]
            self.assertEqual(receive(corrupt), [])
        self.assertEqual(receive(frame[:-6]), [])
        self.assertEqual(receive(frame, omit=(1.18, 1.26)), [])
        self.assertEqual(receive(frame, contrast=0.006), [])
        self.assertEqual(receive('1' * FRAME_LEN), [])

    def test_angle_frames_and_legacy_arc_frames_are_distinct(self):
        self.assertEqual(receive(stimulus_bits(103, BIT_MS)), [])
        self.assertEqual(receive(encode_angles(103, 20), receiver=FrameDecoder(BIT_MS)), [])


if __name__ == '__main__':
    unittest.main()
