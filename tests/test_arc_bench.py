import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

from common.arc_protocol import FrameDecoder, measure_window, stimulus_bits
from Rx import arc_bench


class Rows(list):
    def writerow(self, row):
        self.append(row)


def setup_trial(mode='corrected', steps=2):
    params = dict(mode=mode, bit_ms=50, period_s=4, window_s=measure_window(50),
                  steps=steps, sync_timeout_s=12)
    steps_out, samples, frames, moves = Rows(), Rows(), Rows(), []
    trial = arc_bench.Trial(params, [steps_out, samples, frames], FrameDecoder(50), moves.append)
    return trial, steps_out, samples, frames, moves


class ArcBenchTests(unittest.TestCase):
    def test_complete_waveforms_with_missed_frame_kept_and_no_motion_replay(self):
        for mode, moves_expected in [('corrected', [179, 178, 179]), ('baseline', [179, 178])]:
            with self.subTest(mode=mode):
                trial, steps, raw, frames, moves = setup_trial(mode)
                packets = [stimulus_bits(k, 50) for k in range(3)]
                # Fixed ADC readings and synthetic optical frames. Silence in slot 1.
                for index in range(2700):
                    slot, offset = divmod(index - 200, 800)
                    illuminated = (0 <= slot <= 2 and slot != 1 and
                                   offset // 10 < len(packets[slot]) and packets[slot][offset // 10] == '1')
                    with redirect_stdout(io.StringIO()):
                        trial.sample(index, 20.0 + index*0.005, 500 if illuminated else 25)
                    if trial.completed:
                        break
                self.assertTrue(trial.completed)
                self.assertEqual([row[1] for row in steps], [180, 179, 178])
                self.assertEqual([row[5] for row in steps], [1, 0, 1])
                self.assertEqual([row[6] for row in steps], ['detected', 'scheduled', 'detected'])
                self.assertEqual([row[10] for row in steps], ['ok']*3)
                self.assertEqual([float(row[9]) for row in steps], [2.0, 0.1, 2.0])
                self.assertEqual([row[-1] for row in frames], ['sync', 'accepted'])
                self.assertEqual(moves, moves_expected)
                self.assertGreater(len(raw), 2400)

    def test_duplicate_out_of_slot_and_deadline_never_replay_motion(self):
        trial, _, _, frames, moves = setup_trial()
        trial.frame(0, 10., 11.6)
        trial.previous_time, trial.previous_index = 13.995, 19
        with redirect_stdout(io.StringIO()):
            trial.sample(20, 14.0, 50)
        trial.frame(1, 13.8, 15.4)
        trial.frame(0, 14., 15.6)
        trial.frame(1, 14., 15.6)
        trial.frame(1, 14., 15.65)
        self.assertEqual([row[-1] for row in frames],
                         ['sync', 'outside_slot', 'outside_slot', 'accepted', 'duplicate'])
        with self.assertRaisesRegex(RuntimeError, 'complete slot|deadline'):
            trial.sample(1000, 18.5, 50)
        self.assertEqual(moves, [179, 180])

    def test_gap_clipping_and_partial_coverage(self):
        trial, steps, _, _, _ = setup_trial()
        trial.frame(0, 0., 1.6)
        for index in range(360, 750):
            if index not in (450, 451):
                trial.sample(index, index * .005, 234)
        with redirect_stdout(io.StringIO()):
            trial.finish_slot()
        self.assertEqual((steps[0][8], steps[0][10]), (2, 'sample_gap'))
        with self.assertRaisesRegex(RuntimeError, 'clipped'):
            trial.sample(750, 3.75, 8191)
        trial, steps, _, _, _ = setup_trial()
        trial.frame(0, 0., 1.6)
        with redirect_stdout(io.StringIO()):
            trial.finish_slot(partial=True)
        self.assertEqual(steps[0][10], 'partial')
        self.assertEqual(steps[0][9], '')

    def test_delayed_servo_after_io_is_not_applied(self):
        trial, _, _, _, moves = setup_trial()
        trial.frame(0, 10., 11.6)
        trial.previous_time, trial.previous_index = 13.995, 0
        trial.clock = lambda: 14.3  # e.g. a late CSV flush
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'servo write deadline'):
            trial.sample(1, 14., 25)
        self.assertEqual(moves, [])

    def test_interrupt_persists_aborted_manifest_and_disables_both_pwm(self):
        trial, *_ = setup_trial()
        args = trial.params
        fake_bus = MagicMock()
        fake_bus.write.side_effect = len
        resources = [MagicMock(), MagicMock()]
        for context in resources:
            context.__enter__.return_value = fake_bus
        with tempfile.TemporaryDirectory() as root:
            with patch.object(arc_bench, 'ROOT', Path(root)), patch.object(
                arc_bench, 'i2c_bus', side_effect=resources
            ), patch.object(arc_bench, 'INA219'), patch('builtins.input', side_effect=['START', '']), patch.object(
                arc_bench, 'acquire', side_effect=KeyboardInterrupt('test')
            ), redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
                arc_bench.run(args)
            directory, = (Path(root)/'captures'/'M6-4').iterdir()
            manifest = json.loads((directory/'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'aborted')
            self.assertIs(manifest['pwm_off_writes_completed'], True)
            self.assertEqual([c.args[0] for c in fake_bus.write.call_args_list[-2:]],
                             [bytes((0x09, 0x10)), bytes((0x0D, 0x10))])
            with (directory/'samples.csv').open() as sample_file:
                self.assertIn('monotonic_s', next(csv.reader(sample_file)))
            with (directory/'commands.csv').open() as command_file:
                commands = list(csv.DictReader(command_file))
            self.assertEqual([(r['channel'], r['servo_deg']) for r in commands if r['action'] == 'initial_pose'],
                             [('1', '20'), ('0', '180')])
            self.assertTrue(all(float(r['write_complete_s']) >= float(r['write_start_s']) for r in commands))


if __name__ == '__main__':
    unittest.main()
