import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from common.arc_protocol import stimulus_bits
from Rx import angle_reception as rx


class Rows(list):
    def writerow(self, row):
        self.append(row)


class StaticAngleTest(unittest.TestCase):
    def test_optional_marker_empty_signal_and_missing_frames(self):
        for decoded_command in (None, 7):
            with self.subTest(seq=decoded_command):
                cfg = rx.settings(75, 'a12-075', 12, 2.5)
                frames = Rows()
                rec = rx.Reception(cfg, Rows(), frames)
                bits = stimulus_bits(7, 12)
                start_ms = 7 * 2500
                for i in range(15501):
                    symbol = (i*5-start_ms) // 12
                    on = decoded_command and 0 <= symbol < len(bits) and bits[symbol] == '1'
                    rec.observe(i, 10_000_000_000 + i*5_000_000, 400 if on else 50)
                rec.finish()
                self.assertEqual([row[-1] for row in frames], [] if decoded_command is None else [7])
                self.assertTrue(cfg['operator_bracketed'])
                self.assertTrue(rec.stats()['sampling_contiguous'])
                self.assertEqual(cfg['pan_deg'], 105)

    def test_short_stalled_or_clipped_acquisition(self):
        cfg = rx.settings(10, 'a12-010', 12, 2.5)
        rec = rx.Reception(cfg, Rows(), Rows())
        rec.observe(0, 1_000_000_000, 10)
        with self.assertRaises(RuntimeError):
            rec.finish()
        rec.observe(18000, 91_000_000_000, 10)
        self.assertEqual(rec.stats()['missing_samples'], 17999)
        with self.assertRaises(RuntimeError):
            rec.finish()
        self.assertFalse(cfg['operator_bracketed'])
        self.assertFalse(rec.stats()['sampling_contiguous'])
        with self.assertRaises(RuntimeError):
            rec.observe(18001, 91_005_000_000, 8191)

    def test_angle_and_feasible_period_validation(self):
        for angle in (-1, 181):
            with self.assertRaises(ValueError):
                rx.settings(angle, 'bench', 12, 2.5)
        with self.assertRaises(ValueError):
            rx.settings(90, 'bench', 50, 2.5)
        cfg = rx.settings(90, 'bench', 50, 4)
        self.assertEqual((cfg['steps'], cfg['pan_deg'], cfg['tilt_deg']), (30, 90, 20))
        self.assertAlmostEqual(cfg['minimum_capture_s'], 122.95)
        with patch.object(rx, 'run') as no_hardware, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(rx.main(['--angle', '0', '--pair', 'a12-000', '--plan']), 0)
            no_hardware.assert_not_called()

    def test_interrupt_holds_single_pose_then_releases_and_persists(self):
        bus = MagicMock()
        bus.write.side_effect = len
        def bus_pair(_):
            return contextlib.nullcontext(bus)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(rx, 'i2c_bus', side_effect=bus_pair), patch.object(rx, 'INA219'), \
                    patch('builtins.input', side_effect=['START', '']), \
                    patch.object(rx, 'collect', side_effect=KeyboardInterrupt('test')), \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
                rx.run(rx.settings(45, 'a12-045', 12, 2.5), root=tmp)
            folder, = (Path(tmp)/'captures'/'M6-4').iterdir()
            manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
            self.assertFalse(manifest['operator_bracketed'])
            self.assertTrue(manifest['pwm_off_writes_completed'])
            self.assertEqual(manifest['status'], 'aborted')
            with (folder/'commands.csv').open(encoding='utf-8') as handle:
                cmds = list(csv.DictReader(handle))
            self.assertEqual([(r['channel'], r['servo_deg']) for r in cmds if r['action'] == 'hold_angle'],
                             [('1', '20'), ('0', '135')])
            self.assertEqual([r['action'] for r in cmds[-2:]], ['pwm_off', 'pwm_off'])


if __name__ == '__main__':
    unittest.main()
