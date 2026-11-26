import csv
import json
import tempfile
import unittest
from pathlib import Path

from Tx.Controller.pv_comparison import ComparisonRun


class VoltageComparisonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run = ComparisonRun(Path(temporary.name), {'point_id': 'A', 'xyz_m': [2, 3, 5]})
        self.addCleanup(self.close_trial)

    def close_trial(self):
        if self.run.trial_folder:
            self.run.finish(False, 'test end')

    def sample_window(self, code, indices=range(2000), increment_ns=5_000_000):
        start = 21_000_000_000
        indices = list(indices)
        for i in indices:
            self.run.sample(dict(kind='voltage', token='window', seq=100+i,
                                 rx_monotonic_ns=start+i*increment_ns, bus_raw=code+i%2))
        return dict(kind='measured', token='window', count=len(indices),
                    window_start_ns=start, window_end_ns=start+10_000_000_000)

    def trial(self, method, code, laser_on=True):
        self.run.start(method, {'rx_pan_deg': 103, 'rx_tilt_deg': 20}, laser_on=laser_on)
        self.run.measured(self.sample_window(code))
        return self.run.finish(True, 'OK')

    def test_pairing_and_raw_evidence(self):
        baseline = self.trial('baseline', 200)
        self.assertAlmostEqual(baseline['mean_v'], .802)
        self.assertGreater(baseline['sd_v'], 0)
        result = self.trial('xyz', 230)
        self.assertAlmostEqual(result['delta_v'], .120)
        self.assertEqual(result['baseline_trial_id'], baseline['trial_id'])
        folder = self.run.folder / result['trial_id']
        with (folder/'voltage.csv').open(encoding='utf-8', newline='') as stream:
            samples = list(csv.DictReader(stream))
        self.assertEqual(len(samples), 2000)
        self.assertEqual(samples[0]['pv_v'], '0.920')
        self.assertEqual(samples[-1]['seq'], '2099')
        self.assertEqual(json.loads((folder/'events.jsonl').read_text())['kind'], 'measured')
        replacement = self.trial('baseline', 225)
        self.assertEqual(self.trial('yolo', 220)['baseline_trial_id'], replacement['trial_id'])

    def test_incomplete_attempt_never_supplies_paired_mean(self):
        first = self.run.start('baseline', {}, laser_on=True)
        self.run.log({'event': 'pose', 'pan': 103, 'tilt': 20})
        self.run.save_frame(b'snapshot')
        with (self.run.folder/'summary.csv').open(encoding='utf-8', newline='') as stream:
            self.assertEqual(next(csv.DictReader(stream))['status'], 'running')
        self.sample_window(2, range(3))
        row = self.run.finish(True, 'OK')
        self.assertFalse(row['success'])
        self.assertIsNone(row['mean_v'])
        self.assertEqual(row['n'], 3)
        self.assertEqual((self.run.folder/first/'frame.jpg').read_bytes(), b'snapshot')
        row = self.trial('xyz', 240)
        self.assertTrue(row['success'])
        self.assertIsNone(row['delta_v'])
        with (self.run.folder/'summary.csv').open(encoding='utf-8', newline='') as stream:
            self.assertEqual([r['status'] for r in csv.DictReader(stream)], ['failed', 'complete'])

    def test_xyz_transport_is_saved_with_the_same_voltage_condition(self):
        baseline = self.trial('baseline', 200, False)
        for transport in ('ook', 'ssh'):
            self.run.start('xyz', {'pose_source': {'transport': transport}}, laser_on=False)
            self.run.measured(self.sample_window(230))
            result = self.run.finish(True, 'OK')
            self.assertEqual(result['baseline_trial_id'], baseline['trial_id'])
            path = self.run.folder / result['trial_id'] / 'manifest.json'
            self.assertEqual(json.loads(path.read_text(encoding='utf-8'))['pose_transport'], transport)
        with (self.run.folder/'summary.csv').open(encoding='utf-8', newline='') as stream:
            records = list(csv.DictReader(stream))
        self.assertEqual([(r['method'], r['pose_transport']) for r in records],
                         [('baseline', 'ssh'), ('xyz', 'ook'), ('xyz', 'ssh')])

    def test_short_gaps_are_saved_and_excessive_gaps_rejected(self):
        self.run.start('baseline', {}, laser_on=False)
        self.run.measured(self.sample_window(150, [i for i in range(2000) if i != 20]))
        self.assertEqual(self.run.finish(True, 'OK')['n'], 1999)
        self.run.start('xyz', {}, laser_on=False)
        receipt = self.sample_window(200, [i for i in range(2000) if not 30 <= i < 45])
        with self.assertRaises(ValueError):
            self.run.measured(receipt)
        self.assertFalse(self.run.finish(True, 'OK')['success'])

    def test_mismatched_receipt_and_out_of_order_stream_are_preserved_as_failures(self):
        self.run.start('xyz', {}, laser_on=True)
        receipt = self.sample_window(200)
        with self.assertRaises(ValueError):
            self.run.measured(dict(receipt, count=199))
        self.assertFalse(self.run.finish(True, 'OK')['success'])
        self.run.start('yolo', {}, laser_on=True)
        self.sample_window(100, range(1))
        with self.assertRaises(ValueError):
            self.run.sample(dict(kind='voltage', token='wrong', seq=100, rx_monotonic_ns=0, bus_raw=3))
        self.assertFalse(self.run.finish(True, 'OK')['success'])

    def test_saturated_measurement_is_preserved_in_the_failed_trial(self):
        self.run.start('baseline', {}, laser_on=True)
        folder = self.run.trial_folder
        with self.assertRaises(ValueError):
            self.run.sample(dict(kind='voltage', token='sample', seq=0, rx_monotonic_ns=500, bus_raw=8191))
        self.assertFalse(self.run.finish(True, 'OK')['success'])
        self.assertIn('8191', (folder/'voltage.csv').read_text())

    def test_operator_laser_conditions_pair_only_with_the_matching_baseline(self):
        on = self.trial('baseline', 300, True)
        self.assertIsNone(self.trial('xyz', 100, False)['baseline_trial_id'])
        off = self.trial('baseline', 90, False)
        self.run.start('baseline', {}, laser_on=False)
        self.run.finish(False, 'light changed')
        for method in ('xyz', 'yolo'):
            for light, reference in ((False, off), (True, on)):
                row = self.trial(method, 200, light)
                self.assertEqual(row['baseline_trial_id'], reference['trial_id'])
                self.assertAlmostEqual(row['delta_v'], row['mean_v'] - reference['mean_v'])
                self.assertIs(row['laser_on'], light)
                self.assertEqual((row['rx_pan_deg'], row['rx_tilt_deg']), (103, 20))
        with self.assertRaises(ValueError):
            self.run.start('baseline', {}, laser_on=None)
        self.run.log_station({'event': 'pose_confirmed'})
        self.assertEqual(json.loads((self.run.folder/'events.jsonl').read_text())['event'],
                         'pose_confirmed')
