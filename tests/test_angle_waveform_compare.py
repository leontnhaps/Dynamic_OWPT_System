import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from common.ook_experiment import HEADER, PI_DECODER, PI_RECEIVER, RULES, make_plan
from Tx.Controller.angle_waveform_compare import _angle_header, _reference, compare_angle_waveform
from tests.test_ook_pi_analysis import put_json, signal


def save_wave(folder, t, v):
    with (folder / 'rx.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_seq', 'rx_monotonic_ns', 'bus_raw', 'pv_V'])
        for i, (ms, volts) in enumerate(zip(t, v)):
            writer.writerow([i, 300_000_000_000 + int(ms * 1e6), round(volts / .004), volts])


def basic_trial(root, name, bit_ms, sample_ms, simulated=False):
    folder = root / name / 'trial_00001'
    folder.mkdir(parents=True)
    plan = make_plan([bit_ms], [sample_ms], 1, 31)
    plan.update(receiver=PI_RECEIVER, decoder_version=PI_DECODER)
    put_json(folder.parent / 'experiment.json', plan)
    spec = plan['trials'][0]
    settings = spec['settings']
    put_json(folder / 'config.json',
             dict(receiver=PI_RECEIVER, settings=settings, request_id='test', interval_confirmed=True,
                  confirmed_sample_ms=sample_ms,
                  experiment=dict(spec, rules_version=RULES['version'], decoder_version=PI_DECODER)))
    event = dict(event='laser_pattern', state='completed', request_id='test', simulated=simulated,
                 trace=[dict(index=i, level=int(b)) for i, b in enumerate(settings['bits'])])
    (folder / 'events.jsonl').write_text(json.dumps(event), encoding='utf-8')
    t, v = signal(settings['bits'], bit_ms)
    save_wave(folder, t, v)
    put_json(folder / 'summary.json', dict(receiver=PI_RECEIVER, rows=len(t), invalid_lines=0,
             end_reason='completed', tx_state='completed', missing_records=0, clipped_rows=0))
    return folder


class AngleWaveformTests(unittest.TestCase):
    def test_verified_reference_prefers_matching_sample_rate_then_recency(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            basic_trial(root, 'experiment_01', 8, 1)
            same = basic_trial(root, 'experiment_02', 8, 1)
            other_sampling = basic_trial(root, 'experiment_03', 8, 5)
            basic_trial(root, 'experiment_04', 8, 1, simulated=True)
            self.assertEqual(_reference(root, 8, 1)[0], same)
            self.assertEqual(_reference(root, 8, 5)[0], other_sampling)
            self.assertIsNone(_reference(root, 10, 1))

    def test_failed_trace_cannot_gain_header_origin_from_matching_speed_reference(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ref = basic_trial(root, 'experiment_01', 8, 1)
            attempt = root / 'angle_01'
            attempt.mkdir()
            t, v = signal(HEADER + '0' * 36, 8)
            save_wave(attempt, t, np.full_like(v, .4))
            put_json(attempt / 'summary.json', dict(waveform_complete=True, angle_verified=False,
                                                    end_reason='decode timeout'))
            image, message = compare_angle_waveform(root, attempt, 8)
            metadata = json.loads((attempt / 'comparison.json').read_text(encoding='utf-8'))
            self.assertTrue(image.read_bytes().startswith(b'\x89PNG'))
            self.assertIn('헤더 미확인', message)
            self.assertIsNone(metadata['angle_header_origin_ms'])
            self.assertTrue(metadata['waveform_complete'])
            self.assertFalse(metadata['angle_verified'])
            self.assertEqual(metadata['end_reason'], 'decode timeout')
            self.assertEqual(metadata['reference']['csv'], (ref.relative_to(root) / 'rx.csv').as_posix())
            self.assertEqual(metadata['angle_samples'], len(t))
            self.assertAlmostEqual(_angle_header(t, v, 8), 6 * 8, delta=2)

    def test_no_samples_and_no_reference_are_explicit(self):
        with TemporaryDirectory() as temp:
            path = Path(temp)
            save_wave(path, [], [])
            image, message = compare_angle_waveform(path, path, 1)
            self.assertTrue(image.is_file())
            self.assertIn('성공 기록 없음', message)
            meta = json.loads((path / 'comparison.json').read_text(encoding='utf-8'))
            self.assertIsNone(meta['reference'])
            self.assertEqual(meta['angle_samples'], 0)
