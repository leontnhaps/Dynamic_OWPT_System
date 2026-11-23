import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from common.ook_experiment import HEADER, PI_DECODER, PI_RECEIVER, RULES, make_plan
from Tx.Controller.ook_pi_analysis import aggregate, analyze_experiment, analyze_trial, decode, read_pi_csv


def put_json(path, obj):
    path.write_text(json.dumps(obj), encoding='utf-8')


def signal(bits, bit_ms):
    t = np.arange(0, (len(bits) + 12) * bit_ms, 2.)
    i = np.floor((t - 6 * bit_ms) / bit_ms).astype(int)
    volts = np.full(len(t), .4)
    active = (i >= 0) & (i < len(bits))
    volts[active] += .4 * np.array(list(bits), dtype=int)[i[active]]
    return t, volts


class PiAnalysisTests(unittest.TestCase):
    def test_entire_bit_bins_and_label_independent_alignment(self):
        payload = '01001011' * 4
        t, volts = signal(HEADER + payload + '0000', 20)
        # Remove an entire payload bit's samples, and leave only an edge sample in the next.
        start = (6 + len(HEADER) + 2) * 20
        keep = (t < start) | (t == start + 22) | (t >= start + 40)
        fit = decode(t[keep], volts[keep], 20)
        bits = fit['bits']
        self.assertEqual(bits[2]['decision'], '?')
        self.assertEqual(bits[3]['samples'], 1)
        self.assertEqual(''.join(b['decision'] for b in bits), payload[:2] + '?' + payload[3:])
        self.assertAlmostEqual(fit['threshold_v'], .6)
        self.assertIsNone(decode(t, np.full(len(t), .4), 20))

    def test_missing_and_invalid_frames_are_counted_separately(self):
        plan = make_plan([20], [2], 4, 22)
        plan.update(receiver=PI_RECEIVER, decoder_version=PI_DECODER)
        with TemporaryDirectory() as parent:
            root = Path(parent)
            put_json(root / 'experiment.json', plan)
            for i, spec in enumerate(plan['trials'][:3]):
                folder = root / f'trial_{i+1:05d}'
                folder.mkdir()
                settings = spec['settings']
                put_json(folder / 'config.json', dict(receiver=PI_RECEIVER, settings=settings,
                                                     request_id='scope', interval_confirmed=True,
                                                     experiment=dict(spec, rules_version=RULES['version'],
                                                                     decoder_version=PI_DECODER),
                                                     confirmed_sample_ms=2))
                sent = settings['bits']
                event = dict(event='laser_pattern', state='completed', request_id='scope',
                             trace=[dict(index=k, level=int(b)) for k, b in enumerate(sent)])
                if i == 2:
                    event['request_id'] = 'earlier-trial'
                (folder / 'events.jsonl').write_text(json.dumps(event), encoding='utf-8')
                t, volts = signal(sent, settings['bit_ms'])
                put_json(folder / 'summary.json', dict(end_reason='completed', rows=len(t), invalid_lines=i % 2,
                         receiver=PI_RECEIVER, missing_records=0, clipped_rows=0, tx_state='completed'))
                with (folder / 'rx.csv').open('w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['sample_seq', 'rx_monotonic_ns', 'bus_raw', 'pv_V'])
                    for k, (stamp, v) in enumerate(zip(t, volts)):
                        writer.writerow([k, 1000000000000000 + int(1e6 * stamp), round(v / .004), v])
            result = analyze_experiment(root)
            self.assertTrue((result / 'ook_bitrate_results.png').exists())
            self.assertTrue((result / 'waveform_2ms_20ms.png').exists())
            with (result / 'summary.csv').open(encoding='utf-8-sig', newline='') as f:
                summary = next(csv.DictReader(f))
            self.assertEqual(summary['ber_percent'], '0.0')
            self.assertEqual(summary['block_success_percent'], '50.0')
            for key in ('incomplete_tx', 'rx_invalid', 'not_started'):
                self.assertEqual(summary[key], '1')
            self.assertEqual(summary['erased_bits'], '32')
            self.assertEqual(summary['decided_bits'], '32')
            self.assertEqual(summary['evaluated_blocks'], '2')
            with (result / 'bits.csv').open(encoding='utf-8-sig', newline='') as f:
                exported = list(csv.DictReader(f))
                self.assertEqual(len(exported), 64)
                self.assertEqual(sum(b['decision'] == '?' for b in exported), 32)
            self.assertNotEqual(result, analyze_experiment(root))
            self.assertTrue((result / 'provenance.json').is_file())

    def test_no_rows_do_not_create_false_success_rates(self):
        spec = make_plan([10], [1], 1, 0)['trials'][0]
        row, _, _ = analyze_trial(Path('missing_trial'), spec)
        summary, = aggregate([row])
        self.assertIsNone(summary['ber_percent'])
        self.assertIsNone(summary['block_success_percent'])

    def test_source_timestamp_gaps_and_empty_csv(self):
        with TemporaryDirectory() as parent:
            path = Path(parent) / 'rx.csv'
            header = 'sample_seq,rx_monotonic_ns,bus_raw,pv_V\n'
            path.write_text(header, encoding='utf-8')
            t, _, stats = read_pi_csv(path)
            self.assertEqual(len(t), 0)
            self.assertIsNone(stats['p95_dt_ms'])
            path.write_text(header + '2,100000000,10,0.04\n3,105000000,20,0.08\n5,118000000,20,0.08\n', encoding='utf-8')
            t, _, stats = read_pi_csv(path)
            self.assertEqual(stats['missing_slots'], 3)
            self.assertEqual(stats['p95_dt_ms'], 5)
            self.assertEqual(stats['max_record_dt_ms'], 13)

    def test_step_uses_waveform_only_at_long_sample_interval(self):
        from Tx.Controller.pi_ook_rx import PiCapture, SAMPLE_CONFIG
        spec = make_plan([1000], [1000], 1, 11, mode='step')['trials'][0]
        with TemporaryDirectory() as parent:
            root = Path(parent)
            folder = root / 'trial_00001'
            folder.mkdir()
            active = dict(spec, rules_version=RULES['version'], decoder_version=PI_DECODER)
            capture = PiCapture(folder, spec['settings'], 'fixture', 'step-token', 1000, active)
            capture.confirm(dict(interval_ms=1000, bus=3, address=0x40, config=SAMPLE_CONFIG))
            for seq, level in enumerate(spec['settings']['bits']):
                capture.write_sample(dict(seq=seq, rx_monotonic_ns=(seq + 1) * 10**9,
                    bus_raw=200 + 100 * int(level), pc_receive_monotonic_ns=0, pc_receive_unix_ns=0))
            capture.finish('completed', 'completed')
            event = dict(event='laser_pattern', state='completed', request_id='step-token',
                         trace=[dict(index=i, level=int(b)) for i, b in enumerate(spec['settings']['bits'])])
            (folder / 'events.jsonl').write_text(json.dumps(event), encoding='utf-8')
            row, bits, wave = analyze_trial(folder, spec)
            self.assertEqual(row['status'], 'waveform_only')
            self.assertEqual(row['median_dt_ms'], 1000)
            self.assertEqual(len(wave[0]), 3)
            self.assertEqual(bits, [])
            self.assertEqual(aggregate([row]), [])


if __name__ == '__main__':
    unittest.main()
