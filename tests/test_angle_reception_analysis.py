import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from common.arc_protocol import FRAME_LEN, VERSION
from tools.analysis.angle_reception import analyze
from Tx.Controller.arc_sender import ArcSender


class AngleReceptionAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.period = 2.5
        self.bit = .012
        self.origin = 1002.0
        self.meta = dict(schema_version=1, kind='static_angle_reception', profile=VERSION,
                         steps=30, sample_ms=5, status='completed', operator_bracketed=True,
                         angle_deg=25, pan_deg=155, bit_ms=12, period_s=self.period, pair='angle25')
        self.tx = self.root / 'tx.jsonl'
        self.write_samples()
        self.write_events([])

    def write_samples(self, missing_index=None):
        with (self.root / 'samples.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['sample_index', 'monotonic_s', 'voltage_v', 'missing_before'])
            for i in range(16101):
                if i != missing_index:
                    writer.writerow((i, f'{1000+i*.005:.6f}', 1.0, int(i-1 == missing_index)))
        self.meta.update(first_sample_s=1000, last_sample_s=1080.5,
                         sample_count=16101-int(missing_index is not None),
                         missing_samples=int(missing_index is not None))
        self.write_meta()

    def write_meta(self):
        (self.root / 'manifest.json').write_text(json.dumps(self.meta), encoding='utf-8')

    def write_events(self, events):
        with (self.root / 'frames.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['decoded_at_s', 'frame_start_s', 'sequence'])
            for seq, offset in sorted(events, key=lambda item: item[1]):
                onset = self.origin + offset
                writer.writerow([onset+self.bit*(FRAME_LEN-.10), onset, seq])

    def mocked(self):
        with patch('tools.analysis.angle_reception.audit_tx', return_value={'completed_frames': 31}):
            return analyze(self.root, self.tx)

    def test_marker_independent_and_all_dark(self):
        rows, summary = self.mocked()
        self.assertEqual((summary['success_percent'], summary['missed'], summary['timing_basis']),
                         (0., 30, 'operator_bracket'))
        self.assertEqual(len(rows), 30)
        self.write_events([(2, 2*self.period), (30, 30*self.period)])
        rows, summary = self.mocked()
        self.assertEqual(summary['timing_basis'], 'inferred_sequence')
        self.assertEqual([r['sequence'] for r in rows if r['status']=='received'], [2, 30])
        self.assertFalse(summary['marker_received'])

    def test_gaps_are_not_optical_failures(self):
        self.write_samples(missing_index=int((self.origin+7*self.period+.1-1000)/.005))
        self.assertEqual(self.mocked()[1]['indeterminate'], 30)
        self.write_events([(0, 0), (1, self.period)])
        rows, summary = self.mocked()
        self.assertEqual(summary['indeterminate'], 1)
        self.assertEqual(rows[6]['status'], 'indeterminate')
        self.assertIsNone(summary['success_percent'])

    def test_wrong_slot_duplicate_and_ambiguous_anchor(self):
        self.write_events([(0, 0), (2, self.period), (2, 2*self.period), (2, 2*self.period+.05),
                           (90, 3*self.period)])
        _, s = self.mocked()
        self.assertEqual((s['received'], s['wrong_slot_events'], s['duplicate_events'],
                          s['unexpected_events']), (1, 1, 1, 1))
        self.write_events([(2, 2*self.period), (4, 4*self.period+1)])
        _, s = self.mocked()
        self.assertEqual((s['timing_basis'], s['indeterminate']), ('ambiguous', 30))

    def test_incomplete_bracket_rejected(self):
        self.meta['operator_bracketed'] = False
        self.write_meta()
        with self.assertRaisesRegex(ValueError, 'bracketed'):
            self.mocked()

    def test_missing_event_file_header_does_not_count_as_zero_receptions(self):
        (self.root / 'frames.csv').write_text('')
        with self.assertRaisesRegex(ValueError, 'frame CSV header'):
            self.mocked()

    def test_actual_sender_log_required_for_zero_result(self):
        tx_messages, log = [], []
        now = 0.0
        def send(command):
            tx_messages.append(command)
            return True
        sender = ArcSender(send, log.append, lambda _: None, 12, self.period, 30,
                           pair_id='angle25', mode='A', clock=lambda: now)
        sender.start()
        sender.handle_event(dict(event='laser', request_id=tx_messages[-1]['request_id'], laser_gpio_level=0))
        for seq in range(31):
            now = sender.epoch + seq*self.period
            sender.tick(True)
            cmd = tx_messages[-1]
            sender.handle_event(dict(event='laser_pattern', request_id=cmd['request_id'], state='completed',
                                     off_call_done_ms=len(cmd['bits'])*12,
                                     settings={k:cmd[k] for k in ('bit_ms','bits','repeats')},
                                     trace=[dict(index=i,level=int(b)) for i,b in enumerate(cmd['bits'])]))
        sender.handle_event(dict(event='laser', request_id=tx_messages[-1]['request_id'], laser_gpio_level=0))
        self.tx.write_text(''.join(json.dumps(item)+'\n' for item in log))
        self.assertEqual(analyze(self.root, self.tx)[1]['tx_audit']['completed_frames'], 31)
        self.tx.write_text(''.join(json.dumps(item)+'\n' for item in log[:-1]))
        with self.assertRaisesRegex(ValueError, 'missing completed'):
            analyze(self.root, self.tx)


if __name__ == '__main__':
    unittest.main()
