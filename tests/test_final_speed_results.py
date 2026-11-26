import csv
import json
import tempfile
import unittest
from pathlib import Path

from common.ook_experiment import PI_DECODER, PI_RECEIVER, RULES
from Tx.Controller.final_speed_results import SPEED_COLUMNS, load_speed_results


def condition(bit_ms, sample_ms=5, **changes):
    row = dict(bit_ms=bit_ms, bitrate_bps=1000 / bit_ms, sample_ms=sample_ms, planned=30,
               not_started=0, incomplete_tx=0, evaluated_blocks=30, correct_blocks=30,
               sync_failed=0, decided_bits=960, error_bits=0, erased_bits=0)
    return dict(row, **changes)


class FinalSpeedResultTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)

    def save(self, rows, **metadata):
        provenance = dict(receiver=PI_RECEIVER, decoder_version=PI_DECODER, source_rules=RULES,
                          planned_trials=sum(r['planned'] for r in rows),
                          saved_trials=sum(r['planned'] - r['not_started'] for r in rows))
        (self.path/'provenance.json').write_text(json.dumps(dict(provenance, **metadata)), encoding='utf-8')
        with (self.path/'summary.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=condition(12).keys())
            writer.writeheader()
            writer.writerows(rows)

    def test_fastest_all_thirty_and_partial_runs_are_distinct(self):
        self.save([condition(12), condition(50),
                   condition(5, not_started=3, incomplete_tx=2, evaluated_blocks=25,
                             correct_blocks=22, sync_failed=1, decided_bits=768,
                             error_bits=2, erased_bits=32)])
        headline, rows = load_speed_results(self.path)
        self.assertIn('83.33 bit/s', headline)
        self.assertIn('Rx 5 ms', headline)
        self.assertEqual(rows[-1], ('5', '200', '5', '27/30', '25', '2', '22/25', '1',
                                    '2/768', '32', '0.2604'))
        self.assertEqual(len(SPEED_COLUMNS), len(rows[0]))

    def test_undecided_bits_have_no_ber_and_not_a_zero_error_success(self):
        self.save([condition(5, correct_blocks=0, sync_failed=30, decided_bits=0, erased_bits=960)])
        headline, rows = load_speed_results(self.path)
        self.assertIn('없습니다', headline)
        self.assertEqual(rows[0][-1], '—')

    def test_matching_arduino_or_inconsistent_summary_cannot_label_pi_success(self):
        for rows, overrides in [([condition(12)], {'receiver': 'arduino'}),
                                ([condition(12)], {'saved_trials': 20}),
                                ([condition(12, error_bits=1)], {}),
                                ([condition(12), condition(12)], {})]:
            with self.subTest(rows=rows, overrides=overrides):
                self.save(rows, **overrides)
                with self.assertRaises(ValueError):
                    load_speed_results(self.path)
