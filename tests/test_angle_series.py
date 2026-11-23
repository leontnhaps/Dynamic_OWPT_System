import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from Tx.Controller.ook_panel import OokPanel
from tools.analysis.angle_series import collect, main


class AngleSeriesTest(unittest.TestCase):
    def test_unmatched_runs_are_listed_without_an_optical_score(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rx = root/'angle_one'
            rx.mkdir()
            (rx/'manifest.json').write_text(json.dumps(dict(pair='series-000')))
            (root/'arc_tx_other.jsonl').write_text(json.dumps(dict(event='arc_start', pair_id='series-030')))
            summaries, trials = collect(root, root, 'series-')
            self.assertEqual([row['pair'] for row in summaries], ['series-000', 'series-030'])
            self.assertTrue(all(row['issue'] and 'received' not in row for row in summaries))
            self.assertEqual(trials, [])
            self.assertEqual(main(['--prefix','series-', '--rx-root',folder,'--tx-root',folder,'--out-root',folder]), 1)
            path, = root.glob('angle_series_*/angles.csv')
            with path.open(encoding='utf-8-sig', newline='') as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2)

    def test_requested_preset_uses_existing_guard_and_sender(self):
        panel = Mock()
        OokPanel.start_angle_reception(panel)
        panel.check_ready.assert_called_once_with()
        panel.arc_bit_ms.set.assert_called_once_with('12')
        panel.arc_period_s.set.assert_called_once_with('2.5')
        panel.arc_steps.set.assert_called_once_with('30')
        panel.arc_mode.set.assert_called_once_with('A')
        panel.start_arc.assert_called_once_with()
