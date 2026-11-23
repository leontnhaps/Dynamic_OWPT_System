import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from common.ook_experiment import HEADER, PI_DECODER, PI_RECEIVER, make_pi_comparison_plan
from Tx.Controller.ook_experiment_panel import analyze_saved_experiment


class PresentationPresetTests(unittest.TestCase):
    def test_presentation_conditions_and_reproducibility(self):
        plan = make_pi_comparison_plan(42)
        self.assertEqual(plan, make_pi_comparison_plan(42))
        self.assertEqual(plan['receiver'], PI_RECEIVER)
        self.assertEqual(plan['decoder_version'], PI_DECODER)
        trials = plan['trials']
        periods = [trial['settings']['bit_ms'] for trial in trials]
        self.assertEqual(Counter(periods), {100: 10, 50: 10, 30: 10, 20: 30, 15: 30, 14: 30, 12: 30})
        self.assertNotEqual(periods, sorted(periods))
        self.assertEqual(sum(len(trial['payload']) for trial in trials), 4800)
        for trial in trials:
            self.assertEqual(trial['sample_ms'], 5)
            self.assertEqual(trial['settings']['bits'], HEADER + trial['payload'] + '0000')

    def test_saved_manifest_selects_decoder_independently_of_connected_receiver(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / 'experiment.json'
            for header, module in [({}, 'ook_analysis'), ({'receiver': PI_RECEIVER}, 'ook_pi_analysis')]:
                manifest.write_text(json.dumps(header), encoding='utf-8')
                with patch(f'Tx.Controller.{module}.analyze_experiment', return_value=root) as decode:
                    self.assertEqual(analyze_saved_experiment(root), root)
                    decode.assert_called_once_with(root)
            manifest.write_text('{"receiver":"unknown"}', encoding='utf-8')
            with self.assertRaises(ValueError):
                analyze_saved_experiment(root)
