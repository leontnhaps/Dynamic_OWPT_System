import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from common.ook_experiment import (HEADER, PI_DECODER, PI_LIMIT_PERIODS, PI_LIMIT_PRESET, PI_LIMIT_SAMPLE_MS,
                                   PI_RECEIVER, make_pi_comparison_plan, make_pi_limit_plan)
from Tx.Controller.ook_experiment_panel import analyze_saved_experiment
from tests import test_ook_panel as panel_fixture


class PresentationPresetTests(unittest.TestCase):
    def test_presentation_conditions_and_reproducibility(self):
        plan = make_pi_comparison_plan(42)
        self.assertEqual(plan, make_pi_comparison_plan(42))
        self.assertEqual(plan['receiver'], PI_RECEIVER)
        self.assertEqual(plan['decoder_version'], PI_DECODER)
        trials = plan['trials']
        periods = [trial['settings']['bit_ms'] for trial in trials]
        self.assertEqual(len(trials), 210)
        self.assertEqual(Counter(periods), dict.fromkeys((100, 50, 30, 20, 15, 14, 12), 30))
        self.assertNotEqual(periods, sorted(periods))
        self.assertEqual(sum(len(trial['payload']) for trial in trials), 6720)
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

    def test_limit_sweep_keeps_tx_bit_period_and_rx_sample_interval_independent(self):
        plan = make_pi_limit_plan(PI_LIMIT_PERIODS, PI_LIMIT_SAMPLE_MS, 123)
        self.assertEqual(plan, make_pi_limit_plan('12, 10, 8, 6, 5', '2', 123))
        self.assertEqual(plan['preset'], PI_LIMIT_PRESET)
        self.assertEqual(plan['decoder_version'], PI_DECODER)
        self.assertEqual(plan['receiver'], PI_RECEIVER)
        settings = [(t['settings']['bit_ms'], t['sample_ms']) for t in plan['trials']]
        self.assertEqual(Counter(settings), {(ms, 2): 30 for ms in (12, 10, 8, 6, 5)})
        self.assertNotEqual(settings, sorted(settings))
        self.assertTrue(all(len(t['payload']) == 32 for t in plan['trials']))
        repeat = make_pi_limit_plan('5,4.5', 5)
        self.assertEqual({(t['settings']['bit_ms'], t['sample_ms']) for t in repeat['trials']},
                         {(5, 5), (4.5, 5)})
        for periods, interval in [('5,5', 2), ('0', 2), ('12', '2.5'), ('12', 0)]:
            with self.subTest(periods=periods, interval=interval), self.assertRaises(ValueError):
                make_pi_limit_plan(periods, interval)

    def test_final_tab_starts_pi_only_plan_without_using_hidden_sweep_inputs(self):
        panel = panel_fixture.PanelStateTests().panel()
        panel.check_ready = lambda: None
        panel.arduino = SimpleNamespace(connected=True)
        panel.hardware = {'simulated': False, 'capabilities': ['laser_pattern']}
        panel.sweep_seed.set('previous invalid input')
        with tempfile.TemporaryDirectory() as directory:
            panel.app.stage_dir = lambda stage: Path(directory)
            with self.assertRaises(ValueError):
                panel.start_pi_limit('5', '2')
            panel.arduino.source = PI_RECEIVER
            panel.analysis_events.put((True, Path(directory) / 'older_analysis'))
            panel.start_pi_limit('5', '2', note='point-1')
            manifest = json.loads((panel.experiment_root / 'experiment.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['note'], 'point-1')
            self.assertEqual(manifest['preset'], PI_LIMIT_PRESET)
            self.assertEqual(len(manifest['trials']), 30)
            self.assertIsNone(panel.last_analysis)
            self.assertTrue(panel.analysis_events.empty())
            panel.cancel_experiment('operator_stop')
            result = panel.last_experiment / 'analysis'
            panel.analysis_events.put((True, result))
            panel.tick_experiment()
            self.assertEqual(panel.last_analysis, result)
            panel.start_pi_comparison()
            self.assertEqual(len(panel.experiment['trials']), 210)
            self.assertEqual(panel.sweep_seed.get(), 'previous invalid input')
