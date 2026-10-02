import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from common.control_timing import CONTROL_PERIODS_S
from simulation.config import Config
from Tx.Controller.learning_panel import LearningPanel


class ControlTimingTests(unittest.TestCase):
    def test_new_config_roundtrip_and_legacy_period(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.json'
            self.assertEqual(Config.load().dt, 0.720)
            for period in CONTROL_PERIODS_S:
                Config(dt=period).save(path)
                self.assertEqual(Config.load(path).dt, period)
            path.write_text(json.dumps({'dt': 0.125}))
            self.assertEqual(Config.load(path).dt, 0.125)
            path.write_text('{}')
            self.assertEqual(Config.load(path).dt, 0.1)

    def test_new_model_uses_selected_period_even_with_old_config(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.json'
            Config(dt=0.1).save(path)
            for period in CONTROL_PERIODS_S:
                panel = SimpleNamespace(
                    busy=Mock(), future=None, app=SimpleNamespace(size=(1296, 972)),
                    paths={'config': Mock(get=Mock(return_value=str(path))),
                           'yolo': Mock(get=Mock(return_value='weights.pt'))},
                    new_vars={'dt': Mock(get=Mock(return_value=str(period)))},
                    device=Mock(get=Mock(return_value='cpu')), stop=Mock(),
                    learner=None, widgets=[], executor=Mock(), generation=0,
                    status=Mock(), learning_status=Mock())
                with patch('Tx.Controller.learning_panel.resolve_yolo_path', return_value='weights.pt'):
                    LearningPanel.create_new(panel)
                cfg = panel.executor.submit.call_args.args[5]
                self.assertEqual(cfg.dt, period)
                self.assertEqual(Config.load(path).dt, 0.1)


if __name__ == '__main__':
    unittest.main()
