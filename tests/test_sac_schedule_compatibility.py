"""Resume learned SAC state when an old Python schedule cannot be unpickled."""
import base64
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

import numpy as np

from common.tx_moving import MovingConfig
from Tx.Controller.stationary_learning import SACLearner, RunLog, digest, dump


@unittest.skipUnless(importlib.util.find_spec('stable_baselines3') and importlib.util.find_spec('torch'),
                     'PyTorch / Stable-Baselines3 unavailable')
class ScheduleCompatibilityTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = MovingConfig(visible_bonus=1., loss_mode='confirmed_missing_terminal')
        self.obs = np.zeros(10, np.float32)
        self.learner = SACLearner(self.cfg, seed=7)
        self.learner.add(self.obs, np.zeros(2), .2, self.obs, True)
        log = RunLog(self.root/'source', {})
        try:
            report = self.learner.finish_episode(
                dict(episode=1, reason='manual_episode_end', new_transitions=1), log, 1.)
        finally:
            log.close()
        self.source = Path(report['checkpoint'])
        # Recreate the old SB3 save format, including its pickled schedule.
        self.learner.policy.save(self.source/'model.zip')
        self.rehash()

    def rehash(self):
        manifest = json.loads((self.source/'manifest.json').read_text())
        manifest['files']['model.zip'] = digest(self.source/'model.zip')
        dump(self.source/'manifest.json', manifest)

    def assert_tree_equal(self, left, right):
        import torch
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left: self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right): self.assert_tree_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_unreadable_legacy_schedule_preserves_weights_optimizers_and_resumes_training(self):
        from stable_baselines3 import SAC
        from stable_baselines3.common import save_util
        import torch
        with zipfile.ZipFile(self.source/'model.zip') as archive:
            metadata = json.loads(archive.read('data'))
        payload = base64.b64decode(metadata['lr_schedule'][':serialized:'])
        original_loads = save_util.cloudpickle.loads

        def incompatible_loads(value, *args, **kwargs):
            if value == payload:
                raise TypeError('code expected at most 16 arguments, got 18')
            return original_loads(value, *args, **kwargs)

        weights = deepcopy(self.learner.policy.policy.state_dict())
        optimizers = [deepcopy(opt.state_dict()) for opt in (self.learner.policy.actor.optimizer,
                      self.learner.policy.critic.optimizer, self.learner.policy.ent_coef_optimizer)]
        before = {p.name: digest(p) for p in self.source.iterdir()}
        with patch.object(save_util.cloudpickle, 'loads', side_effect=incompatible_loads):
            with warnings.catch_warnings(record=True) as old_warnings:
                warnings.simplefilter('always')
                old = SAC.load(self.source/'model.zip', device='cpu')
            self.assertTrue(any('Could not deserialize object lr_schedule' in str(w.message)
                                for w in old_warnings))
            del old
            for mode in ('resume', 'evaluate'):
                with warnings.catch_warnings(record=True) as new_warnings:
                    warnings.simplefilter('always')
                    restored = SACLearner(self.cfg, mode, self.source, device='cpu')
                self.assertFalse(any('Could not deserialize' in str(w.message) for w in new_warnings))
                self.assert_tree_equal(weights, restored.policy.policy.state_dict())
                for expected, actual in zip(optimizers, (restored.policy.actor.optimizer,
                        restored.policy.critic.optimizer, restored.policy.ent_coef_optimizer)):
                    self.assert_tree_equal(expected, actual.state_dict())
                self.assertTrue(torch.equal(self.learner.policy.log_ent_coef, restored.policy.log_ent_coef))
                for progress in (0., .5, 1.):
                    self.assertEqual(restored.policy.lr_schedule(progress), .0003)
                self.assertEqual(restored.updates, 1)
                if mode == 'resume':
                    resumed = restored
                    self.assertEqual(restored.total_transitions, 1)
                    np.testing.assert_array_equal(restored.policy.replay_buffer.observations[:1],
                                                  self.learner.policy.replay_buffer.observations[:1])
        resumed.add(self.obs, np.zeros(2), .1, self.obs, True)
        log = RunLog(self.root/'continued', {})
        try:
            report = resumed.finish_episode(
                dict(episode=2, reason='manual_episode_end', new_transitions=1), log, 1.)
        finally:
            log.close()
        self.assertEqual(report['completed_updates'], 1)
        self.assertEqual(resumed.updates, 2)
        new_source = Path(report['checkpoint'])
        with zipfile.ZipFile(new_source/'model.zip') as archive:
            new_data = json.loads(archive.read('data'))
        self.assertNotIn('lr_schedule', new_data)
        self.assertEqual(new_data['learning_rate'], .0003)
        again = SACLearner(self.cfg, 'resume', new_source)
        self.assertEqual(again.total_transitions, 2)
        self.assertEqual(again.updates, 2)
        self.assert_tree_equal(resumed.policy.policy.state_dict(), again.policy.policy.state_dict())
        self.assertEqual(before, {p.name: digest(p) for p in self.source.iterdir()})

    def test_mismatched_numeric_rate_is_rejected_without_loading_policy(self):
        with zipfile.ZipFile(self.source/'model.zip') as archive:
            contents = {name: archive.read(name) for name in archive.namelist()}
        metadata = json.loads(contents['data'])
        metadata['learning_rate'] = .001
        contents['data'] = json.dumps(metadata).encode()
        with zipfile.ZipFile(self.source/'model.zip', 'w') as archive:
            for name, value in contents.items(): archive.writestr(name, value)
        self.rehash()
        with patch('stable_baselines3.SAC.load') as load:
            with self.assertRaisesRegex(ValueError, '고정 학습률'):
                SACLearner(self.cfg, 'resume', self.source)
            load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
