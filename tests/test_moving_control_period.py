"""Moving control-period experiments retain learned state and old replay."""
from copy import deepcopy
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from common.moving_paths import DEFAULT_KALMAN_PATH, load_moving_inputs
from common.tx_bearing import BEARING_MODEL
from common.tx_moving import MOVING_CONTROL_PERIODS_S, MovingConfig, moving_run_config
from common.tx_tracking import TrackingConfig, checkpoint_run_config
from Tx.Controller.stationary_learning import SACLearner, RunLog, digest


class MovingControlPeriodConfigTests(unittest.TestCase):
    def test_selected_period_changes_only_dt_for_resume_and_evaluate(self):
        saved = MovingConfig(visible_bonus=1., loss_mode='confirmed_missing_terminal')
        for mode in ('resume', 'evaluate'):
            self.assertIs(moving_run_config(saved, mode), saved)
            for period in MOVING_CONTROL_PERIODS_S:
                with self.subTest(mode=mode, period=period):
                    cfg = moving_run_config(saved, mode, period)
                    self.assertEqual(asdict(cfg), dict(asdict(saved), dt=period))
                    self.assertEqual(cfg.normalization, saved.normalization)

    def test_invalid_periods_are_rejected(self):
        for period in (True, False, None, float('nan'), float('inf'), 0., .5, .6, 3., '1'):
            with self.subTest(period=period), self.assertRaises(ValueError):
                MovingConfig(dt=period).validate()
        with self.assertRaises(ValueError):
            moving_run_config(MovingConfig(), 'new', 1.)

    def test_stationary_control_period_contract_is_unchanged(self):
        saved = TrackingConfig()
        self.assertEqual(checkpoint_run_config(saved, 'resume', 1.5), saved)
        for period in (.5, .6, .72, .8):
            self.assertEqual(checkpoint_run_config(saved, 'evaluate', period).dt, period)
        for period in (1., 1.2, 1.5, 2.):
            with self.subTest(period=period), self.assertRaises(ValueError):
                TrackingConfig(dt=period).validate()
        with self.assertRaises(ValueError):
            moving_run_config(saved, 'resume', 1.)


@unittest.skipUnless(importlib.util.find_spec('stable_baselines3') and importlib.util.find_spec('torch'),
                     'PyTorch / Stable-Baselines3 unavailable')
class MovingControlPeriodCheckpointTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = MovingConfig(visible_bonus=1., loss_mode='confirmed_missing_terminal')
        self.obs = np.zeros(10, np.float32)
        self.learner = SACLearner(self.cfg, seed=17)
        self.kalman = json.loads(DEFAULT_KALMAN_PATH.read_text(encoding='utf-8'))
        self.learner.experiment_context = dict(
            kalman=deepcopy(self.kalman), bearing_model=deepcopy(BEARING_MODEL),
            normalization=self.cfg.normalization)
        self.learner.add(self.obs, np.array([.1, -.2]), .3, self.obs, True)
        self.learner.add(self.obs, np.array([.2, -.1]), -.4, self.obs, False, terminated=True)
        log = RunLog(self.root/'source', {})
        try:
            report = self.learner.finish_episode(
                dict(episode=1, reason='manual_episode_end', new_transitions=2), log, 1.)
        finally:
            log.close()
        self.source = Path(report['checkpoint'])

    def assert_tree_equal(self, left, right):
        import torch
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(left, right)

    def assert_model_equal(self, left, right):
        self.assert_tree_equal(left.policy.state_dict(), right.policy.state_dict())
        for a, b in zip((left.actor.optimizer, left.critic.optimizer, left.ent_coef_optimizer),
                        (right.actor.optimizer, right.critic.optimizer, right.ent_coef_optimizer)):
            self.assert_tree_equal(a.state_dict(), b.state_dict())
        self.assert_tree_equal(left.log_ent_coef, right.log_ent_coef)

    def test_resume_preserves_model_replay_rng_and_saves_selected_default_with_history(self):
        import torch
        before = {p.name: digest(p) for p in self.source.iterdir()}
        rng = (deepcopy(np.random.get_state()), deepcopy(random.getstate()),
               torch.get_rng_state().clone(), deepcopy(self.learner.rng.bit_generator.state))
        resumed = SACLearner(replace(self.cfg, dt=1.2), 'resume', self.source)
        self.assert_model_equal(self.learner.policy, resumed.policy)
        self.assert_tree_equal(rng[0], np.random.get_state())
        self.assert_tree_equal(rng[1], random.getstate())
        self.assert_tree_equal(rng[2], torch.get_rng_state())
        self.assert_tree_equal(rng[3], resumed.rng.bit_generator.state)
        for key in ('total_transitions', 'updates', 'episodes', 'executed_actions'):
            self.assertEqual(getattr(self.learner, key), getattr(resumed, key))
        for key in ('observations', 'next_observations', 'actions', 'rewards', 'dones', 'timeouts'):
            np.testing.assert_array_equal(getattr(self.learner.policy.replay_buffer, key)[:2],
                                          getattr(resumed.policy.replay_buffer, key)[:2])
        self.assertEqual(resumed.policy.replay_buffer.pos, 2)
        self.assertFalse(resumed.policy.replay_buffer.full)
        history = resumed.experiment_context['control_timing_transfers']
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['source_period_s'], .72)
        self.assertEqual(history[0]['execution_period_s'], 1.2)
        self.assertEqual(history[0]['source_total_transitions'], 2)
        self.assertEqual(history[0]['replay_retention'], 'unchanged')
        self.assertIn('mixed-period', history[0]['assumption'])
        self.assertNotIn('control_timing_transfers', self.learner.experiment_context)

        destination = self.root/'continued'
        destination.mkdir()
        saved = resumed.save_checkpoint(destination)
        inputs = load_moving_inputs(saved, DEFAULT_KALMAN_PATH)
        self.assertEqual(inputs.cfg.dt, 1.2)
        self.assertEqual(inputs.kalman, self.kalman)
        self.assertEqual(inputs.state['experiment_context']['control_timing_transfers'], history)
        with np.load(self.source/'replay.npz') as old, np.load(saved/'replay.npz') as new:
            self.assertEqual(set(old.files), set(new.files))
            for key in old.files:
                np.testing.assert_array_equal(old[key], new[key])
        again = SACLearner(inputs.cfg, 'resume', saved)
        self.assertEqual(again.experiment_context['control_timing_transfers'], history)
        changed_again = SACLearner(replace(inputs.cfg, dt=1.5), 'resume', saved)
        next_history = changed_again.experiment_context['control_timing_transfers']
        self.assertEqual(len(next_history), 2)
        self.assertEqual(next_history[0], history[0])
        self.assertEqual(next_history[1]['source_period_s'], 1.2)
        self.assertEqual(next_history[1]['execution_period_s'], 1.5)
        changed_again.add(self.obs, np.zeros(2), .1, self.obs, True)
        log = RunLog(self.root/'updated', {})
        try:
            result = changed_again.finish_episode(
                dict(episode=2, reason='manual_episode_end', new_transitions=1), log, 1.)
        finally:
            log.close()
        self.assertEqual(result['completed_updates'], 1)
        self.assertEqual(changed_again.updates, 3)
        self.assertEqual(changed_again.total_transitions, 3)
        self.assertEqual(before, {p.name: digest(p) for p in self.source.iterdir()})

    def test_evaluation_uses_selected_period_without_checkpoint_or_history_mutation(self):
        before = {p.name: digest(p) for p in self.source.iterdir()}
        evaluated = SACLearner(replace(self.cfg, dt=2.), 'evaluate', self.source)
        self.assertEqual(evaluated.cfg.dt, 2.)
        self.assert_model_equal(self.learner.policy, evaluated.policy)
        self.assertEqual(evaluated.experiment_context, self.learner.experiment_context)
        self.assertNotIn('control_timing_transfers', evaluated.experiment_context)
        log = RunLog(self.root/'evaluation', {})
        try:
            report = evaluated.finish_episode(
                dict(episode=2, reason='manual_episode_end', new_transitions=0), log, 1.)
        finally:
            log.close()
        self.assertIsNone(report['checkpoint'])
        self.assertFalse((self.root/'evaluation'/'checkpoints').exists())
        self.assertEqual(before, {p.name: digest(p) for p in self.source.iterdir()})

    def test_period_override_does_not_allow_reward_or_angle_changes(self):
        with patch('stable_baselines3.SAC.load') as load:
            for changes in (dict(visible_bonus=2.), dict(loss_pan_weight=5.), dict(pan_min=-80)):
                for mode in ('resume', 'evaluate'):
                    with self.subTest(changes=changes, mode=mode), self.assertRaisesRegex(ValueError, '정규화·보상·각도'):
                        SACLearner(replace(self.cfg, dt=1.5, **changes), mode, self.source)
            load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
