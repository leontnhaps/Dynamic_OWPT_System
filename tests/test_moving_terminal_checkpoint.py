"""Terminal replay is explicit, while legacy tracking and timeout behavior stay intact."""
from dataclasses import asdict, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from common.tx_moving import (MOVING_SCHEMA, MOVING_TERMINAL_SCHEMA, MovingConfig,
                              moving_loss_reward, moving_visible_reward)
from common.tx_tracking import TrackingConfig
from Tx.Controller.stationary_learning import (SACLearner, SAC_SETTINGS, checkpoint_info,
                                                digest, dump, sac_settings)


class MovingRewardContractTests(unittest.TestCase):
    def test_legacy_defaults_and_saved_bonus_leave_geometry_unchanged(self):
        old = MovingConfig()
        old.validate()
        live = replace(old, visible_bonus=1., loss_mode='confirmed_missing_terminal')
        live.validate()
        self.assertEqual(old.schema, MOVING_SCHEMA)
        self.assertEqual(live.schema, MOVING_TERMINAL_SCHEMA)
        self.assertEqual(old.normalization, live.normalization)
        np.testing.assert_array_equal(old.action_scale, live.action_scale)
        self.assertEqual(sac_settings(old), sac_settings(live))
        self.assertAlmostEqual(moving_visible_reward(200., .4, old), .2)
        self.assertAlmostEqual(moving_visible_reward(200., .4, live), 1.2)
        for pan, expected in ((0., 0.), (45., -5.), (-45., -5.), (90., -10.), (-90., -10.)):
            self.assertAlmostEqual(moving_loss_reward(pan, live), expected)

    def test_invalid_policy_or_reward_cannot_become_a_checkpoint_contract(self):
        for changes in (dict(loss_mode='immediate'), dict(visible_bonus=float('nan')),
                        dict(loss_base=-1.), dict(loss_pan_weight=float('inf')),
                        dict(visible_bonus=True)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(MovingConfig(), **changes).validate()


class TerminalAddTests(unittest.TestCase):
    def learner(self, cfg, mode='resume'):
        learner = SACLearner.__new__(SACLearner)
        learner.cfg, learner.mode, learner.total_transitions = cfg, mode, 3
        learner.policy = SimpleNamespace(replay_buffer=Mock(), num_timesteps=3)
        return learner

    def test_true_terminal_masks_bootstrap_and_timeout_does_not(self):
        learner = self.learner(MovingConfig(loss_mode='confirmed_missing_terminal'))
        obs = np.ones(10, np.float32)
        learner.add(obs, [.5, 0], -5., np.zeros(10), False, terminated=True)
        args = learner.policy.replay_buffer.add.call_args.args
        np.testing.assert_array_equal(args[4], [True])
        self.assertEqual(args[5], [{'TimeLimit.truncated': False}])
        self.assertEqual(learner.total_transitions, 4)
        self.assertEqual(learner.policy.num_timesteps, 4)
        learner.add(obs, [0, 0], 1., obs, True)
        args = learner.policy.replay_buffer.add.call_args.args
        np.testing.assert_array_equal(args[4], [True])
        self.assertEqual(args[5], [{'TimeLimit.truncated': True}])

    def test_terminal_rejected_for_legacy_moving_and_stationary_without_mutation(self):
        for cfg in (MovingConfig(), TrackingConfig()):
            learner = self.learner(cfg)
            obs = np.zeros(cfg.observation_size)
            with self.subTest(schema=cfg.schema), self.assertRaises(ValueError):
                learner.add(obs, [0, 0], -1., obs, False, terminated=True)
            learner.policy.replay_buffer.add.assert_not_called()
            self.assertEqual(learner.total_transitions, 3)

    def test_invalid_flags_and_arrays_do_not_modify_replay(self):
        learner = self.learner(MovingConfig(loss_mode='confirmed_missing_terminal'))
        obs = np.zeros(10)
        for changes in (dict(truncated=True, terminated=True), dict(truncated=1),
                        dict(action=[2., 0.]), dict(reward=float('nan')),
                        dict(next_obs=np.zeros(8)), dict(obs=np.full(10, np.inf))):
            values = dict(obs=obs, action=[0, 0], reward=1., next_obs=obs, truncated=False)
            values.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                learner.add(**values)
        learner.policy.replay_buffer.add.assert_not_called()
        self.assertEqual(learner.total_transitions, 3)

    def test_evaluation_never_appends_even_a_valid_terminal(self):
        learner = self.learner(MovingConfig(loss_mode='confirmed_missing_terminal'), 'evaluate')
        self.assertFalse(learner.add(np.zeros(10), [0, 0], -1., np.zeros(10), False, terminated=True))
        learner.policy.replay_buffer.add.assert_not_called()
        self.assertEqual(learner.total_transitions, 3)


class StrictTerminalCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.cfg = MovingConfig(visible_bonus=1., loss_mode='confirmed_missing_terminal')
        self.state = dict(schema=self.cfg.schema, checkpoint_id='terminal-test', tracking=asdict(self.cfg),
                          sac=sac_settings(self.cfg), normalization=self.cfg.normalization,
                          total_transitions=3, updates=8, executed_actions=3, episodes=1)
        self.replay = dict(observations=np.zeros((3, 1, 10), np.float32),
                           next_observations=np.ones((3, 1, 10), np.float32),
                           actions=np.zeros((3, 1, 2), np.float32), rewards=np.array([[1], [-5], [1]], np.float32),
                           dones=np.array([[0], [1], [1]], np.float32),
                           timeouts=np.array([[0], [0], [1]], np.float32),
                           pos=np.asarray(3), full=np.asarray(False),
                           capacity=np.asarray(SAC_SETTINGS['buffer_size']), total_transitions=np.asarray(3))
        (self.folder/'model.zip').write_bytes(b'fixture model; loaded separately by SB3 acceptance audit')
        np.savez_compressed(self.folder/'rng.npz', cpu=np.zeros(10, np.uint8))
        self.write()

    def write(self, schema=None):
        dump(self.folder/'state.json', self.state)
        np.savez_compressed(self.folder/'replay.npz', **self.replay)
        dump(self.folder/'manifest.json', dict(schema=schema or self.cfg.schema, checkpoint_id='terminal-test',
             files={name: digest(self.folder/name) for name in ('model.zip', 'replay.npz', 'state.json', 'rng.npz')}))

    def test_terminal_and_timeout_replay_restore_without_reinterpretation(self):
        path, state, cfg = checkpoint_info(self.folder)
        self.assertEqual((path, state, cfg), (self.folder, self.state, self.cfg))
        learner = SACLearner.__new__(SACLearner)
        learner.cfg, learner.total_transitions = cfg, 3
        buffer = SimpleNamespace(buffer_size=SAC_SETTINGS['buffer_size'])
        for key in ('observations', 'next_observations', 'actions', 'rewards', 'dones', 'timeouts'):
            setattr(buffer, key, np.zeros_like(self.replay[key]))
        learner.policy = SimpleNamespace(replay_buffer=buffer)
        learner.restore_replay(self.folder/'replay.npz')
        for key in ('observations', 'next_observations', 'actions', 'rewards', 'dones', 'timeouts'):
            np.testing.assert_array_equal(getattr(buffer, key), self.replay[key])
        self.assertEqual((buffer.pos, buffer.full), (3, False))

    def test_manifest_hash_detects_unpaired_replay(self):
        (self.folder/'replay.npz').write_bytes(b'other replay')
        with self.assertRaisesRegex(ValueError, '무결성'):
            checkpoint_info(self.folder)

    def test_rehashed_invalid_replay_is_still_rejected(self):
        changes = (
            ('dones', np.array([[0], [.5], [1]], np.float32)),
            ('timeouts', np.array([[1], [0], [1]], np.float32)),
            ('actions', np.full((3, 1, 2), 1.1, np.float32)),
            ('next_observations', np.full((3, 1, 10), np.nan, np.float32)),
            ('observations', np.zeros((3, 1, 8), np.float32)),
            ('pos', np.asarray(2)), ('full', np.asarray(2)),
            ('total_transitions', np.asarray(3.1)), ('capacity', np.asarray(999)),
        )
        for key, value in changes:
            old = self.replay[key]
            self.replay[key] = value
            self.write()
            with self.subTest(key=key), self.assertRaises(ValueError):
                checkpoint_info(self.folder)
            self.replay[key] = old

    def test_schema_and_reward_contract_cannot_be_mixed(self):
        for location, value in (('schema', MOVING_SCHEMA), ('loss_mode', 'exclude')):
            original = json.loads(json.dumps(self.state))
            if location == 'schema':
                self.state['schema'] = value
            else:
                self.state['tracking']['loss_mode'] = value
            self.write()
            with self.subTest(location=location), self.assertRaisesRegex(ValueError, 'schema'):
                checkpoint_info(self.folder)
            self.state = original

    def test_noninteger_or_negative_training_counters_are_rejected(self):
        for key, value in (('updates', -1), ('total_transitions', 3.5), ('episodes', True)):
            old = self.state[key]
            self.state[key] = value
            self.write()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, '카운터'):
                checkpoint_info(self.folder)
            self.state[key] = old


if __name__ == '__main__':
    unittest.main()
