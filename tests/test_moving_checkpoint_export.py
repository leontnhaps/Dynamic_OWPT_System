"""The offline live adapter preserves SAC state and rejects inconsistent inputs."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from common.tx_bearing import BEARING_MODEL
from common.tx_moving import MovingConfig
from tools.analysis import export_moving_live_checkpoint as export
from Tx.Controller.stationary_learning import SACLearner, checkpoint_info, digest, dump, make_policy, sac_settings


class MovingCheckpointExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            policy = make_policy(MovingConfig(), 'cpu', 42)
            policy.num_timesteps = 3
            policy.save(Path(directory)/'model.zip')
            cls.model_bytes = (Path(directory)/'model.zip').read_bytes()
        cls.torch_rng = torch.get_rng_state().numpy().copy()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patch = patch.object(export, 'REPOSITORY_ROOT', self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.bundle = self.root/'source'
        self.checkpoint = self.bundle/'checkpoint'
        self.checkpoint.mkdir(parents=True)
        self.output = self.root/'captures/M5-3/exported'
        cfg = MovingConfig()
        tracking = asdict(cfg)
        for key in ('visible_bonus', 'loss_mode', 'loss_pan_weight', 'loss_base'):
            tracking.pop(key)
        self.kalman = dict(schema='tx-bearing-kalman-v1',
                           state_order=['alpha', 'beta', 'alpha_dot', 'beta_dot'], angle_unit='rad',
                           process_noise_model='continuous_white_acceleration', Q=np.eye(2).tolist(),
                           R=np.eye(2).tolist(), P0=np.eye(4).tolist(), time_source='receive_s',
                           forecast_horizon_s=cfg.dt)
        dump(self.bundle/'kf_receive_time.json', self.kalman)
        self.source_config = dict(schema='tx-moving-solution-study-v1', seed=42,
                                  reward=deepcopy(export.REWARD), tracking=tracking,
                                  normalization=cfg.normalization, sac=sac_settings(cfg),
                                  kalman=dict(sha256=digest(self.bundle/'kf_receive_time.json')),
                                  versions={'fixture': 'synthetic metadata with an actual SAC policy'})
        dump(self.bundle/'source_config.json', self.source_config)
        numpy_rng = np.random.RandomState(8).get_state()
        self.numpy_rng = [numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]]
        self.state = dict(schema=export.SIM_SCHEMA, simulation_only=True,
                          live_controller_compatible=False, tracking=tracking,
                          normalization=cfg.normalization, kalman=self.kalman,
                          bearing_model=deepcopy(BEARING_MODEL), config=self.source_config,
                          total_transitions=3, updates=0, simulated_steps=2, simulated_episodes=1,
                          numpy_rng=dict(algorithm=numpy_rng[0], pos=numpy_rng[2],
                                         has_gauss=numpy_rng[3], cached_gaussian=numpy_rng[4]),
                          python_rng=random.Random(12).getstate())
        self.arrays = dict(observations=np.arange(30, dtype=np.float32).reshape(3, 1, 10)/30,
                           next_observations=np.zeros((3, 1, 10), dtype=np.float32),
                           actions=np.array([[[0.1, 0.2]], [[0.5, -0.5]], [[-0.2, 0.1]]], dtype=np.float32),
                           rewards=np.array([[0.5], [-5.], [0.8]], dtype=np.float32),
                           dones=np.array([[0.], [1.], [1.]], dtype=np.float32),
                           timeouts=np.array([[0.], [0.], [1.]], dtype=np.float32))
        np.savez_compressed(self.checkpoint/'replay.npz', **self.arrays)
        np.savez_compressed(self.checkpoint/'rng.npz', torch_cpu=self.torch_rng, numpy_keys=numpy_rng[1])
        (self.checkpoint/'model.zip').write_bytes(self.model_bytes)
        dump(self.checkpoint/'experiment_runtime.json', dict(replay_position=3, replay_capacity=1_000_000,
                                                            replay_full=False, environment_at_episode_boundary=True,
                                                            next_episode=2,
                                                            environment_rng=np.random.default_rng(42).bit_generator.state))
        self.rehash()

    def rehash(self):
        dump(self.checkpoint/'state.json', self.state)
        files = export.SIM_FILES if self.state['schema'] == export.SIM_SCHEMA else export.LIVE_FILES
        manifest = dict(schema=self.state['schema'], files={name: digest(self.checkpoint/name) for name in files})
        if self.state['schema'] == export.SIM_SCHEMA:
            manifest['simulation_only'] = True
        else:
            manifest['checkpoint_id'] = self.state['checkpoint_id']
        dump(self.checkpoint/'manifest.json', manifest)

    def run_export(self):
        return export.export_checkpoint(self.bundle, self.output)

    def test_simulation_export_is_resumable_without_relabelling_replay(self):
        source_hashes = {path.name: digest(path) for path in self.checkpoint.iterdir()}
        result = self.run_export()
        folder, state, cfg = checkpoint_info(result/'checkpoint')
        self.assertEqual(cfg.loss_mode, 'confirmed_missing_terminal')
        self.assertEqual(cfg.visible_bonus, 1.)
        self.assertEqual(cfg.loss_pan_weight, 10.)
        self.assertEqual((state['total_transitions'], state['updates'], state['episodes']), (3, 0, 0))
        self.assertEqual((folder/'model.zip').read_bytes(), self.model_bytes)
        self.assertEqual(state['numpy_rng'], self.numpy_rng)
        self.assertEqual(state['reset_action_rng'], np.random.default_rng(42).bit_generator.state)
        self.assertEqual(state['python_rng'], json.loads(json.dumps(self.state['python_rng'])))
        with np.load(folder/'rng.npz', allow_pickle=False) as rng:
            np.testing.assert_array_equal(rng['cpu'], self.torch_rng)
        with np.load(folder/'replay.npz', allow_pickle=False) as replay:
            for key in export.ARRAY_FIELDS:
                np.testing.assert_array_equal(replay[key], self.arrays[key])
        provenance = state['experiment_context']['live_export']
        self.assertEqual(provenance['imported_true_terminals'], 1)
        self.assertEqual(provenance['source_simulated_steps'], 2)
        self.assertEqual(provenance['source_simulated_episodes'], 1)
        self.assertEqual(provenance['imported_real_transitions'], 1)
        self.assertTrue(provenance['terminal_domain_shift'])
        self.assertFalse(provenance['replay_terminal_flags_reinterpreted'])
        learner = SACLearner(cfg, mode='resume', checkpoint=folder)
        self.assertEqual(learner.total_transitions, 3)
        self.assertEqual(learner.policy.replay_buffer.dones[1, 0], 1.)
        self.assertEqual(learner.policy.replay_buffer.timeouts[1, 0], 0.)
        self.assertEqual(source_hashes, {path.name: digest(path) for path in self.checkpoint.iterdir()})

    def test_legacy_real_export_preserves_rng_and_live_counters(self):
        self.state = dict(schema='tx-moving-v2', checkpoint_id='source-real',
                          tracking=dict(self.source_config['tracking'], visible_bonus=1.),
                          sac=self.source_config['sac'], normalization=self.source_config['normalization'],
                          experiment_context=dict(kalman=self.kalman, bearing_model=BEARING_MODEL,
                                                  normalization=self.source_config['normalization']),
                          total_transitions=3, updates=0, executed_actions=6, episodes=2,
                          numpy_rng=self.numpy_rng, python_rng=self.state['python_rng'],
                          reset_action_rng=np.random.default_rng(42).bit_generator.state,
                          action_grid='round_half_away_from_zero', versions=self.source_config['versions'],
                          simulation_only=False, live_controller_compatible=True)
        self.arrays['timeouts'] = self.arrays['dones'].copy()
        np.savez_compressed(self.checkpoint/'replay.npz', **self.arrays,
                            pos=3, full=False, capacity=1_000_000, total_transitions=3)
        np.savez_compressed(self.checkpoint/'rng.npz', cpu=self.torch_rng)
        self.rehash()
        rng_bytes = (self.checkpoint/'rng.npz').read_bytes()
        result = self.run_export()
        _, state, _ = checkpoint_info(result/'checkpoint')
        self.assertEqual((state['executed_actions'], state['episodes']), (6, 2))
        self.assertEqual(state['reset_action_rng'], self.state['reset_action_rng'])
        self.assertEqual((result/'checkpoint/rng.npz').read_bytes(), rng_bytes)
        self.assertFalse(state['experiment_context']['live_export']['terminal_domain_shift'])

    def test_corrupted_source_hash_is_rejected_before_output(self):
        (self.checkpoint/'model.zip').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.run_export()
        self.assertFalse(self.output.exists())

    def test_context_and_configuration_mismatch_are_not_defaulted(self):
        self.state['bearing_model']['fx_px'] += 1
        self.rehash()
        with self.assertRaisesRegex(ValueError, 'bearing calibration'):
            self.run_export()
        self.state['bearing_model'] = deepcopy(BEARING_MODEL)
        self.state['config'] = deepcopy(self.source_config)
        self.state['config']['reward']['visible_bonus'] = 0
        self.rehash()
        with self.assertRaisesRegex(ValueError, 'configuration disagrees'):
            self.run_export()

    def test_wrong_kf_file_cannot_be_substituted(self):
        changed = dict(self.kalman, time_source='other')
        dump(self.bundle/'kf_receive_time.json', changed)
        with self.assertRaisesRegex(ValueError, 'KF provenance hash'):
            self.run_export()

    def test_invalid_terminal_flags_and_counters_are_rejected(self):
        self.arrays['timeouts'][0, 0] = 1
        np.savez_compressed(self.checkpoint/'replay.npz', **self.arrays)
        self.rehash()
        with self.assertRaisesRegex(ValueError, 'Timeout without done'):
            self.run_export()
        self.arrays['timeouts'][0, 0] = 0
        np.savez_compressed(self.checkpoint/'replay.npz', **self.arrays)
        self.state['updates'] = 1
        self.rehash()
        with self.assertRaisesRegex(ValueError, 'Model update counter'):
            self.run_export()

    def test_existing_output_is_never_overwritten(self):
        self.output.mkdir(parents=True)
        sentinel = self.output/'keep.txt'
        sentinel.write_text('unchanged', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            self.run_export()
        self.assertEqual(sentinel.read_text(encoding='utf-8'), 'unchanged')

    def test_loader_failure_leaves_no_published_or_pending_output(self):
        with patch.object(export, 'checkpoint_info', side_effect=ValueError('loader rejected')):
            with self.assertRaisesRegex(ValueError, 'loader rejected'):
                self.run_export()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.output.parent.glob('.pending_live_export_*')))

    def test_zip_input_and_unsafe_member_rejection(self):
        archive = self.root/'source.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            for path in self.bundle.rglob('*'):
                if path.is_file():
                    z.write(path, Path('source')/path.relative_to(self.bundle))
        result = export.export_checkpoint(archive, self.output)
        self.assertTrue((result/'checkpoint/model.zip').is_file())
        with zipfile.ZipFile(archive, 'a') as z:
            z.writestr('../outside.txt', 'must not extract')
        with self.assertRaisesRegex(ValueError, 'Unsafe archive'):
            export.export_checkpoint(archive, self.output.parent/'second')
        self.assertFalse((self.root/'outside.txt').exists())


if __name__ == '__main__':
    unittest.main()
