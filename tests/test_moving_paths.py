"""Moving resume inputs must retain the exact saved experiment contract."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from common.model_paths import REPOSITORY_ROOT
from common.moving_paths import (DEFAULT_KALMAN_PATH, DEFAULT_MOVING_CHECKPOINT_PATH,
                                 load_moving_inputs, resolve_kalman_path,
                                 resolve_moving_checkpoint_path)
from common.tx_bearing import BEARING_MODEL
from common.tx_moving import MovingConfig
from common.tx_tracking import TrackingConfig
from Tx.Controller.stationary_learning import sac_settings


class MovingPathsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / 'episode_000013'
        self.folder.mkdir()
        self.kalman = json.loads(DEFAULT_KALMAN_PATH.read_text(encoding='utf-8'))
        self.kalman_path = self.root / 'fixed_kf.json'
        self.kalman_path.write_text(json.dumps(self.kalman), encoding='utf-8')
        self.cfg = MovingConfig()
        self.state = dict(checkpoint_id='paired-test', tracking=asdict(self.cfg),
                          sac=sac_settings(self.cfg), normalization=self.cfg.normalization,
                          experiment_context=dict(kalman=deepcopy(self.kalman),
                                                  bearing_model=deepcopy(BEARING_MODEL),
                                                  normalization=self.cfg.normalization))
        # checkpoint_info verifies these bytes; it never unpickles or loads a model.
        for name in ('model.zip', 'replay.npz', 'rng.npz'):
            (self.folder / name).write_bytes(('fixture ' + name).encode())
        self.write_checkpoint()

    def write_checkpoint(self, schema=None):
        (self.folder / 'state.json').write_text(json.dumps(self.state), encoding='utf-8')
        names = ('model.zip', 'replay.npz', 'state.json', 'rng.npz')
        manifest = dict(schema=schema or self.cfg.schema, checkpoint_id='paired-test',
                        files={name: hashlib.sha256((self.folder / name).read_bytes()).hexdigest()
                               for name in names})
        (self.folder / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')

    def load(self, **kwargs):
        return load_moving_inputs(self.folder, self.kalman_path, **kwargs)

    def test_defaults_and_relative_paths_are_repository_anchored(self):
        self.assertEqual(resolve_kalman_path(), DEFAULT_KALMAN_PATH)
        self.assertEqual(resolve_moving_checkpoint_path('  '), DEFAULT_MOVING_CHECKPOINT_PATH)
        self.assertEqual(resolve_kalman_path('captures/M5-3/custom.json'),
                         REPOSITORY_ROOT / 'captures/M5-3/custom.json')
        self.assertEqual(resolve_moving_checkpoint_path('captures/M5-3/episode'),
                         REPOSITORY_ROOT / 'captures/M5-3/episode')
        self.assertEqual(resolve_kalman_path(self.kalman_path), self.kalman_path)
        self.assertEqual(resolve_moving_checkpoint_path(self.folder), self.folder)

    def test_complete_checkpoint_retains_saved_config_and_kf(self):
        result = self.load()
        self.assertEqual(result.checkpoint, self.folder)
        self.assertEqual(result.state, self.state)
        self.assertEqual(result.cfg, self.cfg)
        self.assertEqual(result.kalman, self.kalman)
        self.assertEqual(result.kalman_identity['path'], str(self.kalman_path))
        self.assertEqual(result.kalman_identity['sha256'],
                         hashlib.sha256(self.kalman_path.read_bytes()).hexdigest())

    def test_evaluation_keeps_the_saved_period(self):
        self.assertEqual(self.load(mode='evaluate').cfg.dt, self.cfg.dt)

    def test_model_file_selection_still_requires_the_paired_checkpoint(self):
        result = load_moving_inputs(self.folder / 'model.zip', self.kalman_path)
        self.assertEqual(result.checkpoint, self.folder)
        (self.folder / 'replay.npz').unlink()
        with self.assertRaisesRegex(ValueError, 'replay.npz'):
            load_moving_inputs(self.folder / 'model.zip', self.kalman_path)

    def test_missing_explicit_input_never_falls_back_to_default(self):
        with self.assertRaisesRegex(ValueError, 'checkpoint가 없습니다'):
            load_moving_inputs(self.root / 'missing', self.kalman_path)
        with self.assertRaisesRegex(ValueError, 'KF 설정 JSON이 없습니다'):
            load_moving_inputs(self.folder, self.root / 'missing.json')

    def test_new_mode_is_not_a_resume_fallback(self):
        with self.assertRaisesRegex(ValueError, '이어서 학습'):
            self.load(mode='new')

    def test_tampered_model_is_rejected_by_manifest(self):
        (self.folder / 'model.zip').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, '무결성 불일치'):
            self.load()

    def test_stationary_checkpoint_is_rejected(self):
        cfg = TrackingConfig()
        self.state.update(tracking=asdict(cfg), sac=sac_settings(cfg), normalization=cfg.normalization)
        self.write_checkpoint(cfg.schema)
        with self.assertRaisesRegex(ValueError, '10차원 이동 checkpoint'):
            self.load()

    def test_missing_context_is_not_replaced_with_current_defaults(self):
        self.state.pop('experiment_context')
        self.write_checkpoint()
        with self.assertRaisesRegex(ValueError, '보정 정보가 없습니다'):
            self.load()

    def test_changed_bearing_model_is_rejected(self):
        self.state['experiment_context']['bearing_model']['fx_px'] += 1
        self.write_checkpoint()
        with self.assertRaisesRegex(ValueError, '각도 보정 모델'):
            self.load()

    def test_changed_context_normalization_is_rejected(self):
        self.state['experiment_context']['normalization']['prediction_scale_deg'][0] = 90
        self.write_checkpoint()
        with self.assertRaisesRegex(ValueError, '정규화'):
            self.load()

    def test_changed_saved_normalization_is_rejected(self):
        self.state['normalization']['prediction_scale_deg'][0] = 90
        self.write_checkpoint()
        with self.assertRaisesRegex(ValueError, '정규화'):
            self.load()

    def test_different_valid_kf_is_rejected(self):
        self.kalman['Q'][0][0] *= 2
        self.kalman_path.write_text(json.dumps(self.kalman), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'KF와 다릅니다'):
            self.load()

    def test_non_receive_clock_is_rejected_even_when_kf_matches(self):
        self.kalman['time_source'] = 'other_clock'
        self.kalman_path.write_text(json.dumps(self.kalman), encoding='utf-8')
        self.state['experiment_context']['kalman'] = deepcopy(self.kalman)
        self.write_checkpoint()
        with self.assertRaisesRegex(ValueError, 'receive_s'):
            self.load()

if __name__ == '__main__':
    unittest.main()
