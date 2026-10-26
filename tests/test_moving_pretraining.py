import csv
from concurrent.futures import Future
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from PIL import Image
import io
import numpy as np
from common.tx_bearing import BEARING_MODEL
from common.tx_moving import MovingConfig, BearingKalman, KALMAN_SCHEMA, moving_observation, validate_kalman_config
from common.tx_tracking import TrackingConfig, action_command
from Tx.Controller.sweep_core import SweepSettings, SweepRun
from Tx.Controller.sweep_training import SweepExperience, build_dataset, pretrain
from Tx.Controller.stationary_learning import SACLearner, sac_settings
import test_sweep_collection as sweep_fixtures


def kalman_config(model='per_observation'):
    # Synthetic test values only; never used as experiment defaults.
    return dict(schema=KALMAN_SCHEMA, state_order=['alpha', 'beta', 'alpha_dot', 'beta_dot'],
                angle_unit='rad', process_noise_model=model,
                Q=(np.eye(4 if model == 'per_observation' else 2)*1e-7).tolist(),
                R=(np.eye(2)*1e-6).tolist(), P0=np.diag([1e-5, 1e-5, .01, .01]).tolist())


def row(step, test=1, detected=True):
    stamp = 10+step
    return dict(test=test, step=step, frame_id=step+1, image=f'T{test}_S{step}.jpeg',
                status='detected' if detected else 'missing', receive_s=stamp,
                processed_s=stamp+.02, saved_s=stamp+.04,
                command_sent_s=stamp-.8, ack_s=stamp-.79, request_id=f'command-{test}-{step}',
                pan_command_deg=-step, tilt_command_deg=0,
                delta_pan_deg=-1 if step else 0, delta_tilt_deg=0,
                u=696+step, v=384, x1=676+step, x2=716+step, y1=364, y2=404,
                width=1296, height=972, confidence=.9,
                alpha_rad=.01*step, beta_rad=.002*step)


def source(folder, rows, period=.72):
    folder.mkdir()
    config = dict(schema='m5-1-sweep-v2', settings=asdict(SweepSettings(dt=period)), bearing_model=BEARING_MODEL)
    (folder/'config.json').write_text(json.dumps(config))
    with (folder/'observations.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['image', 'test', 'step']); writer.writeheader()
        for r in rows:
            writer.writerow({k:r[k] for k in writer.fieldnames})
            metadata = dict(servo_at_capture_start=dict(simulated=False, available=True,
                commanded=dict(pan=r['pan_command_deg'], tilt=r['tilt_command_deg'], command_id=r['request_id'])))
            (folder/r['image']).with_suffix('.json').write_text(json.dumps(dict(observation=r, frame_metadata=metadata)))


class MovingMathTests(unittest.TestCase):
    def test_axis_scaling_quantization_saturation_and_stationary_unchanged(self):
        cfg = MovingConfig(); cfg.validate()
        requested, target, applied = action_command([.1, .2], [0, 0], cfg)
        np.testing.assert_allclose(target, [9, 1])
        requested, target, applied = action_command([1, 1], [80, 39], cfg)
        np.testing.assert_allclose(requested, [90, 5])
        np.testing.assert_allclose(target, [90, 40])
        np.testing.assert_allclose(applied, [10, 1])
        np.testing.assert_allclose(action_command([-1/180, -.1], [0, 0], cfg)[1], [-1, -1])
        np.testing.assert_allclose(action_command([1, 1], [0, 0], TrackingConfig())[1], [5, 5])

    def test_observation_normalization_and_no_prediction_hard_clip(self):
        cfg = MovingConfig()
        sample = SimpleNamespace(target=dict(center=[696, 384]))
        raw, normalized = moving_observation(sample, [45, 12.5], [45, 2.5], None, [180, -10], cfg)
        self.assertEqual(len(raw), 10)
        np.testing.assert_allclose(normalized[4:8], [.5, .5, .5, 0])
        np.testing.assert_allclose(normalized[-2:], np.tanh([2, -2]))
        self.assertTrue(np.all(np.abs(normalized) <= 1))
        np.testing.assert_allclose(raw[-2:], [180, -10])
        self.assertEqual(sac_settings(cfg)['learning_starts'], 0)
        self.assertEqual(sac_settings(TrackingConfig())['learning_starts'], 1000)
        with self.assertRaises(ValueError): replace(cfg, pan_min=-180).validate()


class KalmanTests(unittest.TestCase):
    def test_parameters_required_and_covariances_validated(self):
        for change in [dict(Q=None), dict(R=[[0, 0], [0, 0]]), dict(P0=[[-1]*4]*4),
                       dict(angle_unit='deg'), dict(process_noise_model='guessed')]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_kalman_config(dict(kalman_config(), **change))

    def test_constant_velocity_prediction_and_joseph_covariance(self):
        kf = BearingKalman(kalman_config())
        for i in range(50):
            result = kf.step([.01*i, -.002*i], i*.72, .72)
        np.testing.assert_allclose(result['predicted_displacement_rad'], [.01, -.002], atol=2e-5)
        self.assertGreaterEqual(np.linalg.eigvalsh(kf.P).min(), -1e-12)
        self.assertGreaterEqual(result['nis'], 0)
        np.testing.assert_allclose(kf.Q, kalman_config()['Q'])

    def test_wrap_missing_and_reset(self):
        kf = BearingKalman(kalman_config())
        self.assertFalse(kf.step(None, 0, .72)['initialized'])
        kf.step([math.radians(179), 0], 1, .72)
        result = kf.step([math.radians(-179), 0], 2, .72)
        self.assertAlmostEqual(math.degrees(result['innovation'][0]), 2)
        missing = kf.step(None, 4, .72)
        self.assertIsNone(missing['innovation'])
        self.assertEqual(missing['dt_s'], 2)
        with self.assertRaises(ValueError): kf.step(None, 4, .72)
        kf.reset()
        initial = kf.step([0, 0], 5, .72)
        np.testing.assert_allclose(initial['predicted_displacement_rad'], [0, 0])

    def test_continuous_noise_uses_actual_elapsed_time(self):
        kf = BearingKalman(kalman_config('continuous_white_acceleration'))
        kf.step([0, 0], 0, .72)
        result = kf.step(None, 2, .72)
        q = np.asarray(result['Q_applied'])
        np.testing.assert_allclose(q[:2, :2], np.eye(2)*1e-7*8/3)
        np.testing.assert_allclose(q[:2, 2:], np.eye(2)*1e-7*2)


class ExperienceTests(unittest.TestCase):
    def test_raw_collection_can_be_completed_with_frozen_filter(self):
        raw = SweepExperience(MovingConfig())
        raw.process(row(0)); unfinished = raw.process(row(1))
        self.assertIsNotNone(unfinished['transition'])
        self.assertFalse(unfinished['transition']['ready_for_replay'])
        fixed = SweepExperience(MovingConfig(), kalman_config())
        fixed.process(row(0)); ready = fixed.process(row(1))
        self.assertTrue(ready['transition']['ready_for_replay'])
        self.assertEqual(len(ready['transition']['observation']), 10)
        self.assertAlmostEqual(ready['transition']['action'][0], -1/90)
        self.assertAlmostEqual(ready['reward'], unfinished['reward'])

    def test_missing_and_episode_boundaries_are_not_bridged(self):
        exp = SweepExperience(MovingConfig(), kalman_config())
        exp.process(row(0))
        self.assertIsNone(exp.process(row(1, detected=False))['transition'])
        recovered = exp.process(row(2))
        self.assertIsNone(recovered['transition'])
        np.testing.assert_allclose(recovered['raw8'][2:4], [0, 0])
        self.assertIsNotNone(exp.process(row(3))['transition'])
        initial = exp.process(row(0, test=2))
        self.assertIsNone(initial['transition'])
        np.testing.assert_allclose(initial['raw10'][-2:], [0, 0])

    def test_stale_before_action_and_out_of_range_are_excluded(self):
        for changes, expected in [(dict(command_sent_s=10.9), 'stale_or_unknown_pre_action_observation'),
                                  (dict(delta_pan_deg=2), 'command_delta_mismatch'),
                                  (dict(pan_command_deg=-100), 'outside_moving_contract')]:
            exp = SweepExperience(MovingConfig(), kalman_config()); exp.process(row(0))
            result = exp.process(dict(row(1), **changes))
            self.assertIsNone(result['transition']); self.assertEqual(result['exclusion_reason'], expected)

    def test_future_observations_cannot_change_saved_input(self):
        first, second = SweepExperience(MovingConfig(), kalman_config()), SweepExperience(MovingConfig(), kalman_config())
        for step in range(3):
            a, b = first.process(row(step)), second.process(row(step))
        before = list(a['normalized10'])
        first.process(dict(row(3), alpha_rad=2))
        second.process(dict(row(3), alpha_rad=-2))
        self.assertEqual(before, a['normalized10']); self.assertEqual(a['normalized10'], b['normalized10'])

    def test_dataset_combines_sessions_and_rejects_duplicates_or_mismatch(self):
        with TemporaryDirectory() as temp:
            a, b = Path(temp)/'a', Path(temp)/'b'
            source(a, [row(0), row(1), row(2, detected=False), row(3), row(4)])
            source(b, [row(0, test=2), row(1, test=2)])
            transitions, report = build_dataset([a, b], MovingConfig(), kalman_config())
            self.assertEqual(len(transitions), 3)
            self.assertEqual(len(report['sources']), 2)
            self.assertTrue(transitions[1]['truncated']); self.assertTrue(transitions[2]['truncated'])
            self.assertTrue(report['sources'][0]['metadata_sha256'])
            duplicate = Path(temp)/'duplicate'; shutil.copytree(a, duplicate)
            with self.assertRaisesRegex(ValueError, '중복'): build_dataset([a, duplicate], MovingConfig(), kalman_config())
            with self.assertRaisesRegex(ValueError, '주기'): build_dataset([a], MovingConfig(dt=.8), kalman_config())

    def test_invalid_frame_command_link_and_reordered_steps_rejected(self):
        with TemporaryDirectory() as temp:
            a = Path(temp)/'a'; source(a, [row(0), row(1)])
            path = (a/row(1)['image']).with_suffix('.json')
            data = json.loads(path.read_text()); data['frame_metadata']['servo_at_capture_start']['commanded']['command_id'] = 'wrong'
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, '연결'): build_dataset([a], MovingConfig(), kalman_config())


class OscillationTests(unittest.TestCase):
    def test_requested_triangle_and_leg_reversal_without_extra_tilt_sweep(self):
        helper = sweep_fixtures.SweepTests()
        run = helper.harness(pan=4, pan_step=-1, tilt=-2, tilt_step=1, tilt_motion='oscillate')
        run.start_leg()
        while run.active: helper.capture()
        self.assertEqual([c['tilt'] for c in helper.sent[:9]], [-2, -1, 0, 1, 2, 1, 0, -1, -2])
        self.assertEqual(helper.sent[-1]['pan'], -5)
        previous = len(helper.sent)
        run.start_leg()
        while run.active: helper.capture()
        self.assertEqual(helper.sent[previous]['pan'], -5)
        self.assertEqual(helper.sent[previous+1]['pan'], -4)
        self.assertEqual(helper.sent[-1]['pan'], 5)
        self.assertEqual(run.test, 2)

    def test_oscillation_cannot_run_forever_without_pan(self):
        for changes in [dict(pan_step=0), dict(tilt_step=0), dict(tilt_osc_min=2, tilt_osc_max=-2),
                        dict(tilt=3), dict(tilt_osc_max=41), dict(tilt_osc_min=-1.5)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(SweepSettings(tilt_motion='oscillate', tilt_step=1), **changes).validate()
        cfg = SweepSettings.from_text(dict(tilt_motion='oscillate', tilt_step='1'))
        self.assertEqual(cfg.tilt_osc_min, -2)


class MovingLearningTests(unittest.TestCase):
    def learner(self, total):
        learner = SACLearner.__new__(SACLearner)
        learner.cfg = MovingConfig()
        learner.mode = 'new'; learner.total_transitions = total
        learner.updates = learner.executed_actions = learner.episodes = 0
        learner.cancel = threading.Event(); learner.policy = Mock()
        learner.policy._n_updates = 0
        learner.policy.logger.name_to_value = {k:1. for k in ('train/actor_loss', 'train/critic_loss', 'train/ent_coef', 'train/ent_coef_loss')}
        learner.policy.train.side_effect = lambda **kwargs: setattr(learner.policy, '_n_updates', learner.policy._n_updates+1)
        learner.save_checkpoint = Mock(return_value=Path('checkpoint'))
        return learner

    def test_first_valid_leg_updates_without_1000_threshold(self):
        for total, count in [(3, 3), (100, 2), (100, 0)]:
            learner = self.learner(total)
            report = learner.finish_episode(dict(episode=1, reason='range_complete', new_transitions=count), Mock(), 1)
            self.assertEqual(report['completed_updates'], count)

    def test_initial_pretraining_imports_all_data_and_saves_provenance(self):
        with TemporaryDirectory() as temp:
            a, output = Path(temp)/'source', Path(temp)/'output'
            source(a, [row(0), row(1), row(2)])
            learner = self.learner(0)
            def add(*args): learner.total_transitions += 1
            learner.add = Mock(side_effect=add)
            with patch('Tx.Controller.sweep_training.SACLearner', return_value=learner):
                result = pretrain([a], kalman_config(), MovingConfig(), output)
            self.assertEqual(result['completed_updates'], 2)
            self.assertEqual(learner.add.call_count, 2)
            self.assertEqual(learner.experiment_context['kalman'], kalman_config())
            self.assertTrue((output/'import_report.json').is_file())
            self.assertEqual(len((output/'imported_transitions.jsonl').read_text().splitlines()), 2)

    def test_panel_collection_and_offline_reprocessing_produce_identical_inputs(self):
        p = sweep_fixtures.PanelTests().panel()
        clock = sweep_fixtures.Clock(); raw = sweep_fixtures.jpeg()
        p.cfg = SweepSettings(pan_min=-2, pan_max=2, tilt_step=1, tilt_motion='oscillate')
        p.vars = {key:sweep_fixtures.variable(str(value)) for key,value in asdict(p.cfg).items()}
        p.identity = dict(path=sweep_fixtures.resolve_yolo_path('best.pt'), device='cpu', sha256='test')
        p.prepared_mode = sweep_fixtures.KF_MODE; p.hardware_ready = True
        p.fixed_kalman = kalman_config(); p.kalman_identity = dict(sha256='synthetic')
        p.app.servo.limits = p.cfg.limits; p.app.servo.simulated = False
        p.detector = Mock()
        p.detector.infer.return_value = dict(image=Image.open(io.BytesIO(raw)), count=1, inference_ms=1,
            target=dict(center=[696,384], box=[676,364,716,404], confidence=.9))
        def submit(fn, *args):
            future = Future(); future.set_result(fn(*args)); return future
        p.pool.submit.side_effect = submit
        def frame():
            p.app.received_count += 1
            state = dict(available=True, simulated=False, limits=p.cfg.limits)
            if p.run:
                state['commanded'] = dict(pan=p.run.pose[0], tilt=p.run.pose[1], command_id=p.run.context['request_id'])
            p.app.current = (raw, dict(simulated=False, requested=dict(sweep_fixtures.TX_CAMERA_SETTINGS),
                servo_at_capture_start=state), clock(), 123)
        with TemporaryDirectory() as temp, patch('Tx.Controller.sweep_panel.time.monotonic', clock), \
                patch('Tx.Controller.sweep_panel.SweepRun', side_effect=lambda cfg, log, send:SweepRun(cfg, log, send, clock)):
            p.app.stage_dir = lambda stage:Path(temp)/stage
            for leg in range(2):
                frame(); p.start()
                while p.run.active:
                    pending = p.run.pending
                    p.event(dict(event='servo', operation='move', request_id=pending['token'], available=True,
                        simulated=False, limits=p.cfg.limits, commanded=dict(pan=pending['pose'][0], tilt=pending['pose'][1],
                        command_id=pending['token'], speed=100, acc=1)))
                    clock.now = p.run.frame_after+.001; frame(); p.poll()
                    clock.now += .02; p.poll()
            recorded = [json.loads(line) for line in (p.log.folder/'rl_transitions.jsonl').read_text().splitlines()]
            rebuilt, report = build_dataset([p.log.folder], MovingConfig(), kalman_config())
            self.assertEqual(len(recorded), 3); self.assertEqual(len(rebuilt), 3)
            for online, offline in zip(recorded, rebuilt):
                self.assertEqual(online['observation'], offline['observation'])
                self.assertEqual(online['next_observation'], offline['next_observation'])
                self.assertEqual(online['reward'], offline['reward'])
            self.assertEqual(report['source_episode_count'], 2)
            self.assertFalse(p.busy); self.assertTrue(p.hardware_ready)


if __name__ == '__main__':
    unittest.main()
