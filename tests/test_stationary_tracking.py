import csv
from dataclasses import asdict, replace
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
import numpy as np
from common.tx_tracking import (TrackingConfig, RunSettings, Sample, BeamMap,
                                observation, action_command, invalid_reason, EpisodeMetrics,
                                checkpoint_run_config, RUN_SCHEMA)
from Tx.Controller.stationary_core import StationaryRun
from Tx.Controller.stationary_learning import RunLog, SACLearner, SAC_SETTINGS, checkpoint_info, digest, dump
from Tx.Controller.stationary_panel import StationaryPanel
from Tx.Controller.servo_panel import ServoPanel
from Tx.Controller.Com_main import App


class Clock:
    def __init__(self): self.now = 10.
    def __call__(self): return self.now
    def advance(self, dt): self.now += dt


class Learner:
    mode = 'new'
    updates = 0
    executed_actions = 0

    def __init__(self):
        self.rng = np.random.default_rng(3)
        self.cancel = threading.Event()
        self.transitions = []
        self.behaviors = []

    def action(self, obs, warmup):
        self.behaviors.append(warmup)
        return np.array([0., 0.])

    def add(self, obs, action, reward, next_obs, truncated):
        if self.mode == 'evaluate': return False
        self.transitions.append((obs.copy(), action, reward, next_obs.copy(), truncated))
        return True


def sample(clock, frame=1, center=(696, 384), **kwargs):
    target = None if center is None else dict(center=list(center),
        box=[center[0]-34, center[1]-35, center[0]+34, center[1]+35], confidence=.9)
    return Sample(frame, clock(), clock(), 1296, 972, target, int(target is not None), **kwargs)


class GeometryTests(unittest.TestCase):
    def test_preparation_without_legacy_panels_keeps_servo_interlock(self):
        panel = StationaryPanel.__new__(StationaryPanel)
        panel.app = SimpleNamespace(
            servo=SimpleNamespace(pending=None),
            detection=SimpleNamespace(stop=Mock(), future=None),
            timing=SimpleNamespace(active=False, preparing=None, future=None,
                                   profile=SimpleNamespace(running=False), stop=Mock()))
        panel.stop_other_panels()
        panel.app.detection.stop.assert_called_once()
        panel.app.timing.stop.assert_called_once()
        panel.app.servo.pending = 'in-flight-command'
        with self.assertRaises(ValueError):
            panel.stop_other_panels()

    def test_observation_order_normalization_and_actual_applied_delta(self):
        cfg = TrackingConfig()
        raw, obs = observation(sample(lambda: 0, center=(706, 364)), [20, 10], [3, -2], [6, -12], cfg)
        np.testing.assert_array_equal(raw, [10, -20, 4, -8, 3, -2, 20, 10])
        np.testing.assert_allclose(obs, [10/696, -20/588, 4/1296, -8/972, .6, -.4, 20/180, 2*25/55-1])
        raw, obs = observation(sample(lambda: 0), [-20, -10], [3, -2], None, cfg)
        np.testing.assert_array_equal(raw[2:4], [0, 0])
        np.testing.assert_array_equal(raw[4:6], [3, -2])

    def test_beam_matches_direct_pixel_average(self):
        cfg, box = TrackingConfig(), [660.2, 350.8, 729.3, 420.2]
        mean, count = BeamMap(cfg).mean(box)
        x1, y1, x2, y2 = map(math.ceil, box)
        u, v = np.arange(x1, x2), np.arange(y1, y2)
        direct = np.exp(-2*((u[None]-696)/59.07)**2-2*((v[:, None]-384)/62.255)**2)
        self.assertAlmostEqual(mean, direct.mean(), places=12)
        self.assertEqual(count, direct.size)
        with self.assertRaises(ValueError): BeamMap(cfg).mean([.1, .1, .2, .2])

    def test_quantization_clipping_and_zero(self):
        cfg = TrackingConfig()
        raw, cmd, delta = action_command([.1, -.1], [180, -15], cfg)
        np.testing.assert_array_equal(raw, [.5, -.5])
        np.testing.assert_array_equal(cmd, [180, -15])
        np.testing.assert_array_equal(delta, [0, 0])
        np.testing.assert_array_equal(action_command([.1, -.1], [0, 0], cfg)[1], [1, -1])
        with self.assertRaises(ValueError): action_command([np.nan, 0], [0, 0], cfg)

    def test_freshness_new_frames_and_missing(self):
        cfg, clock = TrackingConfig(), Clock()
        s = sample(clock)
        self.assertIsNone(invalid_reason(s, clock(), cfg))
        self.assertEqual(invalid_reason(s, clock(), cfg, previous=1), 'no_new_frame')
        self.assertEqual(invalid_reason(s, clock(), cfg, after=clock()), 'before_command_or_reset_wait')
        self.assertEqual(invalid_reason(s, clock()+.721, cfg), 'stale')
        self.assertEqual(invalid_reason(sample(clock, center=None), clock(), cfg), 'missing')
        self.assertEqual(invalid_reason(replace(s, width=640), clock(), cfg), 'camera_configuration')

    def test_metrics_exclude_initial_and_reacquired_positions(self):
        m = EpisodeMetrics(TrackingConfig(), 10, 0)
        m.command([0, 0]); m.result(30, .3, 11)
        m.command([5, -2]); m.missing(12); m.recovered(13)
        m.command([-1, 2]); m.result(20, .6, 14)
        r = m.summary(15)
        self.assertAlmostEqual(r['rms_px'], math.sqrt(650))
        self.assertEqual(r['hit_ratio'], .5)
        self.assertEqual(r['first_hit_s'], 0)
        self.assertEqual(r['valid_ratio'], 2/3)
        self.assertEqual(r['pan_total_deg'], 6)
        self.assertEqual(r['unavailable_s'], 1)
        self.assertEqual(r['unavailable_count'], 1)
        self.assertIsNone(EpisodeMetrics(TrackingConfig(), 10, 50).summary(11)['rms_px'])

    def test_invalid_settings_and_limits(self):
        for cfg in (TrackingConfig(dt=.1), TrackingConfig(pan_max=180.5), TrackingConfig(reward_distance=0)):
            with self.assertRaises(ValueError): cfg.validate()
        with self.assertRaises(ValueError): RunSettings().validate(TrackingConfig(pan_min=-10))

    def test_short_periods_are_evaluation_only_and_saved_contract_is_preserved(self):
        saved = TrackingConfig(dt=.8, pan_min=-90, pan_max=90, reward_distance=900)
        self.assertEqual(checkpoint_run_config(saved, 'evaluate'), saved)
        self.assertEqual(checkpoint_run_config(saved, 'resume', .5), saved)
        for period in (.6, .5):
            selected = checkpoint_run_config(saved, 'evaluate', period)
            self.assertEqual(asdict(selected), dict(asdict(saved), dt=period))
            with self.assertRaises(ValueError): selected.validate()
            with self.assertRaises(ValueError): SACLearner(selected, 'new')
        for period in (.1, 0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError): checkpoint_run_config(saved, 'evaluate', period)

    def test_checkpoint_load_rejects_training_period_change_and_other_eval_changes(self):
        saved = TrackingConfig()
        with patch('Tx.Controller.stationary_learning.checkpoint_info', return_value=(Path('checkpoint'), {}, saved)):
            for cfg, mode in [(replace(saved, dt=.8), 'resume'),
                              (replace(saved, dt=.5, reward_distance=900), 'evaluate'),
                              (replace(saved, dt=.6, pan_min=-90), 'evaluate')]:
                with self.subTest(mode=mode, cfg=cfg), self.assertRaisesRegex(ValueError, '저장된'):
                    SACLearner(cfg, mode, 'checkpoint')


class RunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.log = RunLog(Path(self.temp.name)/'run', {})
        self.clock, self.learner, self.commands = Clock(), Learner(), []
        self.cfg, self.settings = TrackingConfig(), RunSettings(max_steps=2, episodes=2)
        self.run = StationaryRun(self.cfg, self.settings, self.learner, self.log,
                                 lambda c: self.commands.append(c) or True, self.clock)

    def tearDown(self):
        self.log.close(); self.temp.cleanup()

    def ack(self):
        c = self.commands[-1]
        self.run.reply(dict(event='servo', operation='move', request_id=c['request_id'],
                            available=True, simulated=False, limits=self.cfg.limits,
                            commanded=dict(pan=c['pan'], tilt=c['tilt'])))

    def begin(self, center=(696, 384)):
        self.run.start(); self.ack()
        self.clock.advance(2.01)
        self.run.latest = sample(self.clock, 1, center)
        self.run.tick()
        self.assertEqual(self.run.phase, 'ready')

    def command(self):
        self.run.tick(); self.ack()

    def result(self, frame=2, center=(696, 384), delay=.721):
        self.clock.advance(delay)
        self.run.latest = sample(self.clock, frame, center)
        self.run.tick()

    def test_hit_does_not_end_episode_and_truncation_keeps_last_obs(self):
        self.begin(); self.command(); self.result()
        self.assertEqual(self.run.phase, 'ready')
        self.command(); self.result(3, (706, 384))
        self.assertEqual(self.run.phase, 'saving')
        self.assertEqual(len(self.learner.transitions), 2)
        self.assertTrue(self.learner.transitions[-1][-1])
        self.assertAlmostEqual(self.learner.transitions[-1][3][0], 10/696)
        self.assertEqual(self.run.save_request['reason'], 'step_limit')
        self.assertEqual(self.run.steps, 2)  # Both zero-degree actions count.

    def test_reset_requires_new_frame_after_wait_and_failure_stops_all(self):
        self.run.start(); self.ack()
        self.run.latest = sample(self.clock)
        self.clock.advance(2.1); self.run.tick()
        self.assertEqual(self.run.phase, 'acquire')
        self.clock.advance(3); self.run.tick()
        self.assertEqual(self.run.save_request['reason'], 'initialization_failed')
        self.run.saved()
        self.assertEqual(self.run.phase, 'complete')
        self.assertEqual(len(self.commands), 1)
        self.assertFalse(self.learner.transitions)

    def test_initial_missing_counts_each_episode_and_stops_at_requested_total(self):
        self.run.settings = replace(self.settings, episodes=3)
        self.run.start()
        for episode in range(1, 4):
            self.assertEqual(self.run.episode, episode)
            command = self.commands[-1]
            self.assertTrue(-27 <= command['pan'] <= -7)
            self.assertTrue(-10 <= command['tilt'] <= 5)
            self.ack()
            self.clock.advance(5.01)
            self.run.latest = sample(self.clock, episode, None)
            self.run.tick()
            summary = self.run.save_request
            self.assertEqual(summary['reason'], 'initial_target_lost')
            self.assertTrue(summary['terminated'])
            self.assertIsNone(summary['initial_observation'])
            self.assertIsNone(summary.get('rms_px'))
            self.assertEqual(summary['new_transitions'], 0)
            self.assertFalse(self.run.stop_requested)
            self.assertFalse(self.learner.cancel.is_set())
            self.run.save_request = None
            self.run.saved()
        self.assertEqual(self.run.phase, 'complete')
        self.assertEqual(len(self.commands), 3)
        self.assertFalse(self.learner.transitions)

    def test_initial_missing_then_tracking_loss_both_count_toward_total(self):
        self.run.start(); self.ack()
        self.clock.advance(5.01)
        self.run.latest = sample(self.clock, 1, None); self.run.tick()
        self.run.save_request = None; self.run.saved(); self.ack()
        self.clock.advance(2.01)
        self.run.latest = sample(self.clock, 2); self.run.tick()
        self.command(); self.result(3, None)
        self.clock.advance(3); self.run.tick()
        self.assertEqual(self.run.save_request['reason'], 'target_lost')
        self.run.save_request = None; self.run.saved()
        self.assertEqual(self.run.phase, 'complete')
        self.assertEqual(self.run.episode, 2)

    def test_reset_ack_timeout_still_stops_all_episodes(self):
        self.run.start(); self.clock.advance(5); self.run.tick()
        self.assertEqual(self.run.save_request['reason'], 'ack_timeout')
        self.run.save_request = None; self.run.saved()
        self.assertEqual(self.run.phase, 'complete')
        self.assertEqual(len(self.commands), 1)

    def test_initial_camera_failure_still_stops_all_episodes(self):
        self.run.start(); self.ack(); self.clock.advance(5.01)
        self.run.latest = sample(self.clock, simulated=True); self.run.tick()
        self.assertEqual(self.run.save_request['reason'], 'initialization_failed')
        self.run.save_request = None; self.run.saved()
        self.assertEqual(self.run.phase, 'complete')

    def test_missing_transition_never_completed_by_reacquisition(self):
        self.begin(); self.command(); self.result(center=None)
        self.assertEqual(self.run.phase, 'missing')
        self.clock.advance(.2)
        self.run.latest = sample(self.clock, 3, (746, 400))
        self.run.tick()
        np.testing.assert_array_equal(self.run.raw[2:4], [0, 0])
        self.assertFalse(self.learner.transitions)
        self.assertEqual(self.run.excluded, 1)
        self.command(); self.result(4)
        self.assertEqual(len(self.learner.transitions), 1)
        self.assertEqual(self.run.save_request['executed_actions'], 2)
        self.assertEqual(self.run.save_request['valid_results'], 1)

    def test_final_step_loss_waits_but_no_extra_action(self):
        self.run.settings = replace(self.settings, max_steps=1)
        self.begin(); self.command(); self.result(center=None)
        self.clock.advance(.2); self.run.latest = sample(self.clock, 3)
        self.run.tick()
        self.assertEqual(self.run.save_request['reason'], 'step_limit')
        self.assertEqual(len(self.commands), 2)
        self.assertFalse(self.learner.transitions)

    def test_timeout_failure_preserves_previous_valid_transitions(self):
        self.begin(); self.command(); self.result()
        self.command(); self.result(3, None)
        self.clock.advance(3); self.run.tick()
        self.assertTrue(self.run.save_request['terminated'])
        self.assertFalse(self.learner.transitions[0][-1])
        self.assertEqual(len(self.learner.transitions), 1)
        self.run.save_request = None; self.run.saved()
        self.assertEqual(self.run.phase, 'reset_reply')
        self.assertEqual(self.run.episode, 2)

    def test_delayed_processing_never_catches_up_commands(self):
        self.begin(); self.command()
        self.clock.advance(.4); self.run.tick()
        self.assertEqual(len(self.commands), 2)
        self.result(delay=.45)
        self.command()
        self.assertGreaterEqual(self.run.inflight['command_interval_s'], .85-1e-8)
        self.clock.advance(.719); self.run.tick()
        self.assertEqual(self.run.phase, 'result')
        self.assertEqual(len(self.commands), 3)

    def test_stop_and_send_failure_do_not_relaunch(self):
        self.begin(); self.command()
        self.run.abort()
        self.assertEqual(self.run.save_request['excluded_transitions'], 1)
        self.run.saved(); self.run.tick()
        self.assertEqual(len(self.commands), 2)
        self.assertTrue(self.learner.cancel.is_set())

    def test_failed_initial_send_is_saved_without_step(self):
        self.run.send = lambda c: False
        self.run.start()
        self.assertEqual(self.run.save_request['reason'], 'send_error')
        self.assertEqual(self.run.steps, 0)

    def test_warmup_remains_fixed_for_entire_episode(self):
        self.begin(); self.command(); self.result()
        self.learner.updates = 100  # Fixed episode behavior even if external count changes.
        self.command()
        self.assertEqual(self.learner.behaviors, [True, True])

    def test_evaluation_has_metrics_without_buffer_entries(self):
        self.learner.mode = 'evaluate'
        self.begin(); self.command(); self.result()
        self.command(); self.result(3)
        self.assertFalse(self.learner.transitions)
        self.assertEqual(self.run.save_request['valid_results'], 2)
        self.assertEqual(self.run.save_request['new_transitions'], 0)
        self.assertEqual(self.learner.behaviors, [False, False])

    def test_short_evaluation_uses_selected_wait_freshness_and_no_catchup(self):
        self.cfg = TrackingConfig(dt=.5)
        self.learner.mode = 'evaluate'
        self.run = StationaryRun(self.cfg, self.settings, self.learner, self.log,
                                 lambda c: self.commands.append(c) or True, self.clock)
        self.begin(); self.command()
        self.clock.advance(.499)
        self.run.latest = sample(self.clock, 2); self.run.tick()
        self.assertEqual(self.run.phase, 'result')
        self.result(3, delay=.002)
        self.assertEqual(self.run.phase, 'ready')
        self.command()
        self.assertAlmostEqual(self.run.inflight['command_interval_s'], .501)
        self.result(4, delay=.8)
        self.assertEqual(self.run.phase, 'saving')
        self.assertEqual(len(self.commands), 3)
        self.assertFalse(self.learner.transitions)
        self.assertEqual(self.learner.behaviors, [False, False])
        self.assertEqual(invalid_reason(sample(self.clock, 5), self.clock()+.501, self.cfg), 'stale')
        with (self.log.folder/'steps.csv').open() as f: rows = list(csv.DictReader(f))
        self.assertAlmostEqual(float(rows[1]['interval_excess_s']), .001)

    def test_latest_missing_prevents_old_success_action(self):
        self.begin()
        self.run.latest = sample(self.clock, 2, None)
        self.run.tick()
        self.assertEqual(self.run.phase, 'missing')
        self.assertEqual(len(self.commands), 1)

    def test_raw_logs_have_null_reward_for_excluded_action(self):
        self.begin(); self.command(); self.result(center=None)
        with (self.log.folder/'steps.csv').open() as f:
            row = next(csv.DictReader(f))
        self.assertEqual(row['reward'], '')
        self.assertEqual(row['valid'], 'False')
        self.assertEqual(row['exclusion_reason'], 'missing')

    def test_slow_policy_discards_action_before_send(self):
        self.begin()
        def slow_action(*args):
            self.clock.advance(1)
            return [1, 1]
        self.learner.action = slow_action
        self.run.tick()
        self.assertEqual(len(self.commands), 1)
        self.assertEqual(self.run.phase, 'missing')

    def test_failed_command_ack_does_not_count_step(self):
        self.begin(); self.run.tick()
        self.run.reply(dict(event='error', request_id=self.commands[-1]['request_id'], message='serial unavailable'))
        self.assertEqual(self.run.steps, 0)
        self.assertEqual(self.run.save_request['reason'], 'command_error')
        self.assertFalse(self.learner.transitions)


class LearningScheduleTests(unittest.TestCase):
    def fake(self, total, mode='new'):
        learner = SACLearner.__new__(SACLearner)
        learner.mode, learner.total_transitions = mode, total
        learner.updates = learner.executed_actions = learner.episodes = 0
        learner.cancel = threading.Event()
        learner.policy = Mock()
        learner.policy._n_updates = 0
        learner.policy.logger.name_to_value = {k: 1. for k in ('train/actor_loss', 'train/critic_loss', 'train/ent_coef', 'train/ent_coef_loss')}
        def train(**kwargs): learner.policy._n_updates += 1
        learner.policy.train.side_effect = train
        learner.save_checkpoint = Mock(return_value=Path('checkpoint'))
        return learner

    def test_threshold_counts_only_new_episode_data(self):
        for total, count, expected in [(999, 100, 0), (1050, 90, 90), (1100, 0, 0)]:
            learner = self.fake(total)
            report = learner.finish_episode(dict(episode=1, reason='step_limit', new_transitions=count), Mock(), 1)
            self.assertEqual(report['completed_updates'], expected)
            self.assertEqual(learner.policy.train.call_count, expected)

    def test_evaluate_stop_and_error_do_not_train(self):
        for mode, reason in [('evaluate', 'step_limit'), ('new', 'user_stop'), ('new', 'command_error')]:
            learner = self.fake(1001, mode)
            report = learner.finish_episode(dict(episode=1, reason=reason, new_transitions=10), Mock(), 1)
            self.assertEqual(report['completed_updates'], 0)
            if mode == 'evaluate': learner.save_checkpoint.assert_not_called()

    def test_initial_missing_saves_and_counts_without_inventing_updates(self):
        learner = self.fake(1001)
        report = learner.finish_episode(dict(episode=1, reason='initial_target_lost', new_transitions=0), Mock(), 1)
        self.assertEqual(report['completed_updates'], 0)
        self.assertEqual(learner.episodes, 1)
        learner.policy.train.assert_not_called()
        learner.save_checkpoint.assert_called_once()

    def test_cancel_between_updates_still_checkpoints(self):
        learner = self.fake(1001)
        learner.cancel.set()
        report = learner.finish_episode(dict(episode=1, reason='target_lost', new_transitions=10), Mock(), 1)
        self.assertEqual(report['planned_updates'], 10)
        self.assertEqual(report['completed_updates'], 0)
        learner.save_checkpoint.assert_called_once()


class LoggingTests(unittest.TestCase):
    def test_summary_is_episode_weighted_and_separates_reasons(self):
        with tempfile.TemporaryDirectory() as d:
            log = RunLog(Path(d)/'run', {})
            for reason, error, reached in [('step_limit', 10, True), ('step_limit', 30, False), ('target_lost', 1, False)]:
                log.episode(dict(mode='evaluate', reason=reason, rms_px=error, reached=reached))
            with (log.folder/'summary.csv').open() as f: rows = list(csv.DictReader(f))
            rms = [r for r in rows if r['metric'] == 'rms_px']
            self.assertEqual(float(rms[0]['mean']), 20.)
            self.assertEqual(rms[1]['sample_std'], '')
            log.close()

    def test_checkpoint_manifest_detects_mixed_files(self):
        from dataclasses import asdict
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            (folder/'model.zip').write_bytes(b'model')
            (folder/'replay.npz').write_bytes(b'replay')
            (folder/'rng.npz').write_bytes(b'rng')
            dump(folder/'state.json', dict(checkpoint_id='paired', tracking=asdict(TrackingConfig()), sac=SAC_SETTINGS))
            dump(folder/'manifest.json', dict(schema='tx-stationary-v1', checkpoint_id='paired',
                files={name: digest(folder/name) for name in ('model.zip','replay.npz','rng.npz','state.json')}))
            self.assertEqual(checkpoint_info(folder)[2], TrackingConfig())
            (folder/'replay.npz').write_bytes(b'different run')
            with self.assertRaisesRegex(ValueError, '무결성'): checkpoint_info(folder)

    def test_compact_replay_restores_full_ring_position(self):
        learner = SACLearner.__new__(SACLearner)
        learner.total_transitions = 7
        shapes = dict(observations=(4,1,8), next_observations=(4,1,8), actions=(4,1,2),
                      rewards=(4,1), dones=(4,1), timeouts=(4,1))
        buffer = SimpleNamespace(buffer_size=4, **{k:np.zeros(s,np.float32) for k,s in shapes.items()})
        learner.policy = SimpleNamespace(replay_buffer=buffer)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'replay.npz'
            np.savez_compressed(path, **{k:np.ones(s,np.float32) for k,s in shapes.items()},
                                pos=3, full=True, capacity=4, total_transitions=7)
            learner.restore_replay(path)
            self.assertEqual(buffer.pos, 3)
            self.assertTrue(buffer.full)
            self.assertTrue(np.all(buffer.observations == 1))


class PanelIntegrationTests(unittest.TestCase):
    def test_foreign_move_is_blocked_during_training(self):
        a = App.__new__(App)
        a.stationary = SimpleNamespace(active=True, run=Mock())
        a.record, a.net = Mock(), Mock()
        self.assertFalse(a.send(dict(cmd='move', pan=0, tilt=0)))
        a.net.send.assert_not_called()
        a.stationary.run.abort.assert_called_once()

    def test_own_command_uses_existing_network_path(self):
        a = App.__new__(App)
        a.stationary = SimpleNamespace(active=True, run=Mock())
        a.record, a.net = Mock(), Mock()
        cmd = dict(cmd='move', request_id='m3-2-unit', pan=0, tilt=0)
        self.assertTrue(a.send(cmd, tracking=True))
        a.net.send.assert_called_once_with(cmd)
        a.stationary.run.abort.assert_not_called()

    def test_camera_configuration_is_checked(self):
        settings = dict(width=1296,height=972,fps=30,quality=80,shutter_speed=None,analogue_gain=None)
        self.assertTrue(StationaryPanel.camera_matches((b'',{'requested':settings})))
        self.assertFalse(StationaryPanel.camera_matches((b'',{'requested':dict(settings, fps=15)})))

    def test_video_inference_continues_during_episode_updates(self):
        from concurrent.futures import Future
        p = StationaryPanel.__new__(StationaryPanel)
        p.preparing = None
        p.closed = False; p.work = Future(); p.inference = None
        p.previewing = True; p.detector = object(); p.cfg = TrackingConfig()
        p.app = SimpleNamespace(current=(b'jpeg', {}, __import__('time').monotonic(), 0), received_count=3)
        p.last_frame = None
        p.infer_pool = Mock(); p.infer_pool.submit.return_value = Future()
        p.run = SimpleNamespace(active=True, phase='saving', episode=1, settings=RunSettings(), steps=2,
                                new_transitions=2, command=np.zeros(2), save_request=None, tick=Mock())
        p.progress = Mock()
        p.poll()
        p.infer_pool.submit.assert_called_once()
        p.run.tick.assert_called_once()


class ManualMovementTests(unittest.TestCase):
    def panel(self):
        p = StationaryPanel.__new__(StationaryPanel)
        p.cfg = TrackingConfig(); p.run = None; p.work = None; p.preparing = None
        p.hardware_ready = True; p.servo_available = True
        p.manual_vars = {k: Mock(get=Mock(return_value=v)) for k, v in
                         dict(pan='-17', tilt='0', step='1').items()}
        p.manual_widgets = [Mock()]
        servo = ServoPanel.__new__(ServoPanel)
        servo.limits = p.cfg.limits; servo.simulated = False; servo.online = True
        servo.last = servo.pending = None; servo.buttons = []; servo.state = Mock()
        servo.vars = {axis: Mock() for axis in ('pan', 'tilt')}
        p.app = SimpleNamespace(servo=servo, send=Mock(return_value=True), record=Mock(),
            detection=SimpleNamespace(stop=Mock(), future=None),
            timing=SimpleNamespace(active=False, preparing=None, future=None,
                                   profile=SimpleNamespace(running=False), stop=Mock()))
        servo.app = p.app
        return p

    def ack(self, p):
        command = p.app.send.call_args.args[0]
        p.app.servo.event(dict(event='servo', request_id=command['request_id'], operation='move',
            simulated=False, available=True, limits=p.cfg.limits,
            commanded={a:command[a] for a in ('pan', 'tilt')}))

    def test_absolute_move_and_jog_share_pending_and_confirmed_command(self):
        p = self.panel(); p.manual_move()
        command = p.app.send.call_args.args[0]
        self.assertEqual((command['pan'], command['tilt'], command['speed'], command['acc']), (-17, 0, 100, 1))
        p.sync_manual_controls(); p.manual_widgets[0].configure.assert_called_with(state='disabled')
        with self.assertRaises(ValueError): p.manual_move('pan', 1)
        with self.assertRaises(ValueError): p.stop_other_panels()
        self.ack(p); p.sync_manual_controls()
        p.manual_widgets[0].configure.assert_called_with(state='normal')
        p.manual_vars['pan'].get.return_value = '90'
        p.manual_move('pan', 1)
        self.assertEqual(p.app.send.call_args.args[0]['pan'], -16)
        self.ack(p); p.manual_move('tilt', -1)
        self.assertEqual(p.app.send.call_args.args[0]['tilt'], -1)

    def test_manual_rejection_does_not_abort_learning_or_send_commands(self):
        from unittest.mock import patch
        p = self.panel(); p.run = SimpleNamespace(active=True, abort=Mock())
        with patch('Tx.Controller.stationary_panel.messagebox.showerror') as error:
            p.manual_guard(p.manual_move)
        error.assert_called_once(); p.run.abort.assert_not_called(); p.app.send.assert_not_called()
        for field, value in [('run', None), ('work', object())]: setattr(p, field, value)
        with self.assertRaises(ValueError): p.manual_move()
        p.work = None; p.preparing = 'model'
        with self.assertRaises(ValueError): p.manual_move()
        p.app.send.assert_not_called()

    def test_invalid_targets_missing_reference_and_disconnection_send_nothing(self):
        p = self.panel()
        with self.assertRaises(ValueError): p.manual_move('pan', 1)
        for key, value in [('pan', '181'), ('pan', '-17.5'), ('pan', 'nan'), ('tilt', '-16')]:
            p = self.panel(); p.manual_vars[key].get.return_value = value
            with self.assertRaises(ValueError): p.manual_move()
            p.app.send.assert_not_called()
        for value in ('0', '0.5', '-1'):
            p = self.panel(); p.app.servo.last = {'pan': -17, 'tilt': 0}
            p.manual_vars['step'].get.return_value = value
            with self.assertRaises(ValueError): p.manual_move('tilt', 1)
            p.app.send.assert_not_called()
        for key, value in [('online', False), ('limits', None), ('simulated', True)]:
            p = self.panel(); setattr(p.app.servo, key, value)
            with self.assertRaises(ValueError): p.manual_move()
            p.app.send.assert_not_called()


class AutomaticPreparationTests(unittest.TestCase):
    def panel(self):
        from common.tx_setup import TX_CAMERA_SETTINGS
        p = StationaryPanel.__new__(StationaryPanel)
        p.cfg = TrackingConfig(); p.preparing = 'model'; p.setup_pending = None
        p.hardware_ready = False; p.previewing = False; p.run = None; p.work = None
        p.identity = {'training_control_period_s': .72}
        p.settings = lambda: RunSettings()
        p.send = Mock(return_value=True)
        p.status = Mock(); p.progress = Mock(); p.lock = Mock()
        p.app = SimpleNamespace(values={k:Mock() for k in TX_CAMERA_SETTINGS},
            current=None, size=None, record=Mock(),
            servo=SimpleNamespace(limits=None, simulated=None,
                vars={k:Mock() for k in (*p.cfg.limits, 'speed', 'acc')}))
        return p

    def ready_reply(self, p, **overrides):
        return dict(dict(event='servo', operation='servo_config', request_id=p.setup_pending,
                         available=True, simulated=False, limits=p.cfg.limits), **overrides)

    def test_cold_start_applies_camera_and_limits_without_moving(self):
        from common.tx_setup import TX_CAMERA_SETTINGS
        p = self.panel()
        p.prepare_hardware()
        commands = [call.args[0] for call in p.send.call_args_list]
        self.assertEqual([c['cmd'] for c in commands], ['preview', 'servo_config'])
        self.assertEqual({k: commands[0][k] for k in TX_CAMERA_SETTINGS}, TX_CAMERA_SETTINGS)
        p.event(self.ready_reply(p))
        self.assertEqual(p.app.servo.limits, p.cfg.limits)
        self.assertEqual(p.preparing, 'frames')
        received = __import__('time').monotonic()
        p.app.current = (b'jpeg', dict(simulated=False, requested=TX_CAMERA_SETTINGS), received, 0)
        p.app.size = (1296, 972)
        p.prepare_poll()
        self.assertTrue(p.hardware_ready)
        self.assertTrue(p.previewing)
        self.assertIsNone(p.preparing)

    def test_previous_frame_and_wrong_metadata_cannot_complete_setup(self):
        from common.tx_setup import TX_CAMERA_SETTINGS
        p = self.panel(); p.prepare_hardware(); p.event(self.ready_reply(p))
        p.app.size = (1296, 972)
        p.app.current = (b'jpeg', dict(simulated=False, requested=TX_CAMERA_SETTINGS), p.camera_requested_at-1, 0)
        p.prepare_poll(); self.assertFalse(p.hardware_ready)
        p.app.current = (b'jpeg', dict(simulated=False, requested=dict(TX_CAMERA_SETTINGS, fps=15)), __import__('time').monotonic(), 0)
        p.prepare_poll(); self.assertFalse(p.hardware_ready)

    def test_resume_applies_saved_limits_directly(self):
        p = self.panel(); p.cfg = TrackingConfig(pan_min=-90, pan_max=90, dt=.8)
        p.prepare_hardware()
        command = p.send.call_args_list[1].args[0]
        self.assertEqual(command['pan_min'], -90)
        self.assertEqual(command['pan_max'], 90)

    def test_unavailable_servo_disconnect_and_stop_prevent_ready(self):
        p = self.panel(); p.prepare_hardware()
        p.event(self.ready_reply(p, available=False))
        self.assertFalse(p.hardware_ready); self.assertIsNone(p.preparing)
        p = self.panel(); p.prepare_hardware()
        p.event(dict(event='network', port=7600, state='disconnected'))
        self.assertFalse(p.hardware_ready); self.assertIsNone(p.preparing)
        p = self.panel(); p.prepare_hardware(); p.stop()
        self.assertIsNone(p.preparing); self.assertFalse(p.previewing)

    def test_stopped_model_loading_does_not_later_configure_devices(self):
        from concurrent.futures import Future
        p = self.panel(); p.closed = False; p.work_kind = 'prepare'
        p.inference = None; p.detector = None
        p.work = Future(); p.stop()
        p.work.set_result((object(), object(), TrackingConfig(), {}))
        p.poll()
        p.send.assert_not_called()
        self.assertFalse(p.hardware_ready)

    def test_prepare_has_no_requirement_for_old_tabs_camera_or_limits(self):
        from concurrent.futures import Future
        from unittest.mock import patch
        p = self.panel(); p.closed = False; p.inference = None
        p.stop_other_panels = Mock(); p.app.servo.online = True
        def var(value): return SimpleNamespace(get=lambda: value)
        p.vars = {'dt': var('.720'), 'seed': var('42')}
        p.mode = var('신규 학습'); p.checkpoint = var(''); p.path = var('weights.pt')
        p.device = var('cpu'); p.sac_device = var('cpu'); p.learn_pool = Mock()
        p.learn_pool.submit.return_value = Future()
        p.preparing = None
        p.prepare()
        self.assertEqual(p.preparing, 'model')
        self.assertIsNone(p.app.current)
        self.assertIsNone(p.app.servo.limits)
        p.learn_pool.submit.assert_called_once()

    def test_prepare_applies_evaluation_period_and_resume_uses_saved_period(self):
        from concurrent.futures import Future
        for mode, selection, expected in [('evaluate', '.500', .5), ('evaluate', '저장값', .8), ('resume', '.500', .8)]:
            p = self.panel(); p.preparing = None; p.inference = None
            p.stop_other_panels = Mock(); p.app.servo.online = True
            def var(value): return SimpleNamespace(get=lambda: value)
            p.vars = {'dt': var(selection), 'seed': var('42')}
            p.mode = var('고정 모델 평가' if mode == 'evaluate' else '이어서 학습')
            p.checkpoint = var('checkpoint'); p.path = var('weights.pt')
            p.device = var('cpu'); p.sac_device = var('cpu'); p.learn_pool = Mock()
            p.learn_pool.submit.return_value = Future()
            saved = TrackingConfig(dt=.8, pan_min=-90, pan_max=90)
            with patch('Tx.Controller.stationary_panel.checkpoint_info', return_value=(Path('checkpoint'), {}, saved)), \
                 patch('Tx.Controller.stationary_panel.PVDetector', return_value=SimpleNamespace(names={0: 'PV'})), \
                 patch('Tx.Controller.stationary_panel.SACLearner', return_value=SimpleNamespace(source='checkpoint')) as learner, \
                 patch('Tx.Controller.stationary_panel.digest', return_value='hash'):
                p.prepare()
                detector, model, selected, identity = p.learn_pool.submit.call_args.args[0]()
            self.assertEqual(selected, replace(saved, dt=expected))
            self.assertEqual(learner.call_args.args[0], selected)
            self.assertEqual(identity['training_control_period_s'], .8)

    def test_start_without_preview_detection_attempts_reset_and_records_new_rules(self):
        p = self.panel(); p.preparing = None; p.servo_available = True
        p.app.servo.online = True; p.app.servo.simulated = False
        p.app.servo.limits = p.cfg.limits
        p.check_prepared = Mock(); p.stop_other_panels = Mock()
        p.check_camera = Mock(return_value=(b'jpeg', {'simulated':False}, 10, 0))
        p.latest = None; p.log = None; p.learner = Learner()
        with tempfile.TemporaryDirectory() as folder:
            p.app.stage_dir = lambda stage: Path(folder)
            p.start()
            try:
                self.assertEqual(p.run.phase, 'reset_reply')
                p.send.assert_called_once()
                config = json.loads((p.log.folder/'config.json').read_text())
                self.assertEqual(config['schema'], RUN_SCHEMA)
                self.assertEqual(config['checkpoint_schema'], 'tx-stationary-v1')
                self.assertEqual([config['run'][k] for k in ('pan_low', 'pan_high', 'tilt_low', 'tilt_high')],
                                 [-27, -7, -10, 5])
            finally:
                p.log.close()

    def test_short_evaluation_logs_training_and_execution_period_separately(self):
        p = self.panel(); p.preparing = None; p.servo_available = True
        p.cfg = TrackingConfig(dt=.6)
        p.app.servo.online = True; p.app.servo.simulated = False; p.app.servo.limits = p.cfg.limits
        p.check_prepared = Mock(); p.stop_other_panels = Mock()
        p.check_camera = Mock(return_value=(b'jpeg', {'simulated':False}, 10, 0))
        p.latest = None; p.log = None; p.learner = Learner(); p.learner.mode = 'evaluate'
        with tempfile.TemporaryDirectory() as folder:
            p.app.stage_dir = lambda stage: Path(folder)
            p.start()
            try:
                config = json.loads((p.log.folder/'config.json').read_text())
                self.assertEqual(config['schema'], 'tx-stationary-run-v3')
                self.assertEqual(config['tracking']['dt'], .6)
                self.assertEqual(config['control_timing'], dict(training_period_s=.72,
                                 execution_period_s=.6, evaluation_override=True))
                self.assertEqual(p.run.cfg.dt, .6)
            finally:
                p.log.close()


@unittest.skipUnless(importlib.util.find_spec('stable_baselines3') and importlib.util.find_spec('torch'),
                     'PyTorch / Stable-Baselines3 unavailable')
class RealSACCheckpointTests(unittest.TestCase):
    def test_optimizer_buffer_rng_resume_and_frozen_evaluation(self):
        import torch
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as d:
            learner = SACLearner(TrackingConfig(), seed=7)
            obs = np.zeros(8, np.float32)
            for _ in range(1000):
                learner.add(obs, learner.action(obs, True), .2, obs, True)
            log = RunLog(Path(d)/'run', {})
            learner.finish_episode(dict(episode=1, reason='step_limit', new_transitions=2), log, 1)
            source = log.folder/json.loads((log.folder/'latest.json').read_text())['checkpoint']
            expected_action = learner.action(obs, False)
            expected_reset = learner.rng.integers(-20, 21, 2)
            restored = SACLearner(TrackingConfig(), 'resume', source)
            np.testing.assert_array_equal(restored.action(obs, False), expected_action)
            np.testing.assert_array_equal(restored.rng.integers(-20, 21, 2), expected_reset)
            self.assertEqual(restored.total_transitions, 1000)
            self.assertEqual(restored.updates, 2)
            self.assertEqual(restored.policy.replay_buffer.size(), 1000)
            self.assertTrue(torch.equal(restored.policy.log_ent_coef, learner.policy.log_ent_coef))
            self.assertTrue(restored.policy.actor.optimizer.state_dict()['state'])
            saved_hashes = {p.name: digest(p) for p in source.iterdir() if p.is_file()}
            eval_model = SACLearner(TrackingConfig(dt=.5), 'evaluate', source)
            before = {k: v.clone() for k, v in eval_model.policy.policy.state_dict().items()}
            self.assertFalse(eval_model.add(obs, [0, 0], .2, obs, True))
            eval_model.action(obs, False)
            report = eval_model.finish_episode(dict(episode=1, reason='step_limit', new_transitions=0), Mock(), 1)
            self.assertEqual(report['completed_updates'], 0)
            self.assertIsNone(report['checkpoint'])
            self.assertEqual(saved_hashes, {p.name: digest(p) for p in source.iterdir() if p.is_file()})
            for key, value in eval_model.policy.policy.state_dict().items(): self.assertTrue(torch.equal(value, before[key]))
            log.close()


if __name__ == '__main__':
    unittest.main()
