"""Clock/transport/policy simulations; no hardware or actual training."""
from dataclasses import replace
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np

from common.tx_moving import MovingConfig, PredictionInputError, KALMAN_SCHEMA
from common.tx_tracking import Sample
from Tx.Controller.moving_core import MovingRun, MovingRunSettings


class Clock:
    def __init__(self):
        self.now = 10.

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def kalman_config():
    # Synthetic covariance values, never defaults for an experiment.
    return dict(schema=KALMAN_SCHEMA, state_order=['alpha', 'beta', 'alpha_dot', 'beta_dot'],
        angle_unit='rad', process_noise_model='continuous_white_acceleration',
        Q=(np.eye(2)*1e-7).tolist(), R=(np.eye(2)*1e-6).tolist(),
        P0=np.diag([1e-5, 1e-5, .01, .01]).tolist())


class Learner:
    def __init__(self):
        self.mode = 'resume'
        self.episodes, self.updates, self.executed_actions = 13, 120, 120
        self.cancel = threading.Event()
        self.transitions = []
        self.action = Mock(return_value=np.zeros(2))

    def add(self, obs, action, reward, next_obs, truncated):
        if self.mode == 'evaluate':
            return False
        self.transitions.append((obs.copy(), list(action), reward, next_obs.copy(), truncated))
        return True


class MovingRunTests(unittest.TestCase):
    def setUp(self):
        self.clock, self.learner, self.log = Clock(), Learner(), Mock()
        self.commands = []
        self.cfg = MovingConfig()
        self.settings = MovingRunSettings(start_pan=-10, start_tilt=2)
        self.run = MovingRun(self.cfg, self.settings, self.learner, self.log,
            lambda command:self.commands.append(command) or True, self.clock, kalman_config())
        self.frame = 0

    def ack(self, **changes):
        command = self.commands[-1]
        event = dict(event='servo', operation='move', request_id=command['request_id'],
            simulated=False, available=True, limits=self.cfg.limits,
            commanded=dict(command_id=command['request_id'], pan=command['pan'], tilt=command['tilt'],
                           speed=100, acc=1))
        event.update(changes)
        self.assertTrue(self.run.reply(event))

    def supply(self, center=(696, 384), received=None, metadata_changes=None, **changes):
        self.frame += 1
        target = None if center is None else dict(center=list(center),
            box=[center[0]-20, center[1]-20, center[0]+20, center[1]+20], confidence=.9)
        received = self.clock() if received is None else received
        sample = Sample(self.frame, received, self.clock(), 1296, 972, target, int(target is not None))
        sample = replace(sample, **changes)
        servo = dict(simulated=False, available=True, limits=self.cfg.limits,
            commanded=dict(command_id=self.run.command_id, pan=float(self.run.command[0]),
                           tilt=float(self.run.command[1]), speed=100, acc=1))
        if metadata_changes:
            servo.update(metadata_changes)
        self.run.supply(sample, dict(servo_at_capture_start=servo))
        return sample

    def begin(self):
        self.run.start(); self.ack()
        self.clock.advance(2.01)
        self.supply(); self.run.tick()
        self.assertEqual(self.run.phase, 'ready')

    def action(self):
        self.run.tick()
        self.assertEqual(self.run.phase, 'action_reply')
        self.ack()
        self.assertEqual(self.run.phase, 'result')

    def result(self, center=(696, 384), delay=.721, **changes):
        self.clock.advance(delay)
        self.supply(center, **changes)
        self.run.tick()

    def returned(self):
        self.run.saved()
        self.assertEqual(self.run.phase, 'return_reply')
        self.ack()
        self.assertEqual(self.run.phase, 'return_settle')
        self.clock.advance(2.001); self.run.tick()
        self.assertEqual(self.run.phase, 'idle')
        self.assertFalse(self.run.active)

    def test_one_manual_episode_updates_only_after_end_returns_and_never_auto_restarts(self):
        self.begin()
        self.assertEqual(self.run.episode, 14)
        self.action(); self.result()
        self.assertEqual(self.learner.transitions, [])  # Last item waits for boundary.
        self.action(); self.result((706, 384))
        self.assertEqual(len(self.learner.transitions), 1)
        self.assertFalse(self.learner.transitions[0][-1])
        before = len(self.commands)
        self.run.manual_end()
        self.assertEqual(self.run.phase, 'saving')
        self.assertEqual(len(self.commands), before)
        self.assertEqual(len(self.learner.transitions), 2)
        self.assertTrue(self.learner.transitions[-1][-1])
        summary = self.run.save_request
        self.assertEqual(summary['reason'], 'manual_episode_end')
        self.assertEqual(summary['new_transitions'], 2)
        self.assertEqual(summary['reward_count'], 2)
        self.assertAlmostEqual(summary['reward_sum'], sum(item[2] for item in self.learner.transitions))
        self.assertFalse(summary['terminated']); self.assertTrue(summary['truncated'])
        self.returned()
        self.clock.advance(20); self.run.tick()
        self.assertEqual(len(self.commands), before+1)
        self.learner.episodes = 14
        self.run.start()
        self.assertEqual(self.run.episode, 15)
        self.assertIsNone(self.run.kalman.x)
        self.assertEqual(len(self.learner.transitions), 2)  # Accumulated replay is retained.

    def test_start_point_change_is_deferred_until_return_and_next_start(self):
        self.begin()
        previous = len(self.commands)
        self.run.set_start(7, 3)
        self.assertEqual(len(self.commands), previous)
        np.testing.assert_allclose(self.run.command, [-10, 2])
        self.run.manual_end(); self.run.saved()
        self.assertEqual([self.commands[-1][key] for key in ('pan', 'tilt')], [7, 3])
        self.ack(); self.clock.advance(2.01); self.run.tick()
        self.run.start()
        self.assertEqual(self.run.initial_command, [7, 3])

    def test_moving_policy_uses_ten_dimensions_without_uniform_warmup(self):
        self.learner.updates = 0
        self.begin(); self.action(); self.result((706, 384))
        self.action(); self.result((716, 384)); self.run.manual_end()
        for call in self.learner.action.call_args_list:
            self.assertEqual(call.args[0].shape, (10,))
            self.assertFalse(call.args[1])
        self.assertNotEqual(self.learner.transitions[-1][3][-2], 0)
        np.testing.assert_array_equal(self.run.kalman.Q, kalman_config()['Q'])

    def test_post_dwell_frame_is_awaited_instead_of_discarding_previous_inference(self):
        self.begin(); self.action()
        sent = self.run.last_command_time
        self.clock.now = sent+.720
        self.supply(received=sent+.690); self.run.tick()
        self.assertEqual(self.run.phase, 'result')
        self.assertEqual(self.run.excluded, 0)
        self.clock.now = sent+.780
        self.supply(received=sent+.750); self.run.tick()
        self.assertEqual(self.run.phase, 'ready')
        self.run.manual_end()
        self.assertEqual(len(self.learner.transitions), 1)
        self.assertEqual(self.run.save_request['excluded_transitions'], 0)

    def test_result_wait_has_a_bound_and_does_not_replay_command_slots(self):
        self.begin(); self.action()
        before = len(self.commands)
        self.clock.now = self.run.result_deadline+.001
        self.run.tick()
        self.assertEqual(self.run.phase, 'missing')
        self.assertEqual(self.run.excluded, 1)
        self.assertEqual(len(self.commands), before)
        self.assertEqual(self.learner.transitions, [])

    def test_missing_gap_breaks_replay_but_preserves_kalman_state_until_next_episode(self):
        self.begin(); self.action(); self.result((706, 384))
        self.action(); self.result(None)
        self.assertEqual(self.run.phase, 'missing')
        self.assertEqual(len(self.learner.transitions), 1)
        self.assertTrue(self.learner.transitions[-1][-1])
        self.assertIsNotNone(self.run.kalman.x)
        before = self.run.kalman.time
        self.clock.advance(.1); self.supply((716, 384)); self.run.tick()
        self.assertEqual(self.run.phase, 'ready')
        self.assertGreater(self.run.kalman.time, before)
        np.testing.assert_allclose(self.run.raw[2:4], [0, 0])
        events = [call.kwargs for call in self.log.event.call_args_list if call.args[0] == 'kalman_observation']
        self.assertFalse(events[-1]['kalman']['initialized_now'])
        self.action(); self.result((726, 384)); self.run.manual_end()
        self.assertEqual(len(self.learner.transitions), 2)
        self.assertEqual(self.run.save_request['excluded_transitions'], 1)
        self.assertTrue(all(len(item[0]) == len(item[3]) == 10 for item in self.learner.transitions))

    def test_invalid_forecast_is_logged_without_clipping_or_replay_bridge(self):
        self.begin(); self.action()
        with patch('Tx.Controller.moving_core.moving_observation', side_effect=PredictionInputError('고도각 오류')):
            self.result()
        self.assertEqual(self.run.phase, 'missing')
        self.assertEqual(self.learner.transitions, [])
        event = [call.kwargs for call in self.log.event.call_args_list if call.args[0] == 'kalman_observation'][-1]
        self.assertEqual(event['exclusion_reason'], 'invalid_kalman_prediction')
        self.assertIsNone(event['raw10']); self.assertIsNotNone(event['kalman'])
        self.clock.advance(.1); self.supply(); self.run.tick()
        self.assertEqual(self.run.phase, 'ready')
        self.action(); self.result(); self.run.manual_end()
        self.assertEqual(len(self.learner.transitions), 1)

    def test_initial_missing_finishes_without_fake_transition_then_returns(self):
        self.run.start(); self.ack()
        self.clock.advance(2.01); self.supply(None); self.run.tick()
        self.clock.advance(3); self.supply(None); self.run.tick()
        self.assertEqual(self.run.save_request['reason'], 'initial_target_lost')
        self.assertEqual(self.run.save_request['new_transitions'], 0)
        self.assertEqual(self.run.save_request['reward_count'], 0)
        self.assertIsNone(self.run.save_request['reward_mean'])
        self.assertIsNone(self.run.save_request['initial_observation'])
        self.assertEqual(self.learner.transitions, [])
        self.returned()
        self.assertEqual(len(self.commands), 2)

    def test_initial_detection_received_after_acquire_deadline_cannot_extend_episode(self):
        self.run.start(); self.ack()
        self.clock.advance(5.1); self.supply(); self.run.tick()
        self.assertEqual(self.run.save_request['reason'], 'initial_target_lost')
        self.assertIsNone(self.run.save_request['initial_observation'])

    def test_missing_three_seconds_finishes_and_does_not_send_tracking_command(self):
        self.begin(); self.action(); self.result(None)
        count = len(self.commands)
        self.clock.advance(3); self.supply(None); self.run.tick()
        self.assertEqual(self.run.save_request['reason'], 'target_lost')
        self.assertFalse(self.learner.cancel.is_set())
        self.assertEqual(len(self.commands), count)
        self.returned()

    def test_reacquisition_received_after_missing_deadline_cannot_avoid_end(self):
        self.begin(); self.action(); self.result(None)
        self.clock.advance(3.1); self.supply(); self.run.tick()
        self.assertEqual(self.run.save_request['reason'], 'target_lost')
        self.assertEqual(self.learner.transitions, [])

    def test_pre_deadline_reacquisition_processed_later_is_usable_when_fresh(self):
        self.begin(); self.action(); self.result(None)
        deadline = self.run.missing_since+3
        self.clock.now = deadline+.1
        self.supply(received=deadline-.1); self.run.tick()
        self.assertEqual(self.run.phase, 'ready')

    def test_camera_timeout_is_an_error_with_no_automatic_return(self):
        self.run.start(); self.ack()
        self.clock.advance(5.1); self.run.tick()
        self.assertEqual(self.run.save_request['reason'], 'initialization_failed')
        self.assertTrue(self.learner.cancel.is_set())
        self.run.saved(); self.run.tick()
        self.assertEqual(self.run.phase, 'error')
        self.assertEqual(len(self.commands), 1)

    def test_manual_end_drains_pending_action_ack_and_post_action_frame(self):
        self.begin(); self.run.tick()
        self.assertEqual(self.run.phase, 'action_reply')
        self.run.manual_end()
        self.assertIsNone(self.run.save_request)
        self.ack(); self.result()
        self.assertEqual(self.run.phase, 'saving')
        self.assertEqual(len(self.commands), 2)
        self.assertEqual(len(self.learner.transitions), 1)
        self.assertTrue(self.learner.transitions[0][-1])
        self.assertEqual(self.run.save_request['reason'], 'manual_episode_end')

    def test_abort_with_pending_action_saves_immediately_and_late_ack_cannot_restart(self):
        self.begin(); self.run.tick()
        self.run.abort('connection_lost')
        self.assertEqual(self.run.phase, 'saving')
        self.assertIsNotNone(self.run.pending)
        self.assertEqual(self.run.save_request['reason'], 'connection_lost')
        self.assertEqual(self.run.save_request['executed_actions'], 0)
        self.assertEqual(self.learner.transitions, [])
        self.run.saved(); self.ack(); self.clock.advance(4); self.run.tick()
        self.assertEqual(self.run.phase, 'error')
        self.assertEqual(len(self.commands), 2)
        with self.assertRaises(ValueError):
            self.run.start()

    def test_abort_during_return_never_issues_another_move(self):
        self.begin(); self.run.manual_end(); self.run.saved()
        count = len(self.commands)
        self.run.abort('user_stop'); self.ack()
        self.clock.advance(5); self.run.tick()
        self.assertEqual(self.run.phase, 'error')
        self.assertIsNone(self.run.save_request)
        self.assertEqual(len(self.commands), count)

    def test_failed_save_prevents_return(self):
        self.begin(); self.run.manual_end(); self.run.saved(False)
        self.assertEqual(self.run.phase, 'error')
        self.assertEqual(len(self.commands), 1)

    def test_ack_timeout_ignores_late_reply_and_never_retries_or_returns(self):
        self.run.start(); self.clock.advance(5); self.run.tick()
        self.assertEqual(self.run.save_request['reason'], 'ack_timeout')
        self.run.saved()
        self.assertFalse(self.run.reply(dict(event='servo', request_id=self.commands[-1]['request_id'])))
        self.run.tick()
        self.assertEqual(len(self.commands), 1)

    def test_ack_requires_real_identity_limits_and_speed(self):
        for changes in (dict(simulated=True), dict(limits={}), dict(commanded={}), dict(available=False)):
            with self.subTest(changes=changes):
                self.setUp(); self.run.start(); self.ack(**changes)
                self.assertEqual(self.run.save_request['reason'], 'command_error')
                self.run.saved(); self.run.tick()
                self.assertEqual(len(self.commands), 1)

    def test_wrong_frame_command_limits_or_speed_never_update_filter_or_replay(self):
        for changes in (dict(limits={}), dict(commanded=dict(command_id='old', pan=-10, tilt=2, speed=100, acc=1)),
                        dict(commanded=dict(command_id=None, pan=-10, tilt=2, speed=1, acc=1))):
            with self.subTest(changes=changes):
                self.setUp(); self.begin(); self.action()
                original_time = self.run.kalman.time
                self.result(metadata_changes=changes)
                self.assertEqual(self.run.phase, 'result')
                self.assertEqual(self.run.kalman.time, original_time)
                self.assertEqual(self.learner.transitions, [])
                self.clock.advance(.05); self.supply(); self.run.tick()
                self.assertEqual(self.run.phase, 'ready')

    def test_irregular_observation_times_drive_kalman_without_command_catchup(self):
        self.begin(); self.action(); self.result(delay=1.1)
        events = [call.kwargs for call in self.log.event.call_args_list if call.args[0] == 'kalman_observation']
        self.assertAlmostEqual(events[-1]['kalman']['dt_s'], 1.1)
        self.action()
        self.assertAlmostEqual(self.run.inflight['command_interval_s'], 1.1)
        self.assertEqual(len(self.commands), 3)

    def test_slow_policy_never_sends_from_a_stale_observation(self):
        self.begin()
        def delayed(*args):
            self.clock.advance(1)
            return np.zeros(2)
        self.learner.action.side_effect = delayed
        self.run.tick()
        self.assertEqual(self.run.phase, 'missing')
        self.assertEqual(len(self.commands), 1)
        self.assertEqual(self.learner.transitions, [])

    def test_log_write_failure_cannot_duplicate_buffered_replay_on_abort(self):
        self.begin(); self.action(); self.result()
        self.log.row.side_effect = OSError('disk full')
        with self.assertRaises(OSError):
            self.run.manual_end()
        self.assertEqual(len(self.learner.transitions), 1)
        self.assertIsNone(self.run.buffered)
        self.run.abort('logging_failed')
        self.run.saved(); self.run.tick()
        self.assertEqual(len(self.learner.transitions), 1)
        self.assertEqual(len(self.commands), 2)
        self.assertEqual(self.run.phase, 'error')

    def test_send_failure_produces_zero_action_summary_without_retry(self):
        self.run.send = Mock(return_value=False)
        self.run.start()
        self.assertEqual(self.run.save_request['reason'], 'send_error')
        self.assertEqual(self.run.save_request['new_transitions'], 0)
        self.run.saved(); self.run.tick()
        self.run.send.assert_called_once()

    def test_settings_reject_invalid_origin_and_keep_three_second_limit(self):
        for changes in (dict(start_pan=91), dict(start_tilt=-20), dict(start_pan=.5),
                        dict(start_tilt=float('nan')), dict(missing_s=4), dict(updates_per_transition=-1)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.settings, **changes).validate(self.cfg)


if __name__ == '__main__':
    unittest.main()
