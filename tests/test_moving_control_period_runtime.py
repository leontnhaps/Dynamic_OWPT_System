"""Long-period moving control with simulated time, transport and observations."""
from dataclasses import replace
import unittest

from tests import test_moving_tracking as fixtures
from Tx.Controller.moving_core import MovingRun


class MovingControlPeriodRuntimeTests(unittest.TestCase):
    ack = fixtures.MovingRunTests.ack
    supply = fixtures.MovingRunTests.supply
    begin = fixtures.MovingRunTests.begin
    action = fixtures.MovingRunTests.action

    def setUp(self):
        fixtures.MovingRunTests.setUp(self)
        self.cfg = replace(self.cfg, dt=2., visible_bonus=1.,
                           loss_mode='confirmed_missing_terminal')
        kalman = fixtures.kalman_config()
        # The fixed filter's original tuning horizon is reference metadata.
        kalman['forecast_horizon_s'] = .72
        self.run = MovingRun(self.cfg, self.settings, self.learner, self.log,
            lambda command: self.commands.append(command) or True, self.clock, kalman)

    def kalman_rows(self):
        return [call.kwargs['kalman'] for call in self.log.event.call_args_list
                if call.args[0] == 'kalman_observation']

    def start_missing_result(self):
        self.begin(); self.action()
        self.clock.advance(2.01); self.supply(None); self.run.tick()
        self.assertEqual(self.run.phase, 'missing')
        self.assertIsNotNone(self.run.pending_loss)

    def test_result_waits_two_seconds_and_forecast_uses_selected_not_observation_interval(self):
        self.begin()
        initial_receive = self.run.observation_sample.received
        self.action()
        sent = self.run.last_command_time
        self.assertAlmostEqual(self.run.wait_until-sent, 2.)
        self.assertAlmostEqual(self.run.result_deadline-sent, 4.)
        self.clock.advance(1.); self.supply((706, 384)); self.run.tick()
        self.assertEqual(self.run.phase, 'result')
        self.assertEqual(len(self.kalman_rows()), 1)
        self.clock.advance(1.); self.supply((706, 384)); self.run.tick()
        # An exposure exactly at the wait boundary is not post-command data.
        self.assertEqual(self.run.phase, 'result')
        self.clock.advance(.25); sample = self.supply((706, 384)); self.run.tick()
        self.assertEqual(self.run.phase, 'ready')
        kf = self.kalman_rows()[-1]
        self.assertAlmostEqual(kf['dt_s'], sample.received-initial_receive)
        self.assertAlmostEqual(kf['dt_s'], 2.25)
        self.assertEqual(kf['horizon_s'], 2.)
        self.assertAlmostEqual(kf['prediction_timestamp_s'], sample.received+2.)
        self.assertEqual(self.run.kalman.config['forecast_horizon_s'], .72)
        self.action()
        self.assertAlmostEqual(self.run.last_command_time-sent, 2.25)

    def test_ready_observation_freshness_uses_selected_two_seconds(self):
        for age, expected in ((1.99, 'action_reply'), (2.01, 'missing')):
            with self.subTest(age=age):
                self.setUp(); self.begin()
                before = len(self.commands)
                self.clock.advance(age); self.run.tick()
                self.assertEqual(self.run.phase, expected)
                self.assertEqual(len(self.commands), before+int(age < 2.))
                self.assertEqual(self.learner.action.call_count, int(age < 2.))

    def test_continuing_missing_frames_confirm_loss_after_three_seconds_not_three_periods(self):
        self.start_missing_result()
        missing_since, command_count = self.run.missing_since, len(self.commands)
        for interval in (1.5, 1.499):
            self.clock.advance(interval); self.supply(None); self.run.tick()
            self.assertEqual(self.run.phase, 'missing')
            self.assertEqual(self.learner.transitions, [])
        self.clock.advance(.002); self.supply(None); self.run.tick()
        self.assertAlmostEqual(self.clock()-missing_since, 3.001)
        self.assertEqual(self.run.phase, 'saving')
        self.assertEqual(self.run.save_request['reason'], 'target_lost')
        self.assertTrue(self.run.save_request['terminated'])
        self.assertEqual(self.learner.terminal_flags, [True])
        self.assertEqual(len(self.learner.transitions), 1)
        self.assertEqual(len(self.commands), command_count)
        self.assertTrue(all(row['horizon_s'] == 2. for row in self.kalman_rows()))

    def test_missing_frame_gap_over_two_seconds_excludes_causal_loss_transition(self):
        self.start_missing_result()
        self.clock.advance(2.001); self.supply(None); self.run.tick()
        self.assertIsNone(self.run.pending_loss)
        self.assertEqual(self.run.excluded, 1)
        self.clock.advance(1.); self.supply(None); self.run.tick()
        self.assertEqual(self.run.save_request['reason'], 'target_lost')
        self.assertFalse(self.run.save_request['terminated'])
        self.assertEqual(self.learner.transitions, [])
        rows = [call.args[1] for call in self.log.row.call_args_list if call.args[0] == 'steps']
        self.assertEqual(rows[-1]['exclusion_reason'], 'missing_evidence_gap')

    def test_missing_timeout_starts_after_result_deadline_and_never_creates_loss_sample(self):
        self.begin(); self.action()
        sent = self.run.last_command_time
        self.clock.now = self.run.result_deadline-.01
        self.run.tick()
        self.assertEqual(self.run.phase, 'result')
        self.clock.advance(.02); self.run.tick()
        self.assertEqual(self.run.phase, 'missing')
        self.assertAlmostEqual(self.run.missing_since-sent, 4.01)
        self.clock.advance(2.99); self.run.tick()
        self.assertEqual(self.run.phase, 'missing')
        self.clock.advance(.02); self.run.tick()
        self.assertEqual(self.run.save_request['reason'], 'observation_timeout')
        self.assertAlmostEqual(self.clock()-sent, 7.02)
        self.assertFalse(self.run.save_request['terminated'])
        self.assertEqual(self.learner.transitions, [])


if __name__ == '__main__':
    unittest.main()
