import unittest
from unittest.mock import patch

from common.arc_protocol import stimulus_bits
from Tx.Controller.arc_sender import ArcSender
from tests import test_ook_panel


class ArcSenderTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.sent = []
        self.events = []
        self.sender = ArcSender(self.send, self.events.append, lambda s: None,
                                50, 4.0, 180, pair_id='P01', mode='A', clock=lambda: self.now)
        self.sender.start()

    def send(self, cmd):
        self.sent.append(cmd)
        return True

    def off_ack(self):
        self.sender.handle_event(dict(event='laser', request_id=self.sent[-1]['request_id'],
                                      laser_gpio_level=0))

    def dispatch(self):
        self.now = self.sender.epoch + self.sender.seq * self.sender.period_s
        self.sender.tick(True)
        return self.sent[-1]

    def pattern_result(self, cmd):
        bits = cmd['bits']
        return dict(event='laser_pattern', request_id=cmd['request_id'], state='completed',
                    settings={k: cmd[k] for k in ('bits', 'bit_ms', 'repeats')},
                    off_call_done_ms=len(bits)*50.0,
                    trace=[dict(index=i, level=int(level)) for i, level in enumerate(bits)])

    def test_full_run_uses_absolute_slots_real_traces_and_confirmed_final_off(self):
        self.assertEqual(self.sent[-1]['cmd'], 'laser')
        self.assertEqual(self.sent[-1]['value'], 0)
        self.off_ack()
        for seq in range(181):
            cmd = self.dispatch()
            self.assertEqual(cmd['bits'], stimulus_bits(seq, 50))
            self.assertEqual(cmd['repeats'], 1)
            self.now += len(cmd['bits']) * 0.05 + .02
            self.sender.handle_event(self.pattern_result(cmd))
        self.assertEqual(self.sent[-1]['cmd'], 'laser')
        self.assertEqual(self.sender.phase, 'stopping')
        self.off_ack()
        self.assertTrue(self.sender.off_confirmed)
        self.assertEqual(self.sender.reason, 'completed')
        self.assertFalse(self.sender.active)
        slots = [e['due_s'] for e in self.events if e['event']=='arc_dispatch']
        self.assertEqual(slots, [11.0 + 4 * seq for seq in range(181)])
        self.assertEqual(self.events[0]['pair_id'], 'P01')
        self.assertEqual(self.events[0]['mode'], 'A')

    def test_late_slot_never_compresses_or_sends_a_frame(self):
        self.off_ack()
        self.now = self.sender.epoch + .3
        self.sender.tick(True)
        self.assertEqual([cmd['cmd'] for cmd in self.sent], ['laser', 'laser'])
        self.assertEqual(self.sender.reason, 'missed_slot')

    def test_stop_ignores_late_pattern_messages(self):
        self.off_ack()
        cmd = self.dispatch()
        self.sender.stop('user_stop')
        self.sender.handle_event(self.pattern_result(cmd))
        self.assertEqual(self.sender.phase, 'stopping')
        self.off_ack()
        self.now += 20
        self.sender.tick(True)
        self.assertEqual(len(self.sent), 3)

    def test_bad_trace_and_stale_ack_timeout(self):
        self.off_ack()
        cmd = self.dispatch()
        event = self.pattern_result(cmd)
        event['trace'].pop()
        self.sender.handle_event(event)
        self.assertEqual(self.sender.reason, 'tx_incomplete_trace')
        self.now += 6
        self.sender.tick(True)
        self.assertFalse(self.sender.off_confirmed)
        self.assertFalse(self.sender.active)

    def test_late_final_terminal_report_and_failed_final_off_are_not_successes(self):
        self.off_ack()
        self.sender.seq = self.sender.steps
        cmd = self.dispatch()
        self.now = self.sender.deadline + .01
        self.sender.handle_event(self.pattern_result(cmd))
        self.assertEqual(self.sender.reason, 'pattern_ack_timeout')
        self.off_ack()
        self.assertEqual(self.events[-1]['reason'], 'pattern_ack_timeout')
        self.assertTrue(self.events[-1]['gpio_off_confirmed'])

    def test_completed_frames_but_failed_final_off(self):
        self.off_ack()
        self.sender.seq = self.sender.steps
        cmd = self.dispatch()
        self.sender.handle_event(self.pattern_result(cmd))
        self.now += 6
        self.sender.tick(True)
        self.assertEqual(self.events[-1]['reason'], 'final_off_unconfirmed')
        self.assertFalse(self.events[-1]['gpio_off_confirmed'])

    def test_lost_preview_and_rejected_off(self):
        self.off_ack()
        self.sender.tick(False)
        self.assertEqual(self.sender.reason, 'tx_not_ready')
        self.sender.handle_event(dict(event='error', request_id=self.sent[-1]['request_id']))
        self.assertFalse(self.sender.active)
        self.assertFalse(self.sender.off_confirmed)

    def test_missing_arming_ack_is_never_treated_as_ready(self):
        self.sender.handle_event(dict(event='laser', request_id='unrelated', laser_gpio_level=0))
        self.now += 6
        self.sender.tick(True)
        self.assertFalse(self.sender.active)
        self.assertFalse(self.sender.off_confirmed)
        self.assertEqual(len(self.sent), 1)

    def test_panel_interlocks_hold_through_wait_and_tab_exit(self):
        panel = test_ook_panel.PanelStateTests().panel()
        panel.arc = self.sender
        panel.hardware = {'simulated': False, 'capabilities': ['laser_pattern']}
        panel.app.current = (None, None, self.now)
        panel.event(dict(event='laser', request_id=self.sent[-1]['request_id'], laser_gpio_level=0))
        self.assertTrue(panel.busy)
        with patch('Tx.Controller.ook_panel.time.monotonic', return_value=self.now):
            with self.assertRaises(ValueError):
                panel.check_ready()
        panel.leave()
        self.assertEqual(self.sent[-1]['cmd'], 'laser')
        panel.event(dict(event='laser', request_id=self.sent[-1]['request_id'], laser_gpio_level=0))
        self.assertIsNone(panel.arc)
        self.assertFalse(panel.busy)
        self.assertEqual(self.sender.reason, 'user_stop')

    def test_panel_disconnect_blocks_retry_and_preserves_uncertain_off(self):
        panel = test_ook_panel.PanelStateTests().panel()
        panel.arc = self.sender
        self.off_ack()
        panel.event(dict(event='network', port=7600, state='disconnected'))
        self.assertIsNone(panel.arc)
        self.assertTrue(panel.output_requested)
        self.assertFalse(self.events[-1]['gpio_off_confirmed'])
        self.assertEqual(self.events[-1]['reason'], 'pi_disconnected')


if __name__ == '__main__':
    unittest.main()
