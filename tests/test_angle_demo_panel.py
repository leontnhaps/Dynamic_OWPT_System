import queue
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from common.angle_demo import BIT_MS, encode_angles
from Tx.Controller.angle_demo_panel import AngleDemoPanel


class DemoTabTests(unittest.TestCase):
    def setUp(self):
        self.panel = p = AngleDemoPanel.__new__(AngleDemoPanel)
        p.rx = Mock(connected=True, terminal_error=None, events=queue.Queue())
        p.app = SimpleNamespace(send=Mock(return_value=True), ook=Mock(
            online=True, hardware={'simulated': False, 'capabilities': ['laser_pattern']}),
            current=(None, None, time.monotonic()))
        p.record = Mock()
        p.phase = p.token = None
        p.pan, p.tilt = Mock(), Mock()
        p.pan.get.return_value, p.tilt.get.return_value = '5', '-1'
        p.off_token = None
        p.applied, p.status, p.rx_status, p.off_status = Mock(), Mock(), Mock(), Mock()
        p.selected = p.prepared = True

    def start(self):
        p = self.panel
        p.transmit()
        token = p.token
        self.assertEqual(p.phase, 'off')
        p.event(dict(event='laser', request_id=token, laser_gpio_level=0))
        p.rx.send.assert_called_once_with('arm', token)
        p.on_rx(dict(kind='armed', token=token))
        p.app.send.assert_called_with(dict(cmd='laser_pattern', request_id=token,
                                           bits=encode_angles(108, 19), bit_ms=BIT_MS, repeats=1))
        return token

    def test_success_requires_both_actual_tx_and_optical_servo_ack_in_either_order(self):
        for optical_first in (False, True):
            with self.subTest(optical_first=optical_first):
                self.setUp()
                p = self.panel
                token = self.start()
                applied = dict(kind='applied', token=token, pan=108, tilt=19)
                complete = dict(event='laser_pattern', state='completed', simulated=False,
                                request_id=token, off_call_done_ms=2050,
                                settings=dict(bits=encode_angles(108, 19), bit_ms=BIT_MS, repeats=1))
                if optical_first:
                    p.on_rx(applied)
                    self.assertEqual(p.phase, 'optical')
                    p.event(complete)
                else:
                    p.event(complete)
                    self.assertEqual(p.phase, 'optical')
                    p.on_rx(applied)
                self.assertIsNone(p.phase)
                self.assertIn('OOK 수신 성공', p.status.set.call_args.args[0])
                p.rx.send.assert_called_once_with('arm', token)

    def test_home_stops_an_inflight_command_and_late_events_do_not_move_the_receiver(self):
        p = self.panel
        old = self.start()
        p.home()
        new = p.token
        p.rx.send.assert_called_with('home', new)
        p.app.send.assert_called_with(dict(cmd='laser', value=0, request_id=old + '-stop'))
        p.on_rx(dict(kind='armed', token=old))
        p.on_rx(dict(kind='applied', token=old, pan=108, tilt=19))
        self.assertEqual(p.phase, 'home')
        p.on_rx(dict(kind='home', token=new, pan=103, tilt=20))
        self.assertIsNone(p.phase)
        p.pan.set.assert_called_once_with('0')
        self.assertEqual(len(p.app.send.call_args_list), 3)

    def test_home_needs_only_rx_and_invalid_offset_sends_nothing(self):
        p = self.panel
        p.pan.get.return_value = '78'
        p.transmit()
        p.rx.send.assert_not_called()
        p.app.send.assert_not_called()
        p.app.ook.online = False
        p.home()
        p.rx.send.assert_called_once_with('home', p.token)
        p.app.send.assert_not_called()

    def test_timeout_and_wrong_pose_never_report_success(self):
        for kind in ('timeout', 'applied'):
            with self.subTest(kind=kind):
                self.setUp()
                p = self.panel
                token = self.start()
                p.on_rx(dict(kind=kind, token=token, pan=111, tilt=20))
                self.assertIsNone(p.phase)
                p.rx.send.assert_called_with('cancel', token)
                self.assertNotIn('성공', p.status.set.call_args.args[0])

    def test_retry_optical_requires_the_confirmed_stop_even_after_leaving_tab(self):
        p = self.panel
        old = self.start()
        p.stop('중지')
        sent = p.app.send.call_count
        p.transmit()
        self.assertEqual(p.app.send.call_count, sent)
        p.event(dict(event='laser', request_id='stale', laser_gpio_level=0))
        self.assertTrue(p.off_token)
        p.selected = False
        p.event(dict(event='laser', request_id=old + '-stop', laser_gpio_level=0))
        self.assertIsNone(p.off_token)
