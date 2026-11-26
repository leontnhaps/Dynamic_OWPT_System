import queue
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from common.angle_demo import BIT_MS, HOME_PAN, HOME_TILT, encode_angles
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

    def test_check_both_steps_wait_for_distinct_acks_without_tx(self):
        p = self.panel
        p.app.ook.online = False
        p.app.current = None
        p.check_both()
        poses = [(HOME_PAN, HOME_TILT), (HOME_PAN+5, HOME_TILT), (HOME_PAN, HOME_TILT),
                 (HOME_PAN, HOME_TILT+5), (HOME_PAN, HOME_TILT)]
        previous = None
        for i, (pan, tilt) in enumerate(poses):
            token = p.token
            self.assertTrue(token and token != previous)
            kind = 'home' if i % 2 == 0 else 'jogged'
            if kind == 'home':
                p.rx.send.assert_called_with('home', token)
            else:
                p.rx.jog.assert_called_with(token, pan - HOME_PAN, tilt - HOME_TILT)
            p.on_rx(dict(kind=kind, token='old', pan=pan, tilt=tilt))
            self.assertEqual(p.token, token)
            p.on_rx(dict(kind=kind, token=token, pan=pan, tilt=tilt))
            self.assertEqual(p.last_servo_pose, (pan, tilt))
            if i < len(poses)-1:
                p.tick()
                self.assertEqual(p.phase, 'check_pause')
                p.check_due = 0
                p.tick()
            previous = token
        self.assertIsNone(p.phase)
        p.app.send.assert_not_called()

    def test_click_jog_uses_acknowledged_pose_and_rejects_out_of_range(self):
        p = self.panel
        p.last_servo_pose = None
        p.jog_size = Mock()
        p.jog_size.get.return_value = 5
        p.jog(0, 1)
        p.rx.jog.assert_not_called()
        p.applied_pose(180, 25)
        p.jog(1, 0)
        p.rx.jog.assert_not_called()
        p.jog(-1, 0)
        p.rx.jog.assert_called_once_with(p.token, -5, 0)
        p.on_rx(dict(kind='jogged', token=p.token, pan=175, tilt=25))
        self.assertIsNone(p.phase)
        p.jog(0, 1)
        p.on_rx(dict(kind='jogged', token=p.token, pan=175, tilt=30))
        self.assertEqual(p.last_servo_pose, (175, 30))
        p.app.send.assert_not_called()

    def test_pose_readout_tracks_current_session_servo_ack(self):
        p = self.panel
        p.on_rx(dict(kind='ready'))
        self.assertIn('Pan —    Tilt —', p.applied.set.call_args.args[0])
        p.applied_pose(HOME_PAN, HOME_TILT)
        p.jog_size = Mock()
        p.jog_size.get.return_value = 1
        p.jog(-1, 0)
        p.on_rx(dict(kind='jogged', token='old', pan=99, tilt=22))
        self.assertIn('Pan 103°    Tilt 20°', p.applied.set.call_args.args[0])
        p.on_rx(dict(kind='jogged', token=p.token, pan=102, tilt=20))
        self.assertIn('원위치 대비 Pan -1°, Tilt +0°', p.applied.set.call_args.args[0])
        p.jog(0, 1)
        p.on_rx(dict(kind='jogged', token=p.token, pan=102, tilt=21))
        self.assertIn('Pan 102°    Tilt 21°', p.applied.set.call_args.args[0])
        p.leave()
        self.assertIn('Pan —    Tilt —', p.applied.set.call_args.args[0])

    def test_stopped_or_mismatched_check_never_dispatches_another_step(self):
        p = self.panel
        p.app.ook.online = False
        p.check_both()
        token = p.token
        p.stop_servo_check()
        p.on_rx(dict(kind='home', token=token, pan=HOME_PAN, tilt=HOME_TILT))
        p.tick()
        p.rx.jog.assert_not_called()
        p.check_both()
        p.on_rx(dict(kind='home', token=p.token, pan=HOME_PAN+1, tilt=HOME_TILT))
        self.assertIsNone(p.phase)
        self.assertIsNone(p.last_servo_pose)
        p.rx.jog.assert_not_called()
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
