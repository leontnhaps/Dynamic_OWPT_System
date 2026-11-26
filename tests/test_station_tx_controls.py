import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from common.tx_setup import TX_OPERATING_LIMITS
from tests import test_ook_panel as panel_fixture
from Tx.Controller.station_tx_controls import StationTxControls


class StationTxControlsTests(unittest.TestCase):
    def setUp(self):
        self.e = panel_fixture.PanelStateTests().panel()
        self.app = self.e.app
        self.app.ook = self.e
        self.app.send = Mock(return_value=True)
        self.app.current = (b'', dict(simulated=False), time.monotonic())
        self.e.hardware = dict(simulated=False, pattern_active=False, laser_gpio_level=0)
        self.panel = SimpleNamespace(app=self.app, ready_tx_motion=Mock(), tx_moved=Mock(),
                                     phase=None, method=None, stop=Mock())
        self.c = StationTxControls.__new__(StationTxControls)
        self.c.app, self.c.panel = self.app, self.panel
        self.c.light = self.c.light_request = self.c.motion = None
        self.c.manual_on = False
        self.c.recover_off = False
        self.c.frame_target = self.c.session_reset_at = None
        self.c.step = panel_fixture.Value(1)
        self.c.readout = panel_fixture.Value()
        self.c.notice = panel_fixture.Value()

    def emit(self, kind, token, **fields):
        packet = dict(event=kind, request_id=token, **fields)
        # The application hands the same event to the OOK engine first.
        self.e.event(packet)
        self.c.event(packet)
        return packet

    def status(self, level, revision=1, token='initial-status'):
        return self.emit('hardware', token, simulated=False, pattern_active=False,
                         laser_gpio_level=level, gpio_revision=revision)

    def last_request(self):
        return self.app.send.call_args.args[0]

    def acknowledge_laser(self, level):
        request = self.last_request()
        self.assertEqual((request['cmd'], request['value']), ('laser', level))
        self.emit('laser', request['request_id'], laser_gpio_level=level)

    def servo(self, *, limits, commanded=None, state=None):
        cmd = self.last_request()
        self.emit('servo', cmd['request_id'], operation=cmd['cmd'], available=True, simulated=False,
                  limits=limits, commanded=commanded, state=state)

    def test_ten_second_measurement_brackets_read_only_and_confirm_same_gpio_revision(self):
        with self.assertRaises(ValueError):
            self.c.light_level()
        self.status(0)
        self.c.on()
        self.assertTrue(self.c.owns_output)
        with self.assertRaises(ValueError):
            self.c.light_level()
        self.acknowledge_laser(1)
        self.assertEqual(self.c.light_level(), 1)
        for token in ('run-before', 'run-after'):
            self.c.probe_light(token)
            self.assertEqual(self.last_request(), dict(cmd='hardware_status', request_id=token))
            event = self.status(1, revision=5, token=token)
            self.assertEqual(self.c.confirmed_light(event, token), (1, 5))
            for changes in [dict(simulated=True), dict(pattern_active=True), dict(request_id='other'),
                            dict(gpio_revision=None), dict(laser_gpio_level=True)]:
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    self.c.confirmed_light(dict(event, **changes), token)
        self.assertEqual([c.args[0]['cmd'] for c in self.app.send.call_args_list],
                         ['laser', 'hardware_status', 'hardware_status'])
        self.c.leave()
        self.assertTrue(self.c.owns_output)
        self.acknowledge_laser(0)
        self.assertFalse(self.c.owns_output)
        self.assertEqual(self.c.light_level(), 0)

    def test_lost_video_after_on_requests_off_and_unknown_reply_is_not_confirmation(self):
        self.status(0)
        self.c.on()
        self.acknowledge_laser(1)
        self.app.current = None
        self.c.tick()
        off = self.last_request()
        self.assertEqual(off['value'], 0)
        self.emit('laser', off['request_id'])
        self.assertIsNone(self.c.light)
        self.assertTrue(self.c.owns_output)
        self.c.off()
        self.acknowledge_laser(0)
        self.assertFalse(self.c.owns_output)

    def test_manual_off_remains_available_with_active_trial_and_pending_on(self):
        self.status(0)
        self.c.on()
        on = self.last_request()
        self.panel.phase = 'measuring'
        self.c.off()
        self.panel.stop.assert_called_once()
        off = self.last_request()
        self.assertEqual(off['value'], 0)
        self.emit('laser', on['request_id'], laser_gpio_level=1)
        self.assertTrue(self.c.busy)
        self.acknowledge_laser(0)
        self.assertEqual(self.c.light_level(), 0)
        self.assertFalse(self.c.owns_output)

    def test_jog_queries_live_pose_and_moves_one_step_within_tighter_configured_limits(self):
        self.c.jog('pan', 1)
        self.assertEqual(self.last_request()['cmd'], 'hardware_status')
        self.status(0, token=self.last_request()['request_id'])
        self.assertEqual(self.last_request()['cmd'], 'servo_status')
        limits = dict(pan_min=-1, pan_max=1, tilt_min=-1, tilt_max=1)
        self.servo(limits=limits, commanded=dict(pan=0, tilt=-.5, command_id='previous'))
        move = self.last_request()
        self.assertEqual((move['cmd'], move['pan'], move['tilt']), ('move', 1, -.5))
        self.panel.tx_moved.assert_called_once()
        self.servo(limits=limits, commanded=dict(pan=1, tilt=-.5, speed=move['speed'], acc=move['acc'],
                                                 command_id=move['request_id']), state='serial_written')
        self.assertFalse(self.c.busy)
        self.assertIn('Pan 1°', self.c.notice.get())
        def frame(command_id, pan):
            return (b'', dict(servo_at_capture_start=dict(commanded=dict(command_id=command_id, pan=pan,
                                                                          tilt=-.5))), time.monotonic())
        for stale in (frame('previous', 1), frame(move['request_id'], 0)):
            with self.assertRaises(ValueError):
                self.c.check_frame(stale)
        self.c.check_frame(frame(move['request_id'], 1))

        self.c.jog('pan', 1)
        self.status(0, token=self.last_request()['request_id'])
        self.servo(limits=limits, commanded=dict(pan=1, tilt=-.5, command_id=move['request_id']))
        self.assertEqual(self.last_request()['cmd'], 'servo_status')
        self.assertFalse(self.c.busy)

    def test_initial_zero_uses_project_limits_and_blocked_or_invalid_requests_never_move(self):
        self.c.jog(None)
        self.status(1, token=self.last_request()['request_id'])
        self.assertFalse(self.c.busy)
        self.assertEqual(self.last_request()['cmd'], 'hardware_status')
        self.c.jog('tilt', -1)
        self.status(0, token=self.last_request()['request_id'])
        self.servo(limits=TX_OPERATING_LIMITS)
        self.assertFalse(self.c.busy)
        self.assertEqual(self.last_request()['cmd'], 'servo_status')
        self.c.jog(None)
        self.status(0, token=self.last_request()['request_id'])
        self.servo(limits=None)
        self.assertEqual(self.last_request()['cmd'], 'servo_config')
        self.assertTrue(TX_OPERATING_LIMITS.items() <= self.last_request().items())
        self.servo(limits=TX_OPERATING_LIMITS)
        move = self.last_request()
        self.assertEqual((move['cmd'], move['pan'], move['tilt']), ('move', 0, 0))
        self.c.leave()
        self.assertFalse(self.c.busy)

    def test_no_move_after_leave_or_late_response_and_no_on_with_simulated_tx(self):
        self.c.jog('pan', 1)
        token = self.last_request()['request_id']
        self.c.leave()
        self.status(0, token=token)
        self.assertEqual(self.app.send.call_count, 1)
        self.e.hardware['simulated'] = True
        self.c.on()
        self.assertEqual(self.app.send.call_count, 1)

    def test_reconnect_rejects_old_frame_and_turns_off_unexpected_persisting_manual_light(self):
        self.status(0)
        self.c.on()
        self.acknowledge_laser(1)
        old_frame = self.app.current
        self.emit('network', None, port=7600, state='disconnected')
        with self.assertRaises(ValueError):
            self.c.check_frame(old_frame)
        self.emit('ready', None, simulated=False)
        probe = self.last_request()
        self.assertEqual(probe['cmd'], 'hardware_status')
        self.status(1, token=probe['request_id'])
        self.assertTrue(self.c.owns_output)
        self.acknowledge_laser(0)
        self.assertFalse(self.c.owns_output)
        self.app.current = (b'', dict(simulated=False), self.c.session_reset_at + .03)
        self.c.check_frame(self.app.current)
        self.assertEqual(self.c.light_level(), 0)

    def test_unknown_light_can_be_requeried_after_timeout_without_cached_hardware_or_video(self):
        self.c.manual_on = True
        self.e.hardware = self.app.current = None
        self.e.output_requested = True
        self.c.refresh()
        request = self.last_request()
        self.assertEqual(request['cmd'], 'hardware_status')
        self.status(0, token=request['request_id'])
        self.assertFalse(self.e.output_requested)
        self.assertFalse(self.c.owns_output)
        self.assertEqual(self.c.light, 0)
        self.assertIs(self.e.hardware['simulated'], False)
