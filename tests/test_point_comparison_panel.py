import json
import queue
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from common.angle_demo import BIT_MS, HOME_PAN, HOME_TILT, encode_angles, payload_from_frame
from Tx.Controller.point_comparison_panel import METHODS, PointComparisonPanel
from Tx.Controller.station_tx_controls import StationTxControls
from tests.test_angle_demo_panel import armed, completed_pattern, decoded_frame, init_wave_test, waveform_receipt
from tests.test_ook_panel import Value


class StationaryPointFlow(unittest.TestCase):
    def setUp(self):
        work = tempfile.TemporaryDirectory()
        self.addCleanup(work.cleanup)
        self.root = Path(work.name)
        p = self.panel = PointComparisonPanel.__new__(PointComparisonPanel)
        p.app = SimpleNamespace(
            current=(b'PV snapshot', dict(simulated=False, session='tx1', servo_at_capture_start={
                'simulated': False, 'commanded': {'pan': 12, 'tilt': 8, 'command_id': 'aim1'}}),
                     time.monotonic(), time.time_ns()),
            stage_dir=lambda stage: self.root / stage, send=Mock(return_value=True),
            ook=Mock(online=True, hardware={'simulated': False, 'capabilities': ['laser_pattern']}))
        p.rx = Mock(connected=True, terminal_error=None, events=queue.Queue())
        init_wave_test(self, p)
        p.phase = p.token = p.off_token = p.logpath = None
        p.run = p.station = p.method = p.station_view = None
        p.pose_method = p.pose_source = p.pose_frame = p.last_servo_pose = None
        p.selected = p.prepared = True
        p.pan, p.tilt = Value('0'), Value('0')
        p.status, p.off_status, p.applied = Value(), Value(), Value()
        p.rx_status, p.location, p.verification = Value(), Value(), Value()
        p.results = {key: Value() for key in METHODS}
        p.targets = {key: Value() for key in ('xyz', 'yolo')}
        p.inputs = Mock()
        p.inputs.snapshot.return_value = {'point_id': 'P1', 'rx_minus_tx_m': [0, 6.3, .4]}
        p.inputs.estimate_xyz.return_value = (103, 24, {'from': 'xyz'})
        p.inputs.poll_yolo.return_value = (108, 21, {'from': 'yolo'})
        p.tx_controls = Mock(busy=False, owns_output=False)
        p.tx_controls.confirmed_light = StationTxControls.confirmed_light
        p.tx_controls.light_level.return_value = 0
        p.tx_controls.probe_light.side_effect = lambda t: p.app.send(dict(cmd='hardware_status', request_id=t))
        p.station_changed()
        self.addCleanup(self.close_trial)

    def close_trial(self):
        p = self.panel
        if p.run and p.run.trial_folder:
            p.stop('test end')

    def home(self):
        p = self.panel
        p.home()
        p.rx.send.assert_called_with('home', p.token)
        p.on_rx(dict(kind='home', token=p.token, pan=HOME_PAN, tilt=HOME_TILT))
        self.assertEqual(p.pose_method, 'baseline')

    def gpio(self, suffix, level, revision):
        p = self.panel
        p.event(dict(event='hardware', request_id=p.token + suffix, simulated=False,
                     laser_gpio_level=level, gpio_revision=revision, pattern_active=False))

    def begin_voltage(self, level):
        p = self.panel
        p.tx_controls.light_level.return_value = level
        p.pose_ready_at = 0
        p.measure()
        self.assertEqual(p.phase, 'light_before', p.status.get())
        self.gpio('-before', level, 80)
        self.assertEqual(p.phase, 'voltage_window')
        p.rx.send.assert_called_with('measure', p.token)

    def voltage_receipt(self, code=250):
        p = self.panel
        start = 180_000_000
        for i in range(2000):
            p.on_rx(dict(kind='voltage', token=p.token, seq=i+100, bus_raw=code,
                         rx_monotonic_ns=start + i*5_000_000))
        p.on_rx(dict(kind='measured', token=p.token, count=2000, window_start_ns=start,
                     window_end_ns=start+10_000_000_000, max_gap_ns=5_000_000))
        self.assertEqual(p.phase, 'light_after')

    def optical_move(self, method):
        p = self.panel
        p.tx_controls.light_level.return_value = 0
        p.move_to(method)
        if method == 'yolo':
            self.assertEqual(p.phase, 'vision')
            p.inputs.start_yolo.assert_called_with(p.station, p.app.current)
            p.tick()
        self.assertEqual(p.phase, 'off', p.status.get())
        token = p.token
        p.event(dict(event='laser', request_id=token, laser_gpio_level=0))
        p.rx.arm_angle.assert_called_with(token, BIT_MS)
        p.on_rx(armed(token))
        goal = (103, 24) if method == 'xyz' else (108, 21)
        bits = encode_angles(*goal)
        pattern = dict(bits=bits, bit_ms=BIT_MS, repeats=1)
        p.app.send.assert_called_with(dict(cmd='laser_pattern', request_id=token, **pattern))
        decoded = decoded_frame(token, *goal)
        p.event(completed_pattern(token, bits))
        p.rx.send.assert_called_with('finish_angle', token)
        waveform_receipt(p, token)
        p.on_rx(decoded)
        p.rx.send.assert_called_with('commit', token)
        self.assertIsNone(p.pose_method)
        p.on_rx(dict(decoded, kind='applied', servo_writes=[
            ['1.0','1.1',0,goal[0],'','applied'], ['1.1','1.2',1,goal[1],'','applied']]))
        self.assertEqual(p.pose_method, method)
        self.assertIsNone(p.phase)
        self.assertEqual(p.last_servo_pose, goal)
        self.assertIn(f'{goal[0]}° / {goal[1]}°', p.targets[method].get())
        return token

    def test_six_independent_windows_never_switch_output_or_move_pose(self):
        p = self.panel
        for method in METHODS:
            if method == 'baseline':
                self.home()
            else:
                self.optical_move(method)
            for level in (0, 1):
                p.app.send.reset_mock()
                p.rx.send.reset_mock()
                self.begin_voltage(level)
                p.on_rx(dict(kind='voltage', token='old', seq=1, rx_monotonic_ns=0, bus_raw=8191))
                self.voltage_receipt(250+level*25)
                self.assertEqual(p.run.results[-1]['status'], 'running')
                self.gpio('-after', level, 80)
                row = p.run.results[-1]
                self.assertEqual((row['method'], row['laser_on'], row['success'], row['n']),
                                 (method, bool(level), True, 2000))
                self.assertAlmostEqual(row['mean_v'], 1 + level*.1)
                self.assertEqual([c.args[0]['cmd'] for c in p.app.send.call_args_list],
                                 ['hardware_status', 'hardware_status'])
                self.assertEqual([c.args[0] for c in p.rx.send.call_args_list], ['measure'])
                self.assertIsNone(p.phase)
                self.assertEqual(p.pose_method, method)
                if method == 'yolo':
                    self.assertEqual((p.run.folder / row['trial_id'] / 'frame.jpg').read_bytes(), b'PV snapshot')
        self.assertIn('6/6', p.location.get())
        events = [json.loads(line) for line in (p.run.folder / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual([e['method'] for e in events if e['event'] == 'angle_verification' and e['success']],
                         ['xyz', 'yolo'])

    def test_new_station_and_changed_tx_cannot_reuse_previous_measurement_group(self):
        p = self.panel
        self.home()
        self.begin_voltage(0)
        p.inputs.snapshot.return_value = {'point_id': 'P2', 'rx_minus_tx_m': [2.7, 7.65, .4]}
        old = p.run
        p.station_changed()
        self.assertFalse(old.results[-1]['success'])
        p.measure()
        self.assertIsNone(p.run)
        self.assertIsNone(p.pose_method)
        self.home()
        self.begin_voltage(1)
        self.assertNotEqual(old.folder, p.run.folder)
        p.app.current[1]['servo_at_capture_start']['commanded']['command_id'] = 'aim2'
        p.tick()
        self.assertFalse(p.run.results[-1]['success'])

    def test_same_laser_level_after_switching_during_measurement_is_rejected(self):
        p = self.panel
        self.home()
        self.begin_voltage(1)
        self.voltage_receipt()
        self.gpio('-after', 1, 82)
        row = p.run.results[-1]
        self.assertFalse(row['success'])
        self.assertEqual(row['n'], 2000)
        self.assertIsNone(row['mean_v'])

    def test_entering_yolo_axis_signs_after_baseline_preserves_reference_and_data(self):
        p = self.panel
        self.home()
        self.begin_voltage(0)
        self.voltage_receipt()
        self.gpio('-after', 0, 80)
        first = p.run
        p.inputs.snapshot.side_effect = ValueError('incomplete calibration')
        p.station_changed()
        self.assertIs(p.run, first)
        p.inputs.snapshot.side_effect = None
        p.inputs.snapshot.return_value = dict(p.station, tx_pan_plus='왼쪽', tx_tilt_plus='위')
        p.station_changed()
        self.assertIs(p.run, first)
        self.assertEqual(p.pose_method, 'baseline')
        self.assertIn('1/6', p.location.get())
        self.optical_move('yolo')
        self.begin_voltage(0)
        self.voltage_receipt(260)
        self.gpio('-after', 0, 80)
        self.assertEqual(first.results[0]['trial_id'], first.results[1]['baseline_trial_id'])
        self.assertAlmostEqual(first.results[1]['delta_v'], .04)

    def test_measure_double_click_or_ook_click_during_window_does_not_abort(self):
        p = self.panel
        self.home()
        self.begin_voltage(0)
        token = p.token
        p.measure()
        p.move_to('xyz')
        p.home()
        self.assertEqual((p.token, p.phase), (token, 'voltage_window'))
        self.assertEqual(p.run.results[-1]['status'], 'running')

    def test_home_is_ssh_only_and_does_not_require_transmitter_or_station(self):
        p = self.panel
        p.app.current = None
        self.home()
        self.assertIsNone(p.run)
        p.app.send.assert_not_called()
        self.assertEqual(p.last_servo_pose, (103, 20))

    def test_ook_move_requires_manual_off_and_rejects_incorrect_decoding(self):
        p = self.panel
        p.tx_controls.light_level.return_value = 1
        p.move_to('xyz')
        p.app.send.assert_not_called()
        self.assertIn('OFF', p.status.get())
        p.tx_controls.light_level.return_value = 0
        p.move_to('xyz')
        token = p.token
        p.event(dict(event='laser', request_id=token, laser_gpio_level=0))
        p.on_rx(armed(token))
        p.event(completed_pattern(token, encode_angles(103, 24)))
        waveform_receipt(p, token)
        p.on_rx(decoded_frame(token, 104, 24))
        self.assertIsNone(p.phase)
        self.assertIsNone(p.pose_method)
        self.assertNotIn('commit', [c.args[0] for c in p.rx.send.call_args_list])
        self.assertFalse(p.run.results)
