import json
import queue
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from common.angle_demo import BIT_MS, HOME_PAN, HOME_TILT, encode_angles, payload_from_frame
from Tx.Controller.point_comparison_panel import FIXED_LIGHT, METHODS, PointComparisonPanel


class Value:
    def __init__(self, value=''):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class StationaryPointFlow(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.panel = p = PointComparisonPanel.__new__(PointComparisonPanel)
        p.app = SimpleNamespace(
            current=(b'PV snapshot', dict(simulated=False, session='tx1', servo_at_capture_start={
                'commanded': {'pan': 12, 'tilt': 8, 'command_id': 42}}), time.monotonic()),
            stage_dir=lambda stage: self.root/stage, send=Mock(return_value=True),
            ook=Mock(online=True, hardware={'simulated': False, 'capabilities': ['laser_pattern']}))
        p.rx = Mock(connected=True, terminal_error=None, events=queue.Queue())
        p.phase = p.token = p.off_token = p.logpath = None
        p.run = p.station = p.method = None
        p.selected = p.prepared = True
        p.pan, p.tilt = Value('0'), Value('0')
        p.status, p.off_status, p.applied = Value(), Value(), Value()
        p.rx_status, p.location, p.verification = Value(), Value(), Value()
        p.results = {key: Value() for key in METHODS}
        p.inputs = Mock()
        p.inputs.snapshot.return_value = {'point_id': 'P1', 'xyz_m': [1, 2, 3]}
        p.inputs.estimate_xyz.return_value = (107, 22, {'from': 'xyz'})
        p.inputs.poll_yolo.return_value = (108, 21, {'from': 'yolo'})
        p.register()
        self.addCleanup(self.close_trial)

    def close_trial(self):
        if self.panel.run.trial_folder:
            self.panel.stop('test end')

    def start_at_home(self, method):
        p = self.panel
        p.start_trial(method)
        p.rx.send.assert_called_with('home', p.token)
        p.rx.events.put(dict(kind='home', token=p.token, pan=HOME_PAN, tilt=HOME_TILT))
        p.tick()
        self.assertEqual(p.phase, 'home_settle')
        self.after_settle()

    def after_settle(self):
        p = self.panel
        p.until = time.monotonic() - 0.1
        frame, meta, _ = p.app.current
        p.app.current = frame, meta, time.monotonic()
        p.tick()

    def begin_voltage(self):
        p = self.panel
        self.assertEqual(p.phase, 'light_start')
        token = p.token
        p.app.send.assert_called_with(dict(cmd='laser_pattern', request_id=token, **FIXED_LIGHT))
        p.event(dict(event='laser_pattern', state='started', request_id=token, simulated=False))
        p.tick()
        p.app.send.assert_called_with(dict(cmd='hardware_status', request_id=token))
        state = dict(event='hardware', request_id=token, simulated=False, pattern_active=True,
                     pattern_request_id=token)
        p.event(dict(state, laser_gpio_level=0))
        self.assertEqual(p.phase, 'light_check')
        p.event(dict(state, laser_gpio_level=1))
        p.rx.send.assert_called_with('measure', token)
        return token

    def complete_light(self, token, off_ms):
        self.panel.event(dict(event='laser_pattern', request_id=token, state='completed',
                              simulated=False, settings=FIXED_LIGHT,
                              trace=[dict(index=0, level=1, gpio_call_done_ms=.13)],
                              off_call_done_ms=off_ms))

    def voltage_receipt(self, token):
        p = self.panel
        start = 180_000_000
        for i in range(200):
            p.rx.events.put(dict(kind='voltage', token=token, seq=i+100, bus_raw=250,
                                 rx_monotonic_ns=start+i*5_000_000))
        p.rx.events.put(dict(kind='measured', token=token, count=200, window_start_ns=start,
                             window_end_ns=start+1_000_000_000, max_gap_ns=5_000_000,
                             pc_receive_monotonic_ns=time.monotonic_ns()))
        p.tick()

    def test_baseline_waits_for_rx_receipt_and_confirmed_bounded_light_off(self):
        p = self.panel
        self.start_at_home('baseline')
        token = self.begin_voltage()
        p.on_rx(dict(kind='voltage', token='old-window', seq=112, rx_monotonic_ns=10, bus_raw=8191))
        self.voltage_receipt(token)
        self.assertEqual(p.run.results[-1]['status'], 'running')
        self.complete_light(token, FIXED_LIGHT['bit_ms'] + .12)
        row = p.run.results[-1]
        self.assertTrue(row['success'])
        self.assertEqual((row['mean_v'], row['n']), (1.0, 200))
        self.assertIn('1.000 V', p.results['baseline'].get())
        self.assertFalse(p.owns_output)

    def test_yolo_and_xyz_each_return_home_then_wait_for_explicit_optical_send(self):
        p = self.panel
        for method, goal in [('yolo', (108, 21)), ('xyz', (107, 22))]:
            with self.subTest(method=method):
                p.app.send.reset_mock()
                self.start_at_home(method)
                if method == 'yolo':
                    self.assertEqual(p.phase, 'vision')
                    p.inputs.start_yolo.assert_called_once_with(p.station, p.app.current)
                    p.tick()
                    self.assertEqual((p.run.trial_folder/'frame.jpg').read_bytes(), p.app.current[0])
                else:
                    p.inputs.estimate_xyz.assert_called_once_with(p.station)
                self.assertEqual(p.phase, 'confirm_estimate')
                p.app.send.assert_not_called()
                p.transmit()
                token = p.token
                p.event(dict(event='laser', request_id=token, laser_gpio_level=0))
                p.rx.send.assert_called_with('arm', token)
                p.rx.events.put(dict(kind='armed', token=token))
                p.tick()
                pattern = dict(bits=encode_angles(*goal), bit_ms=BIT_MS, repeats=1)
                p.app.send.assert_called_with(dict(cmd='laser_pattern', request_id=token, **pattern))
                decoded = dict(kind='decoded', token=token, pan=goal[0], tilt=goal[1],
                               rx_payload_bits=payload_from_frame(pattern['bits']))
                p.rx.events.put(decoded)
                p.tick()
                self.assertEqual(p.phase, 'optical')
                p.event(dict(event='laser_pattern', request_id=token, state='completed', simulated=False,
                             settings=pattern, off_call_done_ms=len(pattern['bits']) * BIT_MS + .1,
                             trace=[dict(index=i, level=int(bit)) for i, bit in enumerate(pattern['bits'])]))
                p.rx.send.assert_called_with('commit', token)
                self.assertEqual(p.phase, 'optical')
                p.rx.events.put(dict(decoded, kind='applied', servo_writes=[
                    ['1.00','1.01',0,goal[0],'','applied'], ['1.01','1.02',1,goal[1],'','applied']]))
                p.tick()
                self.assertEqual(p.phase, 'pose_settle')
                self.after_settle()
                light_token = self.begin_voltage()
                self.assertNotEqual(token, light_token)
                self.voltage_receipt(light_token)
                self.complete_light(light_token, FIXED_LIGHT['bit_ms'] + .1)
                row = p.run.results[-1]
                self.assertTrue(row['success'])
                events = (p.run.folder/row['trial_id']/'events.jsonl').read_text(encoding='utf-8').splitlines()
                estimates = [json.loads(e) for e in events if '"event": "estimate"' in e]
                self.assertEqual((estimates[0]['pan'], estimates[0]['tilt']), goal)
                checks = [json.loads(e) for e in events if '"event": "angle_verification"' in e]
                self.assertEqual(len(checks), 1)
                self.assertTrue(checks[0]['success'])

    def test_early_light_end_fails_and_late_voltage_cannot_reverse_outcome(self):
        p = self.panel
        self.start_at_home('baseline')
        token = self.begin_voltage()
        self.complete_light(token, 400.0)
        self.assertFalse(p.run.results[-1]['success'])
        p.app.send.assert_called_with(dict(cmd='laser', value=0, request_id=token+'-stop'))
        self.voltage_receipt(token)
        self.assertIsNone(p.run.results[-1]['mean_v'])

    def test_tx_disconnect_aborts_and_rx_disconnect_aborts_an_independent_trial(self):
        p = self.panel
        self.start_at_home('baseline')
        token = self.begin_voltage()
        p.event(dict(event='network', port=7600, state='disconnected'))
        self.assertFalse(p.run.results[-1]['success'])
        p.event(dict(event='laser', request_id=token+'-stop', laser_gpio_level=0))
        self.assertIsNone(p.off_token)
        self.start_at_home('xyz')
        p.rx.terminal_error = 'SSH transport stopped'
        p.tick()
        self.assertIsNone(p.rx)
        self.assertFalse(p.run.results[-1]['success'])
        self.assertEqual(len(p.run.results), 2)
