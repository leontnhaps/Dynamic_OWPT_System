import csv
import json
import queue
import tempfile
import time
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock, call

from common.angle_demo import ANGLE_BIT_PERIODS, BIT_MS, HOME_PAN, HOME_TILT, encode_angles, payload_from_frame
from Rx.ina219_ook_stream import SAMPLE_CONFIG
from Tx.Controller.angle_demo_panel import AngleDemoPanel


def init_wave_test(case, panel):
    panel.rate_box, panel.wave_review, panel.target = Mock(), Mock(), Mock()
    panel.target.get.return_value = 'unit@loopback'
    panel.rate_box.current.return_value = ANGLE_BIT_PERIODS.index(BIT_MS)
    def cleanup():
        if panel.wave is not None and not panel.wave.closed:
            panel.wave.file.close()
    case.addCleanup(cleanup)


def armed(token, period=BIT_MS):
    return dict(kind='armed', token=token, bit_ms=period, sample_ms=1, interval_ms=1,
                bus=3, address=0x40, config=SAMPLE_CONFIG, capture_mode='post_tx')


def decoded_frame(token, pan, tilt):
    payload = payload_from_frame(encode_angles(pan, tilt))
    return dict(kind='decoded', token=token, pan=pan, tilt=tilt, rx_payload_bits=payload,
                frame_check=dict(verified=True, header_verified=True, payload_bits=payload,
                                 tail_bits='0000', reason=''))


def completed_pattern(token, bits, bit_ms=BIT_MS):
    return dict(event='laser_pattern', state='completed', simulated=False, request_id=token,
                off_call_done_ms=len(bits)*bit_ms + 0.1,
                trace=[dict(index=i, level=int(bit)) for i, bit in enumerate(bits)],
                settings=dict(bits=bits, bit_ms=bit_ms, repeats=1))


def waveform_receipt(panel, token):
    panel.on_rx(dict(kind='angle_samples', token=token, rows=[[0, 1000000, 100], [2, 3000000, 200]],
                     pc_receive_monotonic_ns=590000, pc_receive_unix_ns=900000))
    panel.on_rx(dict(kind='angle_samples_end', token=token, count=2))


class DemoTabTests(unittest.TestCase):
    def setUp(self):
        work = tempfile.TemporaryDirectory()
        self.addCleanup(work.cleanup)
        self.panel = p = AngleDemoPanel.__new__(AngleDemoPanel)
        p.rx = Mock(connected=True, terminal_error=None, events=queue.Queue())
        p.app = SimpleNamespace(send=Mock(return_value=True), ook=Mock(
            online=True, hardware={'simulated': False, 'capabilities': ['laser_pattern']}),
            current=(None, None, time.monotonic()))
        p.app.stage_dir = lambda stage: Path(work.name) / stage
        init_wave_test(self, p)
        p.record = Mock()
        p.phase = p.token = None
        p.pan, p.tilt = Mock(), Mock()
        p.pan.get.return_value, p.tilt.get.return_value = '5', '-1'
        p.off_token = None
        p.applied, p.status, p.rx_status, p.off_status, p.verification = (Mock() for _ in range(5))
        p.selected = p.prepared = True

    def start(self, period=BIT_MS):
        p = self.panel
        p.rate_box.current.return_value = ANGLE_BIT_PERIODS.index(period)
        p.transmit()
        token = p.token
        self.assertEqual(p.phase, 'off')
        p.event(dict(event='laser', request_id=token, laser_gpio_level=0))
        p.rx.arm_angle.assert_called_once_with(token, period)
        p.on_rx(armed(token, period))
        p.app.send.assert_called_with(dict(cmd='laser_pattern', request_id=token,
                                           bits=encode_angles(108, 19), bit_ms=period, repeats=1))
        return token

    def optical_receipts(self, token):
        bits = encode_angles(108, 19)
        decoded = decoded_frame(token, 108, 19)
        applied = dict(decoded, kind='applied', servo_writes=[
            ['1.00', '1.01', 0, 108, '', 'applied'], ['1.01', '1.02', 1, 19, '', 'applied']])
        return completed_pattern(token, bits), decoded, applied

    def test_verified_tx_finishes_capture_once_and_only_matching_decode_approves(self):
        p = self.panel
        token = self.start()
        complete, decoded, applied = self.optical_receipts(token)
        p.rx.send.assert_not_called()
        p.event(complete)
        p.event(complete)
        p.rx.send.assert_called_once_with('finish_angle', token)
        waveform_receipt(p, token)
        p.on_rx(decoded)
        p.on_rx(decoded)
        p.event(complete)
        self.assertEqual(p.rx.send.call_args_list, [call('finish_angle', token), call('commit', token)])
        self.assertEqual(p.phase, 'optical')
        p.on_rx(applied)
        self.assertIsNone(p.phase)
        self.assertIn('OOK 수신 성공', p.status.set.call_args.args[0])
        report = p.record.call_args.args[0]
        self.assertTrue(report['success'])
        self.assertEqual(report['rx_payload_bits'], report['tx_payload_bits'])
        self.assertEqual(report['rx_angles'], report['servo_angles'])
        self.assertEqual(report['frame_check'], decoded['frame_check'])
        self.assertIn('32/32비트 일치', p.verification.set.call_args.args[0])

    def test_missing_tail_reports_observed_payload_but_cannot_move(self):
        p = self.panel
        token = self.start()
        folder = p.wave.folder
        complete, decoded, applied = self.optical_receipts(token)
        p.event(complete)
        waveform_receipt(p, token)
        frame_check = dict(decoded['frame_check'], verified=False, tail_bits='000?',
                           reason='끝 비트의 표본이 없습니다.')
        p.on_rx(dict(kind='error', token=token, frame_check=frame_check, message=frame_check['reason']))
        p.on_rx(decoded)
        p.on_rx(applied)
        self.assertEqual(p.rx.send.call_args_list, [call('finish_angle', token), call('cancel', token)])
        self.assertIn('32/32비트 일치', p.verification.set.call_args.args[0])
        self.assertIn('000?', p.verification.set.call_args.args[0])
        self.assertIn('끝 비트', p.status.set.call_args.args[0])
        self.assertIsNone(p.decoded_pose)
        self.assertIsNone(p.write_pose)
        summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
        self.assertFalse(summary['angle_verified'])
        self.assertEqual(summary['frame_check'], frame_check)

    def test_early_decode_before_finish_or_capture_end_cannot_move(self):
        for sent_finish in (False, True):
            with self.subTest(sent_finish=sent_finish):
                self.setUp()
                p = self.panel
                token = self.start()
                complete, decoded, applied = self.optical_receipts(token)
                if sent_finish:
                    p.event(complete)
                p.on_rx(decoded)
                p.event(complete)
                waveform_receipt(p, token)
                p.on_rx(applied)
                self.assertFalse(p.commit_sent)
                self.assertIsNone(p.phase)

    def test_selected_rate_is_immutable_until_recording_finishes(self):
        p = self.panel
        token = self.start(20)
        folder = p.wave.folder
        p.rate_box.current.return_value = ANGLE_BIT_PERIODS.index(8)
        self.assertEqual(p.bit_ms, 20)
        self.assertEqual(p.app.send.call_args.args[0]['bit_ms'], 20)
        waveform_receipt(p, token)
        p.on_rx(dict(kind='timeout', token=token))
        p.wave_review.analyze.assert_called_with(folder)
        summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
        self.assertFalse(summary['angle_verified'])
        self.assertTrue(summary['waveform_complete'])
        self.assertEqual(summary['bit_ms'], 20)
        with (folder / 'rx.csv').open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([int(row['rx_monotonic_ns']) for row in rows], [1000000, 3000000])
        self.assertEqual([float(row['pv_V']) for row in rows], [.4, .8])

    def test_stop_preserves_cancelled_capture_until_tail_and_ignores_foreign_samples(self):
        p = self.panel
        token = self.start()
        folder = p.wave.folder
        p.stop('사용자 중지')
        self.assertIsNotNone(p.wave)
        waveform_receipt(p, 'other')
        waveform_receipt(p, token)
        p.wave_review.analyze.assert_called_once_with(folder)
        self.assertIsNone(p.wave)
        self.assertFalse(json.loads((folder/'summary.json').read_text(encoding='utf-8'))['angle_verified'])

    def test_rate_mismatch_blocks_optical_transmission(self):
        p = self.panel
        p.transmit()
        token = p.token
        p.event(dict(event='laser', request_id=token, laser_gpio_level=0))
        p.on_rx(armed(token, 5))
        self.assertIsNone(p.phase)
        self.assertNotIn('laser_pattern', [call.args[0]['cmd'] for call in p.app.send.call_args_list])

    def test_payload_mismatch_or_bad_tx_trace_prevents_commit(self):
        for problem in ('bits', 'pose', 'trace', 'short'):
            with self.subTest(problem=problem):
                self.setUp()
                p = self.panel
                p.applied_pose(105, 22)
                token = self.start()
                complete, decoded, _ = self.optical_receipts(token)
                if problem == 'bits':
                    decoded['rx_payload_bits'] = '1' * 32
                elif problem == 'pose':
                    decoded['pan'] = 110
                elif problem == 'trace':
                    complete['trace'][3]['level'] ^= 1
                else:
                    complete['off_call_done_ms'] = 0
                p.event(complete)
                waveform_receipt(p, token)
                p.on_rx(decoded)
                self.assertFalse(p.commit_sent)
                self.assertIsNone(p.phase)
                self.assertEqual(p.last_servo_pose, (105, 22))
                self.assertNotIn('성공', p.status.set.call_args.args[0])

    def test_stale_approval_or_partial_servo_write_cannot_confirm_receipt(self):
        for problem in ('early', 'missing', 'channel', 'angle', 'order', 'action', 'different_bits'):
            with self.subTest(problem=problem):
                self.setUp()
                p = self.panel
                token = self.start()
                complete, decoded, applied = self.optical_receipts(token)
                p.on_rx(dict(decoded, token='stale'))
                p.event(complete)
                waveform_receipt(p, token)
                self.assertFalse(p.commit_sent)
                if problem != 'early':
                    p.on_rx(decoded)
                    p.rx.send.assert_called_with('commit', token)
                rows = applied['servo_writes']
                if problem == 'missing':
                    rows.pop()
                elif problem == 'channel':
                    rows[1][2] = 0
                elif problem == 'angle':
                    rows[1][3] = 21
                elif problem == 'order':
                    rows[1][0] = '0.99'
                elif problem == 'action':
                    rows[0][5] = 'jogged'
                elif problem == 'different_bits':
                    applied['rx_payload_bits'] = '1' * 32
                p.on_rx(applied)
                self.assertIsNone(p.phase)
                p.rx.send.assert_called_with('cancel', token)
                self.assertFalse(p.rx_done)
                self.assertIsNone(p.last_servo_pose)
                self.assertFalse(p.record.call_args.args[0]['success'])

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
