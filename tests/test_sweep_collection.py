import csv
import io
import json
import math
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from PIL import Image
from common.tx_bearing import BEARING_MODEL, commanded_bearing
from common.tx_setup import TX_CAMERA_SETTINGS, TX_OPERATING_LIMITS
from common.model_paths import resolve_yolo_path
from Tx.Controller.sweep_core import SweepSettings, SweepRun, KF_MODE, LEARNING_MODE
from Tx.Controller.sweep_logging import SweepLog
from Tx.Controller.sweep_panel import SweepPanel
from Tx.Controller.Com_main import App
from Tx.Controller.detection_panel import DetectionPanel
from Tx.Controller.stationary_panel import StationaryPanel
from Tx.Controller.timing_panel import TimingPanel
from Tx.Controller.processing_profile import ProcessingProfile


class Clock:
    def __init__(self): self.now = 10.
    def __call__(self): return self.now


def variable(value=''):
    return SimpleNamespace(get=lambda: value, set=Mock())


def jpeg():
    stream = io.BytesIO()
    Image.new('RGB', (1296, 972), '#234567').save(stream, format='JPEG')
    return stream.getvalue()


class SweepTests(unittest.TestCase):
    def harness(self, **settings):
        self.clock, self.sent, self.log = Clock(), [], Mock()
        cfg = SweepSettings(pan_min=-5, pan_max=5, **settings)
        self.run = SweepRun(cfg, self.log, lambda cmd: self.sent.append(cmd) or True, self.clock)
        return self.run

    def ack(self, **overrides):
        r = self.run
        event = dict(event='servo', operation='move', available=True, simulated=False, limits=r.cfg.limits,
                     request_id=r.pending['token'], commanded=dict(pan=r.pending['pose'][0], tilt=r.pending['pose'][1],
                     command_id=r.pending['token'], speed=r.cfg.speed, acc=r.cfg.acc))
        event.update(overrides)
        r.reply(event)
        return event

    def frame(self):
        r = self.run
        return (b'jpeg', dict(simulated=False, requested=dict(TX_CAMERA_SETTINGS),
            servo_at_capture_start=dict(available=True, simulated=False, limits=r.cfg.limits,
                commanded=dict(pan=r.pose[0], tilt=r.pose[1], command_id=r.context['request_id']))), self.clock(), 123)

    def capture(self, detected=True):
        self.ack()
        self.clock.now = self.run.frame_after+.001
        self.run.tick()
        request = self.run.capture_request(self.frame(), self.run.last_frame+1, (1296, 972))
        self.assertIsNotNone(request)
        self.clock.now += .05
        self.run.captured(dict(status='detected' if detected else 'missing', kf_eligible=detected))

    def test_blank_defaults_and_validation(self):
        cfg = SweepSettings.from_text({key: ' ' for key in ('pan', 'tilt', 'pan_min', 'tilt_max', 'dt')})
        self.assertEqual(cfg, SweepSettings())
        self.assertEqual(cfg.limits, TX_OPERATING_LIMITS)
        self.assertEqual(cfg.dt, .720)
        for changes in [dict(pan_step=0), dict(pan_step=.1), dict(pan_min=-181), dict(pan_max=-1),
                        dict(pan_min=1), dict(tilt_max=91), dict(pan=float('nan')), dict(dt=.5),
                        dict(tilt_min=5, tilt_max=-5), dict(speed=50), dict(pan=-180)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(SweepSettings(), **changes).validate()

    def test_manual_one_way_reverse_clamp_and_no_automatic_restart(self):
        r = self.harness()
        r.start_leg('slow flight')
        while r.active:
            self.capture(detected=False)
        self.assertEqual([c['pan'] for c in self.sent], [0, -2, -4, -5])
        self.assertEqual((r.saved, r.valid, r.phase), (4, 0, 'complete'))
        r.finalize()
        self.assertEqual(self.log.leg.call_args.args[0]['images'], 4)
        self.clock.now += 100
        r.tick()
        self.assertEqual(len(self.sent), 4)
        r.start_leg('reverse flight')
        while r.active:
            self.capture()
        self.assertEqual([c['pan'] for c in self.sent[4:]], [-5, -3, -1, 1, 3, 5])
        self.assertEqual(r.test, 2)
        self.assertEqual(r.direction, -1)
        self.assertTrue(all(c['speed'] == 100 and c['acc'] == 1 for c in self.sent))
        r.start_leg()
        self.assertEqual(self.sent[-1]['pan'], 5)
        self.assertEqual(r.direction, 1)

    def test_axes_clamp_independently_and_zero_axis_is_held(self):
        r = self.harness(pan_step=-10, tilt_step=1, tilt_max=3)
        r.start_leg()
        while r.active:
            self.capture()
        self.assertEqual([(c['pan'], c['tilt']) for c in self.sent], [(0, 0), (-5, 1), (-5, 2), (-5, 3)])
        r = self.harness(tilt=3)
        r.start_leg()
        while r.active:
            self.capture()
        self.assertEqual({c['tilt'] for c in self.sent}, {3})

    def test_no_movement_until_matched_ack_dwell_frame_and_save(self):
        r = self.harness(); r.start_leg()
        r.reply(dict(event='servo', request_id='unrelated'))
        self.clock.now += 1
        r.tick()
        self.assertEqual(r.phase, 'ack')
        self.ack()
        self.clock.now = r.frame_after-.001
        r.tick()
        self.assertIsNone(r.capture_request(self.frame(), 1, (1296, 972)))
        self.clock.now = r.frame_after+.001
        r.tick()
        frame = self.frame()
        frame[1]['servo_at_capture_start']['commanded']['command_id'] = 'old-command'
        self.assertIsNone(r.capture_request(frame, 1, (1296, 972)))
        self.assertIsNotNone(r.capture_request(self.frame(), 2, (1296, 972)))
        self.clock.now += 1.1
        r.tick()
        self.assertEqual(len(self.sent), 1)  # No timer-only movement while save is pending.
        r.captured(dict(status='missing', kf_eligible=False))
        self.assertEqual(len(self.sent), 2)
        self.assertGreater(r.context['command_interval_s'], r.cfg.dt)

    def test_mismatched_servo_and_camera_abort(self):
        for overrides in [dict(simulated=True), dict(available=False), dict(limits={}),
                          dict(operation='servo_config'), dict(commanded={}), dict(event='error')]:
            with self.subTest(overrides=overrides):
                r = self.harness(); r.start_leg(); self.ack(**overrides)
                self.assertFalse(r.active)
                self.assertEqual(len(self.sent), 1)
        r = self.harness(); r.start_leg(); self.ack()
        self.clock.now = r.frame_after+.01; r.tick()
        r.capture_request(self.frame(), 1, (640, 480))
        self.assertEqual(r.reason, 'camera_configuration')

    def test_stale_and_duplicate_frames_are_not_samples(self):
        r = self.harness(); r.start_leg(); self.ack()
        self.clock.now = r.frame_after+.01; r.tick()
        frame = self.frame()
        self.clock.now += 1
        self.assertIsNone(r.capture_request(frame, 1, (1296, 972)))
        r.last_frame = 2
        self.assertIsNone(r.capture_request(self.frame(), 2, (1296, 972)))
        self.clock.now = r.deadline+.001; r.tick()
        self.assertEqual(r.reason, 'camera_timeout')

    def test_stop_drains_ack_and_saved_sample_without_next_move(self):
        r = self.harness(); r.start_leg(); r.abort('user_stop')
        with self.assertRaises(ValueError): r.start_leg()
        self.ack()
        r.tick()
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(r.phase, 'aborted')
        r = self.harness(); r.start_leg(); self.ack()
        self.clock.now = r.frame_after+.001; r.tick()
        r.capture_request(self.frame(), 1, (1296, 972))
        r.abort('user_stop')
        r.captured(dict(status='detected', kf_eligible=True)); r.finalize()
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(self.log.leg.call_args.args[0]['images'], 1)

    def test_timeouts_cannot_be_bypassed_by_late_completion(self):
        r = self.harness(); r.start_leg()
        self.clock.now += 5.1; self.ack()
        self.assertEqual(r.reason, 'servo_ack_timeout')
        r = self.harness(); r.start_leg(); self.ack()
        self.clock.now = r.frame_after+.001; r.tick()
        r.capture_request(self.frame(), 1, (1296, 972))
        self.clock.now += 31
        r.captured(dict(status='detected', kf_eligible=True))
        self.assertEqual(r.reason, 'inference_or_save_timeout')
        self.assertEqual(len(self.sent), 1)

    def test_send_failure_and_detector_error_do_not_move_again(self):
        r = self.harness(); r.send = lambda cmd: False; r.start_leg()
        self.assertEqual(r.reason, 'command_send_failed')
        self.assertIsNone(r.pending)
        r = self.harness(); r.start_leg(); self.ack()
        self.clock.now = r.frame_after+.001; r.tick()
        r.capture_request(self.frame(), 1, (1296, 972))
        r.captured(dict(status='error', kf_eligible=False))
        self.assertEqual(r.reason, 'inference_error')
        self.assertEqual(len(self.sent), 1)


class StorageTests(unittest.TestCase):
    def test_bearing_uses_optical_center_and_expected_signs(self):
        self.assertEqual(commanded_bearing(648, 486, 0, 0), (0, 0))
        a, b = commanded_bearing(648, 486, -17, 5)
        self.assertAlmostEqual(math.degrees(a), -17)
        self.assertAlmostEqual(math.degrees(b), 5)
        a, b = commanded_bearing(658, 496, 0, 0)
        self.assertGreater(a, 0); self.assertLess(b, 0)
        self.assertAlmostEqual(a, math.atan(10/BEARING_MODEL['fx_px']))
        with self.assertRaises(ValueError): commanded_bearing(float('nan'), 0, 0, 0)

    def test_raw_images_metadata_gaps_valid_csv_and_no_overwrite(self):
        raw = jpeg()
        target = dict(center=[648, 486], box=[630, 470, 666, 502], confidence=.9)
        detector = Mock()
        detector.infer.side_effect = [dict(image=Image.open(io.BytesIO(raw)), target=t, count=int(t is not None),
                                          inference_ms=1) for t in (target, None)]
        frame = (raw, dict(session='pi-session', seq=1, camera_metadata={'SensorTimestamp': 7}), 12., 123)
        context = dict(test=1, step=0, pan_command_deg=-2, tilt_command_deg=0, frame_id=1)
        with TemporaryDirectory() as temp:
            log = SweepLog(Path(temp)/'run', dict(schema='test'))
            row, _ = log.capture(detector, SweepSettings(), frame, context, True)
            missing, _ = log.capture(detector, SweepSettings(), frame, dict(context, step=1), True)
            self.assertEqual((log.folder/row['image']).read_bytes(), raw)
            self.assertEqual(row['image'], 'T001_S0000_pan-2_tilt+0_detected.jpeg')
            self.assertTrue(missing['image'].endswith('_missing.jpeg'))
            self.assertNotIn('alpha_rad', missing)
            with (log.folder/'observations.csv').open() as stream: rows = list(csv.DictReader(stream))
            with (log.folder/'kalman_observations.csv').open() as stream: valid = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2); self.assertEqual(len(valid), 1)
            self.assertEqual(rows[1]['u'], '')
            self.assertEqual(rows[0]['actual_pan_deg'], '')
            meta = json.loads((log.folder/row['image']).with_suffix('.json').read_text())
            self.assertEqual(meta['frame_metadata']['camera_metadata']['SensorTimestamp'], 7)
            self.assertAlmostEqual(row['alpha_deg'], -2)
            with self.assertRaises(FileExistsError): SweepLog(log.folder, {})
            detector.infer.side_effect = None
            detector.infer.return_value = dict(image=Image.open(io.BytesIO(raw)), target=target, count=1, inference_ms=1)
            with self.assertRaises(FileExistsError): log.capture(detector, SweepSettings(), frame, context, True)

    def test_detector_failure_preserves_original_and_yolo_only_excludes_kf_rows(self):
        raw = jpeg()
        frame = (raw, {}, 1, 2)
        context = dict(test=1, step=0, pan_command_deg=0, tilt_command_deg=0, frame_id=1)
        detector = Mock(); detector.infer.side_effect = RuntimeError('GPU test error')
        with TemporaryDirectory() as temp:
            log = SweepLog(Path(temp)/'run', {})
            row, image = log.capture(detector, SweepSettings(), frame, context, True)
            self.assertEqual(row['status'], 'error')
            self.assertEqual((log.folder/row['image']).read_bytes(), raw)
            self.assertIsNotNone(image)
            detector.infer.side_effect = None
            detector.infer.return_value = dict(image=image, target=dict(center=[648, 486],
                box=[630, 470, 666, 502], confidence=.8), count=1, inference_ms=1)
            row, _ = log.capture(detector, SweepSettings(), frame, dict(context, test=2), False)
            self.assertFalse(row['kf_eligible'])
            with (log.folder/'kalman_observations.csv').open() as stream:
                self.assertEqual(list(csv.DictReader(stream)), [])


class PanelTests(unittest.TestCase):
    def panel(self):
        p = SweepPanel.__new__(SweepPanel)
        p.future = p.preparing = p.setup_pending = p.run = p.log = None
        p.hardware_ready = p.closed = False
        p.logging_failed = False
        p.display = p.rendered = None
        p.status = Mock(); p.progress = Mock(); p.lock = Mock()
        p.cfg = SweepSettings()
        p.vars = {key: variable() for key in ('pan', 'tilt', 'pan_min', 'pan_max', 'tilt_min', 'tilt_max', 'pan_step', 'tilt_step')}
        p.path = variable('best.pt'); p.device = variable('cpu'); p.mode = variable(KF_MODE); p.note = variable()
        p.pool = Mock(); p.pool.submit.return_value = Future()
        p.app = SimpleNamespace(send=Mock(return_value=True), record=Mock(),
            current=None, size=(1296, 972), received_count=0,
            values={key: variable() for key in TX_CAMERA_SETTINGS},
            servo=SimpleNamespace(pending=None, online=True, limits=None, simulated=None,
                                  vars={key: variable() for key in (*TX_OPERATING_LIMITS, 'speed', 'acc', 'pan', 'tilt')}),
            stationary=SimpleNamespace(busy=False, run=None, stop=Mock(), inference=None),
            detection=SimpleNamespace(stop=Mock(), future=None),
            timing=SimpleNamespace(active=False, preparing=None, future=None, stop=Mock(),
                                   profile=SimpleNamespace(running=False)))
        return p

    def test_late_model_load_after_cancel_cannot_send_commands(self):
        p = self.panel(); p.prepare()
        self.assertEqual(p.preparing, 'model')
        p.stop()
        p.future.set_result((object(), {}))
        p.poll()
        self.assertIsNone(p.preparing)
        self.assertFalse(p.hardware_ready)
        p.app.send.assert_not_called()

    def test_prepares_without_earlier_tab_and_only_after_matching_real_responses(self):
        p = self.panel(); p.prepare()
        p.future.set_result((object(), {'path': 'test'}))
        with patch('Tx.Controller.sweep_panel.time.monotonic', return_value=100): p.poll()
        commands = [call.args[0] for call in p.app.send.call_args_list]
        self.assertEqual([cmd['cmd'] for cmd in commands], ['preview', 'servo_config'])
        self.assertEqual(commands[1]['tilt_min'], -15)
        p.event(dict(event='servo', operation='servo_config', request_id=p.setup_pending,
                     limits=p.cfg.limits, simulated=False, available=True))
        p.app.current = (b'jpeg', dict(simulated=False, requested=dict(TX_CAMERA_SETTINGS)), 100.1, 1)
        with patch('Tx.Controller.sweep_panel.time.monotonic', return_value=100.2): p.poll()
        self.assertTrue(p.hardware_ready)
        self.assertEqual(p.app.servo.limits, p.cfg.limits)
        self.assertFalse(any(cmd['cmd'] == 'move' for cmd in commands))

    def test_bad_setup_and_disconnect_cancel_preparation(self):
        p = self.panel(); p.prepare(); p.future.set_result((object(), {})); p.poll()
        p.event(dict(event='servo', operation='servo_config', request_id=p.setup_pending,
                     limits=p.cfg.limits, simulated=True, available=True))
        self.assertIsNone(p.preparing); self.assertFalse(p.hardware_ready)
        p = self.panel(); p.prepare()
        p.event(dict(event='network', port=7601, state='disconnected'))
        p.future.set_result((object(), {})); p.poll()
        p.app.send.assert_not_called()

    def test_pending_mode_and_invalid_settings_have_no_side_effects(self):
        p = self.panel(); p.mode = variable(LEARNING_MODE)
        with self.assertRaises(ValueError): p.prepare()
        p.pool.submit.assert_not_called(); p.app.send.assert_not_called()
        p = self.panel(); p.vars['pan_step'] = variable('0'); p.vars['tilt_step'] = variable('0')
        with self.assertRaises(ValueError): p.prepare()
        p.pool.submit.assert_not_called()

    def test_foreign_commands_blocked_without_aborting_and_outputs_off_still_available(self):
        a = App.__new__(App); a.sweep = SimpleNamespace(busy=True, hardware_ready=True, invalidate=Mock())
        a.net, a.record = Mock(), Mock()
        for command in ('move', 'preview', 'servo_config', 'ir_cut', 'snap'):
            self.assertFalse(a.send(dict(cmd=command)))
        a.net.send.assert_not_called(); a.sweep.invalidate.assert_not_called()
        self.assertTrue(a.send(dict(cmd='move', request_id='m5-1-move-test')))
        self.assertTrue(a.send(dict(cmd='outputs_off')))
        a.sweep.invalidate.assert_called_once()

    def test_other_tabs_cannot_start_during_sweep(self):
        app = SimpleNamespace(sweep=SimpleNamespace(busy=True))
        for cls, method, args in [(DetectionPanel, 'load', ()), (DetectionPanel, 'start', ()),
                (TimingPanel, 'prepare', (False,)), (ProcessingProfile, 'start', ()),
                (StationaryPanel, 'stop_other_panels', ()), (StationaryPanel, 'check_manual_available', ())]:
            obj = cls.__new__(cls); obj.app = app
            with self.subTest(method=method, cls=cls), self.assertRaises(ValueError): getattr(obj, method)(*args)

    def test_two_legs_through_panel_write_consistent_config_images_and_summaries(self):
        p = self.panel(); clock = Clock(); raw = jpeg()
        p.cfg = SweepSettings(pan_min=-2, pan_max=2)
        p.vars = {key: variable(str(getattr(p.cfg, key))) for key in p.vars}
        p.identity = dict(path=resolve_yolo_path('best.pt'), device='cpu', sha256='test-model')
        p.prepared_mode = KF_MODE; p.hardware_ready = True
        p.app.servo.limits = p.cfg.limits; p.app.servo.simulated = False
        p.detector = Mock()
        p.detector.infer.return_value = dict(image=Image.open(io.BytesIO(raw)), target=None, count=0, inference_ms=1)

        def submit(fn, *args):
            future = Future()
            future.set_result(fn(*args))
            return future

        p.pool.submit.side_effect = submit

        def set_frame():
            p.app.received_count += 1
            state = dict(available=True, simulated=False, limits=p.cfg.limits)
            if p.run:
                state['commanded'] = dict(pan=p.run.pose[0], tilt=p.run.pose[1],
                                          command_id=p.run.context['request_id'])
            p.app.current = (raw, dict(simulated=False, requested=dict(TX_CAMERA_SETTINGS),
                servo_at_capture_start=state, session='test-pi', seq=p.app.received_count), clock(), 123)

        with TemporaryDirectory() as temp, patch('Tx.Controller.sweep_panel.time.monotonic', clock), \
                patch('Tx.Controller.sweep_panel.SweepRun', side_effect=lambda cfg, log, send: SweepRun(cfg, log, send, clock)):
            p.app.stage_dir = lambda stage: Path(temp)/stage
            for leg in range(2):
                set_frame(); p.start()
                while p.run.active:
                    pending = p.run.pending
                    p.event(dict(event='servo', operation='move', request_id=pending['token'],
                        available=True, simulated=False, limits=p.cfg.limits,
                        commanded=dict(pan=pending['pose'][0], tilt=pending['pose'][1],
                                       command_id=pending['token'], speed=100, acc=1)))
                    clock.now = p.run.frame_after+.001
                    set_frame(); p.poll()
                    clock.now += .02
                    p.poll()
                self.assertTrue(p.hardware_ready)
                self.assertFalse(p.busy)
            with (p.log.folder/'observations.csv').open() as stream: rows = list(csv.DictReader(stream))
            with (p.log.folder/'legs.csv').open() as stream: legs = list(csv.DictReader(stream))
            self.assertEqual([float(row['pan_command_deg']) for row in rows], [0, -2, -2, 0, 2])
            self.assertEqual([row['reason'] for row in legs], ['range_complete', 'range_complete'])
            self.assertEqual(len(list(p.log.folder.glob('*.jpeg'))), 5)
            self.assertEqual(len({row['image'] for row in rows}), 5)
            cfg = json.loads((p.log.folder/'config.json').read_text())
            self.assertEqual(cfg['settings']['dt'], .72)
            self.assertIsNone(cfg['kalman']['Q_applied'])
            self.assertEqual(cfg['bearing_model']['cx_px'], 648)
            self.assertTrue(cfg['source_sha256'])

    def test_failed_save_stops_and_does_not_retry_disk_writes_each_poll(self):
        p = self.panel(); p.log = Mock()
        p.run = SweepRun(p.cfg, p.log, p.app.send)
        p.run.start_leg()
        p.run.pending = None; p.run.phase = 'processing'; p.run.deadline = float('inf')
        p.job_kind = 'capture'; p.future = Future()
        p.future.set_exception(OSError('disk full'))
        with patch('Tx.Controller.sweep_panel.messagebox.showerror') as message:
            p.guard(p.poll)
            p.guard(p.poll)
        self.assertFalse(p.run.active)
        self.assertTrue(p.logging_failed)
        self.assertEqual(p.app.send.call_count, 1)
        p.log.leg.assert_not_called()
        message.assert_called_once()


if __name__ == '__main__':
    unittest.main()
