import math
import time
import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

from common.angle_demo import HOME_PAN, HOME_TILT
from Tx.Controller.bench_pose import BenchPoseInputs, CAMERA, detector_pose, frame_pose, xyz_pose


def settings(**fields):
    point = dict(rx_minus_tx_m=[0, 5, 0], rx_home_yaw_deg=0,
                 rx_pan_plus='오른쪽', rx_tilt_plus='아래', tx_pan_plus='오른쪽', tx_tilt_plus='위',
                 tx_ref_pan_deg=0, tx_ref_tilt_deg=0, camera_intrinsics_px=None)
    point.update(fields)
    return point


def preview(command_id='test', when=None):
    meta = dict(seq=5, session='s1', simulated=False, servo_at_capture_start=dict(
        simulated=False, commanded=dict(command_id=command_id, pan=10, tilt=5)))
    return b'jpeg', meta, time.monotonic() if when is None else when, 123


class BenchPoseTests(unittest.TestCase):
    def test_face_to_face_geometry_signs_and_independent_methods(self):
        cfg = settings()
        ref = (CAMERA.laser_u, CAMERA.laser_v)
        self.assertEqual(xyz_pose(cfg)[:2], (HOME_PAN, HOME_TILT))
        self.assertEqual(detector_pose(cfg, 0, 0, ref)[:2], (HOME_PAN, HOME_TILT))
        # Rx higher and to the right of Tx; from the facing Rx the source is right and below.
        yaw = math.radians(15)
        cfg['rx_minus_tx_m'] = [5*math.sin(yaw), 5*math.cos(yaw), 5*math.tan(math.radians(6))]
        expected = (HOME_PAN+15, HOME_TILT+6)
        self.assertEqual(xyz_pose(cfg)[:2], expected)
        cfg['rx_minus_tx_m'] = None
        self.assertEqual(detector_pose(cfg, 15, 6, ref)[:2], expected)
        cfg.update(rx_pan_plus='왼쪽', rx_tilt_plus='위', rx_home_yaw_deg=3)
        self.assertEqual(detector_pose(cfg, 15, 6, ref)[:2], (HOME_PAN-12, HOME_TILT-6))

    def test_accept_aligned_frame_without_lens_and_correct_off_axis_only_with_calibration(self):
        cfg = settings()
        cx, cy = CAMERA.laser_u, CAMERA.laser_v
        uv = [cx+100, cy]
        with self.assertRaises(ValueError):
            detector_pose(cfg, 0, 0, uv)
        cfg['camera_intrinsics_px'] = [1000, 1000, cx, cy]
        self.assertEqual(detector_pose(cfg, 0, 0, uv)[:2], (HOME_PAN+6, HOME_TILT))
        self.assertEqual(detector_pose(cfg, 0, 0, [cx, cy+100])[:2], (HOME_PAN, HOME_TILT-6))
        cfg['camera_intrinsics_px'] = [780, 820, 648, 486]
        self.assertEqual(detector_pose(cfg, 17, 7, [cx, cy])[:2], (HOME_PAN+17, HOME_TILT+7))

    def test_reject_singular_or_unreachable_or_uncalibrated_command(self):
        for xyz in ([0,0,0], [0,0,2], [0,-1,0], [0,1,30], None):
            with self.subTest(xyz=xyz), self.assertRaises(ValueError):
                xyz_pose(settings(rx_minus_tx_m=xyz))
        with self.assertRaises(ValueError):
            detector_pose(settings(tx_tilt_plus='선택'), 0, 0, (CAMERA.laser_u,CAMERA.laser_v))

    def test_only_real_recent_capture_start_pose_is_usable(self):
        self.assertEqual(frame_pose(preview()), ('s1','test',10,5))
        for frame in (preview(when=time.monotonic()-3), preview(None)):
            with self.assertRaises(ValueError):
                frame_pose(frame)
        frame = preview()
        frame[1]['simulated'] = True
        with self.assertRaises(ValueError):
            frame_pose(frame)
        frame = preview()
        frame[1]['servo_at_capture_start']['commanded'] = None
        with self.assertRaises(ValueError):
            frame_pose(frame)

    def test_stale_worker_and_changed_tx_session_cannot_be_used(self):
        ui = BenchPoseInputs.__new__(BenchPoseInputs)
        ui.app = SimpleNamespace(current=preview('new_cmd'))
        ui.pose_stamp = frame_pose(preview('original'))
        ui.future = Future()
        ui.future.set_result((123,22,{}))
        with self.assertRaisesRegex(ValueError,'변경'):
            ui.poll_yolo()
        work = Future()
        work.set_running_or_notify_cancel()
        ui.future = work
        ui.cancel_yolo()
        work.set_result((123,22,{}))
        self.assertIsNone(ui.poll_yolo())

    def test_single_detect_records_frame_metadata_and_pv_selection(self):
        ui = BenchPoseInputs.__new__(BenchPoseInputs)
        ui.model = Mock(names={0:'PV'})
        ui.model_key = ('saved.pt','cpu')
        ui.model.infer.return_value = dict(image=SimpleNamespace(size=(CAMERA.width,CAMERA.height)),
                                           target={'center':[CAMERA.laser_u,CAMERA.laser_v],
                                                   'confidence':.91, 'box':[675,364,717,404]},
                                           count=2, inference_ms=14)
        image = preview()
        p,t,meta = ui.detect(settings(), image, frame_pose(image), 'saved.pt','cpu',.55,0)
        self.assertEqual((p,t), (HOME_PAN+10,HOME_TILT+5))
        self.assertEqual(meta['source'],'servo_at_capture_start.commanded')
        self.assertEqual(meta['receive_unix_ns'],123)
        ui.model.infer.assert_called_once_with(b'jpeg',.55,0)
