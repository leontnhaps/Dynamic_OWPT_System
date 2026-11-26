import math
import time
import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

from common.angle_demo import HOME_PAN, HOME_TILT
from Tx.Controller.bench_pose import BenchPoseInputs, CAMERA, STATION_COORDS_M, detector_pose, frame_pose, xyz_pose


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
    def test_four_station_commands_for_the_observed_receiver_axes(self):
        expected = [(103, 24), (84, 23), (62, 23), (75, 24)]
        for (label, coords), target in zip(STATION_COORDS_M.items(), expected):
            with self.subTest(station=label):
                pan, tilt, meta = xyz_pose(settings(rx_minus_tx_m=coords, rx_pan_plus='왼쪽'))
                self.assertEqual((pan, tilt), target)
                self.assertEqual(meta['rx_minus_tx_m'][2], .4)
                self.assertAlmostEqual(meta['rx_elevation_up_deg'], -meta['elevation_deg'])

    def test_face_to_face_geometry_signs_and_independent_methods(self):
        cfg = settings()
        ref = (648, 486)
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

    def test_old_station_snapshot_replays_a_recorded_m5_observation(self):
        # A detected frame from M5-3/Data/KF/20261102_153654_944906/observations.csv.
        cfg = settings(rx_pan_plus='왼쪽', tx_pan_plus='선택', tx_tilt_plus='선택')
        uv = (215.7223663330078, 809.3411560058594)
        pan, tilt, meta = detector_pose(cfg, 9, 10, uv)
        self.assertAlmostEqual(meta['azimuth_deg'], 0.43418937720356515)
        self.assertAlmostEqual(meta['elevation_deg'], 3.503915979200634)
        self.assertEqual((pan, tilt), (103, 24))
        self.assertEqual(meta['estimator'], 'tx-commanded-bearing-v1')
        self.assertEqual(meta['intrinsics_source'], 'm5')
        self.assertEqual(list(meta['camera_intrinsics_px']), [2857.1428571428573] * 2 + [648, 486])

    def test_custom_lens_uses_its_own_principal_point(self):
        cfg = settings(camera_intrinsics_px=[1000, 500, 558, 402])
        cx, cy = 558, 402
        self.assertEqual(detector_pose(cfg, 17, 7, [cx, cy])[:2], (HOME_PAN+17, HOME_TILT+7))
        self.assertEqual(detector_pose(cfg, 0, 0, [cx+100, cy])[:2], (HOME_PAN+6, HOME_TILT))
        pan, tilt, meta = detector_pose(cfg, 0, 0, [cx, cy+100])
        self.assertEqual((pan, tilt), (HOME_PAN, HOME_TILT-11))
        self.assertEqual(meta['intrinsics_source'], 'override')
        self.assertEqual(list(meta['camera_intrinsics_px']), cfg['camera_intrinsics_px'])

    def test_tx_mount_direction_and_command_zero_overrides_are_applied_once(self):
        cfg = settings(tx_pan_plus='왼쪽', tx_tilt_plus='아래', tx_ref_pan_deg=2, tx_ref_tilt_deg=3)
        pan, tilt, meta = detector_pose(cfg, 17, 7, [648, 486])
        self.assertEqual((pan, tilt), (HOME_PAN-15, HOME_TILT-4))
        self.assertEqual(list(meta['tx_command_deg']), [17, 7])
        self.assertEqual(list(meta['tx_model_angles_deg']), [-15, -4])

    def test_reject_singular_or_unreachable_command(self):
        for xyz in ([0,0,0], [0,0,2], [0,-1,0], [0,1,30], None):
            with self.subTest(xyz=xyz), self.assertRaises(ValueError):
                xyz_pose(settings(rx_minus_tx_m=xyz))
        with self.assertRaises(ValueError):
            detector_pose(settings(), 0, 45, (648, 486))

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
        # The laser reference lies above and to the right of the optical axis.
        self.assertEqual((p,t), (HOME_PAN+11,HOME_TILT+7))
        self.assertEqual(meta['source'],'servo_at_capture_start.commanded')
        self.assertEqual(meta['receive_unix_ns'],123)
        ui.model.infer.assert_called_once_with(b'jpeg',.55,0)
