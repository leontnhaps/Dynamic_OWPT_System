"""Estimate a stationary PV receiver's normal from a surveyed station or one image."""
from concurrent.futures import ThreadPoolExecutor
import math
import time
import tkinter as tk
from tkinter import ttk

from common.angle_demo import HOME_PAN, HOME_TILT, MAX_PAN, MAX_TILT, RX_VIEWPOINT
from common.model_paths import resolve_yolo_path
from common.pv_detection import TARGET_SELECTION, fresh
from common.tx_bearing import BEARING_MODEL, commanded_bearing
from common.tx_tracking import TrackingConfig
from Tx.Controller.detection_panel import PVDetector


CAMERA = TrackingConfig()
PAN_SIGN = {'오른쪽': 1, '왼쪽': -1}
TILT_SIGN = {'위': 1, '아래': -1}
TX_PAN_PLUS, TX_TILT_PLUS = '오른쪽', '위'
LENS_FIELDS = ('fx', 'fy', 'cx', 'cy')
M5_INTRINSICS = tuple(BEARING_MODEL[key + '_px'] for key in LENS_FIELDS)
STATION_COORDS_M = {
    '①': (0.00, 6.30, 0.40),
    '②': (2.70, 7.65, 0.40),
    '③': (5.40, 6.30, 0.40),
    '④': (2.70, 5.00, 0.40),
}


def number(text, name):
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name}에 숫자를 입력하세요.') from exc
    if not math.isfinite(value):
        raise ValueError(f'{name}에는 유한한 수가 필요합니다.')
    return value


def servo_pose(azimuth, elevation, station):
    """Reciprocal sightline: face-to-face heads turn to the same local right, opposite up."""
    yaw = (azimuth - station['rx_home_yaw_deg'] + 180) % 360 - 180
    pitch = -elevation
    pan_relative = PAN_SIGN[station['rx_pan_plus']] * yaw
    tilt_relative = TILT_SIGN[station['rx_tilt_plus']] * pitch
    angles = [int(math.copysign(math.floor(abs(v) + 0.5), v)) for v in (pan_relative, tilt_relative)]
    pan, tilt = HOME_PAN + angles[0], HOME_TILT + angles[1]
    if not 0 <= pan <= MAX_PAN or not 0 <= tilt <= MAX_TILT:
        raise ValueError(f'추정 서보각 Pan {pan}°, Tilt {tilt}°가 가동범위를 벗어났습니다.')
    return pan, tilt, {'rx_bearing_right_deg': yaw, 'rx_elevation_up_deg': pitch,
                       'pan_offset_deg': angles[0], 'tilt_offset_deg': angles[1]}


def xyz_pose(station):
    if station['rx_minus_tx_m'] is None:
        raise ValueError('Tx 광원에서 Rx 셀 중심까지의 X·Y·Z를 m 단위로 모두 입력하세요.')
    x, y, z = station['rx_minus_tx_m']
    horizontal = math.hypot(x, y)
    if horizontal == 0:
        raise ValueError('Tx와 Rx의 수평 좌표가 같아서 Pan을 정할 수 없습니다.')
    azimuth = math.degrees(math.atan2(x, y))
    elevation = math.degrees(math.atan2(z, horizontal))
    pan, tilt, meta = servo_pose(azimuth, elevation, station)
    return pan, tilt, dict(meta, method='xyz', rx_minus_tx_m=[x, y, z],
                           horizontal_m=horizontal, distance_m=math.hypot(horizontal, z),
                           azimuth_deg=azimuth, elevation_deg=elevation)


def frame_pose(frame):
    if not frame or not fresh(frame[2], time.monotonic(), 2):
        raise ValueError('최근 Tx 카메라 영상이 필요합니다.')
    meta = frame[1]
    servo = meta.get('servo_at_capture_start') or {}
    command = servo.get('commanded') or {}
    if meta.get('simulated') is not False or servo.get('simulated') is not False:
        raise ValueError('실제 Tx와 서보 명령 기록이 있는 영상이 필요합니다.')
    if not meta.get('session') or not command.get('command_id'):
        raise ValueError('이 프레임에 Tx Pan·Tilt 명령 기록이 없습니다. 기준 자세를 확인하세요.')
    return (meta['session'], command['command_id'], number(command.get('pan'), 'Tx Pan'),
            number(command.get('tilt'), 'Tx Tilt'))


def detector_pose(station, tx_pan, tx_tilt, uv):
    """Convert only the receiving normal; the transmitter stays at its recorded pose."""
    pan_plus, tilt_plus = station.get('tx_pan_plus'), station.get('tx_tilt_plus')
    # Older station snapshots used '선택' for unspecified mounting directions.
    if pan_plus in (None, '선택'):
        pan_plus = TX_PAN_PLUS
    if tilt_plus in (None, '선택'):
        tilt_plus = TX_TILT_PLUS
    if pan_plus not in PAN_SIGN or tilt_plus not in TILT_SIGN:
        raise ValueError('좌표·축 설정에서 Tx 축 방향을 확인하세요.')
    model_pan = PAN_SIGN[pan_plus] * (tx_pan - station['tx_ref_pan_deg'])
    model_tilt = TILT_SIGN[tilt_plus] * (tx_tilt - station['tx_ref_tilt_deg'])
    u, v = uv
    lens_override = station.get('camera_intrinsics_px')
    lens = M5_INTRINSICS if lens_override is None else tuple(lens_override)
    azimuth, elevation = commanded_bearing(u, v, model_pan, model_tilt, intrinsics=lens)
    azimuth, elevation = math.degrees(azimuth), math.degrees(elevation)
    pan, tilt, meta = servo_pose(azimuth, elevation, station)
    # Retain laser-to-PV pixel residuals for comparisons with earlier bench logs.
    ref_u, ref_v = CAMERA.laser_u, CAMERA.laser_v
    return pan, tilt, dict(meta, method='yolo', estimator=BEARING_MODEL['version'],
                           camera_intrinsics_px=list(lens),
                           intrinsics_source='m5' if lens == M5_INTRINSICS else 'override',
                           azimuth_deg=azimuth, elevation_deg=elevation,
                           tx_model_angles_deg=[model_pan, model_tilt], tx_command_deg=[tx_pan, tx_tilt],
                           center_px=[u, v], reference_px=[ref_u, ref_v],
                           residual_px=math.hypot(u - ref_u, v - ref_v))


class BenchPoseInputs(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = self.pose_stamp = self.model = self.model_key = None
        self.fields = {}
        self.on_change = None
        self.station_coordinates = {key: tuple(f'{n:.2f}' for n in xyz)
                                    for key, xyz in STATION_COORDS_M.items()}
        self.active_point = next(iter(STATION_COORDS_M))
        top = ttk.Frame(self)
        top.pack(anchor='w')
        self.field(top, 'point_id', '측정 지점', self.active_point, 3, STATION_COORDS_M)
        for axis, value in zip('xyz', self.station_coordinates[self.active_point]):
            self.field(top, axis, f'{axis.upper()} (m)', value, 6)
        self.detail = ttk.Frame(self)
        ttk.Button(top, text='좌표·축 설정', command=self.toggle).pack(side='left', padx=4)
        ttk.Label(top, text='Tx 원점 · Tx 뒤에서 +X 우측, +Y 정면, +Z 위').pack(side='left')
        axes = ttk.Frame(self.detail)
        axes.pack(anchor='w')
        for key, label, default, options in [
                ('tx_pan_plus','Tx Pan+', TX_PAN_PLUS, PAN_SIGN), ('tx_tilt_plus','Tx Tilt+', TX_TILT_PLUS, TILT_SIGN),
                ('rx_pan_plus','Rx Pan+', '왼쪽', PAN_SIGN), ('rx_tilt_plus','Rx Tilt+', '아래', TILT_SIGN)]:
            self.field(axes, key, label, default, 6, options)
        self.field(axes, 'tx_p0', 'Tx 정면 Pan°', '0')
        self.field(axes, 'tx_t0', '수평 Tilt°', '0')
        self.field(axes, 'rx_yaw', 'Rx 원위치 방위차°', '0')
        self.rx_axes_observed = tk.BooleanVar(self, value=True)
        ttk.Checkbutton(axes, text='Rx 두 축 방향 실물 확인', variable=self.rx_axes_observed).pack(side='left')
        for key in ('rx_pan_plus', 'rx_tilt_plus'):
            self.fields[key].trace_add('write', lambda *_: self.rx_axes_observed.set(False))
        ttk.Label(self.detail, text='방향은 각 장치 뒤에서 관찰. 수평 발판이 서로 마주보면 방위차 0°, '
                                    'Rx 원위치가 기준보다 자기 오른쪽이면 양수.').pack(anchor='w')
        lens = ttk.Frame(self.detail)
        lens.pack(anchor='w')
        for key, value in zip(LENS_FIELDS, M5_INTRINSICS):
            self.field(lens, key, key+'(px)', str(value), 9)
        ttk.Label(lens, text=f'M5 카메라 모델 · {BEARING_MODEL["width"]}×{BEARING_MODEL["height"]}'
                              ' · 네 칸을 모두 비우면 M5 기본값 사용').pack(side='left')
        self.fields['point_id'].trace_add('write', self.select_point)
        for key, var in self.fields.items():
            if key != 'point_id':
                var.trace_add('write', self.changed)
        self.rx_axes_observed.trace_add('write', self.changed)

    def select_point(self, *_):
        key = self.fields['point_id'].get()
        if key == self.active_point:
            return
        self.station_coordinates[self.active_point] = tuple(self.fields[k].get() for k in 'xyz')
        # Notify the experiment once all three coordinates refer to the new point.
        self.active_point = None
        for axis, value in zip('xyz', self.station_coordinates[key]):
            self.fields[axis].set(value)
        self.active_point = key
        self.changed()

    def changed(self, *_):
        if self.active_point is not None and self.on_change is not None:
            self.on_change()

    def field(self, row, key, label, default, width=5, options=None):
        var = tk.StringVar(self, value=default)
        self.fields[key] = var
        ttk.Label(row, text=label).pack(side='left', padx=(4, 1))
        if options:
            control = ttk.Combobox(row, textvariable=var, width=width, values=tuple(options), state='readonly')
        else:
            control = ttk.Entry(row, textvariable=var, width=width)
        control.pack(side='left')

    def toggle(self):
        if self.detail.winfo_manager():
            self.detail.pack_forget()
        else:
            self.detail.pack(anchor='w')

    def snapshot(self):
        f = {key: var.get().strip() for key, var in self.fields.items()}
        point_id = f['point_id']
        if not point_id or len(point_id) > 80 or not point_id.isprintable():
            raise ValueError('지점 이름을 1~80자로 입력하세요.')
        xyz = [number(f[k], k.upper()) for k in 'xyz'] if any(f[k] for k in 'xyz') else None
        lens_values = [f[k] for k in LENS_FIELDS]
        if any(lens_values) and not all(lens_values):
            raise ValueError('렌즈 보정값 4개를 모두 입력하거나 모두 비우세요.')
        intrinsics = ([number(f[k], k) for k in LENS_FIELDS] if any(lens_values) else list(M5_INTRINSICS))
        if (min(intrinsics[:2]) <= 0 or not 0 <= intrinsics[2] < CAMERA.width
                or not 0 <= intrinsics[3] < CAMERA.height):
            raise ValueError('초점거리는 양수, 주점은 기준 영상 내부여야 합니다.')
        return dict(point_id=point_id, rx_minus_tx_m=xyz, room_axes='Tx origin: +X right, +Y front, +Z up',
                    rx_viewpoint=RX_VIEWPOINT,
                    rx_pan_plus=f['rx_pan_plus'], rx_tilt_plus=f['rx_tilt_plus'],
                    rx_axes_observed=self.rx_axes_observed.get(),
                    tx_pan_plus=f['tx_pan_plus'], tx_tilt_plus=f['tx_tilt_plus'],
                    tx_ref_pan_deg=number(f['tx_p0'], 'Tx 기준 Pan'), tx_ref_tilt_deg=number(f['tx_t0'], 'Tx 기준 Tilt'),
                    rx_home_yaw_deg=number(f['rx_yaw'], 'Rx 원위치 방위차'), camera_intrinsics_px=intrinsics)

    estimate_xyz = staticmethod(xyz_pose)

    def start_yolo(self, snapshot, frame):
        if self.future is not None:
            raise ValueError('이전 YOLO 검출을 기다리세요.')
        stamp = frame_pose(frame)
        detection = self.app.detection
        path = resolve_yolo_path(detection.path.get())
        device = detection.device.get().strip()
        if device not in ('cpu', 'cuda'):
            raise ValueError('YOLO 장치는 cpu 또는 cuda여야 합니다.')
        confidence = number(detection.conf.get(), 'YOLO 신뢰도')
        class_id = int(detection.class_id.get())
        if not 0 < confidence <= 1 or class_id < 0:
            raise ValueError('YOLO 신뢰도·PV class ID를 확인하세요.')
        self.pose_stamp = stamp
        self.future = self.executor.submit(self.detect, snapshot, frame, stamp, path, device, confidence, class_id)

    def detect(self, snapshot, frame, stamp, path, device, confidence, class_id):
        key = path, device
        if self.model is None or self.model_key != key:
            self.model = PVDetector(path, device)
            self.model_key = key
        if class_id not in self.model.names:
            raise ValueError('선택한 YOLO 모델에 PV class ID가 없습니다.')
        result = self.model.infer(frame[0], confidence, class_id)
        if result['image'].size != (CAMERA.width, CAMERA.height):
            raise ValueError(f'영상 보정 해상도 {CAMERA.width}×{CAMERA.height}를 확인하세요.')
        target = result['target']
        if not target:
            raise ValueError('선택한 지점의 프레임에서 PV가 검출되지 않았습니다.')
        pan, tilt, meta = detector_pose(snapshot, stamp[2], stamp[3], target['center'])
        meta.update(target=target, target_selection=TARGET_SELECTION, candidates=result['count'],
                    model=path, device=device, confidence_threshold=confidence, class_id=class_id,
                    inference_ms=result['inference_ms'], frame_seq=frame[1]['seq'], session=stamp[0],
                    tx_command_id=stamp[1], receive_unix_ns=frame[3],
                    width=CAMERA.width, height=CAMERA.height, source='servo_at_capture_start.commanded')
        return pan, tilt, meta

    def poll_yolo(self):
        if self.future is None or not self.future.done():
            return None
        future, self.future = self.future, None
        try:
            result = future.result()
        except Exception as exc:
            raise ValueError(f'YOLO 검출 실패 · {exc}') from exc
        if frame_pose(self.app.current) != self.pose_stamp:
            raise ValueError('검출 중 Tx 명령 또는 연결이 변경됐습니다. 이번 지점에서 다시 측정하세요.')
        return result

    def cancel_yolo(self):
        if self.future is not None:
            self.future.cancel()
            self.future = None
        self.pose_stamp = None

    def close(self):
        self.cancel_yolo()
        self.executor.shutdown(wait=False, cancel_futures=True)
