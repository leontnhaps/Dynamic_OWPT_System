"""Estimate a stationary PV receiver's normal from a surveyed station or one image."""
from concurrent.futures import ThreadPoolExecutor
import math
import time
import tkinter as tk
from tkinter import ttk

from common.angle_demo import HOME_PAN, HOME_TILT, MAX_PAN, MAX_TILT
from common.model_paths import resolve_yolo_path
from common.pv_detection import TARGET_SELECTION, fresh
from common.tx_tracking import TrackingConfig
from Tx.Controller.detection_panel import PVDetector


CAMERA = TrackingConfig()
PAN_SIGN = {'오른쪽': 1, '왼쪽': -1}
TILT_SIGN = {'위': 1, '아래': -1}


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
    return pan, tilt, dict(meta, method='xyz', rx_minus_tx_m=[x, y, z], distance_m=math.hypot(horizontal, z))


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
    if station['tx_pan_plus'] not in PAN_SIGN or station['tx_tilt_plus'] not in TILT_SIGN:
        raise ValueError('좌표·축 설정에서 실물로 확인한 Tx Pan+·Tilt+ 방향을 선택하세요.')
    az = math.radians(PAN_SIGN[station['tx_pan_plus']] * (tx_pan - station['tx_ref_pan_deg']))
    el = math.radians(TILT_SIGN[station['tx_tilt_plus']] * (tx_tilt - station['tx_ref_tilt_deg']))
    u, v = uv
    ref_u, ref_v = CAMERA.laser_u, CAMERA.laser_v
    error = math.hypot(u - ref_u, v - ref_v)
    intrinsics = station['camera_intrinsics_px']
    if intrinsics is None:
        if error > CAMERA.hit_px:
            raise ValueError(f'검출 중심이 기준점에서 {error:.1f}px 떨어졌습니다 '
                             f'(허용 {CAMERA.hit_px:g}px). 영상 보정값이 필요합니다.')
        method = 'confirmed_reference_sightline'
    else:
        # In image coordinates x points right, z points up, y points forward.
        # Construct a local horizontal/vertical basis about the calibrated reference
        # ray, which need not coincide with the principal point.
        fx, fy, cx, cy = intrinsics
        bx, bz = (ref_u-cx)/fx, (cy-ref_v)/fy
        x, z = (u-cx)/fx, (cy-v)/fy
        horizontal = math.hypot(1, bx)
        norm = math.hypot(horizontal, bz)
        right = (x-bx) / horizontal
        forward = (x*bx + 1 + z*bz) / norm
        up = (horizontal*z - (bx*x + 1)*bz/horizontal) / norm
        horizontal_component = forward*math.cos(el) - up*math.sin(el)
        vertical_component = forward*math.sin(el) + up*math.cos(el)
        az += math.atan2(right, horizontal_component)
        el = math.atan2(vertical_component, math.hypot(right, horizontal_component))
        method = 'calibrated_pinhole'
    pan, tilt, meta = servo_pose(math.degrees(az), math.degrees(el), station)
    return pan, tilt, dict(meta, method='yolo', estimator=method, reference_px=[ref_u, ref_v],
                           center_px=[u, v], residual_px=error, tx_command_deg=[tx_pan, tx_tilt])


class BenchPoseInputs(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = self.pose_stamp = self.model = self.model_key = None
        self.fields = {}
        top = ttk.Frame(self)
        top.pack(anchor='w')
        for key, label, default, width in [('point_id','지점','P01',10), ('x','X(m)','',6),
                                           ('y','Y(m)','',6), ('z','Z(m)','',6)]:
            self.field(top, key, label, default, width)
        self.detail = ttk.Frame(self)
        ttk.Button(top, text='좌표·축 설정', command=self.toggle).pack(side='left', padx=4)
        ttk.Label(top, text='Tx 원점 · Tx 뒤에서 +X 우측, +Y 정면, +Z 위').pack(side='left')
        axes = ttk.Frame(self.detail)
        axes.pack(anchor='w')
        for key, label, default, options in [
                ('tx_pan_plus','Tx Pan+', '선택', PAN_SIGN), ('tx_tilt_plus','Tx Tilt+', '선택', TILT_SIGN),
                ('rx_pan_plus','Rx Pan+', '오른쪽', PAN_SIGN), ('rx_tilt_plus','Rx Tilt+', '아래', TILT_SIGN)]:
            self.field(axes, key, label, default, 6, options)
        self.field(axes, 'tx_p0', 'Tx 정면 Pan°', '0')
        self.field(axes, 'tx_t0', '수평 Tilt°', '0')
        self.field(axes, 'rx_yaw', 'Rx 원위치 방위차°', '0')
        self.rx_axes_observed = tk.BooleanVar(self, value=False)
        ttk.Checkbutton(axes, text='Rx 두 축 방향 실물 확인', variable=self.rx_axes_observed).pack(side='left')
        for key in ('rx_pan_plus', 'rx_tilt_plus'):
            self.fields[key].trace_add('write', lambda *_: self.rx_axes_observed.set(False))
        ttk.Label(self.detail, text='방향은 각 장치 뒤에서 관찰. 수평 발판이 서로 마주보면 방위차 0°, '
                                    'Rx 원위치가 기준보다 자기 오른쪽이면 양수.').pack(anchor='w')
        lens = ttk.Frame(self.detail)
        lens.pack(anchor='w')
        for key in ('fx','fy','cx','cy'):
            self.field(lens, key, key+'(px)', '')
        ttk.Label(lens, text=f'미입력 시 기준점 ({CAMERA.laser_u:g}, {CAMERA.laser_v:g}) ±{CAMERA.hit_px:g}px. '
                              '입력 시 수평 카메라·현재 해상도의 핀홀 보정값 사용.').pack(side='left')

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
        lens = ('fx','fy','cx','cy')
        intrinsics = [number(f[k], k) for k in lens] if any(f[k] for k in lens) else None
        if intrinsics and (min(intrinsics[:2]) <= 0 or not 0 <= intrinsics[2] < CAMERA.width
                           or not 0 <= intrinsics[3] < CAMERA.height):
            raise ValueError('초점거리는 양수, 주점은 기준 영상 내부여야 합니다.')
        return dict(point_id=point_id, rx_minus_tx_m=xyz, room_axes='Tx origin: +X right, +Y front, +Z up',
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
        if snapshot['tx_pan_plus'] not in PAN_SIGN or snapshot['tx_tilt_plus'] not in TILT_SIGN:
            if not self.detail.winfo_manager():
                self.toggle()
            raise ValueError('좌표·축 설정에서 Tx Pan+·Tilt+의 실물 방향을 선택하세요.')
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
