"""Independent M5-1 preparation and user-triggered outdoor sweep collection."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import platform
import threading
import time
import uuid
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import io
from common.model_paths import DEFAULT_YOLO_PATH, resolve_yolo_path, REPOSITORY_ROOT
from common.control_timing import CONTROL_PERIODS_S
from common.tx_setup import TX_CAMERA_SETTINGS
from common.tx_bearing import BEARING_MODEL
from common.tx_moving import MovingConfig, load_kalman_config
from common.pv_detection import TARGET_SELECTION
from Tx.Controller.detection_panel import PVDetector, DetectionPanel
from Tx.Controller.sweep_core import (SCHEMA, KF_MODE, YOLO_MODE, LEARNING_MODE,
                                      SweepSettings, SweepRun, camera_matches)
from Tx.Controller.sweep_logging import SweepLog, file_digest


class SweepPanel(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, padding=6)
        self.app = app
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = self.preparing = self.setup_pending = None
        self.run = self.log = self.detector = None
        self.hardware_ready = self.closed = False
        self.logging_failed = False
        self.display = self.rendered = None
        self.widgets = []
        self.path = tk.StringVar(value=str(DEFAULT_YOLO_PATH))
        self.device = tk.StringVar(value='cuda')
        self.mode = tk.StringVar(value=KF_MODE)
        self.note = tk.StringVar()
        self.kalman_path = tk.StringVar()
        self.source_folders = []
        self.training_cancel = threading.Event()
        self.vars = {}
        row = ttk.Frame(self); row.pack(fill='x')
        ttk.Label(row, text='YOLO .pt').pack(side='left')
        entry = ttk.Entry(row, textvariable=self.path, width=75)
        entry.pack(side='left', expand=True, fill='x')
        button = ttk.Button(row, text='선택', command=self.browse)
        button.pack(side='left'); self.widgets.extend([entry, button])
        row = ttk.Frame(self); row.pack(fill='x', pady=3)
        for label, var, values, width in [
                ('모드', self.mode, (KF_MODE, YOLO_MODE, LEARNING_MODE), 43),
                ('장치', self.device, ('cuda', 'cpu'), 7)]:
            ttk.Label(row, text=label).pack(side='left')
            combo = ttk.Combobox(row, textvariable=var, values=values, state='readonly', width=width)
            combo.pack(side='left', padx=4); self.widgets.append(combo)
        self.vars['dt'] = tk.StringVar(value=f'{SweepSettings().dt:.3f}')
        ttk.Label(row, text='제어주기 s').pack(side='left')
        combo = ttk.Combobox(row, textvariable=self.vars['dt'], values=CONTROL_PERIODS_S,
                             state='readonly', width=7)
        combo.pack(side='left'); self.widgets.append(combo)
        box = ttk.LabelFrame(self, text='최초 명령각 / 운용 범위 / 1회 이동각 (°) · 공란은 표시된 기본값 사용', padding=4)
        box.pack(fill='x')
        defaults = asdict(SweepSettings())
        fields = [('pan', '최초 Pan'), ('pan_min', 'Pan min'), ('pan_max', 'Pan max'), ('pan_step', 'Pan 이동각'),
                  ('tilt', '최초 Tilt'), ('tilt_min', 'Tilt min'), ('tilt_max', 'Tilt max'), ('tilt_step', 'Tilt 이동각')]
        for index, (key, label) in enumerate(fields):
            r, c = divmod(index, 4)
            ttk.Label(box, text=f'{label} (기본 {defaults[key]:g})').grid(row=r*2, column=c, sticky='w', padx=8)
            self.vars[key] = tk.StringVar(value=str(defaults[key]))
            entry = ttk.Entry(box, textvariable=self.vars[key], width=18)
            entry.grid(row=r*2+1, column=c, sticky='w', padx=8); self.widgets.append(entry)
        row = ttk.Frame(self); row.pack(fill='x', pady=3)
        ttk.Label(row, text='Tilt 방식').pack(side='left')
        self.vars['tilt_motion'] = tk.StringVar(value='linear')
        combo = ttk.Combobox(row, textvariable=self.vars['tilt_motion'], values=('linear', 'oscillate'),
                             state='readonly', width=11)
        combo.pack(side='left'); self.widgets.append(combo)
        for key, label in (('tilt_osc_min', '왕복 min'), ('tilt_osc_max', '왕복 max')):
            ttk.Label(row, text=label).pack(side='left', padx=4)
            self.vars[key] = tk.StringVar(value=str(defaults[key]))
            entry = ttk.Entry(row, textvariable=self.vars[key], width=6)
            entry.pack(side='left'); self.widgets.append(entry)
        ttk.Label(row, text='linear: 일정 이동 · oscillate: 범위 안 왕복, Pan 끝점에서 편도 종료').pack(side='left', padx=8)
        row = ttk.Frame(self); row.pack(fill='x', pady=3)
        ttk.Label(row, text='고정 KF 설정 JSON (공란: 원자료 수집 후 재처리)').pack(side='left')
        entry = ttk.Entry(row, textvariable=self.kalman_path, width=55)
        entry.pack(side='left', expand=True, fill='x'); self.widgets.append(entry)
        button = ttk.Button(row, text='KF 선택', command=self.browse_kalman)
        button.pack(side='left'); self.widgets.append(button)
        row = ttk.Frame(self); row.pack(fill='x', pady=3)
        ttk.Label(row, text='편도 메모 (속도·비행방향·환경 등, 매 편도 변경 가능)').pack(side='left')
        self.note_entry = ttk.Entry(row, textvariable=self.note, width=55)
        self.note_entry.pack(side='left', expand=True, fill='x')
        ttk.Label(self, text='버튼 1회 = 편도 1회 · 다음 편도는 끝점에서 이동 부호 자동 반전 · 이동각 0인 축은 유지').pack(anchor='w')
        ttk.Label(self, text='축별 한계에서 정지 · 두 축 모두 완료 시 편도 종료 · 미검출이어도 이동/원본 저장 계속').pack(anchor='w')
        ttk.Label(self, text='기본 720 ms / speed 100 / acc 1 · 최초 이동 후 2초 대기 · 지연 시 명령 몰아 보내기 없음').pack(anchor='w')
        row = ttk.Frame(self); row.pack(fill='x', pady=4)
        self.prepare_button = ttk.Button(row, text='자동 준비 (영상 · 범위 · YOLO)', command=lambda: self.action(self.prepare))
        self.prepare_button.pack(side='left', padx=3)
        self.start_button = ttk.Button(row, text='편도 시작 / 다음 편도', command=lambda: self.action(self.start))
        self.start_button.pack(side='left', padx=3)
        ttk.Button(row, text='중지 · 저장', command=lambda: self.guard(self.stop)).pack(side='left', padx=3)
        ttk.Button(row, text='결과 폴더 확인', command=self.show_folder).pack(side='left', padx=3)
        row = ttk.Frame(self); row.pack(fill='x')
        for label, command in [('초기 데이터 폴더 추가', self.add_source),
                               ('폴더 선택 초기화', self.clear_sources),
                               ('수집 데이터로 초기 모델 학습', lambda: self.action(self.pretrain))]:
            button = ttk.Button(row, text=label, command=command)
            button.pack(side='left', padx=3); self.widgets.append(button)
        self.status = tk.StringVar(value='자동 준비 → QGC 비행 준비 → 편도 수집. 초기 모델 학습은 아래 버튼, 실시간 SAC 제어 연결은 후속 단계입니다.')
        self.progress = tk.StringVar(value='목표각은 명령값입니다. 실제 각도/도달/드론 위치는 측정하지 않습니다.')
        for var in (self.status, self.progress):
            ttk.Label(self, textvariable=var, wraplength=1120).pack(anchor='w')
        ttk.Label(self, text='captures/M5-1/날짜_시각/ · 원본 이미지 + JSON + CSV · 아래 영상은 최근 저장 샘플 (준비 후에는 원본 미리보기)').pack(anchor='w')
        self.canvas = ttk.Label(self, anchor='center'); self.canvas.pack(fill='both', expand=True)

    @property
    def busy(self):
        return (self.preparing is not None or self.future is not None
                or self.run is not None and (self.run.active or self.run.pending is not None))

    def lock(self):
        busy = self.busy
        state = (bool(busy), self.hardware_ready)
        if self.__dict__.get('_lock_state') == state:
            return
        self._lock_state = state
        for widget in self.widgets:
            widget.configure(state='disabled' if busy else ('readonly' if isinstance(widget, ttk.Combobox) else 'normal'))
        self.note_entry.configure(state='disabled' if busy else 'normal')
        self.prepare_button.configure(state='disabled' if busy else 'normal')
        self.start_button.configure(state='disabled' if busy or not self.hardware_ready else 'normal')

    def browse(self):
        path = filedialog.askopenfilename(filetypes=[('YOLO weights', '*.pt')])
        if path:
            self.path.set(path)

    def browse_kalman(self):
        path = filedialog.askopenfilename(filetypes=[('칼만필터 설정', '*.json')])
        if path:
            self.kalman_path.set(path)

    def add_source(self):
        path = filedialog.askdirectory(title='초기 데이터로 사용할 편도 수집 세션 폴더 선택')
        if path and str(Path(path).resolve()) not in self.source_folders:
            self.source_folders.append(str(Path(path).resolve()))
            self.status.set(f'초기 데이터 {len(self.source_folders)}개 세션 선택 · {path}')

    def clear_sources(self):
        self.source_folders.clear()
        self.status.set('초기 데이터 폴더 선택을 초기화했습니다. 원본 파일은 유지됩니다.')

    def kalman_text(self):
        var = self.__dict__.get('kalman_path')
        return var.get().strip() if var is not None else ''

    def pretrain(self):
        if self.busy:
            raise ValueError('편도 수집·저장·준비가 끝난 뒤 초기 모델을 학습하세요.')
        if not self.source_folders:
            raise ValueError('초기 데이터 폴더를 한 개 이상 선택하세요.')
        from Tx.Controller.sweep_training import pretrain
        if not self.kalman_text():
            raise ValueError('사전 실험에서 확정한 KF 설정 JSON을 선택하세요. Q/R/P0를 임의로 넣지 않습니다.')
        kalman, _ = load_kalman_config(self.kalman_text())
        cfg = MovingConfig(dt=float(self.vars['dt'].get().strip() or '.720')); cfg.validate()
        self.stop_other_panels()
        self.training_cancel.clear()
        output = self.app.stage_dir('M5-1')/('pretraining_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        self.pretraining_folder = output
        self.pretraining_status = None
        self.job_kind = 'pretrain'
        self.future = self.pool.submit(pretrain, list(self.source_folders), kalman, cfg, output,
                                       'cpu', self.training_cancel)
        self.status.set('원자료를 고정 KF로 시간순 재처리 → 유효 데이터 취합 → 이동 SAC 초기 학습 중 · 이동 명령 없음')
        self.lock()

    def settings(self):
        return SweepSettings.from_text({k: var.get() for k, var in self.vars.items()})

    def stop_other_panels(self):
        a = self.app
        if a.stationary.busy or (a.stationary.run is not None and a.stationary.run.pending):
            raise ValueError('M3-2 실행·저장·명령 응답을 먼저 종료하세요.')
        if a.timing.active or a.timing.preparing or a.timing.profile.running:
            raise ValueError('M1-5 측정을 먼저 종료하세요.')
        if a.servo.pending:
            raise ValueError('기존 서보 명령 응답을 기다리세요.')
        a.stationary.stop(); a.detection.stop(); a.timing.stop()
        if a.stationary.inference is not None or a.detection.future is not None or a.timing.future is not None:
            raise ValueError('기존 검출 작업이 종료된 뒤 다시 누르세요.')

    def prepare(self):
        if self.busy:
            raise ValueError('현재 작업·저장·응답이 끝난 뒤 다시 준비하세요.')
        if self.mode.get() == LEARNING_MODE:
            raise ValueError('실시간 SAC 제어는 후속 연결 단계입니다. 현재는 수집 모드와 아래 초기 모델 학습 버튼을 사용하세요.')
        cfg = self.settings()
        kalman_path = self.kalman_text()
        fixed_kalman, kalman_identity = load_kalman_config(kalman_path) if kalman_path else (None, None)
        path, device, mode = resolve_yolo_path(self.path.get()), self.device.get(), self.mode.get()
        if mode not in (KF_MODE, YOLO_MODE) or device not in ('cuda', 'cpu'):
            raise ValueError('모드와 YOLO 장치를 확인하세요.')
        self.stop_other_panels()
        if not self.app.servo.online:
            raise ValueError('서버·Pi를 실행하고 제어 연결을 확인하세요.')
        if self.run:
            self.run.finalize()
        self.cfg, self.prepared_mode = cfg, mode
        self.fixed_kalman, self.kalman_identity, self.prepared_kalman_path = fixed_kalman, kalman_identity, kalman_path
        self.logging_failed = False
        self.hardware_ready = False
        self.detector = self.run = self.log = self.display = self.rendered = None
        self.preparing, self.job_kind = 'model', 'load'
        self.setup_deadline = time.monotonic()+120
        self.status.set('YOLO 준비 중 · 이동 명령 없음')

        def load():
            detector = PVDetector(path, device)
            if cfg.class_id not in detector.names:
                raise ValueError('YOLO PV class ID 0을 확인하세요.')
            return detector, dict(path=str(path), sha256=file_digest(path), device=device,
                                  class_names=detector.names)

        self.future = self.pool.submit(load)
        self.lock()

    def prepare_hardware(self):
        self.preparing = 'limits'
        self.setup_pending = 'm5-1-setup-'+uuid.uuid4().hex
        self.setup_deadline = time.monotonic()+5
        self.camera_requested_at = time.monotonic()
        for key, value in TX_CAMERA_SETTINGS.items():
            self.app.values[key].set('' if value is None else str(value))
        if not self.app.send(dict(cmd='preview', enable=True, request_id='m5-1-camera-'+uuid.uuid4().hex, **TX_CAMERA_SETTINGS)):
            raise ValueError('카메라 시작 실패')
        if not self.app.send(dict(cmd='servo_config', request_id=self.setup_pending, **self.cfg.limits)):
            raise ValueError('서보 범위 설정 실패')
        self.status.set('카메라 설정 · 실제 서보 응답 · 새 영상 확인 중')

    def check_camera(self):
        frame = self.app.current
        if not frame or not camera_matches(frame, self.app.size) or not 0 <= time.monotonic()-frame[2] <= self.cfg.dt:
            raise ValueError('설정과 일치하는 최신 실제 카메라 영상이 필요합니다. 자동 준비를 확인하세요.')
        return frame

    def start(self):
        if self.busy:
            raise ValueError('편도·저장·명령 응답이 끝난 뒤 다음 편도를 시작하세요.')
        if not self.hardware_ready:
            raise ValueError('이 탭에서 자동 준비를 완료하세요.')
        if (self.settings() != self.cfg or self.mode.get() != self.prepared_mode
                or str(resolve_yolo_path(self.path.get())) != self.identity['path']
                or self.device.get() != self.identity['device']):
            raise ValueError('설정이 바뀌었습니다. 자동 준비로 새 세션을 시작하세요.')
        if self.kalman_text() != self.__dict__.get('prepared_kalman_path', ''):
            raise ValueError('KF 설정 파일이 바뀌었습니다. 자동 준비로 새 세션을 시작하세요.')
        self.stop_other_panels()
        frame = self.check_camera()
        servo = self.app.servo
        if not servo.online or servo.simulated is not False or servo.limits != self.cfg.limits:
            raise ValueError('실제 Pi와 적용 범위를 확인하고 다시 자동 준비하세요.')
        if self.run is None:
            self.log = SweepLog(self.app.stage_dir('M5-1')/datetime.now().strftime('%Y%m%d_%H%M%S_%f'), self.config(frame))
            self.run = SweepRun(self.cfg, self.log, self.app.send)
        self.run.start_leg(self.note.get().strip())
        self.pretraining_status = None
        self.status.set(f'T{self.run.test:03d} 편도 진행 중 · 저장: {self.log.folder}')
        self.lock()

    def config(self, frame):
        packages = {}
        for package in ('ultralytics', 'torch', 'Pillow', 'numpy'):
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                packages[package] = None
        sources = ['Tx/Controller/'+name for name in ('sweep_core.py', 'sweep_logging.py', 'sweep_panel.py', 'detection_panel.py')]
        sources += ['common/'+name for name in ('tx_bearing.py', 'pv_detection.py', 'tx_setup.py', 'control_timing.py',
                                               'tx_moving.py', 'tx_tracking.py')]
        sources += ['Tx/Controller/sweep_training.py', 'Tx/Controller/stationary_learning.py']
        moving_cfg = MovingConfig(dt=self.cfg.dt)
        return dict(schema=SCHEMA, mode=self.prepared_mode, settings=asdict(self.cfg), model=self.identity,
            target_selection=TARGET_SELECTION, bearing_model=BEARING_MODEL, camera=TX_CAMERA_SETTINGS,
            initial_camera_metadata=frame[1], python=platform.python_version(), packages=packages,
            moving_training=dict(tracking=asdict(moving_cfg), normalization=moving_cfg.normalization,
                purpose='fixed sweep experience for later initial replay; not learned control',
                replay_policy='valid adjacent observations only; reset history each leg; no missing-gap bridges'),
            fixed_kalman=self.__dict__.get('fixed_kalman'), kalman_file=self.__dict__.get('kalman_identity'),
            source_sha256={name: file_digest(REPOSITORY_ROOT/name) for name in sources},
            created_local=datetime.now().astimezone().isoformat(), created_unix_ns=time.time_ns(),
            timing=dict(nominal_period_s=self.cfg.dt, frame_policy='after ACK and command dwell; matching Pi command_id',
                scheduling='one in-flight observation; no catch-up commands', host_clock='monotonic',
                actual_arrival_verified=False, pi_clock_synchronized=False),
            kalman=dict(stage='fixed_parameter_filter' if self.__dict__.get('fixed_kalman') else 'observation_collection_only',
                Q_applied=(self.__dict__.get('fixed_kalman') or {}).get('Q'),
                R_applied=(self.__dict__.get('fixed_kalman') or {}).get('R'),
                P0_applied=(self.__dict__.get('fixed_kalman') or {}).get('P0'),
                measurement_order=['alpha', 'beta'], measurement_units='rad',
                time_source='receive_s; use actual differences, reset at each test; azimuth requires unwrap',
                missing_policy='retain gaps in observations.csv; never fill with zero'),
            sweep_policy='clamp each axis independently; stop when all moving axes reach bounds; reverse only on user click',
            image_policy='one original JPEG per sampled pose including initial/end; no automatic ground-truth labels',
            hardware_validated=False, drone_control='external QGC; no telemetry or commands in this application')

    def invalidate(self, reason):
        self.hardware_ready = False
        self.preparing = self.setup_pending = None
        if self.run:
            try:
                self.run.abort(reason)
            except OSError:
                # abort changes the motion state before writing its audit event.
                # A full disk must not prevent cancellation or break the UI poll.
                reason += ' (중단 로그 저장 실패)'
                self.logging_failed = True
        self.status.set(reason+' · 추가 명령 중지. 작업 완료 후 자동 준비로 다시 시작하세요.')

    def stop(self):
        cancel = self.__dict__.get('training_cancel')
        if cancel is not None:
            cancel.set()
        self.invalidate('user_stop')
        self.status.set('추가 명령 중지 · 진행 중 저장/응답은 마무리합니다. 이미 전달된 서보 이동을 즉시 정지시키지는 않습니다.')
        self.lock()

    def event(self, event):
        kind = event.get('event')
        disconnected = kind == 'network' and event.get('state') != 'connected'
        foreign = kind == 'servo' and event.get('operation') in ('move', 'servo_config')
        if self.preparing and self.setup_pending and event.get('request_id') == self.setup_pending and kind == 'servo':
            if (event.get('operation') != 'servo_config' or event.get('simulated') is not False
                    or event.get('available') is not True or event.get('limits') != self.cfg.limits):
                self.invalidate('invalid_setup_reply')
                return
            servo = self.app.servo
            servo.limits, servo.simulated = dict(self.cfg.limits), False
            for key, value in dict(self.cfg.limits, speed=self.cfg.speed, acc=self.cfg.acc).items():
                servo.vars[key].set(str(value))
            self.setup_pending = None
            self.preparing = 'frames'
            self.setup_deadline = time.monotonic()+15
            return
        if self.run and self.run.reply(event):
            if self.run.last_reply_valid:
                self.app.servo.last = event.get('commanded')
                for axis, value in zip(('pan', 'tilt'), self.run.pose):
                    self.app.servo.vars[axis].set(str(value))
            if self.run.phase == 'aborted':
                self.hardware_ready = False
            return
        if disconnected or kind in ('hello', 'agent', 'ready', 'error') or foreign:
            if self.busy or self.hardware_ready:
                if self.log:
                    self.log.event('device_event', detail=event)
                self.invalidate('connection_or_device_change')

    def poll(self):
        if self.closed:
            return
        if self.future is not None and self.future.done():
            future, self.future = self.future, None
            if self.job_kind == 'load':
                loaded = future.result()
                if self.preparing == 'model':
                    self.detector, self.identity = loaded
                    self.prepare_hardware()
                # Cancelled loads are deliberately discarded without any command.
            elif self.job_kind == 'pretrain':
                try:
                    result = future.result()
                except ValueError:
                    if not self.training_cancel.is_set():
                        raise
                    self.pretraining_status = '초기 데이터 학습을 취소했습니다. 원본 수집 자료는 유지됩니다.'
                else:
                    self.pretraining_status = (f"초기 모델 저장 · 유효 {result['total_transitions']}개 / "
                                               f"업데이트 {result['completed_updates']}/{result['planned_updates']}회 · {result['checkpoint']}")
                self.status.set(self.pretraining_status)
            else:
                row, image = future.result()
                self.display = (image, row)
                if self.run.active:
                    try:
                        self.check_camera()
                    except ValueError:
                        self.run.abort('camera_stale_or_changed')
                self.run.captured(row)
        now = time.monotonic()
        if self.preparing:
            if now > self.setup_deadline:
                raise ValueError('자동 준비 시간 초과: '+self.preparing)
            frame = self.app.current
            if (self.preparing == 'frames' and frame and frame[2] > self.camera_requested_at
                    and 0 <= now-frame[2] <= self.cfg.dt and camera_matches(frame, self.app.size)):
                self.preparing = None
                self.hardware_ready = True
                self.status.set('자동 준비 완료 · 최초 이동은 편도 시작 버튼에서 실행합니다.')
        if self.run:
            self.run.tick()
            if self.future is None and self.run.active:
                frame = self.app.current
                context = self.run.capture_request(frame, self.app.received_count, self.app.size)
                if context:
                    self.job_kind = 'capture'
                    self.future = self.pool.submit(self.log.capture, self.detector, self.cfg,
                                                   frame, context, self.prepared_mode == KF_MODE)
            if not self.run.active and self.future is None:
                if not self.logging_failed:
                    self.run.finalize()
                if self.run.phase == 'complete' and self.hardware_ready:
                    self.status.set(self.__dict__.get('pretraining_status') or
                                    f'T{self.run.test:03d} 완료 · 다음 버튼은 반대 방향 · {self.log.folder}')
                elif self.run.phase == 'aborted':
                    self.hardware_ready = False
                    self.status.set(f'T{self.run.test:03d} 중단: {self.run.reason} · 저장 후 자동 준비 필요')
            self.progress.set(f'T{self.run.test:03d} · {self.run.phase} · 명령 Pan/Tilt {self.run.pose}° · '
                              f'저장 {self.run.saved} / 검출 {self.run.detected} / KF 관측 {self.run.valid}')
        self.lock()

    def action(self, fn):
        # Rejected button clicks must not abort an already running experiment.
        try:
            fn()
        except ValueError as exc:
            messagebox.showerror('M5-1 설정', str(exc))
        except Exception as exc:
            self.fail(exc)

    def guard(self, fn):
        try:
            fn()
        except Exception as exc:
            self.fail(exc)

    def fail(self, exc):
        if isinstance(exc, OSError):
            self.logging_failed = True
        self.invalidate('error: '+str(exc))
        try:
            self.app.record(dict(event='m5_1_error', message=str(exc)))
        except OSError:
            pass
        self.lock()
        messagebox.showerror('M5-1', str(exc))

    def render(self):
        if self.display:
            image, row = self.display
            mark = ('saved', row['frame_id'])
            if image is None or mark == self.rendered:
                return
            image = DetectionPanel.annotated(image, row)
        elif self.hardware_ready and self.app.current:
            frame = self.app.current
            mark = ('preview', frame[2])
            if mark == self.rendered or time.monotonic()-frame[2] > self.cfg.dt:
                return
            image = Image.open(io.BytesIO(frame[0])).convert('RGB')
        else:
            return
        image.thumbnail((1000, 340))
        photo = ImageTk.PhotoImage(image)
        self.canvas.configure(image=photo); self.canvas.image = photo
        self.rendered = mark

    def show_folder(self):
        messagebox.showinfo('M5-1 결과', str(self.__dict__.get('pretraining_folder') or self.log.folder)
                            if self.log or self.__dict__.get('pretraining_folder') else '아직 수집한 결과가 없습니다.')

    def close(self):
        self.closed = True
        cancel = self.__dict__.get('training_cancel')
        if cancel is not None:
            cancel.set()
        self.invalidate('application_close')
        self.pool.shutdown(wait=True)
        try:
            if self.future is not None and self.job_kind == 'capture':
                row, _ = self.future.result()
                self.run.captured(row)
            if self.run:
                self.run.pending = None
                if not self.logging_failed:
                    self.run.finalize()
        except Exception as exc:
            self.app.record(dict(event='m5_1_close_error', message=str(exc)))
