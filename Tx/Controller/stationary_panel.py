"""M3-2 tab: camera preview, stationary SAC training, resume and evaluation."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
import time
import uuid
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from PIL import ImageDraw, ImageTk
from common.model_paths import DEFAULT_YOLO_PATH, resolve_yolo_path
from common.control_timing import CONTROL_PERIODS_S, PRIMARY_CONTROL_PERIOD_S, EVALUATION_CONTROL_PERIODS_S
from common.tx_tracking import (SCHEMA, RUN_SCHEMA, TrackingConfig, RunSettings, Sample,
                                invalid_reason, checkpoint_run_config)
from common.servo import move_from, number
from common.pv_detection import TARGET_SELECTION
from common.tx_setup import TX_CAMERA_SETTINGS
from Tx.Controller.detection_panel import PVDetector, DetectionPanel
from Tx.Controller.stationary_core import StationaryRun
from Tx.Controller.stationary_learning import SACLearner, RunLog, SAC_SETTINGS, checkpoint_info, digest


class StationaryPanel(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, padding=6)
        self.app = app
        self.infer_pool = ThreadPoolExecutor(max_workers=1)
        self.learn_pool = ThreadPoolExecutor(max_workers=1)
        self.inference = self.work = None
        self.detector = self.learner = self.run = self.log = None
        self.previewing = False
        self.last_frame = self.rendered = None
        self.latest = self.display = None
        self.closed = False
        self.servo_available = None
        self.preparing = None
        self.setup_pending = None
        self.hardware_ready = False
        self.widgets = []
        self.path = tk.StringVar(value=str(DEFAULT_YOLO_PATH))
        self.checkpoint = tk.StringVar()
        self.mode = tk.StringVar(value='신규 학습')
        self.device = tk.StringVar(value='cuda')
        self.sac_device = tk.StringVar(value='cpu')
        self.vars = {}
        for label, var, choose in [('YOLO .pt', self.path, self.browse_yolo),
                                   ('Checkpoint 폴더', self.checkpoint, self.browse_checkpoint)]:
            row = ttk.Frame(self); row.pack(fill='x', pady=2)
            ttk.Label(row, text=label, width=18).pack(side='left')
            entry = ttk.Entry(row, textvariable=var, width=76)
            entry.pack(side='left', fill='x', expand=True)
            button = ttk.Button(row, text='선택', command=choose); button.pack(side='left')
            self.widgets.extend([entry, button])
        row = ttk.Frame(self); row.pack(fill='x', pady=3)
        for label, var, values in [('모드', self.mode, ('신규 학습', '이어서 학습', '고정 모델 평가')),
                                    ('YOLO 장치', self.device, ('cuda', 'cpu')),
                                    ('SAC 장치', self.sac_device, ('cpu', 'cuda'))]:
            ttk.Label(row, text=label).pack(side='left')
            combo = ttk.Combobox(row, textvariable=var, values=values, state='readonly', width=15)
            combo.pack(side='left', padx=4); self.widgets.append(combo)
            if var is self.mode:
                combo.bind('<<ComboboxSelected>>', self.mode_changed)
        period = tk.StringVar(value=f'{PRIMARY_CONTROL_PERIOD_S:.3f}'); self.vars['dt'] = period
        ttk.Label(row, text='제어주기 s').pack(side='left')
        combo = ttk.Combobox(row, textvariable=period, values=[f'{v:.3f}' for v in CONTROL_PERIODS_S], state='readonly', width=7)
        combo.pack(side='left'); self.widgets.append(combo)
        self.period_combo = combo
        self.timing_note = tk.StringVar()
        ttk.Label(self, textvariable=self.timing_note, wraplength=1120).pack(anchor='w')
        box = ttk.LabelFrame(self, text='Episode 설정 · 최대 step과 반복은 초기 실험용 입력값', padding=4)
        box.pack(fill='x')
        defaults = asdict(RunSettings())
        defaults['seed'] = 42
        fields = [('max_steps', '최대 step'), ('episodes', '반복 횟수'),
                  ('settle_s', '초기 이동 대기 s'), ('updates_per_transition', '경험당 업데이트'),
                  ('seed', 'Seed'), ('pan_low', '초기 Pan min'), ('pan_high', 'max'),
                  ('tilt_low', '초기 Tilt min'), ('tilt_high', 'max')]
        for i, (key, label) in enumerate(fields):
            r, c = divmod(i, 5)
            ttk.Label(box, text=label).grid(row=r*2, column=c*2, sticky='w')
            var = tk.StringVar(value=str(defaults[key])); self.vars[key] = var
            entry = ttk.Entry(box, textvariable=var, width=10)
            entry.grid(row=r*2+1, column=c*2, padx=5, sticky='w'); self.widgets.append(entry)
        ttk.Label(self, text='8차원 관측 · Δ ±5° / 1° 단위 · reward = −거리/1000 + intensity mean · 적중 ≤23 px 후 유지').pack(anchor='w')
        ttk.Label(self, text='유효 경험 1,000개부터 episode 종료 후 학습 · batch 256 · 256×256 · 미검출 3초 · captures/M3-2/').pack(anchor='w')
        row = ttk.Frame(self); row.pack(fill='x', pady=4)
        for label, fn in [('자동 준비 (영상 · 서보 · 모델)', self.prepare), ('검출 미리보기', self.preview),
                          ('학습 / 평가 시작', self.start), ('전체 중지 · 저장', self.stop), ('결과 폴더 확인', self.show_folder)]:
            ttk.Button(row, text=label, command=lambda f=fn: self.guard(f)).pack(side='left', padx=3)
        manual = ttk.LabelFrame(self, text='Pan / Tilt 수동 이동 · 자동 준비 후 사용 · 학습/평가/저장 중 잠금', padding=4)
        manual.pack(fill='x')
        row = ttk.Frame(manual); row.pack(fill='x')
        self.manual_vars = {}; self.manual_widgets = []
        for key, label, value in [('pan', 'Pan °', '-17'), ('tilt', 'Tilt °', '0'), ('step', 'Step °', '1')]:
            ttk.Label(row, text=label).pack(side='left')
            var = tk.StringVar(value=value); self.manual_vars[key] = var
            entry = ttk.Entry(row, textvariable=var, width=7)
            entry.pack(side='left', padx=3); self.manual_widgets.append(entry)
        button = ttk.Button(row, text='입력 각도로 이동', command=lambda: self.manual_guard(self.manual_move))
        button.pack(side='left', padx=3); self.manual_widgets.append(button)
        for axis in ('pan', 'tilt'):
            for sign in (-1, 1):
                button = ttk.Button(row, text=f'{axis.title()} {"+" if sign > 0 else "−"}',
                    command=lambda a=axis, s=sign: self.manual_guard(lambda: self.manual_move(a, s)))
                button.pack(side='left', padx=3); self.manual_widgets.append(button)
        ttk.Label(manual, textvariable=self.app.servo.state, wraplength=1120).pack(anchor='w')
        ttk.Label(manual, text='±는 직전 전송 명령에서 1° 단위 증감 · 입력값은 목표각이며 실제각/도달 여부는 미측정').pack(anchor='w')
        self.status = tk.StringVar(value='이 탭에서 자동 준비 → PV 검출 확인 → 학습 / 평가 시작')
        self.progress = tk.StringVar(value='목표각은 명령값이며 실제 서보 각도·도달 피드백이 아닙니다.')
        self.result_text = tk.StringVar()
        for var in (self.status, self.progress, self.result_text):
            ttk.Label(self, textvariable=var, wraplength=1120).pack(anchor='w')
        self.canvas = ttk.Label(self, anchor='center'); self.canvas.pack(fill='both', expand=True)
        self.sync_period_control()
        self.sync_manual_controls()

    @property
    def active(self):
        return self.run is not None and self.run.active

    @property
    def busy(self):
        return self.active or self.work is not None or self.preparing is not None

    def lock(self, locked):
        for widget in self.widgets:
            widget.configure(state='disabled' if locked else ('readonly' if isinstance(widget, ttk.Combobox) else 'normal'))
        self.sync_period_control(locked)
        self.sync_manual_controls()

    def mode_changed(self, event=None):
        self.vars['dt'].set(f'{PRIMARY_CONTROL_PERIOD_S:.3f}' if self.mode.get() == '신규 학습' else '저장값')
        self.sync_period_control()

    def sync_period_control(self, locked=False):
        if 'period_combo' not in self.__dict__:
            return
        mode = self.mode.get()
        if mode == '신규 학습':
            values = tuple(f'{v:.3f}' for v in CONTROL_PERIODS_S)
            note = '새 학습: 선택한 주기로 학습합니다.'
        elif mode == '이어서 학습':
            values = ('저장값',)
            note = '이어서 학습: checkpoint에 저장된 주기를 자동 적용합니다.'
        else:
            values = ('저장값', *(f'{v:.3f}' for v in EVALUATION_CONTROL_PERIODS_S))
            note = '고정 모델 평가: 저장값 또는 비교 주기를 선택하세요. 0.600 / 0.500 s는 이동 중 명령 갱신이 가능한 실험값입니다.'
        self.period_combo.configure(values=values, state='disabled' if locked or mode == '이어서 학습' else 'readonly')
        self.timing_note.set(note)

    def check_manual_available(self):
        if self.busy:
            raise ValueError('자동 준비·학습·평가·저장이 끝난 뒤 수동 이동하세요.')
        if not self.hardware_ready:
            raise ValueError('이 탭에서 자동 준비를 먼저 완료하세요.')
        servo = self.app.servo
        if (not servo.online or self.servo_available is not True or servo.simulated is not False
                or servo.limits != self.cfg.limits):
            raise ValueError('실제 Pi 연결과 운용 범위를 확인하고 자동 준비를 다시 실행하세요.')
        if servo.pending or (self.run is not None and self.run.pending):
            raise ValueError('전송된 명령의 응답을 기다리세요.')
        if self.app.timing.active or self.app.timing.preparing or self.app.timing.profile.running:
            raise ValueError('M1-5 측정을 먼저 종료하세요.')

    def sync_manual_controls(self):
        if 'manual_widgets' not in self.__dict__:
            return
        try:
            self.check_manual_available()
            state = 'normal'
        except ValueError:
            state = 'disabled'
        if self.__dict__.get('_manual_state') == state:
            return
        self._manual_state = state
        for widget in self.manual_widgets:
            widget.configure(state=state)

    def manual_guard(self, fn):
        # A rejected manual click must never cancel a running episode.
        try:
            fn()
        except (ValueError, TypeError, OSError, KeyError) as exc:
            messagebox.showerror('M3-2 수동 이동', str(exc))
        finally:
            self.sync_manual_controls()

    def manual_move(self, axis=None, sign=0):
        self.check_manual_available()
        servo = self.app.servo
        if axis is None:
            target = {a: number(self.manual_vars[a].get(), a) for a in ('pan', 'tilt')}
        else:
            if axis not in ('pan', 'tilt') or sign not in (-1, 1):
                raise ValueError('Pan/Tilt 증감 방향을 확인하세요.')
            if servo.last is None:
                raise ValueError('먼저 입력 각도로 이동하여 직전 명령을 확인하세요.')
            step = number(self.manual_vars['step'].get(), 'step')
            if step < 1 or not step.is_integer():
                raise ValueError('Step은 1° 이상의 정수입니다.')
            target = {a: number(servo.last[a], a) for a in ('pan', 'tilt')}
            target[axis] += sign*step
        if not all(v.is_integer() for v in target.values()):
            raise ValueError('Pan/Tilt 목표각은 1° 단위 정수입니다.')
        move = move_from(dict(target, speed=self.cfg.speed, acc=self.cfg.acc), self.cfg.limits)
        self.stop_other_panels()
        # Reuse shared request/response tracking so every tab sees the pending move.
        servo.request(dict(cmd='move', **move))
        if servo.pending:
            self.app.record(dict(event='m3_2_manual_move', request_id=servo.pending[0],
                                 command=move, actual_angle=None, arrival_verified=False))

    def guard(self, fn):
        try:
            fn()
        except Exception as exc:
            self.cancel_preparation()
            if self.active:
                self.run.abort('execution_error', str(exc))
            self.status.set('오류: '+str(exc))
            self.app.record(dict(event='m3_2_error', message=str(exc)))
            messagebox.showerror('M3-2', str(exc))

    def browse_yolo(self):
        path = filedialog.askopenfilename(filetypes=[('YOLO', '*.pt')])
        if path: self.path.set(path)

    def browse_checkpoint(self):
        path = filedialog.askdirectory(title='manifest.json이 있는 episode checkpoint 폴더 선택')
        if path: self.checkpoint.set(path)

    def stop_other_panels(self):
        if self.app.timing.active or self.app.timing.preparing or self.app.timing.profile.running:
            raise ValueError('M1-5 측정을 먼저 종료하세요.')
        if self.app.servo.pending:
            raise ValueError('기존 명령 응답이 끝난 뒤 시작하세요.')
        self.app.detection.stop()
        self.app.timing.stop()
        if self.app.detection.future is not None or self.app.timing.future is not None:
            raise ValueError('기존 검출 작업 종료 후 다시 시작하세요.')

    def settings(self):
        keys = set(RunSettings.__dataclass_fields__)
        floats = {'settle_s', 'updates_per_transition'}
        values = {k: (float(v.get()) if k in floats else int(v.get())) for k, v in self.vars.items() if k in keys}
        return RunSettings(**values)

    def prepare(self):
        if self.busy or self.inference is not None:
            raise ValueError('현재 작업을 중지하고 완료를 기다리세요.')
        self.stop_other_panels()
        self.previewing = False
        if not self.app.servo.online:
            raise ValueError('서버와 Raspberry Pi를 실행하고 연결을 확인하세요.')
        mode = {'신규 학습': 'new', '이어서 학습': 'resume', '고정 모델 평가': 'evaluate'}[self.mode.get()]
        period_text = self.vars['dt'].get()
        period = None if period_text == '저장값' else float(period_text)
        cfg = None
        if mode == 'new':
            cfg = TrackingConfig(dt=period, **(self.app.servo.limits or {}))
            cfg.validate()
        source = self.checkpoint.get().strip()
        path, yolo_device, sac_device = resolve_yolo_path(self.path.get()), self.device.get(), self.sac_device.get()
        seed = int(self.vars['seed'].get())
        if not 0 <= seed < 2**32:
            raise ValueError('Seed는 0~2^32−1 정수입니다.')
        self.learner = self.detector = None
        self.latest = self.display = None
        self.rendered = None
        self.preparing = 'model'
        self.hardware_ready = False
        self.setup_pending = None
        self.setup_deadline = time.monotonic()+120
        self.lock(True)
        self.status.set('YOLO / SAC 준비 중 · 이동 명령 없음')
        self.work_kind = 'prepare'

        def load():
            selected = cfg
            training_period = cfg.dt if cfg is not None else None
            if mode != 'new':
                _, _, saved_cfg = checkpoint_info(source)
                training_period = saved_cfg.dt
                selected = checkpoint_run_config(saved_cfg, mode, period)
            detector = PVDetector(path, yolo_device)
            if selected.class_id not in detector.names:
                raise ValueError('YOLO PV class ID 0을 확인하세요.')
            learner = SACLearner(selected, mode, source or None, sac_device, seed)
            identity = dict(yolo_path=path, yolo_sha256=digest(path), yolo_device=yolo_device,
                            sac_device=sac_device, source_checkpoint=learner.source, selected_checkpoint=source, mode=mode, seed=seed,
                            training_control_period_s=training_period)
            return detector, learner, selected, identity

        self.work = self.learn_pool.submit(load)

    def cancel_preparation(self):
        if self.preparing is not None:
            self.preparing = None
            self.setup_pending = None
            self.hardware_ready = False
            self.previewing = False
            if self.work is None:
                self.lock(False)

    def prepare_hardware(self):
        """Configure camera and command limits; no reset/motion before Start."""
        self.settings().validate(self.cfg)
        self.preparing = 'limits'
        self.setup_pending = 'm3-2-setup-'+uuid.uuid4().hex
        self.setup_deadline = time.monotonic()+5
        self.camera_requested_at = time.monotonic()
        for key, value in TX_CAMERA_SETTINGS.items():
            self.app.values[key].set('' if value is None else str(value))
        if not self.send(dict(cmd='preview', enable=True, request_id='m3-2-camera-'+uuid.uuid4().hex, **TX_CAMERA_SETTINGS)):
            raise ValueError('카메라 자동 시작 실패')
        if not self.send(dict(cmd='servo_config', request_id=self.setup_pending, **self.cfg.limits)):
            raise ValueError('서보 운용 범위 자동 적용 실패')
        self.status.set('자동 준비: 카메라 시작 · 실제 서보 연결/운용 범위 확인 중')

    def prepare_poll(self):
        if self.preparing is None:
            return
        now = time.monotonic()
        if now >= self.setup_deadline:
            raise ValueError('자동 준비 시간 초과: '+self.preparing)
        if self.preparing != 'frames':
            return
        frame = self.app.current
        if not frame or frame[2] <= self.camera_requested_at or not 0 <= now-frame[2] <= self.cfg.dt:
            return
        if frame[1].get('simulated') is not False:
            raise ValueError('실제 카메라 영상이 필요합니다.')
        if self.app.size != (self.cfg.width, self.cfg.height) or not self.camera_matches(frame):
            return
        self.preparing = None
        self.hardware_ready = True
        self.previewing = True
        self.last_frame = None
        self.lock(False)
        self.progress.set(f'자동 적용: Pan {self.cfg.pan_min}~{self.cfg.pan_max}°, Tilt {self.cfg.tilt_min}~{self.cfg.tilt_max}° · {self.cfg.dt*1000:.0f} ms')
        training_period = self.identity['training_control_period_s']
        self.status.set(f'자동 준비 완료 · 학습 주기 {training_period:.3f} s / 실행 주기 {self.cfg.dt:.3f} s · 학습 / 평가 시작을 누르세요.')

    def check_camera(self):
        frame = self.app.current
        if (not frame or self.app.size != (1296, 972) or frame[1].get('simulated') is not False
                or not 0 <= time.monotonic()-frame[2] <= self.cfg.dt):
            raise ValueError('1296×972 실제 카메라의 최신 영상이 필요합니다.')
        if not self.camera_matches(frame):
            raise ValueError('카메라 설정이 달라졌습니다. 이 탭에서 자동 준비를 다시 실행하세요.')
        return frame

    @staticmethod
    def camera_matches(frame):
        expected = TX_CAMERA_SETTINGS
        requested = frame[1].get('requested', {})
        return all(k in requested and requested[k] == value for k, value in expected.items())

    def check_prepared(self):
        if self.learner is None:
            raise ValueError('이 탭에서 자동 준비부터 실행하세요.')
        if self.busy:
            raise ValueError('현재 작업이 끝난 뒤 시작하세요.')
        if not self.hardware_ready:
            raise ValueError('이 탭에서 자동 준비를 완료하세요.')
        if (resolve_yolo_path(self.path.get()) != self.identity['yolo_path']
                or self.device.get() != self.identity['yolo_device'] or self.sac_device.get() != self.identity['sac_device']
                or self.checkpoint.get().strip() != self.identity['selected_checkpoint']
                or int(self.vars['seed'].get()) != self.identity['seed']
                or self.vars['dt'].get() == '저장값' or float(self.vars['dt'].get()) != self.cfg.dt
                or {'신규 학습': 'new', '이어서 학습': 'resume', '고정 모델 평가': 'evaluate'}[self.mode.get()] != self.learner.mode):
            raise ValueError('변경된 설정으로 자동 준비를 다시 실행하세요.')
        if self.run is not None and self.run.pending:
            raise ValueError('전송된 명령 응답 확인이 필요합니다. Pi 상태를 확인하세요.')

    def preview(self):
        self.check_prepared()
        self.stop_other_panels()
        self.check_camera()
        self.previewing = True
        self.last_frame = None
        self.status.set('PV 미리보기 · 서보 이동 없음')

    def start(self):
        self.check_prepared()
        if self.run is not None:
            raise ValueError('새 실행은 자동 준비를 다시 실행하세요. 이어서 학습은 저장된 checkpoint를 선택하세요.')
        self.stop_other_panels()
        frame = self.check_camera()
        servo = self.app.servo
        if not servo.online or self.servo_available is not True or servo.simulated is not False or servo.limits != self.cfg.limits:
            raise ValueError('실제 Pi 연결과 저장 모델의 동일한 운용 범위를 확인하세요.')
        settings = self.settings(); settings.validate(self.cfg)
        if self.log is not None:
            self.log.close()
        folder = self.app.stage_dir('M3-2')/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        config = dict(schema=RUN_SCHEMA, checkpoint_schema=SCHEMA,
                      tracking=asdict(self.cfg), run=asdict(settings), sac=SAC_SETTINGS,
                      control_timing=dict(training_period_s=self.identity['training_control_period_s'],
                                          execution_period_s=self.cfg.dt,
                                          evaluation_override=self.learner.mode == 'evaluate' and
                                          self.cfg.dt != self.identity['training_control_period_s']),
                      episode_policy='count_initial_target_lost_and_target_lost; stop_on_device_or_camera_failure',
                      model=self.identity, target_selection=TARGET_SELECTION, camera_metadata=frame[1],
                      observation_order=['e_u', 'e_v', 'delta_e_u', 'delta_e_v', 'last_delta_pan', 'last_delta_tilt', 'pan', 'tilt'],
                      normalization=[696, 588, 1296, 972, 5, 5], angle_normalization=self.cfg.limits,
                      action_grid='round_half_away_from_zero', beam_pixels='ceil bounds, half-open integer pixel centers',
                      metrics='post-action observations only; normalized variation includes first action after reset',
                      source_references=['https://arxiv.org/abs/1812.05905', 'https://spinningup.openai.com/en/latest/algorithms/sac.html'],
                      hardware_validated=False)
        self.log = RunLog(folder, config)
        self.learner.cancel.clear()
        self.run = StationaryRun(self.cfg, settings, self.learner, self.log, self.send, time.monotonic)
        self.run.latest = self.latest
        self.previewing = True
        self.lock(True)
        self.run.start()
        self.status.set('실행 시작 · 저장: '+str(folder))

    def send(self, command):
        return self.app.send(command, tracking=True)

    def stop(self):
        preparing = self.preparing is not None
        self.cancel_preparation()
        self.previewing = False
        if self.active:
            self.run.abort('user_stop')
            self.status.set('추가 명령 중지 · 완료된 데이터 저장 중 (이미 전달된 이동은 즉시 정지되지 않음)')
        else:
            self.status.set('자동 준비 중단 · 추가 설정/이동 명령 없음' if preparing else '미리보기 정지')

    def show_folder(self):
        messagebox.showinfo('M3-2 결과', str(self.log.folder) if self.log else '실행 결과가 아직 없습니다.')

    def event(self, event):
        if event.get('event') == 'servo':
            self.servo_available = event.get('available')
        if event.get('event') in ('hello', 'agent', 'ready') or (event.get('event') == 'network' and event.get('state') != 'connected'):
            self.servo_available = None
            self.hardware_ready = False
        if self.preparing is not None:
            kind = event.get('event')
            disconnected = kind == 'network' and event.get('state') != 'connected'
            foreign = (kind == 'servo' and event.get('operation') in ('move', 'servo_config')
                       and event.get('request_id') != self.setup_pending)
            if disconnected or foreign or kind in ('hello', 'agent', 'ready', 'error'):
                self.cancel_preparation()
                self.status.set('자동 준비 중단: 연결/장치 상태를 확인하고 자동 준비를 다시 실행하세요.')
                self.app.record(dict(event='m3_2_setup_error', detail=event))
                return
            if self.setup_pending is not None and event.get('request_id') == self.setup_pending and kind == 'servo':
                if (event.get('operation') != 'servo_config' or event.get('simulated') is not False
                        or event.get('available') is not True or event.get('limits') != self.cfg.limits):
                    self.cancel_preparation()
                    self.status.set('자동 준비 실패: 실제 서보 연결과 운용 범위 응답을 확인하세요.')
                    self.app.record(dict(event='m3_2_setup_error', detail=event))
                    return
                servo = self.app.servo
                servo.limits = dict(self.cfg.limits)
                servo.simulated = False
                for key, value in self.cfg.limits.items():
                    servo.vars[key].set(str(value))
                for key in ('speed', 'acc'):
                    servo.vars[key].set(str(getattr(self.cfg, key)))
                self.setup_pending = None
                self.preparing = 'frames'
                self.setup_deadline = time.monotonic()+15
                self.status.set('자동 준비: 새 카메라 영상 확인 중')
            return
        if self.run is None:
            return
        if self.run.reply(event):
            if event.get('event') == 'servo' and event.get('commanded'):
                self.app.servo.last = event['commanded']
                for axis in ('pan', 'tilt'):
                    self.app.servo.vars[axis].set(str(event['commanded'][axis]))
            return
        if not self.active:
            return
        kind = event.get('event')
        disconnected = kind == 'network' and event.get('state') != 'connected'
        # A Pi reconnect/restart also invalidates pending state and requires a new run.
        device_change = kind in ('hello', 'agent', 'ready')
        foreign = kind == 'servo' and event.get('operation') in ('move', 'servo_config')
        if disconnected or device_change or foreign or kind == 'error':
            self.run.abort('connection_or_device_error', str(event))
            self.previewing = False

    def poll(self):
        if self.closed:
            return
        self.sync_manual_controls()
        if self.work is not None and self.work.done():
            future, self.work = self.work, None
            if self.work_kind == 'prepare':
                try:
                    loaded = future.result()
                    if self.preparing == 'model':
                        self.detector, self.learner, self.cfg, self.identity = loaded
                        self.run = None
                        self.vars['dt'].set(f'{self.cfg.dt:.3f}')
                        self.latest = None
                        self.prepare_hardware()
                    else:
                        self.lock(False)  # Stopped while loading: discard completion, send nothing.
                except Exception:
                    self.cancel_preparation()
                    self.lock(False)
                    raise
            else:
                try:
                    report = future.result()
                except Exception:
                    self.run.phase = 'error'
                    self.previewing = False
                    self.lock(False)
                    raise
                self.result_text.set(f"Episode {report['episode']} · {report['reason']} · RMS {report.get('rms_px')} px · 적중 비율 {report.get('hit_ratio')} · 유효 경험 {report['total_transitions']} · 업데이트 {report['completed_updates']}/{report['planned_updates']}")
                self.run.saved()
                if not self.active:
                    self.lock(False)
                    self.status.set('실행 종료 · '+str(self.log.folder))
                    self.previewing = False
        self.prepare_poll()
        if self.inference is not None and self.inference.done():
            future, self.inference = self.inference, None
            result, frame, frame_id, processed = future.result()
            sample = Sample(frame_id, frame[2], processed, result['image'].width, result['image'].height,
                            result['target'], result['count'], frame[1].get('seq'), frame[3],
                            result['inference_ms'], frame[1].get('simulated', True))
            self.latest = sample
            self.display = (result['image'], sample)
            if self.run is not None and self.active:
                if not self.camera_matches(frame):
                    self.run.abort('camera_configuration', '실행 중 카메라 설정 변경')
                self.run.latest = sample
                self.record_detection(sample)
        if self.previewing and self.inference is None and self.detector is not None:
            frame = self.app.current
            if frame and frame[2] != self.last_frame and time.monotonic()-frame[2] <= self.cfg.dt:
                self.last_frame = frame[2]
                frame_id = self.app.received_count
                def infer(detector=self.detector, frame=frame, frame_id=frame_id):
                    result = detector.infer(frame[0], self.cfg.confidence, self.cfg.class_id)
                    return result, frame, frame_id, time.monotonic()
                self.inference = self.infer_pool.submit(infer)
        if self.active:
            self.run.tick()
            self.progress.set(f'Episode {self.run.episode}/{self.run.settings.episodes} · {self.run.phase} · step {self.run.steps}/{self.run.settings.max_steps} · 저장 {self.run.new_transitions}개 · 목표각 {self.run.command.tolist()}°')
            if self.run.save_request is not None and self.work is None:
                summary, self.run.save_request = self.run.save_request, None
                self.work_kind = 'save'
                self.work = self.learn_pool.submit(self.learner.finish_episode, summary, self.log, self.run.settings.updates_per_transition)

    def record_detection(self, sample):
        reason = invalid_reason(sample, time.monotonic(), self.cfg)
        row = dict(episode=self.run.episode, phase=self.run.phase, frame_id=sample.frame_id, seq=sample.seq,
                   receive_unix_ns=sample.receive_unix_ns, receive_s=sample.received, processed_s=sample.processed,
                   width=sample.width, height=sample.height, simulated=sample.simulated, status=reason or 'detected',
                   count=sample.count, inference_ms=sample.inference_ms)
        # Preserve detected raw coordinates even for a stale result; status controls use.
        if sample.target is not None:
            row.update(confidence=sample.target['confidence'])
            row.update(zip(('u', 'v'), sample.target['center']))
            row.update(zip(('x1', 'y1', 'x2', 'y2'), sample.target['box']))
        self.log.row('detections', row)

    def render(self):
        if self.display is None:
            return
        image, sample = self.display
        stale = time.monotonic()-sample.received > self.cfg.dt
        key = (sample.frame_id, stale)
        if self.rendered == key:
            return
        row = {'status': 'missing'}
        if sample.target is not None and not stale:
            row.update(status='detected', **dict(zip(('u', 'v'), sample.target['center'])),
                       **dict(zip(('x1', 'y1', 'x2', 'y2'), sample.target['box'])))
        image = DetectionPanel.annotated(image, row)
        draw = ImageDraw.Draw(image)
        u, v = self.cfg.laser_u, self.cfg.laser_v
        draw.line((u-12, v, u+12, v), fill='red', width=2)
        draw.line((u, v-12, u, v+12), fill='red', width=2)
        draw.ellipse((u-23, v-23, u+23, v+23), outline='yellow', width=2)
        draw.text((8, 8), 'STALE' if stale else ('PV detected' if sample.target else 'PV missing'), fill='yellow')
        image.thumbnail((1000, 340))
        photo = ImageTk.PhotoImage(image)
        self.canvas.configure(image=photo); self.canvas.image = photo
        self.rendered = key

    def close(self):
        self.stop()
        self.closed = True
        if self.run is not None and self.run.save_request is not None and self.work is None:
            summary, self.run.save_request = self.run.save_request, None
            self.work = self.learn_pool.submit(self.learner.finish_episode, summary, self.log, self.run.settings.updates_per_transition)
        if self.log is not None:
            self.learn_pool.submit(self.log.close)
        self.infer_pool.shutdown(wait=False, cancel_futures=True)
        self.learn_pool.shutdown(wait=False, cancel_futures=False)
