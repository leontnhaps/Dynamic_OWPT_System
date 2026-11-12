"""M5 manual episodes: fixed-KF moving SAC, paired saves and explicit restart."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import time
import uuid
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from common.model_paths import DEFAULT_YOLO_PATH, resolve_yolo_path
from common.moving_paths import (DEFAULT_KALMAN_PATH, DEFAULT_MOVING_CHECKPOINT_PATH,
    resolve_kalman_path, resolve_moving_checkpoint_path, load_moving_inputs)
from common.tx_tracking import Sample, invalid_reason
from common.tx_bearing import BEARING_MODEL
from common.tx_setup import TX_CAMERA_SETTINGS
from common.pv_detection import TARGET_SELECTION
from Tx.Controller.detection_panel import PVDetector
from Tx.Controller.stationary_panel import StationaryPanel
from Tx.Controller.stationary_learning import SACLearner, sac_settings, digest
from Tx.Controller.moving_core import MovingRun, MovingRunSettings
from Tx.Controller.moving_logging import MovingRunLog


MODES = {'이어서 학습': 'resume', '고정 모델 평가': 'evaluate'}


class MovingPanel(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, padding=6)
        self.app = app
        self.infer_pool = ThreadPoolExecutor(max_workers=1)
        self.learn_pool = ThreadPoolExecutor(max_workers=1)
        self.inference = self.work = None
        self.detector = self.learner = self.run = self.log = None
        self.preparing = self.setup_pending = None
        self.previewing = self.hardware_ready = self.closed = False
        self.servo_available = None
        self.latest = self.display = self.last_frame = self.rendered = None
        self._generation = 0
        self._start_requested = False
        self._save_submit_failed = False
        self.widgets = []
        self.path = tk.StringVar(value=str(DEFAULT_YOLO_PATH))
        self.kalman_path = tk.StringVar(value=str(DEFAULT_KALMAN_PATH))
        self.checkpoint = tk.StringVar(value=str(DEFAULT_MOVING_CHECKPOINT_PATH))
        self.mode = tk.StringVar(value='이어서 학습')
        self.device = tk.StringVar(value='cuda')
        self.sac_device = tk.StringVar(value='cpu')
        self.period_text = tk.StringVar(value='모델에 저장된 주기 사용')
        self.start_pan = tk.StringVar(value='0')
        self.start_tilt = tk.StringVar(value='0')
        self.start_note = tk.StringVar(value='시작점: Pan 0°, Tilt 0° · 변경 후 적용을 누르세요.')
        self.note = tk.StringVar()
        for label, var, choose in [('YOLO 모델', self.path, self.browse_yolo),
                                  ('고정 KF JSON', self.kalman_path, self.browse_kalman),
                                  ('SAC checkpoint 폴더', self.checkpoint, self.browse_checkpoint)]:
            row = ttk.Frame(self); row.pack(fill='x', pady=2)
            ttk.Label(row, text=label, width=22).pack(side='left')
            entry = ttk.Entry(row, textvariable=var, width=75)
            entry.pack(side='left', fill='x', expand=True)
            button = ttk.Button(row, text='선택', command=choose); button.pack(side='left')
            self.widgets.extend([entry, button])
        row = ttk.Frame(self); row.pack(fill='x', pady=4)
        for label, var, values in [('모드', self.mode, tuple(MODES)),
                                  ('YOLO 장치', self.device, ('cuda', 'cpu')),
                                  ('SAC 장치', self.sac_device, ('cpu', 'cuda'))]:
            ttk.Label(row, text=label).pack(side='left')
            combo = ttk.Combobox(row, textvariable=var, values=values, state='readonly', width=14)
            combo.pack(side='left', padx=4); self.widgets.append(combo)
        ttk.Label(row, textvariable=self.period_text).pack(side='left', padx=10)
        box = ttk.LabelFrame(self, text='시작 / 복귀 지점 (°)', padding=5); box.pack(fill='x')
        for label, var in [('Pan', self.start_pan), ('Tilt', self.start_tilt)]:
            ttk.Label(box, text=label).pack(side='left')
            ttk.Entry(box, textvariable=var, width=8).pack(side='left', padx=5)
        ttk.Button(box, text='시작점 적용', command=lambda: self.guard(self.apply_start)).pack(side='left', padx=5)
        ttk.Label(box, textvariable=self.start_note).pack(side='left', padx=8)
        ttk.Label(self, text='추적 중 시작점을 바꾸면 다음 복귀 / 시작에 적용됩니다. 현재 추적 위치는 유지합니다.').pack(anchor='w')
        row = ttk.Frame(self); row.pack(fill='x', pady=3)
        ttk.Label(row, text='에피소드 메모 (속도 · 방향 · 환경)').pack(side='left')
        self.note_entry = ttk.Entry(row, textvariable=self.note)
        self.note_entry.pack(side='left', fill='x', expand=True, padx=5)
        row = ttk.Frame(self); row.pack(fill='x', pady=5)
        self.prepare_button = ttk.Button(row, text='자동 준비 (영상 · 서보 · 모델)', command=lambda: self.guard(self.prepare))
        self.prepare_button.pack(side='left', padx=3)
        self.start_button = ttk.Button(row, text='에피소드 시작', command=lambda: self.guard(self.start))
        self.start_button.pack(side='left', padx=3)
        self.end_button = ttk.Button(row, text='에피소드 종료 · 학습 · 복귀', command=lambda: self.guard(self.end_episode))
        self.end_button.pack(side='left', padx=3)
        ttk.Button(row, text='전체 중지 · 저장 (복귀 없음)', command=lambda: self.guard(self.stop)).pack(side='left', padx=3)
        ttk.Button(row, text='결과 폴더 확인', command=self.show_folder).pack(side='left', padx=3)
        ttk.Label(self, text='사용자 종료 또는 연속 미검출 3초 → 학습·저장 → 시작점 복귀 → 다음 시작 대기').pack(anchor='w')
        ttk.Label(self, text='UAV는 QGC에서 직접 조종합니다. 다음 에피소드는 시작 버튼을 눌러야 실행됩니다.').pack(anchor='w')
        self.status = tk.StringVar(value='기존 YOLO · KF · 초기 모델 경로를 확인하고 자동 준비를 누르세요.')
        self.progress = tk.StringVar(value='captures/M5-3/moving_<날짜_시각>/ · 원본 영상 · 에피소드 기록 · 학습 그래프')
        self.result_text = tk.StringVar()
        for var in (self.status, self.progress, self.result_text):
            ttk.Label(self, textvariable=var, wraplength=1120).pack(anchor='w')
        self.canvas = ttk.Label(self, anchor='center'); self.canvas.pack(fill='both', expand=True)
        self.lock()

    @property
    def active(self):
        return self.run is not None and self.run.active

    @property
    def busy(self):
        return bool(self.active or self.work is not None or self.inference is not None
                    or self.preparing is not None or self._start_requested
                    or self.run is not None and self.run.pending is not None)

    def lock(self):
        locked = bool(self.active or self.work is not None or self.preparing is not None or self._start_requested)
        if self.__dict__.get('_locked') != locked:
            self._locked = locked
            for widget in self.widgets:
                widget.configure(state='disabled' if locked else ('readonly' if isinstance(widget, ttk.Combobox) else 'normal'))
            self.note_entry.configure(state='disabled' if locked else 'normal')
        self.prepare_button.configure(state='disabled' if locked else 'normal')
        ready_to_start = self.run is None or self.run.phase == 'idle'
        self.start_button.configure(state='normal' if self.hardware_ready and not locked and ready_to_start else 'disabled')
        can_end = self.active and self.run.phase not in ('saving', 'return_reply', 'return_settle', 'draining')
        self.end_button.configure(state='normal' if can_end else 'disabled')

    def guard(self, fn):
        try:
            return fn()
        except (ValueError, TypeError) as exc:
            # Bad input or a rejected extra click must not abort valid tracking.
            messagebox.showerror('M5-3 이동 학습', str(exc))
        except Exception as exc:
            self.fail(exc)

    def fail(self, exc):
        try:
            self.invalidate('실행 오류: '+str(exc))
        except Exception:
            # A failing/full log must never keep hardware control enabled.
            self.cancel_preparation()
            if self.run is not None:
                self.run.stop_requested = True
            if self.learner is not None:
                self.learner.cancel.set()
            self.status.set('실행 오류: '+str(exc)+' · 추가 이동 / 자동 복귀 중지')
        try:
            self.app.record(dict(event='m5_moving_error', message=str(exc)))
        except Exception:
            pass
        messagebox.showerror('M5-3 이동 학습', str(exc))

    def browse_yolo(self):
        path = filedialog.askopenfilename(filetypes=[('YOLO', '*.pt')])
        if path: self.path.set(path)

    def browse_kalman(self):
        path = filedialog.askopenfilename(filetypes=[('KF', '*.json')])
        if path: self.kalman_path.set(path)

    def browse_checkpoint(self):
        path = filedialog.askdirectory(title='manifest.json이 있는 이동 모델 episode 폴더 선택')
        if path: self.checkpoint.set(path)

    def start_values(self):
        try:
            pan, tilt = float(self.start_pan.get()), float(self.start_tilt.get())
            if not pan.is_integer() or not tilt.is_integer():
                raise ValueError()
            return int(pan), int(tilt)
        except (ValueError, OverflowError):
            raise ValueError('시작 Pan/Tilt는 유한한 1° 단위 정수로 입력하세요.') from None

    def settings(self):
        pan, tilt = self.start_values()
        return MovingRunSettings(start_pan=pan, start_tilt=tilt)

    def apply_start(self):
        settings = self.settings()
        if self.learner is not None:
            settings.validate(self.cfg)
        elif not -90 <= settings.start_pan <= 90 or not -15 <= settings.start_tilt <= 40:
            raise ValueError('시작점은 Pan -90~90°, Tilt -15~40° 안에 입력하세요.')
        if self.run is not None:
            self.run.set_start(settings.start_pan, settings.start_tilt)
        self.start_note.set(f'적용된 시작점: Pan {settings.start_pan}°, Tilt {settings.start_tilt}° · 다음 복귀 / 시작에 사용')

    def stop_other_panels(self):
        a = self.app
        if getattr(getattr(a, 'ook', None), 'busy', False):
            raise ValueError('OOK 시험 중지 및 레이저 OFF 응답을 먼저 확인하세요.')
        if getattr(getattr(a, 'sweep', None), 'busy', False):
            raise ValueError('M5 데이터 수집·초기학습 작업을 먼저 종료하세요.')
        if a.stationary.busy or (a.stationary.run is not None and a.stationary.run.pending):
            raise ValueError('M3 정지 학습·저장·응답 대기를 먼저 종료하세요.')
        if a.timing.active or a.timing.preparing or a.timing.profile.running:
            raise ValueError('M1-5 측정을 먼저 종료하세요.')
        if a.servo.pending:
            raise ValueError('기존 서보 명령 응답을 기다리세요.')
        a.stationary.stop(); a.detection.stop(); a.timing.stop()
        if a.stationary.inference is not None or a.detection.future is not None or a.timing.future is not None:
            raise ValueError('기존 검출 작업이 끝난 뒤 다시 누르세요.')

    def prepare(self):
        if (self.active or self.work is not None or self.preparing is not None or self._start_requested
                or self.run is not None and self.run.pending is not None):
            raise ValueError('현재 에피소드·저장·준비 작업을 먼저 종료하세요.')
        if self.inference is not None:
            self.previewing = False
            raise ValueError('미리보기를 정지했습니다. 검출 작업 완료 후 자동 준비를 다시 누르세요.')
        self.stop_other_panels()
        if not self.app.servo.online:
            raise ValueError('서버와 Raspberry Pi의 제어 연결을 확인하세요.')
        settings = self.settings()
        selection = self.selection()
        if selection['mode'] not in ('resume', 'evaluate'):
            raise ValueError('실행 모드를 확인하세요.')
        if selection['yolo_device'] not in ('cpu', 'cuda') or selection['sac_device'] not in ('cpu', 'cuda'):
            raise ValueError('모델 실행 장치를 확인하세요.')
        if self.log is not None:
            self.log.close(); self.log = None
        self._generation += 1
        self._save_submit_failed = False
        self.detector = self.learner = self.run = None
        self.latest = self.display = self.rendered = None
        self.hardware_ready = self.previewing = False
        self.preparing = 'model'
        self.setup_deadline = time.monotonic()+120
        self.work_kind = 'prepare'
        self.status.set('기존 모델 · KF 확인 및 로딩 중 · 이동 명령 없음')

        def load():
            inputs = load_moving_inputs(selection['checkpoint'], selection['kalman_path'], selection['mode'])
            settings.validate(inputs.cfg)
            detector = PVDetector(selection['yolo_path'], selection['yolo_device'])
            if inputs.cfg.class_id not in detector.names:
                raise ValueError('YOLO PV class ID 0을 확인하세요.')
            learner = SACLearner(inputs.cfg, selection['mode'], inputs.checkpoint, selection['sac_device'])
            identity = dict(selection, yolo_sha256=digest(selection['yolo_path']),
                            source_checkpoint=str(inputs.checkpoint), kalman_file=inputs.kalman_identity,
                            source_episode_count=learner.episodes)
            return detector, learner, inputs, identity

        try:
            self.work = self.learn_pool.submit(load)
        except Exception:
            self.cancel_preparation()
            raise
        self.lock()

    def selection(self):
        return dict(yolo_path=resolve_yolo_path(self.path.get()),
                    checkpoint=str(resolve_moving_checkpoint_path(self.checkpoint.get())),
                    kalman_path=str(resolve_kalman_path(self.kalman_path.get())),
                    mode=MODES.get(self.mode.get()), yolo_device=self.device.get(), sac_device=self.sac_device.get())

    def cancel_preparation(self):
        self.preparing = self.setup_pending = None
        self.hardware_ready = self.previewing = False
        self._start_requested = False

    def prepare_hardware(self):
        self.settings().validate(self.cfg)
        self.preparing = 'limits'
        self.setup_pending = 'm5-1-moving-setup-'+uuid.uuid4().hex
        self.setup_deadline = time.monotonic()+5
        self.camera_requested_at = time.monotonic()
        for key, value in TX_CAMERA_SETTINGS.items():
            self.app.values[key].set('' if value is None else str(value))
        if not self.send(dict(cmd='preview', enable=True, request_id='m5-1-moving-camera-'+uuid.uuid4().hex, **TX_CAMERA_SETTINGS)):
            raise ValueError('카메라 자동 시작 실패')
        if not self.send(dict(cmd='servo_config', request_id=self.setup_pending, **self.cfg.limits)):
            raise ValueError('서보 운용 범위 자동 적용 실패')
        self.status.set('자동 준비: 카메라 · 실제 서보 연결 · 저장 모델의 운용 범위 확인 중')

    camera_matches = staticmethod(StationaryPanel.camera_matches)
    check_camera = StationaryPanel.check_camera
    render = StationaryPanel.render

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
        self.hardware_ready = self.previewing = True
        self.last_frame = None
        self.status.set('자동 준비 완료 · PV 위치와 시작점을 확인한 뒤 에피소드 시작을 누르세요.')

    def check_prepared(self):
        if self.learner is None or not self.hardware_ready:
            raise ValueError('이 탭에서 자동 준비를 완료하세요.')
        if self.active or self.work is not None or self.preparing is not None:
            raise ValueError('현재 에피소드·저장·준비가 끝난 뒤 시작하세요.')
        if any(value != self.identity.get(key) for key, value in self.selection().items()):
            raise ValueError('모델·KF·장치 설정이 바뀌었습니다. 자동 준비를 다시 실행하세요.')
        if self.run is not None and self.run.pending is not None:
            raise ValueError('전송된 명령의 응답을 기다리세요.')
        if self.run is not None and self.run.phase != 'idle':
            raise ValueError('중단 또는 오류 후에는 자동 준비를 다시 실행하세요.')

    def start(self):
        self.check_prepared()
        if self._start_requested:
            raise ValueError('에피소드 시작을 준비하고 있습니다.')
        self.settings().validate(self.cfg)
        self.stop_other_panels()
        self.check_camera()
        self.previewing = False
        self._start_requested = True
        self.status.set('미리보기 작업 완료 후 지정 시작점에서 에피소드를 시작합니다.')
        if self.inference is None:
            try:
                self._start_episode()
            except Exception:
                self._start_requested = False
                raise
        self.lock()

    def _start_episode(self):
        self.check_prepared()
        self.stop_other_panels()
        frame = self.check_camera()
        servo = self.app.servo
        if (not servo.online or self.servo_available is not True or servo.simulated is not False
                or servo.limits != self.cfg.limits):
            raise ValueError('실제 Pi 연결과 저장 모델의 운용 범위를 확인하세요.')
        settings = self.settings(); settings.validate(self.cfg)
        if self.run is None:
            folder = self.app.stage_dir('M5-3')/('moving_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            config = dict(schema='tx-moving-run-v1', checkpoint_schema=self.cfg.schema,
                tracking=asdict(self.cfg), run=asdict(settings), sac=sac_settings(self.cfg),
                model=self.identity, kalman=self.kalman, normalization=self.cfg.normalization,
                bearing_model=BEARING_MODEL, target_selection=TARGET_SELECTION,
                camera=TX_CAMERA_SETTINGS, initial_camera_metadata=frame[1],
                episode_policy='manual start; manual end or continuous missing; update and return; no auto restart',
                metrics='post-action valid observations; missing data are not zero reward',
                hardware_validated=False, actual_angle_feedback=False)
            self.log = MovingRunLog(folder, config)
            self.run = MovingRun(self.cfg, settings, self.learner, self.log, self.send, time.monotonic, self.kalman)
        else:
            self.run.set_start(settings.start_pan, settings.start_tilt)
        self.learner.cancel.clear()
        self._generation += 1
        self.latest = None
        self.last_frame = None
        self._start_requested = False
        self.previewing = True
        self.log.event('operator_episode_start', note=self.note.get().strip(), start_pan=settings.start_pan,
                       start_tilt=settings.start_tilt)
        self.run.start()
        self.status.set('에피소드 진행 중 · 종료 버튼 또는 연속 미검출 3초까지 추적합니다.')

    def end_episode(self):
        if not self.active:
            raise ValueError('진행 중인 에피소드가 없습니다.')
        self.run.manual_end()
        self.status.set('에피소드 종료 요청 · 진행 중 명령·관측 완료 후 학습·저장하고 시작점으로 복귀합니다.')

    def send(self, command):
        return self.app.send(command, tracking=True)

    def stop(self):
        self._generation += 1
        self.cancel_preparation()
        if self.run is not None and (self.run.active or self.run.pending is not None):
            self.run.abort('user_stop')
        if self.learner is not None:
            self.learner.cancel.set()
        self.status.set('추가 명령 중지 · 저장 후 대기 · 자동 복귀 없음 · 다시 실행하려면 자동 준비하세요.')

    def invalidate(self, reason):
        self._generation += 1
        self.cancel_preparation()
        if self.run is not None and (self.run.active or self.run.pending is not None):
            self.run.abort('connection_or_device_error', str(reason))
        if self.learner is not None:
            self.learner.cancel.set()
        self.status.set(str(reason)+' · 추가 이동 / 자동 복귀 중지')

    def on_tab_left(self):
        if self.busy or self.hardware_ready or self.previewing:
            self.stop()

    def show_folder(self):
        messagebox.showinfo('M5-3 이동 학습 결과', str(self.log.folder) if self.log else '실행 결과가 아직 없습니다.')

    def handle_event(self, event):
        kind = event.get('event')
        disconnected = kind == 'network' and event.get('state') != 'connected'
        device_change = kind in ('hello', 'agent', 'ready')
        if kind == 'servo':
            self.servo_available = event.get('available')
        if disconnected or device_change:
            self.servo_available = None
            self.invalidate('연결 또는 Pi 상태 변경')
            return
        if self.preparing is not None:
            foreign = (kind == 'servo' and event.get('operation') in ('move', 'servo_config')
                       and event.get('request_id') != self.setup_pending)
            if foreign or kind == 'error':
                self.invalidate('자동 준비 중 장치 오류 또는 외부 명령')
                return
            if self.setup_pending and event.get('request_id') == self.setup_pending and kind == 'servo':
                if (event.get('operation') != 'servo_config' or event.get('simulated') is not False
                        or event.get('available') is not True or event.get('limits') != self.cfg.limits):
                    self.invalidate('자동 준비 실패: 실제 서보와 운용 범위 응답을 확인하세요.')
                    return
                servo = self.app.servo
                servo.limits = dict(self.cfg.limits); servo.simulated = False
                for key, value in self.cfg.limits.items(): servo.vars[key].set(str(value))
                for key in ('speed', 'acc'): servo.vars[key].set(str(getattr(self.cfg, key)))
                self.setup_pending = None
                self.preparing = 'frames'
                self.setup_deadline = time.monotonic()+15
                self.status.set('자동 준비: 설정이 적용된 새 영상 확인 중')
            return
        if self.run is not None and self.run.reply(event):
            if kind == 'servo' and event.get('commanded'):
                self.app.servo.last = event['commanded']
                for axis in ('pan', 'tilt'): self.app.servo.vars[axis].set(str(event['commanded'][axis]))
            return
        foreign = kind == 'servo' and event.get('operation') in ('move', 'servo_config')
        if self.hardware_ready and (foreign or kind == 'error'):
            self.invalidate('장치 오류 또는 다른 탭의 명령 감지')

    event = handle_event

    def _finish_and_plot(self, summary):
        report = self.learner.finish_episode(summary, self.log, self.run.settings.updates_per_transition)
        try:
            plots = self.log.generate_plots()
        except Exception as exc:
            plots = dict(files=[], errors=[str(exc)])
        return report, plots

    def select_saved_checkpoint(self, report):
        # Use this worker's confirmed save, never an arbitrary latest-folder search.
        if self.learner.mode == 'resume' and report.get('checkpoint'):
            selected = str(resolve_moving_checkpoint_path(report['checkpoint']))
            self.checkpoint.set(selected)
            self.identity['checkpoint'] = selected

    def poll(self):
        if self.closed:
            return
        try:
            self._poll()
        except Exception as exc:
            self.fail(exc)
        finally:
            self.lock()

    def _poll(self):
        if self.work is not None and self.work.done():
            future, self.work = self.work, None
            if self.work_kind == 'prepare':
                loaded = future.result()
                if self.preparing == 'model':
                    self.detector, self.learner, inputs, self.identity = loaded
                    self.cfg, self.kalman = inputs.cfg, inputs.kalman
                    self.period_text.set(f'저장 모델 주기 {self.cfg.dt:.3f} s')
                    self.prepare_hardware()
            else:
                try:
                    report, plots = future.result()
                    self.select_saved_checkpoint(report)
                    if (report['completed_updates'] != report['planned_updates']
                            and not self.learner.cancel.is_set()):
                        raise RuntimeError('업데이트 횟수가 계획과 다릅니다. 부분 checkpoint를 보존하고 자동 복귀를 중지했습니다.')
                except Exception:
                    self.run.saved(False)
                    self.hardware_ready = self.previewing = False
                    raise
                self.result_text.set(f"Episode {report['episode']} · {report['reason']} · 보상합 {report.get('reward_sum')} · "
                    f"RMS {report.get('rms_px')} px · 업데이트 {report['completed_updates']}/{report['planned_updates']} · {self.log.folder}")
                if plots.get('errors'):
                    self.result_text.set(self.result_text.get()+' · 그래프 일부 실패 (원본 로그 유지)')
                if report.get('checkpoint'):
                    self.result_text.set(self.result_text.get()+' · 다음 재개 모델 경로 갱신됨')
                self.run.saved()
                if self.run.stop_requested or self.run.phase == 'error':
                    self.hardware_ready = self.previewing = False
                    self.status.set('중단 결과 저장 완료 · 자동 복귀 없음 · 다시 실행하려면 자동 준비하세요.')
                else:
                    self.status.set('저장 완료 · 시작점 복귀 중')
        self.prepare_poll()
        if self.inference is not None and self.inference.done():
            future, self.inference = self.inference, None
            result, sample, frame, generation = future.result()
            if generation == self._generation:
                self.latest = sample
                self.display = (result['image'], sample)
                if self.run is not None and self.run.active and self.run.inference_allowed:
                    if not self.camera_matches(frame):
                        self.invalidate('실행 중 카메라 설정 변경')
                    else:
                        self.run.supply(sample, frame[1])
        if self._start_requested and self.inference is None:
            self._start_episode()
        if self.active:
            old_phase = self.run.phase
            self.run.tick()
            self.progress.set(f'Episode {self.run.episode} · {self.run.phase} · step {self.run.steps} · '
                              f'저장 {self.run.new_transitions}개 · 명령각 {self.run.command.tolist()}°')
            if old_phase != 'idle' and self.run.phase == 'idle':
                self.status.set('시작점 복귀 대기 완료 · 다음 에피소드는 시작 버튼을 누르세요. 실제 도달 여부는 미측정입니다.')
                self.previewing = self.hardware_ready
        if (self.run is not None and self.run.save_request is not None
                and self.work is None and self.inference is None and not self._save_submit_failed):
            summary, self.run.save_request = self.run.save_request, None
            self.work_kind = 'save'
            try:
                self.work = self.learn_pool.submit(self._finish_and_plot, summary)
            except Exception:
                self.run.save_request = summary
                self._save_submit_failed = True
                raise
            self.status.set('에피소드 학습 · checkpoint · 그래프 저장 중')
        can_infer = self.previewing and self.detector is not None and self.work is None and not self._start_requested
        if self.run is not None and self.run.active:
            can_infer = can_infer and self.run.inference_allowed
        if can_infer and self.inference is None:
            frame = self.app.current
            if frame and frame[2] != self.last_frame and 0 <= time.monotonic()-frame[2] <= self.cfg.dt:
                self.last_frame = frame[2]
                frame_id, generation = self.app.received_count, self._generation
                context = (self.run.episode, self.run.phase, self.run.steps) if self.active else None
                detector, cfg, log = self.detector, self.cfg, self.log

                def infer():
                    result = detector.infer(frame[0], cfg.confidence, cfg.class_id)
                    sample = Sample(frame_id, frame[2], time.monotonic(), result['image'].width, result['image'].height,
                        result['target'], result['count'], frame[1].get('seq'), frame[3],
                        result['inference_ms'], frame[1].get('simulated', True))
                    if context is not None:
                        log.record_frame(context[0], context[1], sample, frame[1], frame[0], step=context[2],
                                         detection={k:v for k,v in result.items() if k != 'image'},
                                         status=invalid_reason(sample, sample.processed, cfg) or 'detected')
                    return result, sample, frame, generation

                self.inference = self.infer_pool.submit(infer)

    def close(self):
        if self.closed:
            return
        self.stop()
        self.closed = True
        pending_inference = self.inference
        summary = self.run.save_request if self.run is not None else None
        if self.run is not None: self.run.save_request = None
        log, learner, run = self.log, self.learner, self.run

        def finish_close():
            try:
                if pending_inference is not None:
                    try: pending_inference.result()
                    except Exception: pass
                if summary is not None:
                    learner.finish_episode(summary, log, run.settings.updates_per_transition)
                    log.generate_plots()
            finally:
                if log is not None: log.close()

        self.learn_pool.submit(finish_close)
        self.infer_pool.shutdown(wait=False, cancel_futures=False)
        self.learn_pool.shutdown(wait=False, cancel_futures=False)
