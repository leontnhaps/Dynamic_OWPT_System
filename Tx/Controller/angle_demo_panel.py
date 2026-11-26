"""Single optical pose command from the PC to the stationary Rx demonstration rig."""
import json
import math
import queue
import time
import tkinter as tk
import uuid
from datetime import datetime
from tkinter import ttk

from common.angle_demo import BIT_MS, HOME_PAN, HOME_TILT, MAX_PAN, MAX_TILT, RX_VIEWPOINT, encode_angles
from Tx.Controller.demo_ssh import DemoSsh
from Tx.Controller.pi_ook_rx import DEFAULT_RX_TARGET


# Each step is relative to the last acknowledged pose. None requests the calibrated home.
SERVO_CHECK_STEPS = (None, (5, 0), None, (0, 5), None)
CHECK_DWELL_S = 1.0


class AngleDemoPanel(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, padding=8)
        self.app = app
        self.rx = self.token = self.phase = self.logpath = self.off_token = None
        self.selected = self.prepared = False
        self.target = tk.StringVar(value=DEFAULT_RX_TARGET)
        self.password = tk.StringVar()
        self.pan = tk.StringVar(value='0')
        self.tilt = tk.StringVar(value='0')
        self.rx_status = tk.StringVar(value='Rx 미연결')
        self.off_status = tk.StringVar()
        self.status = tk.StringVar(value='원위치 기준으로 각도를 입력하세요.')
        self.applied = tk.StringVar(self)
        self.clear_servo_pose('Rx 미연결')
        row = ttk.Frame(self)
        row.pack(anchor='w', pady=4)
        for label, variable, width, mask in [('Rx SSH', self.target, 24, ''),
                                             ('비밀번호 (키 인증은 빈칸)', self.password, 14, '*')]:
            ttk.Label(row, text=label).pack(side='left', padx=3)
            ttk.Entry(row, textvariable=variable, width=width, show=mask).pack(side='left')
        ttk.Button(row, text='SSH 연결', command=self.connect).pack(side='left', padx=5)
        ttk.Label(row, textvariable=self.rx_status).pack(side='left', padx=4)
        check_box = ttk.LabelFrame(self, text=f'Rx 서보 점검 (SSH) · {RX_VIEWPOINT}', padding=4)
        check_box.pack(anchor='w', pady=3)
        readout = ttk.Frame(check_box)
        readout.pack(anchor='w')
        ttk.Label(readout, text='마지막 확인 명령각').pack(side='left', padx=3)
        ttk.Label(readout, textvariable=self.applied, font='TkHeadingFont').pack(side='left', padx=6)
        check = ttk.Frame(check_box)
        check.pack(anchor='w')
        ttk.Button(check, text='두 축 한 번에 확인', command=self.check_both).pack(side='left', padx=3)
        ttk.Button(check, text='점검 중지', command=self.stop_servo_check).pack(side='left', padx=3)
        self.jog_size = tk.IntVar(value=1)
        for step in (1, 5):
            ttk.Radiobutton(check, text=f'{step}°', variable=self.jog_size, value=step).pack(side='left')
        for axis, dp, dt in (('오른쪽 (Pan −)', -1, 0), ('왼쪽 (Pan +)', 1, 0),
                             ('위 (Tilt −)', 0, -1), ('아래 (Tilt +)', 0, 1)):
            ttk.Button(check, text=axis,
                       command=lambda p=dp, t=dt: self.jog(p, t)).pack(side='left', padx=2)
        row = ttk.Frame(self)
        row.pack(anchor='w', pady=4)
        for label, variable, low, high in [('Pan', self.pan, -HOME_PAN, MAX_PAN-HOME_PAN),
                                           ('Tilt', self.tilt, -HOME_TILT, MAX_TILT-HOME_TILT)]:
            ttk.Label(row, text=f'{label} ({low}° ~ +{high}°)').pack(side='left', padx=4)
            ttk.Spinbox(row, from_=low, to=high, increment=1, width=6,
                        textvariable=variable).pack(side='left')
        ttk.Button(row, text='전송 (OOK)', command=self.transmit).pack(side='left', padx=6)
        ttk.Button(row, text='원위치 (SSH)', command=self.home).pack(side='left', padx=3)
        ttk.Button(row, text='송신 중지', command=lambda: self.stop('사용자 중지')).pack(side='left', padx=3)
        ttk.Button(row, text='Tx 카메라 다시 준비', command=self.retry_preview).pack(side='left', padx=3)
        ttk.Label(self, text=f'입력 (0°, 0°) = 발판 중앙 · 셀 수직 '
                             f'(서보 Pan {HOME_PAN}°, Tilt {HOME_TILT}°)').pack(anchor='w')
        ttk.Label(self, textvariable=self.status).pack(anchor='w')
        ttk.Label(self, textvariable=self.off_status).pack(anchor='w')

    @property
    def optical(self):
        return self.phase in ('off', 'arming', 'optical')

    def record(self, event):
        try:
            if self.logpath is None:
                stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                self.logpath = self.app.stage_dir('M6-4') / f'angle_demo_{stamp}.jsonl'
            with self.logpath.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(event, pc_unix_ns=time.time_ns()), ensure_ascii=False) + '\n')
        except OSError as exc:
            self.rx_status.set(f'시연 기록 저장 오류 · {exc}')

    def connect(self):
        if self.rx is not None:
            self.status.set('Rx SSH 연결 상태를 확인하세요.')
            return
        old = self.app.ook
        if old.busy:
            self.status.set('진행 중인 OOK 시험을 먼저 종료하세요.')
            return
        try:
            if old.arduino is not None:
                old.disconnect_rx()  # The final tab takes ownership of the selected Rx.
            self.rx = DemoSsh(self.target.get(), self.password.get())
            self.password.set('')
            self.rx_status.set('Rx SSH 연결 중…')
        except (ValueError, OSError) as exc:
            self.rx_status.set(str(exc))

    def retry_preview(self):
        if not self.optical:
            self.prepared = False
            self.status.set('Tx 연결 및 카메라 준비 중…')

    def ready_rx(self):
        if self.rx is None or not self.rx.connected:
            raise ValueError('먼저 Rx SSH를 연결하세요.')

    def begin(self, phase):
        self.token = 'm6-demo-' + uuid.uuid4().hex
        self.phase = phase
        self.deadline = time.monotonic() + 5

    def check_both(self):
        self.start_servo_check(SERVO_CHECK_STEPS)

    def jog(self, pan, tilt):
        step = self.jog_size.get()
        self.start_servo_check(((pan * step, tilt * step),))

    def start_servo_check(self, steps):
        if self.phase or self.off_token:
            self.status.set('현재 동작과 OFF 확인이 끝난 뒤 서보를 점검하세요.')
            return
        try:
            self.ready_rx()
            if steps[0] is not None and self.last_servo_pose is None:
                raise ValueError('먼저 원위치 (SSH) 또는 두 축 한 번에 확인을 누르세요.')
            self.check_steps = list(steps)
            self.next_check_step()
        except (ValueError, OSError) as exc:
            self.stop(str(exc))

    def next_check_step(self):
        delta = self.check_steps.pop(0)
        if delta is None:
            self.check_goal = (HOME_PAN, HOME_TILT)
            self.begin('check_home')
            self.rx.send('home', self.token)
        else:
            pan, tilt = self.last_servo_pose
            self.check_goal = (pan + delta[0], tilt + delta[1])
            if not (0 <= self.check_goal[0] <= MAX_PAN and 0 <= self.check_goal[1] <= MAX_TILT):
                raise ValueError(f'점검 범위는 Pan 0~{MAX_PAN}°, Tilt 0~{MAX_TILT}°입니다.')
            self.begin('check_jog')
            self.rx.jog(self.token, *delta)
        self.record(dict(event='servo_check_requested', token=self.token, pan=self.check_goal[0],
                         tilt=self.check_goal[1], rx_viewpoint=RX_VIEWPOINT))
        self.status.set(f'SSH 점검 · Pan {self.check_goal[0]}°, Tilt {self.check_goal[1]}° 응답 대기')

    def stop_servo_check(self):
        if self.phase in ('check_home', 'check_jog', 'check_pause'):
            self.stop('서보 점검 중지 · 다음 동작을 취소했습니다.')

    def transmit(self):
        if self.phase:
            return
        try:
            self.ready_rx()
            if self.off_token:
                raise ValueError('이전 송신 OFF 미확인 · 송신 중지를 눌러 OFF를 다시 확인하세요.')
            try:
                pan, tilt = HOME_PAN + int(self.pan.get()), HOME_TILT + int(self.tilt.get())
            except ValueError as exc:
                raise ValueError('Pan과 Tilt는 1° 단위 정수로 입력하세요.') from exc
            if not (0 <= pan <= MAX_PAN and 0 <= tilt <= MAX_TILT):
                raise ValueError('입력칸 옆에 표시된 Pan·Tilt 범위 안에서 선택하세요.')
            self.bits = encode_angles(pan, tilt)
            self.app.ook.check_ready()
            hw = self.app.ook.hardware
            if hw.get('simulated') is not False or 'laser_pattern' not in hw.get('capabilities', []):
                raise ValueError('광신호 전송에는 실제 Tx 연결이 필요합니다.')
            self.goal = (pan, tilt)
            self.tx_done = self.rx_done = False
            self.off_status.set('')
            self.begin('off')
            self.record(dict(event='requested', token=self.token, pan=pan, tilt=tilt, bit_ms=BIT_MS))
            self.status.set('Tx OFF 확인 중…')
            if not self.app.send(dict(cmd='laser', value=0, request_id=self.token)):
                self.stop('Tx OFF 요청 실패')
        except (ValueError, OSError) as exc:
            self.status.set(str(exc))

    def home(self):
        if self.phase == 'home':
            return
        try:
            self.ready_rx()
            if self.optical:
                self.stop('원위치 요청으로 광송신 중지')
            self.begin('home')
            self.rx.send('home', self.token)
            self.record(dict(event='home_requested', token=self.token))
            self.status.set('SSH로 Rx 원위치 명령 확인 중…')
        except (ValueError, OSError) as exc:
            self.stop(str(exc))

    def stop(self, reason):
        optical, old_token = self.optical, self.token
        if self.phase in ('check_home', 'check_jog', 'home', 'bench_home', 'optical'):
            self.clear_servo_pose('마지막 명령 미확인 · 원위치 (SSH)로 다시 확인하세요.')
        self.phase = self.token = None  # Late ACKs must not dispatch a frame.
        if optical:
            if self.rx is not None and self.rx.connected:
                try:
                    self.rx.send('cancel', old_token)
                except OSError:
                    pass
            self.off_token = old_token + '-stop'
        if self.off_token:
            self.off_status.set('Tx OFF 미확인 · 송신 중지 버튼으로 다시 요청할 수 있습니다.')
            self.app.send(dict(cmd='laser', value=0, request_id=self.off_token))
        if old_token:
            self.record(dict(event='stopped', token=old_token, reason=reason))
        self.status.set(reason)

    def applied_pose(self, pan, tilt):
        self.last_servo_pose = (pan, tilt)
        self.applied.set(f'Pan {pan}°    Tilt {tilt}°     (원위치 대비 Pan {pan-HOME_PAN:+d}°, '
                         f'Tilt {tilt-HOME_TILT:+d}°)')

    def clear_servo_pose(self, reason):
        self.last_servo_pose = None
        self.applied.set(f'Pan —    Tilt —     ({reason})')

    def on_rx(self, event):
        kind = event['kind']
        self.record(event)
        if kind == 'ready':
            self.rx_status.set('Rx SSH 연결됨')
            self.clear_servo_pose('원위치 또는 두 축 확인을 눌러 시작하세요.')
            return
        if kind == 'error' and not event.get('token'):
            self.stop('Rx 오류 · ' + event.get('message', ''))
            return
        if not self.token or event.get('token') != self.token:
            return
        if kind in ('error', 'timeout', 'cancelled'):
            self.stop(event.get('message') or '유효한 광명령 수신을 확인하지 못했습니다.')
        elif ((self.phase == 'check_home' and kind == 'home') or
              (self.phase == 'check_jog' and kind == 'jogged')):
            if (event.get('pan'), event.get('tilt')) != self.check_goal:
                self.stop('점검 요청과 Rx가 응답한 서보 각도가 다릅니다.')
                return
            self.applied_pose(*self.check_goal)
            self.token = None
            if self.check_steps:
                self.phase = 'check_pause'
                self.check_due = time.monotonic() + CHECK_DWELL_S
            else:
                self.phase = None
                self.status.set('SSH 점검 완료 · Pan과 Tilt의 서보 명령을 확인했습니다.')
        elif kind == 'home' and self.phase == 'home':
            if (event.get('pan'), event.get('tilt')) != (HOME_PAN, HOME_TILT):
                self.stop('Rx 원위치 응답 각도 불일치')
                return
            self.applied_pose(HOME_PAN, HOME_TILT)
            self.pan.set('0')
            self.tilt.set('0')
            self.phase = self.token = None
            self.status.set('원위치 · Rx 서보 명령 적용 확인 (SSH)')
        elif kind == 'armed' and self.phase == 'arming':
            self.app.ook.check_ready()
            self.phase = 'optical'
            self.deadline = time.monotonic() + 9
            self.status.set('OOK 전송 중 · Rx 광복호 대기')
            if not self.app.send(dict(cmd='laser_pattern', request_id=self.token,
                                      bits=self.bits, bit_ms=BIT_MS, repeats=1)):
                self.stop('광프레임 송신 요청 실패')
        elif kind == 'applied' and self.phase == 'optical':
            pan, tilt = event['pan'], event['tilt']
            self.applied_pose(pan, tilt)
            if (pan, tilt) != self.goal:
                self.stop('Rx가 적용한 각도와 요청한 각도가 다릅니다.')
            else:
                self.rx_done = True
                self.finish_if_confirmed()

    def finish_if_confirmed(self):
        if self.tx_done and self.rx_done:
            self.phase = self.token = None
            self.status.set('OOK 수신 성공 · Rx 광복호 및 서보 명령 적용 확인')
        elif self.tx_done:
            self.status.set('Tx 프레임 송신 완료 · Rx 광복호 응답 대기')

    def event(self, event):
        if self.off_token and event.get('event') == 'ready':
            self.app.send(dict(cmd='laser', value=0, request_id=self.off_token))
        if self.off_token and event.get('request_id') == self.off_token:
            self.record(event)
            if event.get('event') == 'laser' and event.get('laser_gpio_level') == 0:
                self.off_status.set('Tx OFF 확인')
                self.off_token = None
            return
        if not self.optical:
            return
        kind = event.get('event')
        if (kind in ('outputs_off', 'hello', 'ready', 'agent') or
                (kind == 'network' and event.get('port') == 7600 and event.get('state') != 'connected')):
            self.stop('Tx 연결 또는 출력 상태가 바뀌어 광송신을 중지했습니다.')
            return
        if event.get('request_id') != self.token:
            if kind in ('laser', 'laser_pattern'):
                # A delayed confirmation of our own previous stop is already OFF.
                if not (kind == 'laser' and event.get('laser_gpio_level') == 0 and
                        str(event.get('request_id')).endswith('-stop')):
                    self.stop('다른 Tx 출력 명령이 들어와 광송신을 중지했습니다.')
            return
        self.record(event)
        try:
            if kind == 'error':
                self.stop('Tx 오류 · ' + event.get('message', ''))
            elif kind == 'laser' and self.phase == 'off':
                if event.get('laser_gpio_level') != 0:
                    self.stop('Tx OFF 응답을 확인하지 못했습니다.')
                    return
                self.phase = 'arming'
                self.deadline = time.monotonic() + 5
                self.rx.send('arm', self.token)
                self.status.set('Rx 광수신 준비 중…')
            elif kind == 'laser_pattern' and self.phase == 'optical' and event.get('state') != 'started':
                off = event.get('off_call_done_ms')
                if (event.get('state') != 'completed' or event.get('simulated') is not False
                        or type(off) not in (int, float) or not math.isfinite(off) or off < 0
                        or event.get('settings') != dict(bits=self.bits, bit_ms=BIT_MS, repeats=1)):
                    self.stop('Tx 송신 완료 또는 종료 OFF를 확인하지 못했습니다.')
                    return
                self.tx_done = True
                self.finish_if_confirmed()
        except (ValueError, OSError) as exc:
            self.stop(str(exc))

    def tick(self):
        if not self.selected:
            return
        ook = self.app.ook
        if not ook.online:
            self.prepared = False
        elif (not self.prepared and not ook.busy and not ook.pending and not self.phase and not self.off_token
              and not self.app.stationary.active and not self.app.stationary.preparing
              and not self.app.timing.active and not self.app.timing.preparing and not self.app.timing.profile.running):
            self.prepared = True
            ook.prepare()
        if self.rx:
            try:
                while True:
                    self.on_rx(self.rx.events.get_nowait())
            except queue.Empty:
                pass
            except (ValueError, OSError) as exc:
                self.stop(str(exc))
            if self.rx.terminal_error:
                detail = self.rx.terminal_error
                self.stop('Rx SSH 연결 종료')
                self.rx_status.set(detail)
                self.rx.close()
                self.rx = None
                self.clear_servo_pose('SSH 연결 종료')
        if self.phase == 'check_pause' and time.monotonic() >= self.check_due:
            try:
                self.next_check_step()
            except (ValueError, OSError) as exc:
                self.stop(str(exc))
        elif self.phase and self.phase != 'check_pause' and time.monotonic() > self.deadline:
            self.stop('응답 시간 초과 · Rx 서보 적용 상태를 확인하세요.')
        elif self.optical and (not self.app.current or time.monotonic()-self.app.current[2] > 2):
            self.stop('Tx 카메라 영상 수신 중단')

    def leave(self):
        self.selected = self.prepared = False
        if self.phase:
            self.stop('탭을 닫아 시연 중지')
        if self.rx:
            self.rx.close()
            self.rx = None
            self.clear_servo_pose('Rx 미연결')
            self.rx_status.set('Rx 미연결')
