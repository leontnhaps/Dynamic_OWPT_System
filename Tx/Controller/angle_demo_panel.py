"""Single optical pose command from the PC to the stationary Rx demonstration rig."""
import json
import math
import queue
import time
import tkinter as tk
import uuid
from datetime import datetime
from tkinter import ttk

from common.angle_demo import (ANGLE_BIT_PERIODS, BIT_MS, DECODE_SAMPLE_MS, HOME_PAN, HOME_TILT,
                               MAX_PAN, MAX_TILT, RX_VIEWPOINT, encode_angles, payload_from_frame)
from Rx.ina219_ook_stream import SAMPLE_CONFIG
from Tx.Controller.angle_waveform import AngleWaveform, WaveformReview
from Tx.Controller.demo_ssh import DemoSsh, optical_servo_writes
from Tx.Controller.pi_ook_rx import DEFAULT_RX_TARGET


# Each step is relative to the last acknowledged pose. None requests the calibrated home.
SERVO_CHECK_STEPS = (None, (5, 0), None, (0, 5), None)
CHECK_DWELL_S = 1.0
RATE_LABELS = {ms: f'{1000/ms:.2f}'.rstrip('0').rstrip('.') + f' bit/s  ({ms} ms/bit)'
               for ms in ANGLE_BIT_PERIODS}


class AngleDemoPanel(ttk.Frame):
    wave = wave_review = draining_rx = None

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
        self.verification = tk.StringVar(self)
        self.clear_servo_pose('Rx 미연결')
        row = ttk.Frame(self)
        row.pack(anchor='w', pady=4)
        for label, variable, width, mask in [('Rx SSH', self.target, 24, ''),
                                             ('비밀번호 (키 인증은 빈칸)', self.password, 14, '*')]:
            ttk.Label(row, text=label).pack(side='left', padx=3)
            ttk.Entry(row, textvariable=variable, width=width, show=mask).pack(side='left')
        ttk.Button(row, text='SSH 연결', command=self.connect).pack(side='left', padx=5)
        ttk.Label(row, textvariable=self.rx_status).pack(side='left', padx=4)
        ttk.Button(row, text='Rx 0°, 0° (SSH)', command=self.home).pack(side='left', padx=4)
        self.diagnostic_toggle = ttk.Button(row, text='서보·OOK 점검 열기', command=self.toggle_diagnostics)
        self.diagnostic_toggle.pack(side='left', padx=4)
        readout = ttk.Frame(self)
        readout.pack(anchor='w')
        ttk.Label(readout, text='Rx 마지막 확인 명령각').pack(side='left', padx=3)
        ttk.Label(readout, textvariable=self.applied, font='TkHeadingFont').pack(side='left', padx=6)
        rates = ttk.Frame(self)
        rates.pack(anchor='w', fill='x')
        ttk.Label(rates, text='각도 OOK 속도').pack(side='left', padx=3)
        self.rate_box = ttk.Combobox(rates, width=26, values=tuple(RATE_LABELS.values()), state='readonly')
        self.rate_box.set(RATE_LABELS[BIT_MS])
        self.rate_box.pack(side='left', padx=3)
        self.wave_review = WaveformReview(rates)
        self.wave_review.pack(side='left', padx=4)
        self.diagnostic_slot = ttk.Frame(self)
        self.diagnostic_slot.pack(fill='x')
        self.diagnostics = ttk.Frame(self.diagnostic_slot)
        check_box = ttk.LabelFrame(self.diagnostics, text=f'Rx 서보 점검 (SSH) · {RX_VIEWPOINT}', padding=4)
        check_box.pack(anchor='w', pady=3)
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
        row = ttk.Frame(self.diagnostics)
        row.pack(anchor='w', pady=4)
        for label, variable, low, high in [('Pan', self.pan, -HOME_PAN, MAX_PAN-HOME_PAN),
                                           ('Tilt', self.tilt, -HOME_TILT, MAX_TILT-HOME_TILT)]:
            ttk.Label(row, text=f'{label} ({low}° ~ +{high}°)').pack(side='left', padx=4)
            ttk.Spinbox(row, from_=low, to=high, increment=1, width=6,
                        textvariable=variable).pack(side='left')
        ttk.Button(row, text='전송 (OOK)', command=self.transmit).pack(side='left', padx=6)
        ttk.Button(row, text='송신 중지', command=lambda: self.stop('사용자 중지')).pack(side='left', padx=3)
        ttk.Button(row, text='Tx 카메라 다시 준비', command=self.retry_preview).pack(side='left', padx=3)
        ttk.Label(self.diagnostics, text=f'입력 (0°, 0°) = 발판 중앙 · 셀 수직 '
                             f'(서보 Pan {HOME_PAN}°, Tilt {HOME_TILT}°)').pack(anchor='w')
        ttk.Label(self, textvariable=self.status).pack(anchor='w')
        ttk.Label(self, textvariable=self.verification).pack(anchor='w')
        ttk.Label(self, textvariable=self.off_status).pack(anchor='w')

    def toggle_diagnostics(self):
        if self.diagnostics.winfo_manager():
            self.diagnostics.pack_forget()
            self.diagnostic_toggle.configure(text='서보·OOK 점검 열기')
        else:
            self.diagnostics.pack(anchor='w')
            self.diagnostic_toggle.configure(text='서보·OOK 점검 닫기')

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
        if self.draining_rx is not None:
            self.status.set('이전 Rx 파형을 저장한 뒤 다시 연결하세요.')
            return
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

    def selected_period(self):
        index = self.rate_box.current()
        if index < 0:
            raise ValueError('목록에서 실험한 각도 OOK 속도를 선택하세요.')
        return ANGLE_BIT_PERIODS[index]

    def waveform_context(self):
        return dict(method='manual', rx_ssh_target=self.target.get().strip(),
                    pan=self.goal[0], tilt=self.goal[1])

    def transmit(self):
        if self.phase:
            return
        try:
            self.ready_rx()
            if self.wave is not None:
                raise ValueError('이전 송신의 Rx 파형을 저장하고 있습니다.')
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
            self.tx_payload = payload_from_frame(self.bits)
            self.decoded_payload = self.decoded_pose = self.write_pose = self.servo_rows = None
            self.tx_done = self.rx_done = self.commit_sent = False
            self.bit_ms = self.selected_period()
            self.show_verification()
            self.off_status.set('')
            self.begin('off')
            self.arm_sent = False
            self.wave_outcome = None
            try:
                self.wave = AngleWaveform(self.app.stage_dir('M6-4'), self.token, self.bit_ms, self.bits,
                                          context=self.waveform_context())
            except (ValueError, OSError):
                self.phase = self.token = None
                raise
            self.rate_box.state(['disabled'])
            self.wave_review.recording(self.wave.folder)
            requested = dict(event='requested', token=self.token, pan=pan, tilt=tilt, bit_ms=self.bit_ms,
                             tx_frame_bits=self.bits, tx_payload_bits=self.tx_payload,
                             waveform_folder=str(self.wave.folder))
            self.record(requested)
            self.wave.tx_event(requested)
            self.status.set('Tx OFF 확인 중…')
            if not self.app.send(dict(cmd='laser', value=0, request_id=self.token)):
                self.stop('Tx OFF 요청 실패')
        except (ValueError, OSError) as exc:
            if self.optical:
                self.stop(str(exc))
            else:
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
        if (self.phase in ('check_home', 'check_jog', 'home', 'bench_home')
                or (self.phase == 'optical' and self.commit_sent)):
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
        if optical:
            self.log_verification(old_token, False, reason)

    def show_verification(self):
        def pose_text(pose):
            return '—' if pose is None else f'{pose[0]}° / {pose[1]}°'
        detail = ''
        if self.decoded_payload is not None:
            diff = sum(tx != rx for tx, rx in zip(self.tx_payload, self.decoded_payload))
            detail = f' · payload {len(self.tx_payload) - diff}/{len(self.tx_payload)}비트 일치'
        self.verification.set(f'최근 OOK (Pan / Tilt)    Tx 요청 {pose_text(self.goal)}    '
                              f'Rx 광복호 {pose_text(self.decoded_pose)}    '
                              f'Rx 서보 쓰기 {pose_text(self.write_pose)}{detail}')

    def record_verification(self, event):
        self.record(event)

    def log_verification(self, token, success, reason):
        report = dict(event='angle_verification', token=token, success=success, reason=reason,
                      bit_ms=self.bit_ms, tx_completed=self.tx_done, tx_payload_bits=self.tx_payload,
                      tx_angles=self.goal, rx_payload_bits=self.decoded_payload,
                      rx_angles=self.decoded_pose, commit_sent=self.commit_sent,
                      servo_angles=self.write_pose, servo_writes=self.servo_rows)
        if self.wave is not None and token == self.wave.token:
            report['waveform_folder'] = str(self.wave.folder)
            self.wave_outcome = (success, reason)
            self.drain_until = time.monotonic() + 3
            try:
                self.wave.tx_event(report)
            except (ValueError, OSError) as exc:
                self.rx_status.set(f'Rx 파형 결과 기록 오류 · {exc}')
            if not self.arm_sent or self.wave.end_received:
                self.finish_wave()
        self.record_verification(report)

    def finish_wave(self):
        wave, self.wave = self.wave, None
        self.rate_box.state(['!disabled'])
        try:
            wave.finish(*self.wave_outcome)
            self.wave_review.analyze(wave.folder)
        except (ValueError, OSError) as exc:
            self.rx_status.set(f'Rx 파형 저장 오류 · {exc}')

    def wave_packet(self, event):
        if self.wave is None or self.wave.token != event.get('token'):
            return
        if event['kind'] == 'angle_samples':
            self.wave.samples(event)
        else:
            self.wave.end_samples(event)
            if self.wave_outcome is not None:
                self.finish_wave()

    def applied_pose(self, pan, tilt):
        self.last_servo_pose = (pan, tilt)
        self.applied.set(f'Pan {pan}°    Tilt {tilt}°     (원위치 대비 Pan {pan-HOME_PAN:+d}°, '
                         f'Tilt {tilt-HOME_TILT:+d}°)')

    def clear_servo_pose(self, reason):
        self.last_servo_pose = None
        self.applied.set(f'Pan —    Tilt —     ({reason})')

    def on_rx(self, event):
        kind = event['kind']
        if kind in ('angle_samples', 'angle_samples_end'):
            self.wave_packet(event)
            return
        self.record(event)
        if kind == 'armed' and self.wave is not None and self.wave.token == event.get('token'):
            try:
                self.wave.tx_event(event)
            except (ValueError, OSError) as exc:
                if self.optical:
                    self.stop(str(exc))
                else:
                    self.rx_status.set(str(exc))
                return
        if kind == 'ready':
            self.rx_status.set('Rx SSH 연결됨')
            self.clear_servo_pose('원위치 또는 두 축 확인을 눌러 시작하세요.')
            self.verification.set('')
            return
        if kind == 'error' and not event.get('token'):
            self.stop('Rx 오류 · ' + event.get('message', ''))
            return
        if not self.token or event.get('token') != self.token:
            return
        if kind == 'timeout' and self.optical:
            self.stop(self.verification_timeout())
        elif kind in ('error', 'timeout', 'cancelled'):
            self.stop(event.get('message') or '유효한 광명령 수신을 확인하지 못했습니다.')
        elif kind in ('decoded', 'applied') and self.optical and self.phase != 'optical':
            if kind == 'applied':
                self.clear_servo_pose('승인과 대응하지 않는 Rx 서보 응답')
            self.stop('광프레임 송신 요청 전에 Rx 응답이 도착했습니다.')
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
            if (event.get('bit_ms'), event.get('sample_ms'), event.get('interval_ms'),
                    event.get('bus'), event.get('address'), event.get('config')) != (
                    self.bit_ms, DECODE_SAMPLE_MS, DECODE_SAMPLE_MS, 3, 0x40, SAMPLE_CONFIG):
                self.stop('Rx가 확인한 각도 OOK 속도 또는 INA219 설정이 요청과 다릅니다.')
                return
            self.app.ook.check_ready()
            self.phase = 'optical'
            self.deadline = time.monotonic() + 9 + len(self.bits) * self.bit_ms / 1000
            self.status.set('OOK 전송 중 · Rx 광복호 대기')
            if not self.app.send(dict(cmd='laser_pattern', request_id=self.token,
                                      bits=self.bits, bit_ms=self.bit_ms, repeats=1)):
                self.stop('광프레임 송신 요청 실패')
        elif kind == 'decoded' and self.phase == 'optical':
            if not self.wave.end_received:
                self.stop('Rx 각도 파형 종료 응답 전에 복호 결과가 도착했습니다.')
                return
            self.decoded_payload = event['rx_payload_bits']
            self.decoded_pose = (event['pan'], event['tilt'])
            self.show_verification()
            if self.decoded_payload != self.tx_payload or self.decoded_pose != self.goal:
                self.stop('Tx 요청과 Rx 광복호 payload 또는 각도가 다릅니다.')
            else:
                self.commit_if_verified()
        elif kind == 'applied' and self.phase == 'optical':
            if not self.commit_sent:
                self.clear_servo_pose('승인과 대응하지 않는 Rx 서보 응답')
                self.stop('PC의 적용 승인 전에 Rx 서보 응답이 도착했습니다.')
                return
            self.servo_rows = event.get('servo_writes')
            try:
                self.write_pose = optical_servo_writes(event)
            except ValueError as exc:
                self.stop(str(exc))
                return
            self.show_verification()
            if (event.get('rx_payload_bits') != self.decoded_payload
                    or (event.get('pan'), event.get('tilt')) != self.goal or self.write_pose != self.goal):
                self.stop('Rx 서보 쓰기와 검증한 광명령이 일치하지 않습니다.')
                return
            self.applied_pose(*self.write_pose)
            self.rx_done = True
            self.finish_if_confirmed()

    def commit_if_verified(self):
        if not self.tx_done:
            self.status.set('Rx 광복호 일치 · Tx 송신 완료와 OFF 응답 대기')
        elif self.decoded_pose is None:
            self.status.set('Tx 송신 완료 · Rx 광복호 응답 대기')
        elif not self.commit_sent:
            self.rx.send('commit', self.token)
            self.commit_sent = True
            self.deadline = time.monotonic() + 5
            self.status.set('Tx 완료 · Rx 광복호 일치 · 두 축 서보 쓰기 응답 대기')

    def verification_timeout(self):
        if self.commit_sent:
            return '적용 승인 후 Rx 두 축의 서보 쓰기 응답 시간 초과'
        if self.decoded_pose is not None:
            return 'Rx 광복호 일치 · Tx 송신 완료 확인 시간 초과'
        if self.phase == 'optical':
            return 'Rx 각도 payload 복호 응답 시간 초과'
        return 'OOK 전송 준비 응답 시간 초과'

    def finish_if_confirmed(self):
        if self.tx_done and self.rx_done:
            self.log_verification(self.token, True, '')
            self.phase = self.token = None
            self.status.set('OOK 수신 성공 · Rx 광복호 및 서보 명령 적용 확인')

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
            if self.wave is not None and kind == 'laser_pattern':
                self.wave.tx_event(event)
            if kind == 'error':
                self.stop('Tx 오류 · ' + event.get('message', ''))
            elif kind == 'laser' and self.phase == 'off':
                if event.get('laser_gpio_level') != 0:
                    self.stop('Tx OFF 응답을 확인하지 못했습니다.')
                    return
                self.phase = 'arming'
                self.deadline = time.monotonic() + 5
                self.rx.arm_angle(self.token, self.bit_ms)
                self.arm_sent = True
                self.status.set('Rx 광수신 준비 중…')
            elif kind == 'laser_pattern' and self.phase == 'optical' and event.get('state') != 'started':
                off = event.get('off_call_done_ms')
                if (event.get('state') != 'completed' or event.get('simulated') is not False
                        or type(off) not in (int, float) or not math.isfinite(off) or off < len(self.bits) * self.bit_ms
                        or event.get('settings') != dict(bits=self.bits, bit_ms=self.bit_ms, repeats=1)):
                    self.stop('Tx 송신 완료 또는 종료 OFF를 확인하지 못했습니다.')
                    return
                trace = event.get('trace')
                if (not isinstance(trace, list) or len(trace) != len(self.bits)
                        or any(not isinstance(row, dict) or type(row.get('index')) is not int
                               or row['index'] != i or type(row.get('level')) is not int
                               or row['level'] != int(bit) for i, (row, bit) in enumerate(zip(trace, self.bits)))):
                    self.stop('Tx GPIO 비트 기록과 요청한 프레임이 다릅니다.')
                    return
                self.tx_done = True
                self.commit_if_verified()
        except (ValueError, OSError) as exc:
            self.stop(str(exc))

    def drain_samples(self, rx):
        # After the operator stops, only consume telemetry for the old request.
        # Late decode/approval events cannot resume servo commands.
        for _ in range(rx.events.qsize()):
            event = rx.events.get_nowait()
            if event.get('kind') in ('armed', 'angle_samples', 'angle_samples_end'):
                try:
                    if event['kind'] == 'armed':
                        if self.wave is not None and self.wave.token == event.get('token'):
                            self.wave.tx_event(event)
                    else:
                        self.wave_packet(event)
                except (OSError, ValueError) as exc:
                    self.rx_status.set(f'Rx 파형 저장 오류 · {exc}')

    def release_wave(self, *, closing=False):
        if self.draining_rx is not None:
            rx = self.draining_rx
            self.drain_samples(rx)
            if self.wave is not None and (closing or rx.terminal_error or time.monotonic() >= self.drain_until):
                self.finish_wave()
            if self.wave is None:
                rx.close()
                self.draining_rx = None

    def tick(self):
        if self.wave_review is not None:
            self.wave_review.poll()
        self.release_wave()
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
                self.stop(f'Rx SSH 연결 종료 · {detail}')
                self.rx_status.set(detail)
                self.rx.close()
                self.rx = None
                self.clear_servo_pose('SSH 연결 종료')
        if self.wave is not None and self.wave_outcome is not None and (
                not self.rx or time.monotonic() >= self.drain_until):
            self.finish_wave()
        if self.phase == 'check_pause' and time.monotonic() >= self.check_due:
            try:
                self.next_check_step()
            except (ValueError, OSError) as exc:
                self.stop(str(exc))
        elif self.phase and self.phase != 'check_pause' and time.monotonic() > self.deadline:
            self.stop(self.verification_timeout() if self.optical else '응답 시간 초과 · Rx 서보 적용 상태를 확인하세요.')
        elif self.optical and (not self.app.current or time.monotonic()-self.app.current[2] > 2):
            self.stop('Tx 카메라 영상 수신 중단')

    def leave(self):
        self.selected = self.prepared = False
        if self.phase:
            self.stop('탭을 닫아 시연 중지')
        if self.rx:
            if self.wave is not None and self.wave_outcome is not None:
                self.draining_rx = self.rx
            else:
                self.rx.close()
            self.rx = None
            self.clear_servo_pose('Rx 미연결')
            self.rx_status.set('Rx 미연결')
