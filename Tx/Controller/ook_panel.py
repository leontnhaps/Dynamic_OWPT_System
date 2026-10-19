"""M4-4 laser controls and raw OOK tests; no command-frame encoding yet."""
import json
import time
import uuid
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
from common.laser_pattern import pattern_settings
from common.tx_setup import TX_CAMERA_SETTINGS


class OokPanel(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, padding=8)
        self.app = app
        self.online = False
        self.hardware = None
        self.pending = None
        self.run_id = None
        self.deadline = None
        self.output_requested = False
        self.logpath = None
        self.state = tk.StringVar(value='카메라 시작 / 상태 조회를 눌러 준비하세요. 자동 발사 없음.')
        ttk.Label(self, textvariable=self.state).pack(anchor='w')
        row = ttk.Frame(self); row.pack(fill='x', pady=5)
        for label, action in [('카메라 시작 / 상태 조회', self.prepare),
                              ('상태 조회', lambda: self.request(dict(cmd='hardware_status'))),
                              ('레이저 ON', self.laser_on), ('레이저 OFF / 시험 중지', self.off),
                              ('IR LOW', lambda: self.ir(0)), ('IR HIGH', lambda: self.ir(1))]:
            ttk.Button(row, text=label, command=lambda fn=action: self.guard(fn)).pack(side='left', padx=3)
        row = ttk.Frame(self); row.pack(fill='x', pady=5)
        self.bits = tk.StringVar(value='10101010')
        self.bit_ms = tk.StringVar(value='500')
        self.repeats = tk.StringVar(value='1')
        for label, variable, width in [('ON/OFF 패턴', self.bits, 30),
                                       ('한 비트 길이(ms)', self.bit_ms, 8), ('반복', self.repeats, 5)]:
            ttk.Label(row, text=label).pack(side='left', padx=3)
            ttk.Entry(row, textvariable=variable, width=width).pack(side='left')
        ttk.Button(row, text='패턴 송신', command=lambda: self.guard(self.start)).pack(side='left', padx=6)
        ttk.Label(self, text='1 = 레이저 ON, 0 = OFF · Pi에서 반복 실행 · 종료 후 OFF\n'
                  '500 ms는 초기 시험값입니다. 통신 속도·수신 성공은 아직 검증되지 않았습니다.\n'
                  '현재는 원시 ON/OFF 시험이며, 7비트 각도 명령·Manchester·오류 검사 프레임은 미구현입니다.\n'
                  '아래 공통 영상으로 조준 상태를 확인하세요. GPIO 상태는 실제 광출력 피드백이 아닙니다.').pack(anchor='w')

    @property
    def busy(self):
        return self.output_requested or self.run_id is not None

    def guard(self, fn):
        try:
            fn()
        except (ValueError, OSError) as exc:
            messagebox.showerror('M4-4', str(exc))

    def request(self, cmd):
        token = 'm4-4-' + uuid.uuid4().hex
        if self.app.send(dict(cmd, request_id=token)):
            self.pending = (token, time.monotonic(), 'laser_off' if cmd.get('cmd')=='laser' and cmd.get('value')==0 else cmd['cmd'])
            return token
        return None

    def check_ready(self):
        if not self.online or not self.hardware or self.pending:
            raise ValueError('Pi 연결 및 상태 조회 응답을 확인하세요.')
        frame = self.app.current
        if not frame or time.monotonic() - frame[2] > 2:
            raise ValueError('카메라 시작 후 최근 영상을 확인하세요.')
        if self.busy or self.hardware.get('pattern_active'):
            raise ValueError('레이저 OFF / 시험 중지 후 다시 실행하세요.')

    def prepare(self):
        if self.busy or self.pending:
            raise ValueError('진행 중인 명령을 중지하거나 응답을 기다리세요.')
        if self.app.send(dict(cmd='preview', enable=True, **TX_CAMERA_SETTINGS)):
            self.request(dict(cmd='hardware_status'))

    def laser_on(self):
        self.check_ready()
        if self.request(dict(cmd='laser', value=1)):
            self.output_requested = True
            self.state.set('레이저 ON 응답 대기')

    def ir(self, level):
        if self.busy or self.pending:
            raise ValueError('레이저 OFF 및 응답 확인 후 IR을 변경하세요.')
        self.request(dict(cmd='ir_cut', level=level))

    def off(self):
        # Available during pending requests; stale responses no longer match.
        if self.request(dict(cmd='laser', value=0)):
            self.state.set('중지 / OFF 응답 대기')

    def start(self):
        self.check_ready()
        if 'laser_pattern' not in self.hardware.get('capabilities', []):
            raise ValueError('Pi도 M4-4 코드로 업데이트하고 재시작하세요.')
        settings = pattern_settings(dict(bits=self.bits.get().strip(),
                                         bit_ms=float(self.bit_ms.get()), repeats=int(self.repeats.get())))
        token = 'm4-4-' + uuid.uuid4().hex
        folder = self.app.stage_dir('M4-4') / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + token[-6:])
        folder.mkdir()
        self.logpath = folder / 'events.jsonl'
        self.write_log(dict(event='requested', request_id=token, settings=settings, hardware=self.hardware))
        if self.app.send(dict(cmd='laser_pattern', request_id=token, **settings)):
            self.run_id = token
            self.pending = (token, time.monotonic(), 'laser_pattern')
            self.deadline = time.monotonic() + len(settings['bits']) * settings['repeats'] * settings['bit_ms'] / 1000 + 5
            self.output_requested = True
            self.state.set('패턴 송신 응답 대기')
        else:
            self.write_log(dict(event='send_failed', request_id=token))

    def write_log(self, event):
        if self.logpath:
            with self.logpath.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(event, gui_unix_ns=time.time_ns()), ensure_ascii=False) + '\n')

    def event(self, event):
        kind = event.get('event')
        if kind in ('hello', 'agent', 'ready'):
            self.online = kind == 'ready' or event.get('agent_state', event.get('state')) == 'connected'
            self.hardware = None
            if self.online:
                self.request(dict(cmd='hardware_status'))
        disconnected = (kind == 'network' and event.get('port') == 7600 and event.get('state') != 'connected') or (
            kind in ('hello', 'agent') and not self.online)
        if disconnected:
            self.write_log(dict(event='connection_lost', request_id=self.run_id))
            self.online = False; self.hardware = None; self.pending = None
            self.run_id = None; self.deadline = None; self.output_requested = False
            self.state.set('연결 끊김 · 실제 소등 상태는 확인되지 않음')
            return
        if kind == 'laser_pattern' and event.get('request_id') == self.run_id:
            self.write_log(event)
            self.state.set('패턴: ' + event.get('state', '?') + ' · GPIO 명령 기록이며 광출력 미측정')
            if event.get('state') != 'started':
                self.run_id = None; self.deadline = None
                self.output_requested = event.get('state') == 'off_unconfirmed'
                if self.hardware:
                    self.hardware['pattern_active'] = False
        if not self.pending or event.get('request_id') != self.pending[0]:
            return
        operation = self.pending[2]
        self.pending = None
        if kind == 'error':
            self.write_log(event)
            self.state.set('요청 오류: ' + event.get('message', ''))
            if operation in ('laser', 'laser_pattern'):
                self.off()
            return
        if kind == 'hardware':
            self.hardware = event
            self.output_requested = event.get('laser_gpio_level') == 1 or event.get('pattern_active', False)
            self.state.set(('SIMULATION · ' if event.get('simulated') else '') +
                           f"GPIO 명령 상태: laser={event.get('laser_gpio_level')} · 실제 광출력 미측정")
        elif kind == 'laser':
            self.output_requested = event.get('laser_gpio_level') != 0
            self.state.set(f"레이저 GPIO 명령 상태: {event.get('laser_gpio_level')} · 실제 광출력 미측정")

    def tick(self):
        now = time.monotonic()
        if self.pending and now - self.pending[1] > 5:
            operation = self.pending[2]
            self.pending = None
            self.write_log(dict(event='response_timeout', operation=operation))
            self.hardware = None
            if self.busy:
                self.app.send(dict(cmd='laser', value=0))
            self.state.set('응답 시간 초과 · OFF 요청 · 상태 재조회 필요')
        if self.deadline and now > self.deadline:
            self.deadline = None
            self.off()

    def leave(self):
        if self.busy:
            self.off()
