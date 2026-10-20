"""M4-4 laser controls and raw OOK tests; no command-frame encoding yet."""
import json
import time
import uuid
import queue
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
from common.laser_pattern import pattern_settings
from common.tx_setup import TX_CAMERA_SETTINGS
from Tx.Controller.arduino_capture import ArduinoSerial, RxCapture, parse_reading, serial_ports


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
        self.trial_paths = {}
        self.arduino = None
        self.capture = None
        self.rx_phase = None
        self.rx_deadline = None
        self.rx_start_sent = False
        self.rx_last_seen = None
        self.rx_last_ms = None
        self.trial_settings = None
        self.trial_id = None
        self.tx_state = None
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
        row = ttk.Frame(self); row.pack(fill='x', pady=3)
        self.rx_port = tk.StringVar()
        self.rx_save = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text='Arduino 수신 자동 저장', variable=self.rx_save).pack(side='left')
        ttk.Label(row, text='COM 포트').pack(side='left', padx=3)
        self.port_box = ttk.Combobox(row, textvariable=self.rx_port, width=12)
        self.port_box.pack(side='left')
        for label, action in [('포트 새로고침', self.refresh_ports), ('Arduino 연결', self.connect_rx),
                              ('연결 해제', self.disconnect_rx)]:
            ttk.Button(row, text=label, command=lambda fn=action: self.guard(fn)).pack(side='left', padx=3)
        self.rx_status = tk.StringVar(value='Arduino 미연결 · 9600 baud · IDE 시리얼 모니터를 닫고 연결하세요.')
        self.saved_status = tk.StringVar(value='저장: captures/M4-4/<시험 폴더>/')
        ttk.Label(self, textvariable=self.rx_status, wraplength=1100).pack(anchor='w')
        ttk.Label(self, textvariable=self.saved_status, wraplength=1100).pack(anchor='w')
        ttk.Label(self, text='1 = 레이저 ON, 0 = OFF · Pi에서 반복 실행 · 종료 후 OFF\n'
                  '자동 저장: OFF 확인 → Arduino 기록 시작 확인 → 바로 송신 (WAIT 대기 없음) → 종료 후 2.5초 기록\n'
                  '수신 CSV는 250ms 간격 출력값입니다. 전체 10ms ADC 샘플이나 복원된 비트열이 아닙니다.\n'
                  '전압 비교 시험: BASELINE/NA여도 송신·저장합니다. HIGH/LOW는 판정 결과로 사용하지 않습니다.\n'
                  '현재는 원시 ON/OFF 시험이며, 7비트 각도 명령·Manchester·오류 검사 프레임은 미구현입니다.\n'
                  '아래 공통 영상으로 조준 상태를 확인하세요. GPIO 상태는 실제 광출력 피드백이 아닙니다.').pack(anchor='w')

    @property
    def busy(self):
        return self.output_requested or self.run_id is not None or self.rx_phase is not None

    def guard(self, fn):
        try:
            fn()
        except ValueError as exc:
            messagebox.showerror('M4-4', str(exc))
        except OSError as exc:
            if self.capture:
                self.abort_capture('operation_error')
            messagebox.showerror('M4-4', str(exc))

    def refresh_ports(self):
        ports = serial_ports()
        self.port_box.configure(values=ports)
        if not self.rx_port.get() and len(ports) == 1:
            self.rx_port.set(ports[0])

    def connect_rx(self):
        if self.busy:
            raise ValueError('시험 중지 후 Arduino 연결을 변경하세요.')
        if self.arduino is not None:
            if self.arduino.worker.is_alive():
                raise ValueError('기존 Arduino 연결을 해제하고 잠시 기다리세요.')
            self.arduino = None
        port = self.rx_port.get().strip()
        if not port:
            raise ValueError('Arduino COM 포트를 선택하세요.')
        self.arduino = ArduinoSerial(port)
        self.rx_status.set('Arduino 연결 중 · 보드 초기화 대기…')

    def disconnect_rx(self):
        if self.capture:
            self.abort_capture('arduino_disconnected')
        if self.arduino:
            self.arduino.close()
        self.rx_status.set('Arduino 연결 해제')

    def finish_capture(self, reason):
        capture, self.capture = self.capture, None
        self.rx_phase = self.rx_deadline = None
        self.rx_start_sent = False
        if self.arduino and self.arduino.connected:
            try:
                self.arduino.send('stop')
            except OSError:
                pass
        if capture:
            try:
                capture.finish(reason, self.tx_state)
                self.saved_status.set(f'저장 완료 ({reason}, {capture.rows}행): {capture.folder}')
            except OSError as exc:
                self.saved_status.set(f'저장 오류 · 일부 파일만 남았을 수 있음: {exc}')

    def abort_capture(self, reason):
        # Cancel before consuming any more serial records. OFF remains independent
        # of disk/USB failures; late serial messages cannot launch a pattern.
        self.rx_phase = None
        self.deadline = None
        self.run_id = None
        try:
            if self.request(dict(cmd='laser', value=0)):
                self.state.set('시험 중단 / OFF 응답 대기: ' + reason)
        except OSError as exc:
            self.state.set('OFF 요청/기록 오류 · 장치 상태 확인 필요: ' + str(exc))
        finally:
            self.finish_capture(reason)

    def poll_rx(self):
        if self.arduino is None:
            return
        for _ in range(200):
            try:
                event = self.arduino.events.get_nowait()
            except queue.Empty:
                break
            kind = event['kind']
            if kind == 'connected':
                self.rx_status.set(f"Arduino 연결됨: {event['port']} · 9600 baud")
            elif kind == 'error':
                self.rx_status.set('Arduino 오류: ' + event['message'])
                if self.capture:
                    self.abort_capture('arduino_error')
            if not self.capture:
                continue
            self.write_log(dict(event='arduino', **event))
            if kind == 'command_sent' and event.get('command') == 'start' and event.get('token') == self.trial_id:
                self.rx_start_sent = True
                continue
            if kind != 'line' or not self.rx_start_sent:
                continue
            line = event['line']
            if line.startswith('M4-4 PV'):
                self.abort_capture('arduino_reset')
                continue
            try:
                reading = parse_reading(line)
            except (ValueError, OverflowError):
                self.capture.invalid_lines += 1
                self.abort_capture('invalid_arduino_record')
                continue
            if reading is None:
                continue
            self.capture.write(event, reading)
            now = time.monotonic()
            # The UI can lag the reader; stale queued data must not arm TX.
            age = (time.monotonic_ns() - event['pc_receive_monotonic_ns']) / 1e9
            self.rx_last_seen = event['pc_receive_monotonic_ns'] / 1e9
            ms = reading['arduino_ms']
            if self.rx_last_ms is not None and ms < self.rx_last_ms and self.rx_last_ms - ms < 0x80000000:
                self.abort_capture('arduino_clock_reset')
                continue
            self.rx_last_ms = ms
            self.rx_status.set(f"Rx: {reading['state']} / {reading['level']} · PV={reading['pv_V']} V · "
                               f"OFF={reading['off_V']} V · ON={reading['on_V']} V · {self.capture.rows}행")
            if reading['clipped']:
                self.abort_capture('adc_clipped')
                continue
            # Raw voltage experiment: only confirm that fresh samples arrive.
            # Detector calibration and HIGH/LOW classification do not gate TX.
            if (self.rx_phase == 'recording_start' and reading['state'] != 'IDLE'
                    and reading['pv_V'] is not None and age < 1.0):
                if not self.online or self.pending or not self.app.current or now - self.app.current[2] > 2:
                    self.abort_capture('tx_not_ready')
                    continue
                self.rx_phase = 'sending'
                self.rx_deadline = None
                self.transmit(self.trial_settings, self.trial_id)
        if self.capture and not self.arduino.worker.is_alive():
            self.abort_capture('arduino_connection_lost')

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
        if self.capture:
            self.abort_capture('user_stop')
            return
        if self.request(dict(cmd='laser', value=0)):
            self.state.set('중지 / OFF 응답 대기')

    def start(self):
        self.check_ready()
        if 'laser_pattern' not in self.hardware.get('capabilities', []):
            raise ValueError('Pi도 M4-4 코드로 업데이트하고 재시작하세요.')
        settings = pattern_settings(dict(bits=self.bits.get().strip(),
                                         bit_ms=float(self.bit_ms.get()), repeats=int(self.repeats.get())))
        save_rx = self.rx_save.get()
        if save_rx and (self.arduino is None or not self.arduino.connected):
            raise ValueError('Arduino를 연결하세요. 원시 송신만 하려면 수신 자동 저장을 해제하세요.')
        token = 'm4-4-' + uuid.uuid4().hex
        folder = self.app.stage_dir('M4-4') / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + token[-6:])
        folder.mkdir()
        self.logpath = folder / 'events.jsonl'
        self.trial_paths[token] = self.logpath
        self.write_log(dict(event='requested', request_id=token, settings=settings, hardware=self.hardware))
        if save_rx:
            self.capture = RxCapture(folder, settings, self.arduino.port, token)
            self.saved_status.set(f'기록 중: {folder}')
            self.trial_settings, self.trial_id = settings, token
            self.tx_state = 'not_sent'
            self.rx_last_seen = self.rx_last_ms = None
            self.rx_start_sent = False
            self.rx_phase = 'off'
            self.rx_deadline = time.monotonic() + 5
            if not self.request(dict(cmd='laser', value=0)):
                self.finish_capture('off_request_failed')
                return
            self.state.set('자동 저장 준비: 레이저 OFF 응답 대기')
            return
        self.transmit(settings, token)

    def transmit(self, settings, token):
        self.write_log(dict(event='pattern_send_requested', request_id=token, settings=settings))
        if self.app.send(dict(cmd='laser_pattern', request_id=token, **settings)):
            self.tx_state = 'requested'
            self.run_id = token
            self.pending = (token, time.monotonic(), 'laser_pattern')
            self.deadline = time.monotonic() + len(settings['bits']) * settings['repeats'] * settings['bit_ms'] / 1000 + 5
            self.output_requested = True
            self.state.set('패턴 송신 응답 대기')
        else:
            self.write_log(dict(event='send_failed', request_id=token))
            self.finish_capture('tx_send_failed')

    def write_log(self, event):
        path = self.trial_paths.get(event.get('request_id'), self.logpath)
        if path:
            with path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(event, gui_unix_ns=time.time_ns()), ensure_ascii=False) + '\n')

    def event(self, event):
        try:
            self.handle_event(event)
        except OSError as exc:
            if self.capture:
                self.abort_capture('recording_error')
            self.rx_status.set('이벤트 저장 오류: ' + str(exc))

    def handle_event(self, event):
        kind = event.get('event')
        # Keep terminal GPIO traces even if the user already stopped capture.
        if kind == 'laser_pattern' and event.get('request_id') in self.trial_paths:
            self.write_log(event)
        if kind in ('hello', 'agent', 'ready'):
            self.online = kind == 'ready' or event.get('agent_state', event.get('state')) == 'connected'
            self.hardware = None
            if self.online:
                self.request(dict(cmd='hardware_status'))
        disconnected = (kind == 'network' and event.get('port') == 7600 and event.get('state') != 'connected') or (
            kind in ('hello', 'agent') and not self.online)
        if disconnected:
            if self.capture:
                self.finish_capture('pi_disconnected')
            self.write_log(dict(event='connection_lost', request_id=self.run_id))
            self.online = False; self.hardware = None; self.pending = None
            self.run_id = None; self.deadline = None; self.output_requested = False
            self.state.set('연결 끊김 · 실제 소등 상태는 확인되지 않음')
            return
        if kind == 'laser_pattern' and event.get('request_id') == self.run_id:
            self.tx_state = event.get('state')
            self.state.set('패턴: ' + event.get('state', '?') + ' · GPIO 명령 기록이며 광출력 미측정')
            if event.get('state') != 'started':
                self.run_id = None; self.deadline = None
                self.output_requested = event.get('state') == 'off_unconfirmed'
                if self.hardware:
                    self.hardware['pattern_active'] = False
                if self.capture:
                    if event.get('state') == 'completed':
                        self.rx_phase = 'tail'
                        self.rx_deadline = time.monotonic() + 2.5
                    else:
                        self.finish_capture('tx_' + event.get('state', 'unknown'))
        if not self.pending or event.get('request_id') != self.pending[0]:
            return
        operation = self.pending[2]
        self.pending = None
        if kind == 'error':
            self.write_log(event)
            self.state.set('요청 오류: ' + event.get('message', ''))
            if self.capture:
                self.abort_capture('pi_request_error')
            elif operation in ('laser', 'laser_pattern'):
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
            if self.capture and self.rx_phase == 'off' and operation == 'laser_off':
                if event.get('laser_gpio_level') != 0:
                    self.abort_capture('off_not_confirmed')
                    return
                try:
                    self.arduino.send('start', self.trial_id)
                except OSError:
                    self.abort_capture('arduino_start_failed')
                    return
                self.rx_phase = 'recording_start'
                self.rx_deadline = time.monotonic() + 5
                self.state.set('Arduino 첫 전압 기록 확인 후 바로 송신 · WAIT 대기 없음')

    def tick(self):
        try:
            self.tick_capture()
        except OSError as exc:
            self.abort_capture('recording_error')
            self.rx_status.set('수신/저장 오류: ' + str(exc))

    def tick_capture(self):
        self.poll_rx()
        now = time.monotonic()
        if self.capture and self.rx_deadline and now > self.rx_deadline:
            if self.rx_phase == 'tail':
                self.finish_capture('completed')
            else:
                self.abort_capture('rx_preparation_timeout')
        if (self.capture and self.rx_phase in ('sending', 'tail') and
                (self.rx_last_seen is None or now - self.rx_last_seen > 3)):
            self.abort_capture('rx_data_timeout')
        if self.pending and now - self.pending[1] > 5:
            operation = self.pending[2]
            self.pending = None
            self.write_log(dict(event='response_timeout', operation=operation))
            self.hardware = None
            if self.capture:
                self.abort_capture('pi_response_timeout')
            if self.busy:
                self.app.send(dict(cmd='laser', value=0))
            self.state.set('응답 시간 초과 · OFF 요청 · 상태 재조회 필요')
        if self.deadline and now > self.deadline:
            self.deadline = None
            self.off()

    def leave(self):
        if self.busy:
            self.off()

    def close(self):
        self.leave()
        if self.arduino:
            self.arduino.close()
