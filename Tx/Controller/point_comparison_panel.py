"""Operator-triggered, stationary-point comparison of three PV receiver poses."""
import math
import time
import tkinter as tk
from tkinter import ttk

from common.angle_demo import (BIT_MS, FRAME_LEN, HOME_PAN, HOME_TILT, VOLTAGE_SETTLE_NS,
                               VOLTAGE_WINDOW_NS, encode_angles, payload_from_frame)
from common.ook_experiment import SYNC_WORD, THRESHOLD_BITS, TRAILING_ZEROS
from Tx.Controller.angle_demo_panel import AngleDemoPanel
from Tx.Controller.bench_pose import BenchPoseInputs
from Tx.Controller.final_speed_panel import FinalSpeedPanel
from Tx.Controller.pv_comparison import ComparisonRun


METHODS = {'baseline': '무보정', 'yolo': 'YOLO', 'xyz': '실측 XYZ'}
SERVO_SETTLE_S = 1.0
# One bounded HIGH bit on the existing fixed Tx. The Pi always ends the pattern OFF.
FIXED_LIGHT = dict(bits='1', bit_ms=3000, repeats=1)


class PointComparisonPanel(AngleDemoPanel):
    speed = None

    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.run = self.station = self.method = None
        self.inputs = BenchPoseInputs(self, app)
        self.inputs.pack(fill='x', pady=3)
        actions = ttk.Frame(self)
        actions.pack(anchor='w')
        ttk.Button(actions, text='지점 등록', command=self.register).pack(side='left')
        for method, label in METHODS.items():
            text = label + (' 측정' if method == 'baseline' else ' 각도 계산')
            ttk.Button(actions, text=text, command=lambda m=method: self.start_trial(m)).pack(side='left', padx=3)
        ttk.Button(actions, text='계산각 OOK 전송 · 측정', command=self.send_estimate).pack(side='left', padx=4)
        axis_row = ttk.Frame(self)
        axis_row.pack(anchor='w')
        ttk.Label(axis_row, text='OOK 단일 축 전송 (원위치 기준)').pack(side='left')
        for axis in ('Pan', 'Tilt'):
            for offset in (-1, 1):
                ttk.Button(axis_row, text=f'{axis} {offset:+d}°', width=8,
                           command=lambda a=axis, n=offset: self.check_axis(a, n)).pack(side='left', padx=2)
        self.frame_label = tk.StringVar(self)
        ttk.Label(self, textvariable=self.frame_label).pack(anchor='w')
        self.pan.trace_add('write', self.show_frame)
        self.tilt.trace_add('write', self.show_frame)
        self.show_frame()
        self.location = tk.StringVar(value='측정 지점을 등록하세요.')
        self.results = {key: tk.StringVar(value=label + ' —') for key, label in METHODS.items()}
        ttk.Label(self, textvariable=self.location).pack(anchor='w')
        results = ttk.Frame(self)
        results.pack(anchor='w')
        for variable in self.results.values():
            ttk.Label(results, textvariable=variable).pack(side='left', padx=(0, 18))
        self.speed = FinalSpeedPanel(self)
        self.speed.pack(fill='x', pady=3)

    def show_frame(self, *_):
        try:
            bits = encode_angles(HOME_PAN+int(self.pan.get()), HOME_TILT+int(self.tilt.get()))
        except ValueError:
            self.frame_label.set('각도 프레임 · 범위 안의 정수 각도를 입력하세요.')
            return
        payload = payload_from_frame(bits)
        self.frame_label.set(f'각도 명령 {FRAME_LEN} bit · {1000/BIT_MS:g} bit/s ({BIT_MS} ms/bit)\n'
                             f'레벨 {THRESHOLD_BITS}  |  동기 {SYNC_WORD}  |  Pan {payload[:16]}  |  '
                             f'Tilt {payload[16:]}  |  끝 {TRAILING_ZEROS}')

    def record_verification(self, event):
        super().record_verification(event)
        if self.run is not None and self.run.trial_folder is not None:
            self.run.log(event)

    def check_axis(self, axis, offset):
        if self.phase or self.method or self.off_token:
            self.status.set('진행 중인 동작을 끝낸 후 축 방향을 확인하세요.')
            return
        self.pan.set(str(offset if axis == 'Pan' else 0))
        self.tilt.set(str(offset if axis == 'Tilt' else 0))
        self.transmit()

    @property
    def owns_output(self):
        return self.method is not None or self.phase == 'speed' or self.optical or bool(self.off_token)

    def allows_pattern(self, request_id):
        if not request_id:
            return False
        if self.phase == 'speed':
            ook = self.app.ook
            return ook.experiment is not None and ook.capture is not None and request_id == ook.trial_id
        return self.token == request_id and (self.optical or self.method is not None)

    def register(self):
        if self.phase or self.off_token:
            self.status.set('현재 측정 또는 OFF 확인이 끝난 후 지점을 등록하세요.')
            return
        try:
            snapshot = self.inputs.snapshot()
            run = ComparisonRun(self.app.stage_dir('M6-4'), snapshot)
        except (ValueError, OSError) as exc:
            self.status.set(str(exc))
            return
        self.station, self.run = snapshot, run
        self.station_view = None
        for key, var in self.results.items():
            var.set(METHODS[key] + ' —')
        self.location.set(f'지점 {snapshot["point_id"]} · 저장 위치 {run.folder.parent.name}/{run.folder.name}')
        self.status.set('무보정 측정 또는 보정 각도 계산을 선택하세요.')

    def current_view(self):
        frame = self.app.current
        if not frame or time.monotonic() - frame[2] > 2 or frame[1].get('simulated') is not False:
            raise ValueError('실제 Tx의 최근 카메라 영상이 필요합니다.')
        meta = frame[1]
        cmd = (meta.get('servo_at_capture_start') or {}).get('commanded') or {}
        if not meta.get('session'):
            raise ValueError('Tx 영상의 세션을 확인하지 못했습니다.')
        return dict(session=meta['session'], pan=cmd.get('pan'), tilt=cmd.get('tilt'),
                    command_id=cmd.get('command_id'))

    def check_station(self):
        view = self.current_view()
        if self.station_view is not None and view != self.station_view:
            raise ValueError('지점 등록 이후 Tx 자세 또는 연결이 바뀌었습니다. 새 지점으로 등록하세요.')
        return view

    def start_trial(self, method):
        if self.phase or self.off_token:
            self.status.set('현재 작업이 끝난 후 다음 조건을 시작하세요.')
            return
        try:
            self.ready_rx()
            self.app.ook.check_ready()
            if self.app.ook.hardware.get('simulated') is not False:
                raise ValueError('실제 Tx가 연결되어야 합니다.')
            if self.run is None:
                self.register()
                if self.run is None:
                    return
            if self.inputs.snapshot() != self.station:
                raise ValueError('입력값이 변경됐습니다. 변경한 조건으로 지점을 다시 등록하세요.')
            view = self.check_station()
            self.run.start(method, dict(tx_at_start=view, home=[HOME_PAN, HOME_TILT],
                                        bit_ms=BIT_MS, servo_settle_s=SERVO_SETTLE_S,
                                        light_pattern=FIXED_LIGHT))
            self.method, self.station_view = method, view
            self.begin('bench_home')
            self.rx.send('home', self.token)
            self.run.log(dict(event='return_to_home', token=self.token))
            self.status.set(f'{METHODS[method]} · Rx 원위치 응답 대기')
        except (ValueError, OSError) as exc:
            if self.method:
                self.stop(str(exc))
            else:
                self.status.set(str(exc))

    def estimate_ready(self, result):
        pan, tilt, metadata = result
        self.estimated_goal = (pan, tilt)
        self.run.log(dict(event='estimate', method=self.method, pan=pan, tilt=tilt, inputs=metadata))
        self.pan.set(str(pan-HOME_PAN))
        self.tilt.set(str(tilt-HOME_TILT))
        self.phase = 'confirm_estimate'
        self.deadline = math.inf
        self.status.set(f'{METHODS[self.method]} 계산 · 서보 Pan {pan}°, Tilt {tilt}° · '
                        '계산각 OOK 전송 · 측정을 누르세요.')

    def send_estimate(self):
        if self.phase != 'confirm_estimate':
            self.status.set('먼저 이 지점에서 보정 각도를 계산하세요.')
            return
        try:
            if self.station != self.inputs.snapshot():
                raise ValueError('계산 후 지점 입력값이 바뀌었습니다. 새 지점을 등록해 다시 계산하세요.')
            self.check_station()
            pan, tilt = self.estimated_goal
            if (int(self.pan.get()), int(self.tilt.get())) != (pan-HOME_PAN, tilt-HOME_TILT):
                raise ValueError('계산 후 표시 각도가 수정되어 비교 측정을 중단했습니다.')
            self.phase = None
            super().transmit()
            if self.phase != 'off':
                raise ValueError(self.status.get())
            self.run.log(dict(event='optical_angle_command', token=self.token, pan=pan, tilt=tilt))
        except (OSError, ValueError) as exc:
            self.stop(str(exc))

    def transmit(self):
        if self.phase == 'confirm_estimate':
            self.send_estimate()
        elif self.method is None:
            super().transmit()

    def home(self):
        if self.phase == 'speed':
            self.status.set('속도 측정을 중지한 뒤 원위치 버튼을 누르세요.')
            return
        if self.method is not None:
            self.stop('원위치 버튼으로 비교 측정 중단')
        super().home()

    def stop(self, reason):
        if self.phase == 'speed':
            self.speed.stop()
            return
        method, self.method = self.method, None
        # Attempt remote cancel/OFF even when logging is the reason we got here.
        if method and not self.optical and self.token:
            if self.rx and self.rx.connected:
                try:
                    self.rx.send('cancel', self.token)
                except OSError:
                    pass
            self.off_token = self.token + '-stop'
        try:
            super().stop(reason)
        finally:
            if method:
                self.inputs.cancel_yolo()
                try:
                    self.run.finish(False, reason)
                except (OSError, ValueError) as exc:
                    self.status.set(f'{reason} · 기록 오류 {exc}')
                self.results[method].set(f'{METHODS[method]} 실패 · {reason}')

    def finish_if_confirmed(self):
        complete = self.method is not None and self.tx_done and self.rx_done
        super().finish_if_confirmed()
        if complete:
            self.begin('pose_settle')
            self.until = time.monotonic() + SERVO_SETTLE_S
            self.status.set('두 축 서보 명령 적용 확인 · 정착 대기')

    def start_light(self):
        self.app.ook.check_ready()
        self.begin('light_start')
        self.measured_ok = self.light_done = False
        self.run.log(dict(event='fixed_light_request', token=self.token, **FIXED_LIGHT))
        self.safe_end_ns = time.monotonic_ns() + FIXED_LIGHT['bit_ms'] * 1_000_000
        if not self.app.send(dict(cmd='laser_pattern', request_id=self.token, **FIXED_LIGHT)):
            raise OSError('고정광 요청 실패')
        self.status.set('고정광 ON 응답 대기')

    def on_rx(self, event):
        if event['kind'] in ('started', 'sample', 'stopped'):
            return  # A late raw-speed reply can follow an operator stop.
        if self.method is None or event.get('token') != self.token:
            # Never pass a late batch of raw samples to the demonstration event log.
            if event['kind'] not in ('voltage', 'measured'):
                super().on_rx(event)
            return
        kind = event['kind']
        if kind == 'voltage':
            if self.phase == 'voltage_window':
                self.run.sample(event)
        elif kind == 'measured' and self.phase == 'voltage_window':
            self.run.measured(event)
            if event['pc_receive_monotonic_ns'] >= self.safe_end_ns:
                raise ValueError('Rx 측정 완료가 보장된 고정광 시간 범위를 벗어났습니다.')
            self.measured_ok = True
            self.finish_measurement()
        else:
            self.run.log(event)
            if kind == 'home' and self.phase == 'bench_home':
                if (event.get('pan'), event.get('tilt')) != (HOME_PAN, HOME_TILT):
                    raise ValueError('Rx 원위치 각도 불일치')
                self.applied_pose(HOME_PAN, HOME_TILT)
                self.phase = 'home_settle'
                self.until = time.monotonic() + SERVO_SETTLE_S
                self.deadline = self.until + 5
                self.status.set('Rx 원위치 확인 · 정착 대기')
            else:
                super().on_rx(event)

    def finish_measurement(self):
        if not (self.light_done and self.measured_ok):
            return
        method = self.method
        row = self.run.finish(True, '')
        self.method = self.phase = self.token = None
        if not row['success']:
            self.results[method].set(f'{METHODS[method]} 실패 · {row["reason"]}')
            self.status.set(row['reason'])
            return
        delta = row['delta_v']
        label = f'{METHODS[method]} {row["mean_v"]:.3f} V'
        if method != 'baseline':
            label += f' (Δ {delta:+.3f} V)' if delta is not None else ' (무보정 기준 미측정)'
        else:
            # Existing deltas on disk remain paired with their original baseline.
            for key in ('xyz', 'yolo'):
                self.results[key].set(METHODS[key] + ' — 현재 기준으로 미측정')
        self.results[method].set(label)
        self.status.set(f'저장 완료 · {label} · 표준편차 {row["sd_v"]:.3f} V · {row["n"]}개 표본')

    def light_event(self, event):
        kind = event['event']
        if kind == 'error':
            raise OSError(event.get('message', '고정광 송신 오류'))
        if kind == 'laser_pattern' and event.get('state') == 'started' and self.phase == 'light_start':
            if event.get('simulated') is not False:
                raise ValueError('모의 광원으로 전압 측정을 시작할 수 없습니다.')
            self.phase = 'light_check'
            self.next_probe = 0
        elif kind == 'hardware' and self.phase == 'light_check':
            if (event.get('simulated') is False and event.get('pattern_request_id') == self.token
                    and event.get('pattern_active') and event.get('laser_gpio_level') == 1):
                self.phase = 'voltage_window'
                self.rx.send('measure', self.token)
                self.status.set('고정광 ON 확인 · INA219 전압 측정 중')
        elif kind == 'laser_pattern' and event.get('state') != 'started':
            if (self.phase != 'voltage_window' or event.get('state') != 'completed'
                    or event.get('settings') != FIXED_LIGHT or event.get('simulated') is not False
                    or not isinstance(event.get('trace'), list) or len(event['trace']) != 1
                    or not isinstance(event['trace'][0], dict) or event['trace'][0].get('level') != 1
                    or not isinstance(event.get('off_call_done_ms'), (float, int))
                    or not math.isfinite(event['off_call_done_ms'])
                    or event['off_call_done_ms'] < FIXED_LIGHT['bit_ms']):
                raise ValueError('고정광 유지 시간 또는 종료 OFF를 확인하지 못했습니다.')
            self.light_done = True
            self.finish_measurement()

    def event(self, event):
        if (self.off_token and event.get('request_id') == self.off_token
                and event.get('event') == 'laser' and event.get('laser_gpio_level') == 0):
            self.app.ook.output_requested = False
        try:
            if self.method is not None:
                kind = event.get('event')
                if (kind in ('outputs_off', 'hello', 'ready', 'agent')
                        or (kind == 'network' and event.get('port') == 7600 and event.get('state') != 'connected')):
                    raise OSError('Tx 세션이 변경되어 측정을 중단했습니다.')
                if kind in ('laser', 'laser_pattern') and event.get('request_id') != self.token:
                    raise ValueError('다른 광출력 명령이 들어와 측정을 중단했습니다.')
                if event.get('request_id') == self.token:
                    self.run.log(event)
                    if self.phase in ('light_start', 'light_check', 'voltage_window'):
                        self.light_event(event)
                        return
                    if kind == 'laser' and not self.optical:
                        raise ValueError('측정 중 광출력 상태가 변경됐습니다.')
            super().event(event)
        except (ValueError, OSError) as exc:
            self.stop(str(exc))

    def tick_comparison(self):
        self.check_station()
        now = time.monotonic()
        if self.phase == 'home_settle' and now >= self.until:
            if self.method == 'baseline':
                self.start_light()
            elif self.method == 'xyz':
                self.estimate_ready(self.inputs.estimate_xyz(self.station))
            elif self.app.current[2] >= self.until:
                frame = self.app.current
                self.run.save_frame(frame[0])
                self.inputs.start_yolo(self.station, frame)
                self.phase = 'vision'
                self.deadline = now + 120
                self.status.set('원위치 영상에서 PV 검출 중…')
        elif self.phase == 'vision':
            result = self.inputs.poll_yolo()
            if result is not None:
                self.estimate_ready(result)
        elif self.phase == 'pose_settle' and now >= self.until:
            self.start_light()
        elif self.phase == 'light_check' and now >= self.next_probe:
            if time.monotonic_ns() + VOLTAGE_SETTLE_NS + VOLTAGE_WINDOW_NS >= self.safe_end_ns:
                raise ValueError('고정광 ON 확인이 늦어 전압 측정을 중단했습니다.')
            self.next_probe = now + 0.10
            if not self.app.send(dict(cmd='hardware_status', request_id=self.token)):
                raise OSError('고정광 상태 조회 실패')

    def tick(self):
        if self.speed is not None:
            self.speed.tick()
            if self.speed.running:
                return
        super().tick()
        if self.selected and self.method is not None:
            try:
                self.tick_comparison()
            except (ValueError, OSError) as exc:
                self.stop(str(exc))

    def leave(self):
        if self.speed is not None and self.speed.running:
            self.speed.stop()
            pending = self.app.ook.pending
            self.speed.release()
            if pending:
                self.off_token = pending[0]
                self.off_status.set('속도 측정 중지 · Tx OFF 확인 대기')
            else:
                self.off_token = 'm6-speed-' + self.speed.last_root.name + '-stop'
                self.stop('속도 측정 중지 · Tx OFF 재확인')
        super().leave()

    def close(self):
        try:
            self.leave()
        finally:
            self.inputs.close()
