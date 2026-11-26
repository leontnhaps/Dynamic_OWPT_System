"""Operator-triggered pose and voltage comparison at a stationary PV receiver."""
import time
import tkinter as tk
from tkinter import ttk

from common.angle_demo import FRAME_LEN, HOME_PAN, HOME_TILT, encode_angles, payload_from_frame
from common.ook_experiment import SYNC_WORD, THRESHOLD_BITS, TRAILING_ZEROS
from Tx.Controller.angle_demo_panel import AngleDemoPanel
from Tx.Controller.bench_pose import BenchPoseInputs, frame_pose
from Tx.Controller.final_speed_panel import FinalSpeedPanel
from Tx.Controller.pv_comparison import ComparisonRun
from Tx.Controller.station_tx_controls import StationTxControls


METHODS = {'baseline': '원위치', 'xyz': '실측 XYZ', 'yolo': 'YOLO'}
SERVO_SETTLE_S = 1.0
MEASURING = ('pose_settle', 'light_before', 'voltage_window', 'light_after')
YOLO_SETTINGS = {'tx_pan_plus', 'tx_tilt_plus', 'tx_ref_pan_deg', 'tx_ref_tilt_deg', 'camera_intrinsics_px'}


def pose_label(pan, tilt):
    return (f'상대 {pan-HOME_PAN:+d}° / {tilt-HOME_TILT:+d}°'
            f'  (서보 {pan}° / {tilt}°)')


class PointComparisonPanel(AngleDemoPanel):
    speed = tx_controls = None

    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.run = self.station = self.station_view = self.method = None
        self.pose_method = self.pose_source = self.pose_frame = None
        self.pose_ready_at = 0
        self.yolo_results_since = 0
        self.inputs = BenchPoseInputs(self, app)
        self.inputs.pack(fill='x', pady=3)
        self.tx_controls = StationTxControls(self, self)
        self.tx_controls.pack(fill='x')
        actions = ttk.Frame(self)
        actions.pack(anchor='w', pady=(4, 0))
        for method in ('xyz', 'yolo'):
            ttk.Button(actions, text=f'{METHODS[method]} 각도로 이동 · OOK',
                       command=lambda m=method: self.move_to(m)).pack(side='left', padx=3)
        ttk.Button(actions, text='현재 전압 10초 측정', command=self.measure).pack(side='left', padx=(9, 2))
        self.active_condition = tk.StringVar(self)
        ttk.Label(actions, textvariable=self.active_condition).pack(side='left', padx=6)
        self.targets = {m: tk.StringVar(self) for m in ('xyz', 'yolo')}
        for m, var in self.targets.items():
            row = ttk.Frame(self)
            row.pack(anchor='w')
            ttk.Label(row, text=f'{METHODS[m]} 목표 Pan / Tilt', width=22).pack(side='left')
            ttk.Label(row, textvariable=var).pack(side='left')
        self.location = tk.StringVar(self)
        ttk.Label(self, textvariable=self.location).pack(anchor='w')
        results = ttk.Frame(self)
        results.pack(anchor='w')
        self.results = {}
        for column, method in enumerate(METHODS):
            var = self.results[method] = tk.StringVar(self)
            ttk.Label(results, textvariable=var).grid(row=0, column=column, padx=(0, 16), sticky='w')
        self.inputs.on_change = self.station_changed
        self.station_changed()
        self.frame_label = tk.StringVar(self)
        self.speed = FinalSpeedPanel(self)
        extra = ttk.Frame(self)
        extra.pack(anchor='w')
        self.extra_button = ttk.Button(extra, text='OOK 속도 시험 · 프레임 보기', command=self.toggle_extra)
        self.extra_button.pack(side='left')
        self.frame_info = ttk.Label(self, textvariable=self.frame_label)
        axis = ttk.Frame(self.diagnostics)
        axis.pack(anchor='w')
        ttk.Label(axis, text='OOK 단일 축 점검 (원위치 기준)').pack(side='left')
        for name in ('Pan', 'Tilt'):
            for offset in (-1, 1):
                ttk.Button(axis, text=f'{name} {offset:+d}°', width=8,
                           command=lambda a=name, d=offset: self.check_axis(a, d)).pack(side='left')
        self.pan.trace_add('write', self.show_frame)
        self.tilt.trace_add('write', self.show_frame)
        self.rate_box.bind('<<ComboboxSelected>>', self.show_frame)
        self.show_frame()

    def toggle_extra(self):
        if self.speed.winfo_manager():
            self.speed.pack_forget()
            self.frame_info.pack_forget()
        else:
            self.frame_info.pack(anchor='w')
            self.speed.pack(fill='x', pady=3)

    def show_frame(self, *_):
        try:
            bits = encode_angles(HOME_PAN + int(self.pan.get()), HOME_TILT + int(self.tilt.get()))
        except ValueError:
            self.frame_label.set('각도 프레임 · 범위 안의 정수 각도를 입력하세요.')
            return
        payload = payload_from_frame(bits)
        bit_ms = self.selected_period()
        self.frame_label.set(f'{FRAME_LEN} bit · {1000/bit_ms:g} bit/s ({bit_ms} ms/bit) · '
                             f'레벨 {THRESHOLD_BITS} | 동기 {SYNC_WORD} | Pan {payload[:16]} | '
                             f'Tilt {payload[16:]} | 끝 {TRAILING_ZEROS}')

    @property
    def owns_output(self):
        return (self.method is not None or self.phase == 'speed' or self.optical or bool(self.off_token)
                or bool(self.tx_controls and self.tx_controls.owns_output))

    def allows_pattern(self, request_id):
        if not request_id:
            return False
        if self.phase == 'speed':
            ook = self.app.ook
            return ook.experiment is not None and ook.capture is not None and request_id == ook.trial_id
        return self.optical and self.token == request_id

    def ready_tx_motion(self):
        if self.phase or self.off_token:
            raise ValueError('현재 동작과 응답 확인이 끝나면 조작할 수 있습니다.')

    def ready_action(self):
        self.ready_tx_motion()
        if self.tx_controls and self.tx_controls.busy:
            raise ValueError('Tx 조작 응답을 기다려 주세요.')
        self.ready_rx()

    def check_axis(self, axis, offset):
        if self.phase or self.off_token:
            self.status.set('현재 동작이 끝나면 축 방향을 확인할 수 있습니다.')
            return
        self.pan.set(str(offset if axis == 'Pan' else 0))
        self.tilt.set(str(offset if axis == 'Tilt' else 0))
        self.transmit()

    def station_changed(self, force_new=False):
        if self.method is not None:
            self.stop('지점 또는 보정 설정이 변경되어 이번 동작을 중지했습니다.')
        try:
            station = self.inputs.snapshot()
        except ValueError as exc:
            # A partially typed entry must not discard earlier measurements.
            # All actions that can capture data validate the completed snapshot.
            self.targets['xyz'].set(str(exc))
            self.location.set('지점과 좌표를 확인하세요.')
            return
        if self.station is not None and not force_new:
            changed = {key for key in station.keys() | self.station.keys()
                       if station.get(key) != self.station.get(key)}
            if changed <= YOLO_SETTINGS:
                if changed:
                    self.targets['yolo'].set('변경한 보정값으로 다시 계산하세요.')
                    if self.pose_method == 'yolo':
                        self.pose_method = self.pose_source = self.pose_frame = None
                    self.yolo_results_since = len(self.run.results)
                    self.station = station
                    try:
                        self.run.log_station(dict(event='yolo_settings_updated', settings=station))
                    except OSError as exc:
                        self.status.set(f'보정값 기록 오류 · {exc}')
                self.preview_xyz(station)
                self.show_results()
                return
        self.run = self.station = self.station_view = None
        self.pose_method = self.pose_source = self.pose_frame = None
        self.yolo_results_since = 0
        self.targets['yolo'].set('현재 지점 영상으로 계산 전')
        self.preview_xyz(station)
        self.location.set(f'지점 {station["point_id"]} · Tx 조준 후 Rx (0°, 0°) 복귀')
        self.show_results()

    def preview_xyz(self, station):
        try:
            pan, tilt, _ = self.inputs.estimate_xyz(station)
            self.targets['xyz'].set(pose_label(pan, tilt))
        except ValueError as exc:
            self.targets['xyz'].set(str(exc))

    def tx_moved(self):
        # A new aim starts a fresh comparison even at the same numbered point.
        self.station_changed(force_new=True)

    def ensure_station(self):
        snapshot = self.inputs.snapshot()
        view = self.current_view()
        if self.run is None:
            self.run = ComparisonRun(self.app.stage_dir('M6-4'), snapshot)
            self.station, self.station_view = snapshot, view
            self.show_results()
        elif snapshot != self.station or view != self.station_view:
            self.station_changed(force_new=True)
            raise ValueError('Tx 조준 또는 지점이 바뀌었습니다. Rx (0°, 0°)에서 새 측정을 시작하세요.')
        return view

    def current_view(self):
        session, command_id, pan, tilt = frame_pose(self.app.current)
        if self.tx_controls:
            self.tx_controls.check_frame(self.app.current)
        return dict(session=session, pan=pan, tilt=tilt, command_id=command_id)

    def check_station(self):
        if self.inputs.snapshot() != self.station or self.current_view() != self.station_view:
            raise ValueError('측정 중 Tx 조준, 연결 또는 지점이 바뀌었습니다.')

    def show_results(self):
        completed = {}
        if self.run:
            for index, row in enumerate(self.run.results):
                if row.get('success') and (row['method'] != 'yolo' or index >= self.yolo_results_since):
                    completed[row['method'], row['laser_on']] = row['mean_v']
            self.location.set(f'지점 {self.station["point_id"]} · {len(completed)}/6조건 측정 · '
                              f'{self.run.folder.parent.name}/{self.run.folder.name}')
        for key, var in self.results.items():
            levels = []
            for state in (False, True):
                v = completed.get((key, state))
                levels.append('—' if v is None else f'{v:.3f} V')
            var.set(f'{METHODS[key]}  OFF {levels[0]}  |  ON {levels[1]}')

    def home(self):
        # Independent SSH operation, including when Tx is disconnected.
        try:
            if self.phase or self.off_token:
                raise ValueError('진행 중인 동작을 종료한 후 Rx 원위치로 복귀하세요.')
            self.ready_rx()
        except ValueError as exc:
            self.status.set(str(exc))
            return
        self.pose_method = self.pose_source = self.pose_frame = None
        super().home()

    def move_to(self, method):
        started = False
        try:
            self.ready_action()
            if self.wave is not None:
                raise ValueError('이전 송신의 Rx 파형을 저장하고 있습니다.')
            if self.tx_controls.light_level():
                raise ValueError('Laser OFF를 누른 뒤 계산각을 OOK로 전송하세요.')
            self.app.ook.check_ready()
            self.ensure_station()
            self.method = method
            started = True
            self.pose_method = self.pose_source = self.pose_frame = None
            if method == 'xyz':
                self.send_estimate(self.inputs.estimate_xyz(self.station))
            else:
                self.begin('vision')
                self.deadline = time.monotonic() + 120
                self.pose_frame = self.app.current[0]
                self.inputs.start_yolo(self.station, self.app.current)
                self.status.set('현재 영상에서 PV 검출 중 · 계산 후 Rx에 OOK 전송')
        except (ValueError, OSError) as exc:
            if started:
                self.stop(str(exc))
            else:
                self.status.set(str(exc))

    def send_estimate(self, result):
        pan, tilt, meta = result
        self.check_station()
        self.targets[self.method].set(pose_label(pan, tilt))
        self.pose_source = dict(method=self.method, estimate=meta, tx_view=self.station_view,
                                pose_settings=self.station)
        self.run.log_station(dict(event='estimate', **self.pose_source, pan=pan, tilt=tilt))
        self.pan.set(str(pan-HOME_PAN))
        self.tilt.set(str(tilt-HOME_TILT))
        self.phase = None
        super().transmit()
        if self.phase != 'off':
            raise ValueError(self.status.get())
        self.run.log_station(dict(event='optical_angle_command', token=self.token, method=self.method,
                                  pan=pan, tilt=tilt, bit_ms=self.bit_ms, waveform_folder=str(self.wave.folder)))

    def waveform_context(self):
        context = super().waveform_context()
        if self.method in ('xyz', 'yolo'):
            context.update(method=self.method, station=self.station, point_id=self.station['point_id'],
                           station_folder=str(self.run.folder))
        return context

    def transmit(self):
        if self.phase or self.off_token:
            self.status.set('현재 동작이 끝나면 전송할 수 있습니다.')
            return
        if self.tx_controls and (self.tx_controls.busy or self.tx_controls.owns_output):
            self.status.set('Laser OFF 및 Tx 응답 확인 후 전송하세요.')
            return
        self.pose_method = self.pose_source = self.pose_frame = None
        super().transmit()

    def applied_pose(self, pan, tilt):
        super().applied_pose(pan, tilt)
        self.pose_method = 'baseline' if self.phase == 'home' else None
        self.pose_ready_at = time.monotonic() + SERVO_SETTLE_S

    def clear_servo_pose(self, reason):
        super().clear_servo_pose(reason)
        self.pose_method = self.pose_source = self.pose_frame = None

    def record_verification(self, event):
        super().record_verification(event)
        if self.run is not None and self.method in ('xyz', 'yolo'):
            self.run.log_station(dict(event, method=self.method))

    def finish_if_confirmed(self):
        method, token = self.method, self.token
        super().finish_if_confirmed()
        if method in ('xyz', 'yolo') and self.phase is None and self.tx_done and self.rx_done:
            self.pose_method, self.method = method, None
            self.pose_source['ook_token'] = token
            self.tx_controls.probe_light(token + '-pose-ready')
            self.status.set(f'{METHODS[method]} · {pose_label(*self.last_servo_pose)} 적용 확인 · '
                            '현재 전압 10초 측정을 누르세요.')

    def measure(self):
        started = False
        try:
            self.ready_action()
            if not self.pose_method or self.last_servo_pose is None:
                raise ValueError('먼저 Rx (0°, 0°) 복귀 또는 계산각 이동의 응답을 확인하세요.')
            view = self.ensure_station()
            level = self.tx_controls.light_level()
            pan, tilt = self.last_servo_pose
            self.run.start(self.pose_method, dict(tx_at_start=view, rx_pan_deg=pan, rx_tilt_deg=tilt,
                           pose_source=self.pose_source, home=[HOME_PAN, HOME_TILT],
                           servo_settle_s=SERVO_SETTLE_S), laser_on=bool(level))
            self.method = self.pose_method
            self.begin('pose_settle')
            started = True
            self.measure_level = level
            if self.pose_frame is not None and self.method == 'yolo':
                self.run.save_frame(self.pose_frame)
            self.status.set('Rx 자세 정착 후 현재 조명 상태에서 10초간 측정합니다.')
            self.tick_measure()
        except (OSError, ValueError) as exc:
            if started:
                self.stop(str(exc))
            else:
                self.status.set(str(exc))

    def tick_measure(self):
        if self.phase == 'pose_settle' and time.monotonic() >= self.pose_ready_at:
            self.phase = 'light_before'
            self.deadline = time.monotonic() + 5
            self.tx_controls.probe_light(self.token + '-before')
        elif self.phase == 'voltage_window':
            elapsed = min(10, max(0, time.monotonic() - self.window_request - .5))
            line = (f'{METHODS[self.method]} · Laser {"ON" if self.measure_level else "OFF"} · '
                    f'전압 측정 {elapsed:.1f} / 10초')
            if self.status.get() != line:
                self.status.set(line)

    def receive_light(self, event):
        before = self.phase == 'light_before'
        expected = self.token + ('-before' if before else '-after')
        if event.get('request_id') != expected:
            return
        if event.get('event') == 'error':
            raise ValueError('Tx 조명 상태 조회 실패 · ' + event.get('message', ''))
        state = self.tx_controls.confirmed_light(event, expected)
        self.run.log(event)
        if before:
            if state[0] != self.measure_level:
                raise ValueError('측정 버튼을 누른 뒤 레이저 상태가 변경됐습니다.')
            self.light_at_start = state
            self.phase = 'voltage_window'
            self.window_request = time.monotonic()
            self.deadline = self.window_request + 18
            self.rx.send('measure', self.token)
        elif state != self.light_at_start:
            raise ValueError('10초 수집 중 Tx GPIO 상태가 변경되어 이 측정을 실패로 기록했습니다.')
        else:
            self.check_station()
            row = self.run.finish(True, '')
            self.phase = self.token = self.method = None
            self.show_results()
            if row['success']:
                self.status.set(f'{METHODS[row["method"]]} · Laser {"ON" if row["laser_on"] else "OFF"} · '
                                f'10초 평균 {row["mean_v"]:.3f} V · 표준편차 {row["sd_v"]:.3f} V · '
                                f'{row["n"]}개 표본 저장')
            else:
                self.status.set(row['reason'])

    def on_rx(self, event):
        kind = event['kind']
        if kind in ('started', 'sample', 'stopped'):
            return
        if kind in ('voltage', 'measured'):
            if self.phase == 'voltage_window' and event.get('token') == self.token:
                if kind == 'voltage':
                    self.run.sample(event)
                else:
                    self.run.measured(event)
                    self.phase = 'light_after'
                    self.deadline = time.monotonic() + 5
                    self.tx_controls.probe_light(self.token + '-after')
            return
        super().on_rx(event)

    def event(self, event):
        if self.tx_controls:
            self.tx_controls.event(event)
        if (self.off_token and event.get('request_id') == self.off_token
                and event.get('event') == 'laser' and event.get('laser_gpio_level') == 0):
            self.app.ook.output_requested = False
        try:
            if self.method is not None:
                kind = event.get('event')
                if (kind in ('outputs_off', 'hello', 'ready', 'agent')
                        or (kind == 'network' and event.get('port') == 7600 and event.get('state') != 'connected')):
                    raise ValueError('Tx 세션 또는 출력 상태가 변경되어 동작을 중지했습니다.')
                if kind == 'servo' and event.get('operation') in ('move', 'servo_config'):
                    raise ValueError('Tx 자세 또는 서보 설정이 변경되어 동작을 중지했습니다.')
                if self.phase in MEASURING:
                    if kind in ('laser', 'laser_pattern', 'ir_cut', 'move'):
                        raise ValueError('측정 중 Tx 자세 또는 광출력 명령이 변경됐습니다.')
                    if self.phase in ('light_before', 'light_after'):
                        self.receive_light(event)
                    return
            super().event(event)
        except (ValueError, OSError) as exc:
            self.stop(str(exc))

    def stop(self, reason):
        if self.phase == 'speed':
            self.speed.stop()
            return
        was_measuring = self.phase in MEASURING
        if reason.startswith('응답 시간 초과'):
            if self.phase in ('light_before', 'light_after'):
                reason = 'Tx 조명 상태 조회 응답 시간 초과'
            elif self.phase == 'voltage_window':
                reason = 'Rx의 10초 전압 수집 완료 응답 시간 초과'
            elif self.phase == 'vision':
                reason = '현재 지점의 YOLO 검출 응답 시간 초과'
        if self.phase == 'vision':
            self.inputs.cancel_yolo()
        if self.phase == 'voltage_window' and self.rx and self.rx.connected:
            try:
                self.rx.send('cancel', self.token)
            except OSError:
                pass
        try:
            super().stop(reason)
        finally:
            self.method = None
            if was_measuring and self.run and self.run.trial_folder:
                try:
                    self.run.finish(False, reason)
                except (OSError, ValueError) as exc:
                    self.status.set(f'{reason} · 기록 오류 {exc}')

    def tick(self):
        if self.tx_controls and self.selected:
            self.tx_controls.tick()
            if hasattr(self, 'active_condition'):
                method = METHODS.get(self.pose_method, '자세 미확인')
                light = {0: 'OFF', 1: 'ON'}.get(self.tx_controls.light, '미확인')
                text = f'현재 조건 · {method} / Laser {light}'
                if self.active_condition.get() != text:
                    self.active_condition.set(text)
        if self.speed:
            self.speed.tick()
            if self.speed.running:
                return
        super().tick()
        if self.selected and self.method is not None:
            try:
                self.check_station()
                if self.phase == 'vision':
                    result = self.inputs.poll_yolo()
                    if result is not None:
                        self.send_estimate(result)
                elif self.phase in MEASURING:
                    self.tick_measure()
            except (ValueError, OSError) as exc:
                self.stop(str(exc))

    def leave(self):
        if self.speed and self.speed.running:
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
        if self.tx_controls:
            self.tx_controls.leave()

    def close(self):
        try:
            self.leave()
        finally:
            self.release_wave(closing=True)
            self.wave_review.close()
            self.inputs.close()
