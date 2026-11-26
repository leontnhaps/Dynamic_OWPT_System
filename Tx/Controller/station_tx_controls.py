"""Operator controls for positioning the stationary transmitter and choosing the light state."""
import time
import tkinter as tk
import uuid
from tkinter import ttk

from common.servo import limits_from, move_from
from common.tx_setup import TX_OPERATING_LIMITS


class StationTxControls(ttk.LabelFrame):
    def __init__(self, parent, panel):
        super().__init__(parent, text='Tx 수동 조절', padding=4)
        self.panel = panel
        self.app = panel.app
        self.light = self.light_request = self.motion = None
        self.manual_on = False
        self.recover_off = False
        self.frame_target = None
        self.session_reset_at = None
        self.step = tk.IntVar(self, value=1)
        self.notice = tk.StringVar(self, value='레이저 OFF에서 Tx 자세를 조절하세요.')
        self.readout = tk.StringVar(self, value='Tx GPIO 상태 미확인')
        row = ttk.Frame(self)
        row.pack(anchor='w')
        ttk.Button(row, text='Tx 0°, 0°', command=lambda: self.jog(None)).pack(side='left')
        for degrees in (1, 5):
            ttk.Radiobutton(row, text=f'{degrees}°', variable=self.step, value=degrees).pack(side='left')
        for axis in ('pan', 'tilt'):
            for delta, sign in ((-1, '−'), (1, '+')):
                ttk.Button(row, text=f'Tx {axis.title()} {sign}',
                           command=lambda a=axis, d=delta: self.jog(a, d)).pack(side='left', padx=2)
        ttk.Button(row, text='레이저 ON', command=self.on).pack(side='left', padx=(10, 2))
        ttk.Button(row, text='레이저 OFF', command=self.off).pack(side='left')
        ttk.Button(row, text='Tx 상태 조회', command=self.refresh).pack(side='left', padx=2)
        ttk.Label(row, textvariable=self.readout).pack(side='left', padx=5)
        ttk.Label(self, textvariable=self.notice).pack(anchor='w')

    @property
    def busy(self):
        return self.motion is not None or self.light_request is not None

    @property
    def owns_output(self):
        return self.manual_on or self.light_request is not None

    def _real_tx(self):
        e = self.app.ook
        if not e.online or not e.hardware or e.hardware.get('simulated') is not False:
            raise ValueError('실제 Tx 연결과 상태 응답을 확인하세요.')
        frame = self.app.current
        if not frame or frame[1].get('simulated') is not False or time.monotonic() - frame[2] > 2:
            raise ValueError('실제 Tx의 최근 카메라 영상이 필요합니다.')
        return e

    def probe_light(self, token):
        self._real_tx()
        if self.busy:
            raise ValueError('Tx 수동 명령 응답을 기다리세요.')
        if not self.app.send(dict(cmd='hardware_status', request_id=token)):
            raise OSError('Tx GPIO 상태 조회 전송 실패')

    def refresh(self):
        try:
            e = self.app.ook
            if self.busy or e.pending or e.experiment or self.panel.phase:
                raise ValueError('진행 중인 요청이 끝나면 Tx 상태를 조회할 수 있습니다.')
            # A read-only recovery query must also work when an OFF timeout has
            # invalidated cached hardware state or the preview has stopped.
            if not e.online or not e.request(dict(cmd='hardware_status')):
                raise OSError('Tx GPIO 상태 조회 전송 실패')
        except (ValueError, OSError) as exc:
            self.notice.set(str(exc))

    @staticmethod
    def confirmed_light(event, token):
        level, revision = event.get('laser_gpio_level'), event.get('gpio_revision')
        if (not token or event.get('request_id') != token or event.get('event') != 'hardware'
                or event.get('simulated') is not False or event.get('pattern_active') is not False
                or type(level) is not int or level not in (0, 1)
                or type(revision) is not int or revision < 0):
            raise ValueError('실제 Tx에서 고정된 GPIO 상태를 확인하지 못했습니다.')
        return level, revision

    def light_level(self):
        self._real_tx()
        if self.busy or self.light is None:
            raise ValueError('Tx 레이저 상태를 먼저 확인하세요.')
        return self.light

    def check_frame(self, frame):
        if self.session_reset_at is not None and frame[2] <= self.session_reset_at:
            raise ValueError('재연결한 Tx의 새 영상을 기다리세요.')
        if self.frame_target is None:
            return
        token, pan, tilt = self.frame_target
        pose = (frame[1].get('servo_at_capture_start') or {}).get('commanded') or {}
        if (pose.get('command_id'), pose.get('pan'), pose.get('tilt')) != (token, pan, tilt):
            raise ValueError('Tx가 새 명령각에서 촬영한 영상을 기다리세요.')
        self.frame_target = None

    def _light(self, level):
        self.light = level
        label = {None: '미확인', 0: 'OFF', 1: 'ON'}[level]
        self.readout.set('Tx GPIO ' + label)

    def on(self):
        try:
            self.panel.ready_tx_motion()
            e = self._real_tx()
            if self.light_level() != 0 or self.owns_output:
                raise ValueError('Tx 레이저 OFF가 확인된 뒤 켤 수 있습니다.')
            e.laser_on()
            if not e.pending or e.pending[2] != 'laser':
                raise OSError('Tx ON 요청 전송 실패')
            self.manual_on = True
            self.light_request = (e.pending[0], 1, time.monotonic())
            self._light(None)
            self.notice.set('Tx ON 응답 대기')
        except (ValueError, OSError) as exc:
            self.notice.set(str(exc))

    def off(self):
        try:
            if self.panel.phase or self.panel.method:
                self.panel.stop('수동 OFF로 진행 중인 동작을 중단했습니다.')
        finally:
            self._send_off()

    def _send_off(self):
        self.motion = None
        self.manual_on = True  # Keep output ownership until the Pi confirms OFF.
        self._light(None)
        e = self.app.ook
        try:
            if not e.online:
                raise OSError('Tx 연결 끊김 · OFF 미확인')
            if not e.pending or e.pending[2] != 'laser_off':
                e.off()
            if not e.pending or e.pending[2] != 'laser_off':
                raise OSError('Tx OFF 요청 전송 실패')
            self.light_request = (e.pending[0], 0, time.monotonic())
            self.notice.set('Tx OFF 응답 대기')
        except (ValueError, OSError) as exc:
            self.light_request = None
            self.notice.set(str(exc))

    def leave(self):
        self.motion = None
        if self.owns_output:
            self._send_off()

    def jog(self, axis, delta=0):
        try:
            self.panel.ready_tx_motion()
            e = self._real_tx()
            e.check_ready()
            if self.busy or self.owns_output or (axis is not None and
                    (axis not in ('pan', 'tilt') or delta not in (-1, 1) or self.step.get() not in (1, 5))):
                raise ValueError('Tx 수동 조절 설정과 이전 요청의 응답을 확인하세요.')
            self.motion = dict(axis=axis, step=delta * self.step.get())
            # Fetch live GPIO/servo state before every jog. The previous video frame
            # can still contain the pose from *before* the last serial write.
            self._motion_request('hardware_status')
        except (ValueError, OSError) as exc:
            self.notice.set(str(exc))

    def _motion_request(self, command, **settings):
        token = 'm6-tx-' + uuid.uuid4().hex
        self.motion.update(token=token, cmd=command, sent=time.monotonic())
        if command == 'move':
            # A move may have reached the Pi even if its acknowledgement was lost.
            self.frame_target = (token, settings['pan'], settings['tilt'])
        if not self.app.send(dict(cmd=command, request_id=token, **settings)):
            self.motion = None
            raise OSError('Tx 수동 조절 요청 전송 실패')
        self.notice.set(f'Tx {command} 응답 대기')

    def _motion_event(self, event):
        m = self.motion
        if event.get('event') == 'error':
            raise ValueError(event.get('message', 'Tx 요청 오류'))
        self._real_tx()
        if m['cmd'] == 'hardware_status':
            level, _ = self.confirmed_light(event, m['token'])
            if level != 0:
                raise ValueError('Tx 레이저를 끈 뒤 자세를 조절하세요.')
            self._motion_request('servo_status')
            return
        if (event.get('event') != 'servo' or event.get('operation') != m['cmd']
                or event.get('simulated') is not False or event.get('available') is not True):
            raise ValueError('실제 Tx 서보 응답을 확인할 수 없습니다.')
        if self.light != 0:
            raise ValueError('Tx GPIO 상태가 바뀌어 자세 조절을 중단했습니다.')
        if m['cmd'] == 'move':
            sent = event.get('commanded') or {}
            if (event.get('state') != 'serial_written' or sent.get('command_id') != m['token']
                    or any(sent.get(k) != v for k, v in m['target'].items())):
                raise ValueError('Tx 서보의 명령 전송 결과가 요청과 다릅니다.')
            self.motion = None
            self.notice.set(f"Tx 전송 명령각 · Pan {sent['pan']:g}° · Tilt {sent['tilt']:g}°")
            return
        limits = event.get('limits')
        if limits is None and m['cmd'] == 'servo_status':
            self._motion_request('servo_config', **TX_OPERATING_LIMITS)
            return
        if not isinstance(limits, dict):
            raise ValueError('Tx 서보 운용 범위를 확인할 수 없습니다.')
        limits = limits_from(limits)
        pose = dict(pan=0, tilt=0, speed=100, acc=1)
        if m['axis'] is not None:
            previous = event.get('commanded') or {}
            if not previous.get('command_id') or any(type(previous.get(k)) not in (int, float) for k in ('pan', 'tilt')):
                raise ValueError('Tx 직전 명령각이 없습니다. 먼저 Tx 0°, 0°를 누르세요.')
            pose.update(pan=previous['pan'], tilt=previous['tilt'])
            pose[m['axis']] += m['step']
        target = move_from(move_from(pose, limits), TX_OPERATING_LIMITS)
        m['target'] = target
        self.panel.tx_moved()
        self._motion_request('move', **target)

    def event(self, event):
        kind = event.get('event')
        token = event.get('request_id')
        if kind in ('agent', 'hello', 'ready') or (
                kind == 'network' and event.get('port') == 7600 and event.get('state') != 'connected'):
            self.recover_off = self.owns_output
            self.frame_target = None
            self.session_reset_at = time.monotonic()
            self.panel.prepared = False
            self.motion = self.light_request = None
            self._light(None)
            return
        if kind == 'hardware':
            try:
                level, _ = self.confirmed_light(event, token)
            except ValueError:
                level = None
            self._light(level)
            if level == 0 and self.light_request is None:
                self.manual_on = self.recover_off = False
            elif level == 1 and self.recover_off:
                self.recover_off = False
                self._send_off()
        if kind == 'laser_pattern':
            self._light(None)
            self.motion = None
        if kind == 'outputs_off' or (kind == 'laser' and event.get('laser_gpio_level') != 0):
            self.motion = None
        if kind == 'outputs_off' or kind == 'laser':
            level = event.get('laser_gpio_level')
            if type(level) is int and level in (0, 1) and self.app.ook.hardware and self.app.ook.hardware.get('simulated') is False:
                self._light(level)
                if level == 0 and self.light_request is None:
                    self.manual_on = self.recover_off = False
        if self.light_request and self.light_request[0] == token:
            _, expected, _ = self.light_request
            self.light_request = None
            if kind == 'laser' and type(event.get('laser_gpio_level')) is int and event['laser_gpio_level'] == self.light == expected:
                self.manual_on = bool(expected)
                self.recover_off = False
                self.notice.set('Tx GPIO 응답 확인')
            elif expected == 1:
                self._send_off()
            else:
                self._light(None)
                self.notice.set('Tx OFF 응답 미확인 · OFF를 다시 누르세요.')
        if self.motion and self.motion['token'] == token:
            try:
                self._motion_event(event)
            except (ValueError, OSError) as exc:
                self.motion = None
                self.notice.set(str(exc))

    def tick(self):
        now = time.monotonic()
        if self.motion and now - self.motion['sent'] > 5:
            self.motion = None
            self.notice.set('Tx 서보 응답 시간 초과 · 다음 이동은 새 상태를 조회합니다.')
        if self.light_request and now - self.light_request[2] > 5:
            _, expected, _ = self.light_request
            self.light_request = None
            self._light(None)
            if expected == 1:
                self._send_off()
            else:
                self.notice.set('Tx OFF 응답 미확인 · OFF를 다시 누르세요.')
        elif self.manual_on and self.light == 1 and self.light_request is None:
            try:
                self._real_tx()
            except ValueError:
                self._send_off()
