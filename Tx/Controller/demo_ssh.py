"""Launch the matching angle receiver on the Rx Pi over a verified SSH session."""
import base64
import io
import json
import math
import shlex
import zipfile
from pathlib import Path

from common.angle_demo import (DECODE_SAMPLE_MS, HOME_PAN, HOME_TILT, MAX_PAN, MAX_TILT,
                               validate_angle_bit_ms, validate_jog)
from common.ook_experiment import PAYLOAD_BITS, TRAILING_ZEROS, interval_value
from Rx.ina219_ook_stream import SAMPLE_CONFIG
from Tx.Controller.pi_ook_rx import PiSshReceiver


ROOT = Path(__file__).resolve().parents[2]
RECEIVER_MODULES = (
    'Rx/angle_demo.py', 'Rx/arc_bench.py', 'Rx/ina219_ook_stream.py',
    'common/angle_demo.py', 'common/arc_protocol.py', 'common/laser_pattern.py',
    'common/ook_experiment.py', 'common/ook_frame_decode.py',
)


def optical_servo_writes(event):
    """Read back the two successful bus write records in receiver-clock order."""
    rows = event.get('servo_writes')
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError('Rx의 Pan·Tilt 서보 쓰기 기록 두 개가 필요합니다.')
    last_end = 0.0
    angles = []
    for axis, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != 6:
            raise ValueError('Rx 서보 쓰기 기록의 형식이 잘못되었습니다.')
        start, end, channel, angle, step, action = row
        if (type(channel) is not int or channel != axis or type(angle) is not int
                or not 0 <= angle <= (MAX_PAN, MAX_TILT)[axis] or step != '' or action != 'applied'):
            raise ValueError('Rx 서보 쓰기의 채널·각도·동작이 잘못되었습니다.')
        if type(start) not in (str, int, float) or type(end) not in (str, int, float):
            raise ValueError('Rx 서보 쓰기 시각을 읽을 수 없습니다.')
        try:
            start, end = float(start), float(end)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError('Rx 서보 쓰기 시각을 읽을 수 없습니다.') from exc
        if not (math.isfinite(start) and math.isfinite(end) and last_end <= start <= end):
            raise ValueError('Rx 서보 쓰기 시각의 순서가 잘못되었습니다.')
        last_end = end
        angles.append(angle)
    return tuple(angles)


def receiver_archive():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for package in ('Rx', 'common'):
            archive.writestr(f'{package}/__init__.py', '')
        for module in RECEIVER_MODULES:
            archive.write(ROOT / module, module)
        archive.writestr('__main__.py', 'from Rx.angle_demo import main\nraise SystemExit(main())\n')
    return buffer.getvalue()


class DemoSsh(PiSshReceiver):
    startup_timeout_message = 'Rx 각도 시연 수신기 시작 응답 시간 초과'
    closed_message = 'Rx 각도 시연 수신기의 SSH 연결이 종료되었습니다.'

    def _remote_command(self):
        archive = base64.b64encode(receiver_archive()).decode('ascii')
        # Anonymous storage on Raspberry Pi OS vanishes even if SSH is lost. stdin
        # stays available as the control channel while the application is running.
        script = (
            'import base64, runpy, tempfile\n'
            'with tempfile.TemporaryFile() as app:\n'
            f'    app.write(base64.b64decode({archive!r}))\n'
            '    app.flush()\n'
            '    runpy.run_path(f"/proc/self/fd/{app.fileno()}", run_name="__main__")\n'
        )
        return 'exec python3 -I -u -c ' + shlex.quote(script)

    def send(self, command, token):
        if not isinstance(command, str) or not isinstance(token, str) or not 1 <= len(token) <= 128:
            raise ValueError('잘못된 Rx 시연 명령 또는 요청 ID입니다.')
        if command.startswith('raw '):
            self._enqueue(dict(cmd='start', token=token, interval_ms=interval_value(command[4:])))
        elif command in ('arm', 'finish_angle', 'commit', 'home', 'cancel', 'measure', 'stop'):
            self._enqueue(dict(cmd=command, token=token))
        else:
            raise ValueError('잘못된 Rx 시연 명령입니다.')

    def jog(self, token, pan_step, tilt_step):
        if not isinstance(token, str) or not 1 <= len(token) <= 128:
            raise ValueError('잘못된 Rx 점검 요청 ID입니다.')
        validate_jog(pan_step, tilt_step)
        self._enqueue(dict(cmd='jog', token=token, pan_step=pan_step, tilt_step=tilt_step))

    def arm_angle(self, token, bit_ms):
        if not isinstance(token, str) or not 1 <= len(token) <= 128:
            raise ValueError('잘못된 Rx 각도 수신 요청 ID입니다.')
        self._enqueue(dict(cmd='arm', token=token, bit_ms=validate_angle_bit_ms(bit_ms)))

    def _receive(self, line):
        if len(line) > 4096:
            raise OSError('너무 긴 Rx 시연 응답입니다.')
        event = json.loads(line)
        kind = event.get('kind') if isinstance(event, dict) else None
        if kind == 'ready':
            if self.connected:
                raise OSError('Rx 시연 수신기가 예기치 않게 다시 시작했습니다.')
            self.connected = True
        elif kind == 'error':
            if not isinstance(event.get('message'), str):
                raise OSError('Rx 시연 오류 응답에 설명이 없습니다.')
            if not event.get('token'):
                raise OSError(event['message'])
        elif not self.connected or kind not in ('armed', 'decoded', 'applied', 'home', 'jogged', 'cancelled', 'timeout',
                                                'angle_samples', 'angle_samples_end', 'voltage', 'measured',
                                                'started', 'sample', 'stopped'):
            raise OSError('예상하지 못한 Rx 시연 응답입니다.')
        elif not isinstance(event.get('token'), str) or not event['token']:
            raise OSError('Rx 시연 응답에 요청 ID가 없습니다.')
        elif kind == 'armed' and ('bit_ms' in event or 'sample_ms' in event):
            try:
                validate_angle_bit_ms(event.get('bit_ms'))
            except ValueError as exc:
                raise OSError('Rx 각도 OOK 비트 시간 확인 실패') from exc
            if (type(event.get('sample_ms')) is not int or type(event.get('interval_ms')) is not int
                    or (event['sample_ms'], event['interval_ms']) != (DECODE_SAMPLE_MS, DECODE_SAMPLE_MS)
                    or (event.get('bus'), event.get('address'), event.get('config')) != (3, 0x40, SAMPLE_CONFIG)):
                raise OSError('Rx 각도 OOK INA219 설정 확인 실패')
        elif kind == 'angle_samples':
            rows = event.get('rows')
            if not isinstance(rows, list) or not rows:
                raise OSError('Rx 각도 OOK 전압 표본 묶음이 비어 있습니다.')
            for row in rows:
                if (not isinstance(row, list) or len(row) != 3 or
                        any(type(x) is not int or x < 0 for x in row) or row[2] > 8191):
                    raise OSError('Rx 각도 OOK 전압 표본 형식이 잘못되었습니다.')
        elif kind == 'angle_samples_end':
            if type(event.get('count')) is not int or event['count'] < 0:
                raise OSError('Rx 각도 OOK 전체 표본 수가 잘못되었습니다.')
        elif kind == 'started':
            if ((event.get('bus'), event.get('address'), event.get('config')) != (3, 0x40, SAMPLE_CONFIG)
                    or type(event.get('interval_ms')) is not int or not 1 <= event['interval_ms'] <= 1000):
                raise OSError('Rx 속도 측정 설정 응답이 잘못되었습니다.')
        elif kind in ('voltage', 'sample'):
            if (any(type(event.get(k)) is not int or event[k] < 0 for k in ('seq', 'rx_monotonic_ns', 'bus_raw'))
                    or event['bus_raw'] > 8191):
                raise OSError('Rx 전압 표본 형식이 잘못되었습니다.')
        elif kind == 'measured':
            if (any(type(event.get(k)) is not int or event[k] < 0 for k in
                    ('count', 'window_start_ns', 'window_end_ns', 'max_gap_ns'))
                    or not event['count'] or event['window_end_ns'] <= event['window_start_ns']):
                raise OSError('Rx 전압 측정 결과 형식이 잘못되었습니다.')
        elif kind in ('decoded', 'applied', 'home', 'jogged'):
            pan, tilt = event.get('pan'), event.get('tilt')
            if kind in ('decoded', 'applied'):
                payload = event.get('rx_payload_bits')
                if not isinstance(payload, str) or len(payload) != PAYLOAD_BITS or set(payload) - {'0', '1'}:
                    raise OSError(f'Rx 광복호 payload가 올바른 {PAYLOAD_BITS}비트 문자열이어야 합니다.')
            max_pan, max_tilt = (65535, 65535) if kind == 'decoded' else (MAX_PAN, MAX_TILT)
            if (type(pan) is not int or not 0 <= pan <= max_pan
                    or type(tilt) is not int or not 0 <= tilt <= max_tilt):
                raise OSError('Rx 시연 응답의 적용 각도가 잘못되었습니다.')
            if kind == 'home' and (pan, tilt) != (HOME_PAN, HOME_TILT):
                raise OSError('Rx 원위치 응답의 각도가 일치하지 않습니다.')
        if kind in ('decoded', 'error') and 'frame_check' in event:
            check = event['frame_check']
            if (not isinstance(check, dict) or type(check.get('header_verified')) is not bool
                    or check.get('verified') is not (kind == 'decoded')):
                raise OSError('Rx 프레임 확인 상태가 잘못되었습니다.')
            payload, tail = check.get('payload_bits'), check.get('tail_bits')
            if payload is not None and (not isinstance(payload, str) or len(payload) != PAYLOAD_BITS
                                        or set(payload) - {'0', '1'}):
                raise OSError('Rx가 관측한 payload 형식이 잘못되었습니다.')
            if tail is not None and (not isinstance(tail, str) or len(tail) != len(TRAILING_ZEROS)
                                     or set(tail) - {'0', '1', '?'}):
                raise OSError('Rx가 관측한 끝 비트 형식이 잘못되었습니다.')
            if kind == 'decoded' and (not check['header_verified'] or tail != TRAILING_ZEROS
                                      or payload != event['rx_payload_bits']):
                raise OSError('Rx 프레임 확인 결과와 복호 응답이 다릅니다.')
        self._publish(event)
