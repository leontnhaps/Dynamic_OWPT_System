"""Launch the matching angle receiver on the Rx Pi over a verified SSH session."""
import base64
import io
import json
import shlex
import zipfile
from pathlib import Path

from common.angle_demo import HOME_PAN, HOME_TILT, MAX_PAN, MAX_TILT
from Tx.Controller.pi_ook_rx import PiSshReceiver


ROOT = Path(__file__).resolve().parents[2]
RECEIVER_MODULES = (
    'Rx/angle_demo.py', 'Rx/arc_bench.py', 'Rx/ina219_ook_stream.py',
    'common/angle_demo.py', 'common/arc_protocol.py', 'common/laser_pattern.py',
)


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
        if command not in ('arm', 'home', 'cancel') or not isinstance(token, str) or not token:
            raise ValueError('잘못된 Rx 시연 명령 또는 요청 ID입니다.')
        self._enqueue(dict(cmd=command, token=token))

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
            if not event.get('token'):
                self.closed_message = event['message']
        elif not self.connected or kind not in ('armed', 'applied', 'home', 'cancelled', 'timeout'):
            raise OSError('예상하지 못한 Rx 시연 응답입니다.')
        elif not isinstance(event.get('token'), str) or not event['token']:
            raise OSError('Rx 시연 응답에 요청 ID가 없습니다.')
        elif kind in ('applied', 'home'):
            pan, tilt = event.get('pan'), event.get('tilt')
            if (type(pan) is not int or not 0 <= pan <= MAX_PAN
                    or type(tilt) is not int or not 0 <= tilt <= MAX_TILT):
                raise OSError('Rx 시연 응답의 적용 각도가 잘못되었습니다.')
            if kind == 'home' and (pan, tilt) != (HOME_PAN, HOME_TILT):
                raise OSError('Rx 원위치 응답의 각도가 일치하지 않습니다.')
        self._publish(event)
