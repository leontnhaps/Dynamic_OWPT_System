"""Pi-side acquisition over one verified SSH channel. Device and PC clocks are independent."""
import csv
import json
import queue
import shlex
import threading
import time
from pathlib import Path
from common.ook_experiment import PI_RECEIVER
from Rx.ina219_ook_stream import SAMPLE_CONFIG


SAMPLER = Path(__file__).resolve().parents[2] / 'Rx' / 'ina219_ook_stream.py'
DEFAULT_RX_TARGET = 'gmlwns@192.168.219.47'


class PiSshReceiver:
    source = PI_RECEIVER
    startup_timeout_message = 'INA219 수집기 시작 응답 시간 초과'
    closed_message = 'Rx SSH sampler closed'

    def __init__(self, target, password=None):
        username, sep, host = target.strip().rpartition('@')
        if not sep or not host or not username or any(c.isspace() for c in target):
            raise ValueError('Rx SSH 주소는 사용자@IP 형식으로 입력하세요.')
        try:
            import paramiko
        except ImportError as exc:
            raise ValueError('Rx SSH 연결에는 requirements-m4-4.txt의 paramiko가 필요합니다.') from exc
        self.port = target.strip()
        self.events = queue.Queue(maxsize=8192)
        self.commands = queue.Queue(maxsize=16)
        self.stopping = threading.Event()
        self.connected = False
        self.terminal_error = None
        self.worker = threading.Thread(target=self._run, args=(paramiko, host, username, password), daemon=True)
        self.worker.start()

    def send(self, command, token):
        if command == 'stop':
            payload = dict(cmd='stop', token=token)
        elif command.startswith('raw '):
            payload = dict(cmd='start', token=token, interval_ms=int(command[4:]))
        else:
            raise ValueError('Unknown Rx command')
        self._enqueue(payload)

    def _enqueue(self, payload):
        if not self.connected or self.stopping.is_set():
            raise OSError('Rx SSH 연결이 끊겼습니다.')
        try:
            self.commands.put_nowait(payload)
        except queue.Full as exc:
            raise OSError('Rx SSH command queue full') from exc

    def close(self):
        self.stopping.set()

    def _receive(self, line):
        if len(line) > 4096:
            raise OSError('Invalid Rx record size')
        event = json.loads(line)
        if not isinstance(event, dict) or event.get('kind') not in ('ready', 'started', 'sample', 'stopped', 'error'):
            raise OSError('Rx protocol mismatch')
        if event['kind'] == 'error':
            raise OSError(str(event.get('message', 'Rx sampler error')))
        if event['kind'] == 'ready':
            if self.connected or (event.get('bus'), event.get('address')) != (3, 0x40):
                raise OSError('Unexpected INA219 device or restarted sampler')
            self.connected = True
        self._publish(event)

    def _publish(self, event):
        event.update(pc_receive_unix_ns=time.time_ns(), pc_receive_monotonic_ns=time.monotonic_ns())
        try:
            self.events.put_nowait(event)
        except queue.Full as exc:
            raise OSError('Rx 수신 큐가 가득 찼습니다. 수집을 중단합니다.') from exc

    def _remote_command(self):
        return 'exec python3 -u -c ' + shlex.quote(SAMPLER.read_text(encoding='utf-8'))

    def _run(self, paramiko, host, username, password):
        client = paramiko.SSHClient()
        stream = None
        try:
            client.load_system_host_keys()
            # An existing OpenSSH connection supplies known_hosts (including hashed entries).
            client.connect(host, username=username, password=password or None, timeout=6,
                           banner_timeout=6, auth_timeout=10, channel_timeout=6)
            if self.stopping.is_set():
                return
            command = self._remote_command()
            stream = client.get_transport().open_session(timeout=6)
            stream.settimeout(2)
            stream.exec_command(command)
            partial = b''
            err = b''
            ready_by = time.monotonic() + 8
            while not self.stopping.is_set():
                try:
                    payload = self.commands.get_nowait()
                except queue.Empty:
                    payload = None
                if payload is not None:
                    stream.sendall((json.dumps(payload) + '\n').encode('ascii'))
                if stream.recv_stderr_ready():
                    err = (err + stream.recv_stderr(4096))[-4096:]
                if stream.recv_ready():
                    partial += stream.recv(65536)
                    *lines, partial = partial.split(b'\n')
                    for line in lines:
                        self._receive(line)
                    if len(partial) > 4096:
                        raise OSError('Unterminated Rx record')
                elif stream.exit_status_ready() or stream.closed or not client.get_transport().is_active():
                    raise OSError(err.decode('utf-8', 'replace').strip() or self.closed_message)
                elif not self.connected and time.monotonic() > ready_by:
                    raise OSError(self.startup_timeout_message)
                else:
                    self.stopping.wait(.004)
        except Exception as exc:
            if not self.stopping.is_set():
                self.terminal_error = str(exc) or type(exc).__name__
                if 'not found in known_hosts' in self.terminal_error:
                    self.terminal_error += (f' · PC 터미널에서 ssh {self.port}로 연결해 '
                                            'Rx 호스트 키 지문을 확인하고 다시 시도하세요.')
        finally:
            self.connected = False
            if stream is not None:
                try:
                    # EOF on the Pi closes the sampler and restores the INA219 configuration.
                    stream.shutdown_write()
                    end = time.monotonic() + 1
                    while not stream.exit_status_ready() and time.monotonic() < end:
                        time.sleep(.02)
                except (OSError, EOFError):
                    pass
            client.close()


class PiCapture:
    columns = ('sample_seq', 'rx_monotonic_ns', 'bus_raw', 'pv_V', 'pc_receive_monotonic_ns',
               'pc_receive_unix_ns', 'sequence_gap', 'sample_dt_ms')

    def __init__(self, folder, settings, target, token, sample_ms, experiment):
        self.folder = folder
        self.sample_ms = sample_ms
        self.rows = self.invalid_lines = self.missing_records = self.clipped_rows = 0
        self.max_sample_dt_ms = 0
        self.last_seq = self.last_ns = None
        self.config = dict(schema_version=1, receiver=PI_RECEIVER, request_id=token,
                           rx_ssh_target=target, settings=settings, experiment=experiment,
                           requested_sample_ms=sample_ms, confirmed_sample_ms=None, interval_confirmed=False,
                           bus=3, address=0x40, lsb_volts=.004,
                           timing_note='rx_monotonic_ns is Pi I2C readout time. Tx, Rx and PC clocks are independent.')
        self._write_config()
        self.file = (folder / 'rx.csv').open('x', newline='', encoding='utf-8')
        try:
            self.writer = csv.writer(self.file)
            self.writer.writerow(self.columns)
            self.file.flush()
        except OSError:
            self.file.close()
            raise

    def _write_config(self):
        (self.folder / 'config.json').write_text(json.dumps(self.config, indent=2), encoding='utf-8')

    def confirm(self, event):
        if (self.config['interval_confirmed'] or event.get('interval_ms') != self.sample_ms
                or (event.get('bus'), event.get('address'), event.get('config')) != (3, 0x40, SAMPLE_CONFIG)):
            raise OSError('Rx 수집기 설정 응답이 요청과 다릅니다.')
        self.config.update(interval_confirmed=True, confirmed_sample_ms=self.sample_ms,
                           ina219_config=event['config'])
        self._write_config()

    def write_sample(self, event):
        seq, ns, code = (event.get(k) for k in ('seq', 'rx_monotonic_ns', 'bus_raw'))
        if (not self.config['interval_confirmed'] or any(type(value) is not int for value in (seq, ns, code))
                or seq < 0 or ns < 0 or not 0 <= code <= 8191
                or (self.last_seq is not None and (seq <= self.last_seq or ns <= self.last_ns))):
            self.invalid_lines += 1
            raise OSError('Invalid or out-of-order INA219 sample')
        gap = seq if self.last_seq is None else seq - self.last_seq - 1
        dt_ms = None if self.last_ns is None else (ns - self.last_ns) / 1e6
        self.writer.writerow((seq, ns, code, f'{code * .004:.3f}', event['pc_receive_monotonic_ns'],
                              event['pc_receive_unix_ns'], gap, dt_ms))
        self.clipped_rows += code == 8191
        self.missing_records += gap
        self.max_sample_dt_ms = max(self.max_sample_dt_ms, dt_ms or 0)
        self.last_seq, self.last_ns = seq, ns
        self.rows += 1
        if self.rows % 64 == 0:
            self.file.flush()

    def finish(self, reason, tx_state):
        self.file.close()
        summary = dict(schema_version=1, end_reason=reason, tx_state=tx_state, rows=self.rows,
                       receiver=PI_RECEIVER, missing_records=self.missing_records, invalid_lines=self.invalid_lines,
                       clipped_rows=self.clipped_rows, max_sample_dt_ms=self.max_sample_dt_ms)
        (self.folder / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
