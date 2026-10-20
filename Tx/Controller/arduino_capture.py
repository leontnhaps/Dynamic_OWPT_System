"""USB serial evidence capture. Arduino, PC and Pi clocks are NOT synchronized."""
import csv
import json
import math
import queue
import threading
import time

ARDUINO_BAUD = 38400
ADC_REFERENCE_V = 1.1
DIVIDER_GAIN = 5.7


def serial_ports():
    try:
        from serial.tools import list_ports
    except ImportError as exc:
        raise ValueError('GUI Python 환경에 pyserial을 설치하세요: python -m pip install pyserial') from exc
    return [p.device for p in list_ports.comports()]


def parse_reading(line):
    """Return only complete current M4-4 records; NA is missing, never zero."""
    if line.startswith('R1,'):
        parts = line.split(',')
        if len(parts) != 4 or any(not x.isascii() or not x.isdigit() for x in parts[1:]):
            raise ValueError('Invalid R1 record')
        seq, ms, adc = map(int, parts[1:])
        if not 0 <= seq <= 0xffffffff or not 0 <= ms <= 0xffffffff or not 0 <= adc <= 1023:
            raise ValueError('Invalid R1 counter/ADC')
        clipped = adc == 1023
        a2 = adc * ADC_REFERENCE_V / 1024
        return dict(sample_seq=seq, arduino_ms=ms, adc=adc,
                    a2_V=None if clipped else round(a2, 6),
                    pv_V=None if clipped else round(a2 * DIVIDER_GAIN, 6),
                    off_V=None, on_V=None, th_V=None, state='CLIPPED' if clipped else 'RAW',
                    level='NA', clipped=clipped)
    if not line.startswith('ms='):
        return None
    fields = dict(part.split('=', 1) for part in line.split() if '=' in part)
    required = ('ms', 'adc', 'a2_V', 'pv_V', 'state', 'off_V', 'on_V', 'th_V', 'level')
    if any(key not in fields for key in required):
        raise ValueError('Incomplete/old Arduino record')
    result = dict(arduino_ms=int(fields['ms']), adc=int(fields['adc']),
                  state=fields['state'], level=fields['level'],
                  clipped='CLIPPED' in line.split() or fields['state'] == 'CLIPPED')
    if not 0 <= result['arduino_ms'] <= 0xffffffff or not 0 <= result['adc'] <= 1023:
        raise ValueError('Invalid Arduino counter/ADC')
    if result['state'] not in ('IDLE', 'BASELINE', 'WAIT', 'RISE', 'TRACK', 'CLIPPED'):
        raise ValueError('Unknown detector state')
    if result['level'] not in ('NA', 'LOW', 'HIGH'):
        raise ValueError('Unknown detector level')
    for key in ('a2_V', 'pv_V', 'off_V', 'on_V', 'th_V'):
        value = None if fields[key] == 'NA' else float(fields[key])
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError('Invalid voltage')
        result[key] = value
    if result['adc'] == 1023:
        result['clipped'] = True
    if result['clipped']:
        result['a2_V'] = result['pv_V'] = None
    return result


class ArduinoSerial:
    """One worker owns the port; no serial blocking operations in Tk callbacks."""
    def __init__(self, port, factory=None, settle_seconds=2.0):
        if factory is None:
            try:
                import serial
            except ImportError as exc:
                raise ValueError('pyserial 필요: python -m pip install pyserial') from exc
            factory = serial.Serial
        self.port = port
        self.events = queue.Queue(maxsize=4096)
        self.commands = queue.Queue(maxsize=16)
        self.stopping = threading.Event()
        self.connected = False
        self.terminal_error = None
        self.worker = threading.Thread(target=self._run, args=(factory, settle_seconds), daemon=True)
        self.worker.start()

    def emit(self, **event):
        event.update(pc_receive_unix_ns=time.time_ns(), pc_receive_monotonic_ns=time.monotonic_ns())
        try:
            self.events.put_nowait(event)
        except queue.Full as exc:
            raise OSError('Arduino receive queue overflow; capture stopped') from exc

    def send(self, command, token=None):
        if command not in ('raw', 'start', 'stop'):
            raise ValueError('Unsupported Arduino command')
        if not self.connected or self.stopping.is_set():
            raise OSError('Arduino가 연결되지 않았습니다.')
        try:
            self.commands.put_nowait((command, token))
        except queue.Full as exc:
            raise OSError('Arduino command queue full') from exc

    def close(self):
        self.stopping.set()

    def _run(self, factory, settle_seconds):
        port = None
        try:
            port = factory(self.port, baudrate=ARDUINO_BAUD, timeout=0.1, write_timeout=0.5)
            # Opening a USB adapter may reset the Pro Mini. Do not send start
            # while its bootloader/startup help is still running.
            if self.stopping.wait(settle_seconds):
                return
            port.reset_input_buffer()
            self.connected = True
            self.emit(kind='connected', port=self.port)
            partial = b''
            while not self.stopping.is_set():
                try:
                    command, token = self.commands.get_nowait()
                except queue.Empty:
                    command = None
                if command:
                    if command in ('raw', 'start'):
                        port.reset_input_buffer()
                        partial = b''
                    data = (command + '\n').encode('ascii')
                    if port.write(data) != len(data):
                        raise OSError('Incomplete Arduino serial write')
                    self.emit(kind='command_sent', command=command, token=token)
                chunk = port.read(max(1, min(512, port.in_waiting)))
                partial += chunk
                while b'\n' in partial:
                    raw, partial = partial.split(b'\n', 1)
                    if len(raw) > 512:
                        raise OSError('Arduino line too long; check firmware/baud')
                    self.emit(kind='line', line=raw.rstrip(b'\r').decode('ascii', errors='replace'))
                if len(partial) > 512:
                    raise OSError('Arduino unterminated line; check firmware/baud')
        except Exception as exc:
            self.terminal_error = str(exc)
            try:
                self.emit(kind='error', message=str(exc))
            except OSError:
                pass
        finally:
            self.connected = False
            if port is not None:
                try:
                    port.write(b'stop\n')
                except Exception:
                    pass
                try:
                    port.close()
                except Exception:
                    pass


class RxCapture:
    columns = ('pc_receive_unix_ns', 'pc_receive_monotonic_ns', 'arduino_ms',
               'adc', 'a2_V', 'pv_V', 'off_V', 'on_V', 'th_V', 'state', 'level',
               'clipped', 'sample_seq', 'sequence_gap', 'sample_dt_ms', 'raw_line')

    def __init__(self, folder, settings, port, request_id):
        self.folder = folder
        self.rows = 0
        self.high_rows = 0
        self.low_rows = 0
        self.clipped_rows = 0
        self.invalid_lines = 0
        self.last_seq = None
        self.last_ms = None
        self.missing_records = 0
        self.max_sample_dt_ms = 0
        self.long_sample_gaps = 0
        self.closed = False
        self.file = None
        config = dict(schema_version=3, request_id=request_id, settings=settings,
                      capture_mode='raw_voltage_pattern',
                      tx_start_condition='first_fresh_voltage_record_after_raw; no WAIT gate',
                      detector_levels_validated=False,
                      arduino_port=port, arduino_baud=ARDUINO_BAUD,
                      created_pc_unix_ns=time.time_ns(),
                      rx_adc_interval_ms=10, rx_report_interval_ms=10,
                      rx_rise_pv_v=None, rx_firmware='M4-4 RAW10 R1',
                      serial_format='R1,sequence,Arduino_ms,ADC',
                      adc_reference_v=ADC_REFERENCE_V, divider_gain=DIVIDER_GAIN,
                      timing_note='Arduino ms, PC receive times and Pi GPIO times have independent clocks.',
                      scope='Raw 10ms voltage comparison; no calibration or level classification. Sequence gaps indicate missing records; timestamps show sampling gaps. No BER or optical latency measurement.')
        (folder / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
        self.file = (folder / 'rx.csv').open('w', newline='', encoding='utf-8-sig')
        try:
            self.writer = csv.DictWriter(self.file, fieldnames=self.columns)
            self.writer.writeheader()
            self.file.flush()
        except OSError:
            self.file.close()
            raise

    def write(self, event, reading):
        row = dict(reading, pc_receive_unix_ns=event['pc_receive_unix_ns'],
                   pc_receive_monotonic_ns=event['pc_receive_monotonic_ns'], raw_line=event['line'])
        seq = reading.get('sample_seq')
        if seq is not None:
            delta = ((seq - self.last_seq) & 0xffffffff) if self.last_seq is not None else None
            if delta is not None and (delta == 0 or delta >= 0x80000000):
                raise OSError('Arduino sequence reset/duplicate; restart capture')
            gap = seq if delta is None else delta - 1
            dt = None if self.last_ms is None else (reading['arduino_ms'] - self.last_ms) & 0xffffffff
            row.update(sequence_gap=gap, sample_dt_ms=dt)
            self.missing_records += gap
            if dt is not None:
                self.max_sample_dt_ms = max(self.max_sample_dt_ms, dt)
                self.long_sample_gaps += dt > 20
            self.last_seq, self.last_ms = seq, reading['arduino_ms']
        self.writer.writerow(row)
        self.file.flush()
        self.rows += 1
        self.high_rows += reading['level'] == 'HIGH'
        self.low_rows += reading['level'] == 'LOW'
        self.clipped_rows += reading['clipped']

    def finish(self, reason, tx_state):
        if self.closed:
            return
        try:
            self.file.close()
            result = dict(schema_version=3, end_reason=reason, tx_state=tx_state,
                          rows=self.rows, missing_records=self.missing_records,
                          max_sample_dt_ms=self.max_sample_dt_ms, long_sample_gaps=self.long_sample_gaps,
                          high_rows=self.high_rows, low_rows=self.low_rows,
                          clipped_rows=self.clipped_rows, invalid_lines=self.invalid_lines,
                          finished_pc_unix_ns=time.time_ns(),
                          capture_mode='raw_voltage_pattern', detector_levels_validated=False,
                          bit_decoding_implemented=False, ber=None,
                          note='HIGH/LOW counts are reported sample counts, not received bit counts.')
            (self.folder / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        finally:
            self.closed = True
