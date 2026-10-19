import csv
import json
import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from Tx.Controller.arduino_capture import ArduinoSerial, parse_reading
from tests import test_ook_panel as panel_fixture


def reading(ms=1250, state='WAIT', level='NA', adc=100):
    return (f'ms={ms} adc={adc} a2_V=0.107 pv_V=0.612 state={state} '
            f'off_V=0.612 on_V=NA th_V=NA level={level}')


class FakeArduino:
    port = 'COM_TEST'
    connected = True
    worker = SimpleNamespace(is_alive=lambda: True)

    def __init__(self):
        self.events = queue.Queue()
        self.commands = []

    def send(self, command, token=None):
        self.commands.append((command, token))

    def emit(self, **event):
        event.update(pc_receive_unix_ns=time.time_ns(), pc_receive_monotonic_ns=time.monotonic_ns())
        self.events.put(event)

    def close(self):
        self.connected = False


class CaptureTests(unittest.TestCase):
    def panel(self):
        p = panel_fixture.PanelStateTests().panel()
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.sent = []
        p.app = SimpleNamespace(send=lambda cmd: self.sent.append(cmd) or True,
                                stage_dir=lambda _: folder,
                                current=(b'', {}, time.monotonic(), 0))
        p.hardware = dict(capabilities=['laser_pattern'], laser_gpio_level=0)
        p.arduino = FakeArduino()
        self.p = p
        return p

    def arm(self, p):
        p.start()
        self.assertEqual([x['cmd'] for x in self.sent], ['laser'])
        self.assertEqual(p.arduino.commands, [])
        token = p.pending[0]
        p.event(dict(event='laser', request_id=token, laser_gpio_level=0))
        self.assertEqual(p.arduino.commands, [('start', p.trial_id)])

    def ready(self, p):
        p.arduino.emit(kind='command_sent', command='start', token=p.trial_id)
        p.arduino.emit(kind='line', line='OK BASELINE: laser OFF; wait for state=WAIT')
        p.arduino.emit(kind='line', line=reading())
        p.tick()

    def test_complete_trial_saves_settings_times_values_and_trace(self):
        p = self.panel(); self.arm(p)
        folder = p.capture.folder
        # Old queued WAIT alone must not trigger a transmission.
        p.arduino.emit(kind='line', line=reading())
        p.tick(); self.assertEqual(len(self.sent), 1)
        self.ready(p)
        self.assertEqual(self.sent[-1]['cmd'], 'laser_pattern')
        self.assertEqual(self.sent[-1]['bits'], '00100110')
        token = p.run_id
        p.event(dict(event='laser_pattern', state='started', request_id=token))
        p.arduino.emit(kind='line', line=reading(1500, 'TRACK', 'HIGH', 300))
        p.tick()
        p.event(dict(event='laser_pattern', state='completed', request_id=token, trace=[dict(index=0, level=0)]))
        self.assertEqual(p.rx_phase, 'tail')
        p.rx_deadline = time.monotonic() - 1
        p.tick()
        self.assertIsNone(p.capture)
        self.assertEqual(p.arduino.commands[-1][0], 'stop')
        config = json.loads((folder / 'config.json').read_text())
        self.assertEqual(config['settings']['bits'], '00100110')
        with (folder / 'rx.csv').open(encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['arduino_ms'], '1250')
        self.assertEqual(rows[0]['on_V'], '')
        self.assertTrue(rows[0]['pc_receive_monotonic_ns'])
        summary = json.loads((folder / 'summary.json').read_text())
        self.assertEqual(summary['end_reason'], 'completed')
        self.assertEqual(summary['high_rows'], 1)
        self.assertIsNone(summary['ber'])
        self.assertIn('"trace"', (folder / 'events.jsonl').read_text())

    def test_stop_during_baseline_never_fires_on_late_wait(self):
        p = self.panel(); self.arm(p)
        folder = p.capture.folder
        self.ready_events(p)
        p.off(); p.tick()
        self.assertFalse(any(x['cmd'] == 'laser_pattern' for x in self.sent))
        self.assertEqual(json.loads((folder / 'summary.json').read_text())['end_reason'], 'user_stop')

    def ready_events(self, p):
        p.arduino.emit(kind='command_sent', command='start', token=p.trial_id)
        p.arduino.emit(kind='line', line=reading(250, 'BASELINE'))
        p.arduino.emit(kind='line', line=reading())

    def test_clipping_keeps_row_and_aborts(self):
        p = self.panel(); self.arm(p); self.ready(p)
        folder = p.capture.folder
        p.arduino.emit(kind='line', line=reading(1500, 'CLIPPED', adc=1023) + ' CLIPPED')
        p.tick()
        self.assertIsNone(p.capture)
        self.assertEqual(self.sent[-1]['cmd'], 'laser')
        result = json.loads((folder / 'summary.json').read_text())
        self.assertEqual(result['end_reason'], 'adc_clipped')
        self.assertEqual(result['clipped_rows'], 1)

    def test_disconnect_and_timeout_save_partial(self):
        for failure in ('disconnect', 'timeout'):
            with self.subTest(failure=failure):
                p = self.panel(); self.arm(p); self.ready(p)
                folder = p.capture.folder
                if failure == 'disconnect':
                    p.arduino.emit(kind='error', message='unplugged')
                else:
                    p.rx_last_seen = time.monotonic() - 4
                p.tick()
                self.assertIsNone(p.capture)
                self.assertTrue((folder / 'summary.json').exists())
                self.assertEqual(self.sent[-1]['value'], 0)

    def test_no_wait_times_out_without_sending(self):
        p = self.panel(); self.arm(p)
        p.rx_deadline = time.monotonic() - 1; p.tick()
        self.assertFalse(any(x['cmd'] == 'laser_pattern' for x in self.sent))
        self.assertIsNone(p.capture)

    def test_save_error_stops_instead_of_silent_continue(self):
        p = self.panel(); self.arm(p); self.ready(p)
        with patch.object(p.capture, 'write', side_effect=OSError('disk full')):
            p.arduino.emit(kind='line', line=reading(1500)); p.tick()
        self.assertIsNone(p.capture)
        self.assertEqual(self.sent[-1]['value'], 0)

    def test_raw_mode_remains_available(self):
        p = self.panel(); p.arduino = None; p.rx_save.set(False); p.start()
        self.assertEqual(self.sent[-1]['cmd'], 'laser_pattern')
        self.assertIsNone(p.capture)

    def test_off_log_failure_still_closes_capture(self):
        p = self.panel(); self.arm(p); self.ready(p)
        capture = p.capture
        with patch.object(p.app, 'send', side_effect=OSError('log disk full')):
            p.abort_capture('recording_error')
        self.assertTrue(capture.closed)
        self.assertIsNone(p.capture)
        self.assertIsNone(p.rx_phase)

    def test_late_terminal_trace_saved_after_stop(self):
        p = self.panel(); self.arm(p); self.ready(p)
        token, path = p.trial_id, p.logpath
        p.off()
        p.event(dict(event='laser_pattern', request_id=token, state='stopped', trace=[1]))
        self.assertIn('"trace": [1]', path.read_text())

    def test_parser_na_and_rejects_incomplete_nonfinite(self):
        self.assertIsNone(parse_reading('help | pv | start | stop'))
        self.assertIsNone(parse_reading(reading())['on_V'])
        for line in ('ms=100 adc=1', reading().replace('pv_V=0.612', 'pv_V=nan')):
            with self.assertRaises(ValueError):
                parse_reading(line)


class SerialWorkerTests(unittest.TestCase):
    def test_fragmented_serial_and_start_command(self):
        class Port:
            def __init__(self, *args, **kwargs):
                self.chunks = queue.Queue(); self.writes = []; self.closed = False
                self.in_waiting = 1
            def reset_input_buffer(self): pass
            def read(self, _):
                try: return self.chunks.get(timeout=.02)
                except queue.Empty: return b''
            def write(self, data): self.writes.append(data); return len(data)
            def close(self): self.closed = True
        port = Port()
        serial = ArduinoSerial('TEST', factory=lambda *a, **k: port, settle_seconds=0)
        self.assertEqual(serial.events.get(timeout=1)['kind'], 'connected')
        serial.send('start', 'trial')
        event = serial.events.get(timeout=1)
        self.assertEqual((event['kind'], event['token']), ('command_sent', 'trial'))
        line = reading()
        port.chunks.put(line[:20].encode()); port.chunks.put((line[20:]+'\r\n').encode())
        event = serial.events.get(timeout=1)
        self.assertEqual(event['line'], line)
        self.assertTrue(event['pc_receive_unix_ns'])
        serial.close(); serial.worker.join(timeout=1)
        self.assertFalse(serial.worker.is_alive())
        self.assertTrue(port.closed)
        self.assertIn(b'start\n', port.writes)
        self.assertEqual(port.writes[-1], b'stop\n')


if __name__ == '__main__': unittest.main()
