import csv
import json
import queue
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from tests import test_arduino_capture as capture_fixture
from Tx.Controller.pi_ook_rx import PI_RECEIVER, PiSshReceiver


class SshWorkerTests(unittest.TestCase):
    def test_reuses_verified_client_for_fragmented_stream_and_control_messages(self):
        lib = MagicMock()
        client = lib.SSHClient.return_value
        channel = client.get_transport.return_value.open_session.return_value
        chunks = queue.Queue()
        chunks.put(b'{"kind":"rea')
        chunks.put(b'dy","bus":3,"address":64}\n')
        channel.recv_ready.side_effect = lambda: not chunks.empty()
        channel.recv.side_effect = lambda count: chunks.get_nowait()
        channel.recv_stderr_ready.return_value = False
        closed = threading.Event()
        channel.shutdown_write.side_effect = closed.set
        channel.exit_status_ready.side_effect = closed.is_set
        channel.closed = False

        def reply(raw):
            cmd = json.loads(raw)
            self.assertEqual(cmd['token'], 'trial1')
            if cmd['cmd'] == 'start':
                chunks.put(b'{"kind":"started","token":"trial1","interval_ms":5,"bus":3,"address":64,"config":8606}\n'
                           b'{"kind":"sample","token":"trial1","seq":0,"rx_monotonic_ns":1234,"bus_raw":1}\n')
            elif cmd['cmd'] == 'stop':
                chunks.put(b'{"kind":"stopped","token":"trial1"}\n')

        channel.sendall.side_effect = reply
        with patch.dict('sys.modules', paramiko=lib):
            rx = PiSshReceiver('test@pi.local', 'fixture')
            try:
                self.assertEqual(rx.events.get(timeout=3)['kind'], 'ready')
                rx.send('raw 5', 'trial1')
                self.assertEqual(rx.events.get(timeout=3)['kind'], 'started')
                self.assertEqual(rx.events.get(timeout=3)['bus_raw'], 1)
                rx.send('stop', 'trial1')
                self.assertEqual(rx.events.get(timeout=3)['kind'], 'stopped')
            finally:
                rx.close()
                rx.worker.join(3)
        self.assertFalse(rx.worker.is_alive())
        self.assertFalse(rx.connected)
        self.assertIsNone(rx.terminal_error)
        client.load_system_host_keys.assert_called_once()
        client.connect.assert_called_once()
        channel.exec_command.assert_called_once()
        self.assertIn('exec python3 -u -c', channel.exec_command.call_args.args[0])
        client.close.assert_called_once()


class PiTransportTests(unittest.TestCase):
    def setUp(self):
        self.panel = capture_fixture.CaptureTests.panel(self)
        self.rx = self.panel.arduino
        self.rx.source = PI_RECEIVER
        self.rx.terminal_error = None
        self.assertEqual(self.panel.stage_name, 'M6-4')
        capture_fixture.CaptureTests.arm(self, self.panel)
        self.folder = self.panel.capture.folder
        self.addCleanup(lambda: self.panel.capture and self.panel.capture.file.close())

    def started(self, token=None):
        self.rx.emit(kind='started', token=token or self.panel.trial_id, interval_ms=5,
                     bus=3, address=64, config=0x219e)

    def sample(self, seq, token=None, code=126):
        self.rx.emit(kind='sample', token=token or self.panel.trial_id, seq=seq,
                     rx_monotonic_ns=10**13 + seq * 5000000, bus_raw=code)

    def test_headers_and_late_records_are_bound_to_trial_until_stop_ack(self):
        p = self.panel
        self.started('old')
        self.sample(0, 'old')
        p.tick()
        self.assertFalse(p.capture.config['interval_confirmed'])
        self.assertEqual(len(self.sent), 1)
        self.started()
        self.sample(0)
        p.tick()
        self.assertEqual(self.sent[-1]['cmd'], 'laser_pattern')
        p.event(dict(event='laser_pattern', state='started', request_id=p.run_id))
        p.event(dict(event='laser_pattern', state='completed', request_id=p.run_id))
        p.rx_deadline = time.monotonic() - 1
        p.tick()
        self.assertEqual(p.rx_phase, 'stopping_rx')
        self.assertEqual(self.rx.commands[-1], ('stop', p.trial_id))
        self.sample(2, code=0)
        self.rx.emit(kind='stopped', token='old')
        p.tick()
        self.assertFalse((self.folder / 'summary.json').exists())
        self.rx.emit(kind='stopped', token=p.trial_id)
        p.tick()
        self.assertIsNone(p.capture)
        with (self.folder / 'rx.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row['pv_V'] for row in rows], ['0.504', '0.000'])
        self.assertEqual(rows[1]['rx_monotonic_ns'], str(10**13 + 10**7))
        result = json.loads((self.folder / 'summary.json').read_text())
        self.assertEqual((result['end_reason'], result['missing_records']), ('completed', 1))

    def test_ssh_failure_preempts_queued_samples(self):
        self.started()
        self.sample(0)
        self.rx.terminal_error = 'disconnected'
        self.panel.tick()
        self.assertIsNone(self.panel.capture)
        self.assertFalse(any(cmd['cmd'] == 'laser_pattern' for cmd in self.sent))
        self.assertEqual(self.sent[-1]['value'], 0)

    def test_host_parser_requires_known_device_and_reports_terminal_errors(self):
        receiver = PiSshReceiver.__new__(PiSshReceiver)
        receiver.connected = False
        receiver.events = queue.Queue()
        with self.assertRaises(OSError):
            receiver._receive(b'{"kind":"ready","bus":1,"address":64}')
        receiver._receive(b'{"kind":"ready","bus":3,"address":64}')
        self.assertTrue(receiver.connected)
        self.assertIn('pc_receive_monotonic_ns', receiver.events.get_nowait())
        with self.assertRaisesRegex(OSError, 'restore'):
            receiver._receive(b'{"kind":"error","message":"restore failed"}')
