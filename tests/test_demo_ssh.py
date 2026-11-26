import os
import json
import queue
import shlex
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from Tx.Controller.demo_ssh import DemoSsh, receiver_archive


class DemoSshTests(unittest.TestCase):
    def setUp(self):
        self.rx = DemoSsh.__new__(DemoSsh)
        self.rx.connected = False
        self.rx.commands = queue.Queue()
        self.rx.events = queue.Queue()
        self.rx.stopping = threading.Event()

    def test_command_channel_cannot_carry_a_pose(self):
        self.rx._receive(b'{"kind":"ready"}')
        for command in ('arm', 'home', 'cancel', 'measure'):
            self.rx.send(command, 'request-1')
            self.assertEqual(self.rx.commands.get_nowait(), {'cmd': command, 'token': 'request-1'})
        with self.assertRaises(ValueError):
            self.rx.send('move 103 20', 'request-1')

    def test_jog_carries_only_bounded_relative_steps_and_checks_reply(self):
        self.rx._receive(b'{"kind":"ready"}')
        self.rx.events.get_nowait()
        self.rx.jog('J', 5, -1)
        self.assertEqual(self.rx.commands.get_nowait(), dict(cmd='jog', token='J', pan_step=5, tilt_step=-1))
        for token, pan, tilt in [('J', 103, 20), ('J', 0, 0), ('J', 1, True), ('', 0, 1)]:
            with self.assertRaises(ValueError):
                self.rx.jog(token, pan, tilt)
        self.assertTrue(self.rx.commands.empty())
        reply = dict(kind='jogged', token='J', pan=108, tilt=19,
                     servo_writes=[['1.0', '1.1', 0, 108, '', 'jogged'], ['1.1', '1.2', 1, 19, '', 'jogged']])
        self.rx._receive(json.dumps(reply))
        self.assertEqual(self.rx.events.get_nowait()['servo_writes'], reply['servo_writes'])
        for invalid in (dict(tilt=36), dict(pan=True), dict(token='')):
            with self.assertRaises(OSError):
                self.rx._receive(json.dumps(dict(reply, **invalid)))

    def test_same_connection_accepts_matching_raw_sampler_protocol(self):
        self.rx._receive(b'{"kind":"ready"}')
        self.rx.events.get_nowait()
        self.rx.send('raw 2', 'run1')
        self.assertEqual(self.rx.commands.get_nowait(), dict(cmd='start', token='run1', interval_ms=2))
        self.rx._receive(b'{"kind":"started","token":"run1","bus":3,"address":64,"config":8606,"interval_ms":2}')
        self.rx._receive(b'{"kind":"sample","token":"run1","seq":3,"rx_monotonic_ns":20,"bus_raw":600}')
        self.rx.send('stop', 'run1')
        self.assertEqual(self.rx.commands.get_nowait(), dict(cmd='stop', token='run1'))
        self.rx._receive(b'{"kind":"stopped","token":"run1"}')
        records = [self.rx.events.get_nowait() for _ in range(3)]
        self.assertEqual([r['kind'] for r in records], ['started', 'sample', 'stopped'])
        self.assertEqual(records[1]['rx_monotonic_ns'], 20)
        self.assertIn('pc_receive_monotonic_ns', records[1])
        for invalid in ('raw 0', 'raw 1001', 'raw 1.2'):
            with self.assertRaises(ValueError):
                self.rx.send(invalid, 'run1')
        with self.assertRaises(OSError):
            self.rx._receive(b'{"kind":"started","token":"run1","bus":1,"address":64,"config":8606,"interval_ms":2}')

    def test_voltage_stream_preserves_rx_time_and_rejects_malformed_rows(self):
        self.rx._receive(b'{"kind":"ready"}')
        self.rx.events.get_nowait()
        self.rx._receive(b'{"kind":"voltage","token":"t","seq":101,"rx_monotonic_ns":5555,"bus_raw":777}')
        sample = self.rx.events.get_nowait()
        self.assertEqual((sample['seq'], sample['rx_monotonic_ns'], sample['bus_raw']), (101, 5555, 777))
        self.rx._receive(b'{"kind":"measured","token":"t","count":200,"window_start_ns":1,"window_end_ns":1000000001,"max_gap_ns":5000000}')
        self.assertEqual(self.rx.events.get_nowait()['count'], 200)
        with self.assertRaises(OSError):
            self.rx._receive(b'{"kind":"voltage","token":"t","seq":true,"rx_monotonic_ns":1,"bus_raw":777}')
        with self.assertRaises(OSError):
            self.rx._receive(b'{"kind":"measured","token":"t"}')

    def test_failed_arm_still_allows_home_and_unsolicited_ack_fails(self):
        with self.assertRaises(OSError):
            self.rx._receive(b'{"kind":"applied","token":"1","pan":103,"tilt":20}')
        self.rx._receive(b'{"kind":"ready"}')
        self.rx._receive(b'{"kind":"error","token":"1","message":"INA219 unavailable"}')
        self.rx._receive(b'{"kind":"home","token":"2","pan":103,"tilt":20}')
        self.assertTrue(self.rx.connected)
        self.assertEqual([self.rx.events.get_nowait()['kind'] for _ in range(3)], ['ready', 'error', 'home'])
        with self.assertRaises(OSError):
            self.rx._receive(b'{"kind":"home","pan":103,"tilt":20}')

    def test_pi_bundle_imports_without_a_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            app = Path(directory) / 'demo.pyz'
            app.write_bytes(receiver_archive())
            script = 'import sys; sys.path.insert(0,sys.argv[1]); from Rx.angle_demo import main; assert callable(main)'
            result = subprocess.run([sys.executable, '-I', '-c', script, str(app)], cwd=directory,
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(os.path.isdir('/proc/self/fd'), 'Raspberry Pi OS uses Linux procfs')
    def test_anonymous_remote_launcher_leaves_stdin_for_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / 'fixture.pyz'
            with ZipFile(fixture, 'w') as archive:
                archive.writestr('__main__.py', 'print(input(), flush=True)')
            with patch('Tx.Controller.demo_ssh.receiver_archive', return_value=fixture.read_bytes()):
                _, _, *args = shlex.split(self.rx._remote_command())
            result = subprocess.run([sys.executable, *args], cwd=directory, input='home\n', capture_output=True,
                                    text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), 'home')
