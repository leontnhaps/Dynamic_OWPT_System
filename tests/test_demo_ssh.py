import os
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
        for command in ('arm', 'home', 'cancel'):
            self.rx.send(command, 'request-1')
            self.assertEqual(self.rx.commands.get_nowait(), {'cmd': command, 'token': 'request-1'})
        with self.assertRaises(ValueError):
            self.rx.send('move 103 20', 'request-1')

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
