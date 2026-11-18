"""Real command/framing/relay/client startup with fake hardware boundaries.

No camera, motor, detector or SAC learning performance is established by these
tests. In particular, simulated pixels and detections are explicit fixtures;
servo ACKs and capture metadata are produced by the actual Pi implementation.
"""
import io
import queue
import socket
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from common.tx_moving import MovingConfig
from common.tx_setup import TX_CAMERA_SETTINGS
from common.tx_tracking import Sample
from Tx.Controller.moving_core import MovingRun, MovingRunSettings
from Tx.Controller.network_client import Network
from Tx.RaspberryPi.Rasp_main import Agent
from Server import Server_main as relay


class SerialBoundary:
    def __init__(self):
        self.writes = []

    def write(self, data):
        self.writes.append(data)
        return len(data)

    def close(self):
        pass


class CameraBoundary:
    def __init__(self, blocked=False):
        self.started = threading.Event()
        self.release_first = threading.Event()
        self.shutdown = threading.Event()
        self.blocked = blocked
        self.calls = 0
        output = io.BytesIO()
        Image.new('RGB', (1296, 972)).save(output, format='JPEG')
        self.jpeg = output.getvalue()

    def capture_request(self, flush):
        self.calls += 1
        if self.blocked:
            if self.calls == 1:
                self.started.set()
                if not self.release_first.wait(5):
                    raise RuntimeError('Test did not release the first capture')
            else:
                self.shutdown.wait(5)
                raise RuntimeError('Test capture stream deliberately stalled')
        jpeg = self.jpeg
        return SimpleNamespace(config={'main': {'size': (1296, 972)}},
            get_metadata=lambda: {'SensorTimestamp': time.monotonic_ns()},
            save=lambda name, output, format: output.write(jpeg), release=lambda: None)

    def stop(self):
        pass

    def close(self):
        self.shutdown.set()


class MovingWireStartupTests(unittest.TestCase):
    def setUp(self):
        self.net = None
        self.pi_thread = None
        self.servers = []
        self.errors = []
        self.camera = CameraBoundary(blocked=self._testMethodName.startswith('test_old'))
        self.serial = SerialBoundary()
        self.addCleanup(self.cleanup_wire)
        # Use ephemeral ports so the test never connects to a running apparatus.
        relay.agent = relay.gui = relay.latest = None
        relay.version = 0
        relay.snapshots.clear()
        addresses = {}
        for port, handler in ((7500, relay.PiControl), (7501, relay.PiImages),
                              (7600, relay.GuiControl), (7601, relay.GuiImages)):
            server = relay.Server(('127.0.0.1', 0), handler)
            self.servers.append(server)
            addresses[port] = server.server_address
            threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01}, daemon=True).start()
        connect = socket.create_connection

        def local_connection(address, *args, **kwargs):
            if address[0] != 'moving-wire-test':
                raise AssertionError('Unexpected test network destination')
            return connect(addresses[address[1]], *args, **kwargs)

        self.connection_patch = patch.object(socket, 'create_connection', side_effect=local_connection)
        self.connection_patch.start()
        self.net = Network('moving-wire-test')
        self.wait_event(lambda e: e.get('event') == 'hello')
        self.pi = Agent(SimpleNamespace(server='moving-wire-test', simulate=False, servo_port=None,
                                        ir_pin=17, laser_pin=23))
        self.pi.servo.serial = self.serial
        self.pi.gpio = SimpleNamespace(output=lambda *_: None)
        self.pi.configure_camera = lambda camera, cfg, still=False: self.camera

        def serve_pi():
            try:
                self.pi.connection()
            except Exception as exc:
                self.errors.append(exc)

        self.pi_thread = threading.Thread(target=serve_pi, daemon=True)
        self.pi_thread.start()
        self.wait_event(lambda e: e.get('event') == 'ready')
        self.cfg = MovingConfig()
        self.request(dict(cmd='servo_config', request_id='configure', **self.cfg.limits))
        self.request(dict(cmd='move', request_id='before-episode', pan=-5, tilt=0, speed=100, acc=1))
        self.net.send(dict(cmd='preview', enable=True, **TX_CAMERA_SETTINGS))
        self.wait_event(lambda e: e.get('event') == 'preview' and e.get('state') == 'started')
        self.transitions = []
        self.learner = SimpleNamespace(mode='resume', episodes=0, updates=0, executed_actions=0,
            cancel=threading.Event(), action=Mock(return_value=np.array([1/90, 0.])),
            add=lambda *args, **kwargs: self.transitions.append((args, kwargs)) or True)
        self.log = Mock()
        kalman = dict(schema='tx-bearing-kalman-v1', state_order=['alpha', 'beta', 'alpha_dot', 'beta_dot'],
            angle_unit='rad', process_noise_model='continuous_white_acceleration',
            Q=(np.eye(2)*1e-7).tolist(), R=(np.eye(2)*1e-6).tolist(),
            P0=np.diag([1e-5, 1e-5, .01, .01]).tolist())
        self.run = MovingRun(self.cfg, MovingRunSettings(start_pan=-5, settle_s=0),
            self.learner, self.log, lambda cmd: self.net.send(cmd) or True, time.monotonic, kalman)

    def cleanup_wire(self):
        self.camera.release_first.set()
        self.camera.shutdown.set()
        if self.net is not None:
            self.net.close()
        if self.pi_thread is not None:
            self.pi.stop.set()
            self.pi_thread.join(3)
        for server in self.servers:
            server.shutdown()
            server.server_close()
        if hasattr(self, 'connection_patch'):
            self.connection_patch.stop()

    def wait_event(self, predicate):
        deadline = time.monotonic()+5
        while time.monotonic() < deadline:
            try:
                event = self.net.events.get(timeout=.1)
            except queue.Empty:
                continue
            if predicate(event):
                return event
        self.fail('Control event did not arrive')

    def request(self, command):
        self.net.send(command)
        event = self.wait_event(lambda e: e.get('request_id') == command['request_id'])
        self.assertEqual(event['event'], 'servo')
        return event

    def pump(self):
        while True:
            try:
                self.run.reply(self.net.events.get_nowait())
            except queue.Empty:
                break
        frame, count = self.net.pop()
        if frame is not None:
            target = dict(center=[696, 384], box=[676, 364, 716, 404], confidence=.9)
            sample = Sample(count, frame[2], time.monotonic(), 1296, 972, target, 1,
                            seq=frame[1]['seq'], simulated=frame[1]['simulated'])
            self.run.supply(sample, frame[1])
        self.run.tick()

    def until(self, predicate, timeout=5):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            self.pump()
            if predicate():
                return
            time.sleep(.005)
        self.fail(f'Run did not reach expected state: {self.run.phase}, {self.run.stop_detail}')

    def test_actual_pi_metadata_allows_first_action_and_transition(self):
        self.run.start()
        self.until(lambda: self.run.phase == 'ready')
        self.assertIsNone(self.run.frame_command_diagnostic)
        self.assertEqual(self.run.command.tolist(), [-5, 0])
        self.until(lambda: self.run.steps == 1 and self.run.phase == 'ready')
        self.run.manual_end()
        self.assertEqual(self.run.phase, 'saving')
        self.assertEqual(self.run.save_request['new_transitions'], 1)
        self.assertEqual(len(self.transitions), 1)
        self.learner.action.assert_called_once()
        self.assertEqual(len(self.serial.writes), 3)  # initial, reset at same angle, first action
        confirmed = [c.kwargs for c in self.log.event.call_args_list if c.args[0] == 'command_confirmed']
        self.assertEqual(len(confirmed), 2)
        self.assertEqual(confirmed[0]['commanded'], [-5, 0])
        self.assertEqual(confirmed[1]['commanded'], [-4, 0])

    def test_old_frame_after_new_ack_stops_before_policy_without_fake_experience(self):
        self.assertTrue(self.camera.started.wait(2))
        self.run.start()
        self.until(lambda: self.run.phase == 'acquire')
        expected_id = self.run.command_id
        self.camera.release_first.set()
        self.until(lambda: self.run.phase == 'saving')
        self.assertEqual(self.run.save_request['reason'], 'initialization_failed')
        self.assertEqual(self.run.stop_detail, 'frame_command_mismatch')
        self.assertEqual(self.run.steps, 0)
        self.assertEqual(self.transitions, [])
        self.learner.action.assert_not_called()
        self.assertEqual(len(self.serial.writes), 2)  # reset sent despite unchanged target
        diagnostics = [c.kwargs for c in self.log.event.call_args_list if c.args[0] == 'frame_command_mismatch']
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0]['mismatched_fields'], ['command_id'])
        self.assertEqual(diagnostics[0]['expected']['command_id'], expected_id)
        self.assertEqual(diagnostics[0]['observed']['command_id'], 'before-episode')
        self.assertEqual(self.run.frame_command_diagnostic['prior_sample_reason'], 'stale')


if __name__ == '__main__':
    unittest.main()
