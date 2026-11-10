"""M4 OOK and M5 collection share hardware without interrupting each other."""
import queue
import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock, patch

from common.model_paths import resolve_yolo_path
from Tx.Controller.Com_main import App
from Tx.Controller.ook_panel import OokPanel
from Tx.Controller.sweep_core import KF_MODE, SweepSettings
from Tx.Controller.sweep_panel import SweepPanel


def variable(value):
    return SimpleNamespace(get=lambda: value, set=Mock())


class M4M5IntegrationTests(unittest.TestCase):
    def app(self):
        app = App.__new__(App)
        app.net = Mock()
        app.record = Mock()
        app.stationary = SimpleNamespace(active=False, preparing=None, busy=False,
            run=None, stop=Mock(), inference=None, close=Mock())
        app.timing = SimpleNamespace(active=False, preparing=None, future=None,
            profile=SimpleNamespace(running=False), stop=Mock(), close=Mock())
        app.detection = SimpleNamespace(stop=Mock(), future=None, close=Mock())
        app.servo = SimpleNamespace(pending=None, online=True)
        app.ook = SimpleNamespace(busy=False, pending=None)
        app.sweep = self.sweep(app)
        return app

    def sweep(self, app):
        panel = SweepPanel.__new__(SweepPanel)
        panel.app = app
        panel.preparing = panel.future = panel.run = panel.setup_pending = None
        panel.hardware_ready = True
        panel.cfg = SweepSettings()
        panel.settings = lambda: panel.cfg
        panel.mode = variable(KF_MODE)
        panel.path = variable('test.pt')
        panel.device = variable('cpu')
        panel.prepared_mode = KF_MODE
        panel.identity = dict(path=resolve_yolo_path('test.pt'), device='cpu')
        panel.detector = object()
        panel.pool = Mock()
        panel.invalidate = Mock()
        return panel

    def ook(self, app):
        panel = OokPanel.__new__(OokPanel)
        panel.app = app
        panel.online = True
        panel.hardware = dict(capabilities=['laser_pattern'], pattern_active=False)
        panel.pending = panel.run_id = panel.rx_phase = panel.capture = None
        panel.experiment = panel.arduino = None
        panel.output_requested = False
        panel.state = Mock()
        app.current = (b'jpeg', {}, 10, 0)
        app.ook = panel
        return panel

    def set_busy(self, panel, state):
        if state == 'prepare':
            panel.preparing = 'model'
        elif state == 'background':
            panel.future = Future()
        else:
            panel.run = SimpleNamespace(active=state == 'active',
                                        pending=object() if state == 'reply' else None)

    def test_laser_start_is_rejected_in_every_m5_busy_phase(self):
        for state in ('prepare', 'background', 'active', 'reply'):
            for command in (dict(cmd='laser', value=1),
                            dict(cmd='laser_pattern', bits='10', bit_ms=20, repeats=1)):
                with self.subTest(state=state, command=command):
                    app = self.app()
                    self.set_busy(app.sweep, state)
                    self.assertFalse(app.send(command))
                    app.net.send.assert_not_called()
                    app.sweep.invalidate.assert_not_called()
                    self.assertTrue(app.sweep.busy)

    def test_laser_off_remains_available_without_aborting_m5(self):
        for state in ('prepare', 'background', 'active', 'reply'):
            with self.subTest(state=state):
                app = self.app()
                self.set_busy(app.sweep, state)
                app.ook.busy = True
                command = dict(cmd='laser', value=0)
                self.assertTrue(app.send(command))
                app.net.send.assert_called_once_with(command)
                app.sweep.invalidate.assert_not_called()

    def test_rejected_ook_start_creates_no_batch_or_capture(self):
        for action in ('laser_on', 'start', 'start_experiment'):
            with self.subTest(action=action):
                app = self.app()
                self.set_busy(app.sweep, 'active')
                panel = self.ook(app)
                app.stage_dir = Mock()
                panel.begin_trial = Mock()
                args = ('step',) if action == 'start_experiment' else ()
                with patch('Tx.Controller.ook_panel.time.monotonic', return_value=10.1):
                    with self.assertRaises(ValueError):
                        getattr(panel, action)(*args)
                app.net.send.assert_not_called()
                app.stage_dir.assert_not_called()
                panel.begin_trial.assert_not_called()
                self.assertIsNone(panel.experiment)
                self.assertFalse(panel.output_requested)

    def test_ook_busy_rejects_m5_buttons_before_changing_prepared_session(self):
        for action in ('prepare', 'start'):
            with self.subTest(action=action):
                app = self.app()
                app.ook.busy = True
                panel = app.sweep
                panel.run = SimpleNamespace(active=False, pending=None, finalize=Mock())
                previous = (panel.run, panel.detector, panel.cfg, panel.identity)
                with self.assertRaises(ValueError):
                    getattr(panel, action)()
                self.assertTrue(panel.hardware_ready)
                self.assertEqual((panel.run, panel.detector, panel.cfg, panel.identity), previous)
                panel.run.finalize.assert_not_called()
                panel.pool.submit.assert_not_called()
                app.net.send.assert_not_called()
                app.stationary.stop.assert_not_called()
                app.detection.stop.assert_not_called()

    def test_ook_rejection_does_not_invalidate_idle_m5_preparation(self):
        for command in ('move', 'servo_config', 'preview', 'snap', 'ir_cut'):
            with self.subTest(command=command):
                app = self.app()
                app.ook.busy = True
                self.assertFalse(app.send(dict(cmd=command)))
                app.sweep.invalidate.assert_not_called()
                self.assertTrue(app.sweep.hardware_ready)
                app.net.send.assert_not_called()

    def test_m5_internal_move_passes_when_ook_is_idle(self):
        app = self.app()
        self.set_busy(app.sweep, 'active')
        command = dict(cmd='move', request_id='m5-1-move-test', pan=0, tilt=0)
        self.assertTrue(app.send(command))
        app.net.send.assert_called_once_with(command)
        app.sweep.invalidate.assert_not_called()

    def test_poll_routes_events_and_background_work_to_both_panels(self):
        app = App.__new__(App)
        app.closing = False
        app.root = Mock()
        app.record = Mock()
        app.links = {}
        app.pings = {}
        app.connection = Mock()
        app.current = None
        app.mark = 10
        app.servo = Mock()
        app.ook = Mock()
        for name in ('timing', 'stationary', 'detection', 'sweep'):
            panel = Mock(name=name)
            panel.guard.side_effect = lambda callback: callback()
            setattr(app, name, panel)
        event = dict(event='hardware', laser_gpio_level=0)
        events = queue.Queue()
        events.put(event)
        app.net = SimpleNamespace(events=events, pop=lambda: (None, 0))
        with patch('Tx.Controller.Com_main.time.monotonic', return_value=10.1):
            app.poll()
        app.ook.event.assert_called_once_with(event)
        app.sweep.event.assert_called_once_with(event)
        app.ook.tick.assert_called_once_with()
        app.sweep.poll.assert_called_once_with()
        app.root.after.assert_called_once_with(5, app.poll)

    def test_close_requests_laser_off_before_waiting_for_m5_worker(self):
        app = self.app()
        app.root = Mock()
        panel = self.ook(app)
        panel.output_requested = True
        completed = []

        def close_sweep():
            commands = [call.args[0] for call in app.net.send.call_args_list]
            self.assertTrue(any(cmd.get('cmd') == 'laser' and cmd.get('value') == 0
                                for cmd in commands))
            completed.append('sweep')

        app.sweep.close = Mock(side_effect=close_sweep)
        app.close()
        self.assertEqual(completed, ['sweep'])
        app.stationary.close.assert_called_once_with()
        app.timing.close.assert_called_once_with()
        app.detection.close.assert_called_once_with()
        app.net.close.assert_called_once_with()
        app.root.destroy.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
