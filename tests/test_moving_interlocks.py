"""The live moving controller owns shared hardware through loading and return."""
import queue
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from Tx.Controller.Com_main import App
from Tx.Controller.detection_panel import DetectionPanel
from Tx.Controller.ook_panel import OokPanel
from Tx.Controller.processing_profile import ProcessingProfile
from Tx.Controller.stationary_panel import StationaryPanel
from Tx.Controller.sweep_panel import SweepPanel
from Tx.Controller.timing_panel import TimingPanel


class MovingInterlockTests(unittest.TestCase):
    def app(self, *, moving_busy=True, sweep_busy=False):
        app = App.__new__(App)
        app.net, app.record = Mock(), Mock()
        app.moving = SimpleNamespace(busy=moving_busy, hardware_ready=True,
                                    invalidate=Mock(), close=Mock())
        app.sweep = SimpleNamespace(busy=sweep_busy, hardware_ready=False,
                                   invalidate=Mock(), close=Mock())
        app.stationary = SimpleNamespace(active=False, preparing=None, close=Mock())
        app.ook = SimpleNamespace(busy=False, close=Mock())
        app.timing = SimpleNamespace(active=False, profile=SimpleNamespace(running=False), close=Mock())
        app.detection = SimpleNamespace(close=Mock())
        return app

    def test_external_control_and_ook_are_blocked_without_aborting_live_run(self):
        app = self.app()
        commands = [dict(cmd=name) for name in ('move', 'servo_config', 'preview', 'snap', 'ir_cut')]
        commands += [dict(cmd='laser', value=1), dict(cmd='laser_pattern', bits='10')]
        # The old collection prefix must not bypass the new controller's lock.
        commands += [dict(cmd='move', request_id='m5-1-sweep-test')]
        for command in commands:
            with self.subTest(command=command):
                self.assertFalse(app.send(command))
        app.net.send.assert_not_called()
        app.moving.invalidate.assert_not_called()
        app.sweep.invalidate.assert_not_called()

    def test_live_commands_pass_and_cannot_bypass_active_collection(self):
        for name in ('move', 'servo_config', 'preview'):
            command = dict(cmd=name, request_id='m5-1-moving-test')
            app = self.app()
            self.assertTrue(app.send(command))
            app.net.send.assert_called_once_with(command)
            app.moving.invalidate.assert_not_called()
            app = self.app(moving_busy=False, sweep_busy=True)
            self.assertFalse(app.send(command))
            app.net.send.assert_not_called()
            app.moving.invalidate.assert_not_called()
            app.sweep.invalidate.assert_not_called()

    def test_laser_off_and_queries_remain_available(self):
        for command in (dict(cmd='laser', value=0), dict(cmd='hardware_status'),
                        dict(cmd='servo_status'), dict(cmd='ping', token='test')):
            with self.subTest(command=command):
                app = self.app()
                app.ook.busy = True
                self.assertTrue(app.send(command))
                app.net.send.assert_called_once_with(command)
                app.moving.invalidate.assert_not_called()

    def test_outputs_off_aborts_live_owner_and_is_still_sent(self):
        app = self.app()
        app.ook.busy = True
        self.assertTrue(app.send(dict(cmd='outputs_off')))
        app.moving.invalidate.assert_called_once_with('external_command: outputs_off')
        app.net.send.assert_called_once_with(dict(cmd='outputs_off'))

    def test_idle_preparation_is_invalidated_only_by_allowed_external_change(self):
        app = self.app(moving_busy=False)
        app.ook.busy = True
        self.assertFalse(app.send(dict(cmd='preview')))
        app.moving.invalidate.assert_not_called()
        app.ook.busy = False
        self.assertTrue(app.send(dict(cmd='preview')))
        app.moving.invalidate.assert_called_once_with('external_command: preview')

    def test_ook_busy_blocks_live_setup_and_move(self):
        for name in ('preview', 'servo_config', 'move'):
            with self.subTest(command=name):
                app = self.app()
                app.ook.busy = True
                self.assertFalse(app.send(dict(cmd=name, request_id='m5-1-moving-test')))
                app.moving.invalidate.assert_not_called()
                app.net.send.assert_not_called()

    def test_other_entry_points_reject_before_mutating_their_state(self):
        app = SimpleNamespace(moving=SimpleNamespace(busy=True))
        entries = [(DetectionPanel, 'load', ()), (DetectionPanel, 'start', ()),
                   (TimingPanel, 'prepare', (False,)), (ProcessingProfile, 'start', ()),
                   (StationaryPanel, 'stop_other_panels', ()),
                   (StationaryPanel, 'check_manual_available', ()),
                   (SweepPanel, 'stop_other_panels', ()),
                   (OokPanel, 'check_ready', ()), (OokPanel, 'prepare', ()),
                   (OokPanel, 'ir', (1,))]
        for cls, method, args in entries:
            with self.subTest(cls=cls.__name__, method=method):
                panel = cls.__new__(cls)
                panel.app = app
                with self.assertRaisesRegex(ValueError, '이동 학습'):
                    getattr(panel, method)(*args)

    def test_poll_delivers_events_and_consumes_live_background_work(self):
        app = App.__new__(App)
        app.closing = False
        app.root, app.record, app.connection = Mock(), Mock(), Mock()
        app.links, app.pings = {}, {}
        app.current = None
        app.mark = 10
        app.servo, app.ook = Mock(), Mock()
        for name in ('timing', 'stationary', 'detection', 'sweep', 'moving'):
            panel = Mock(name=name)
            panel.guard.side_effect = lambda callback: callback()
            setattr(app, name, panel)
        event = dict(event='servo', operation='move', request_id='m5-1-moving-test')
        events = queue.Queue()
        events.put(event)
        app.net = SimpleNamespace(events=events, pop=lambda: (None, 0))
        with patch('Tx.Controller.Com_main.time.monotonic', return_value=10.1):
            app.poll()
        app.moving.handle_event.assert_called_once_with(event)
        app.moving.poll.assert_called_once_with()
        app.root.after.assert_called_once_with(5, app.poll)

    def test_close_requests_ook_off_before_waiting_on_live_workers(self):
        app = self.app()
        app.root = Mock()
        calls = []
        app.ook.close.side_effect = lambda: calls.append('ook_off')
        app.moving.close.side_effect = lambda: calls.append('moving_drain')
        app.close()
        self.assertEqual(calls, ['ook_off', 'moving_drain'])
        app.net.send.assert_called_once_with(dict(cmd='outputs_off'))
        app.net.close.assert_called_once_with()
        app.root.destroy.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
