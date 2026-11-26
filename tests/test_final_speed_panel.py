import csv
import json
import queue
import tempfile
import threading
import time
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import Mock, patch

from common.ook_experiment import PI_RECEIVER
from Rx.ina219_ook_stream import SAMPLE_CONFIG
from Tx.Controller.demo_ssh import DemoSsh
from Tx.Controller.final_speed_panel import FinalSpeedPanel
from Tx.Controller.pi_ook_rx import PiCapture
from Tx.Controller.point_comparison_panel import PointComparisonPanel
from tests import test_ook_panel as panel_fixture


class FinalSpeedPanelTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.captures = Path(directory.name) / 'captures'
        (self.captures / 'M6-4').mkdir(parents=True)

        self.engine = panel_fixture.PanelStateTests().panel()
        self.engine.hardware = dict(simulated=False, capabilities=['laser_pattern'])
        app = self.engine.app
        app.ook = self.engine
        app.stage_dir = lambda stage: self.captures / stage
        app.current = (b'', dict(simulated=False, session='test'), time.monotonic(), time.time_ns())
        app.send = Mock(return_value=True)

        self.rx = DemoSsh.__new__(DemoSsh)
        self.rx.connected = True
        self.rx.port = 'test@rx'
        self.rx.terminal_error = None
        self.rx.stopping = threading.Event()
        self.rx.events = queue.Queue()
        self.rx.commands = queue.Queue()

        p = self.panel = PointComparisonPanel.__new__(PointComparisonPanel)
        p.app, p.rx = app, self.rx
        p.phase = p.off_token = p.token = p.method = p.station = p.station_view = p.run = p.logpath = None
        p.prepared = p.selected = True
        p.status = panel_fixture.Value()
        p.off_status = panel_fixture.Value()
        p.rx_status = panel_fixture.Value()

        speed = p.speed = FinalSpeedPanel.__new__(FinalSpeedPanel)
        speed.panel, speed.engine = p, self.engine
        speed.running = False
        speed.last_root = speed.shown = None
        speed.periods = panel_fixture.Value('12,10,8,6,5')
        speed.sample_ms = panel_fixture.Value('2')
        speed.progress = panel_fixture.Value()
        speed.headline = panel_fixture.Value()
        speed.table, speed.grid_frame = Mock(), Mock()
        speed.graph_button = Mock()
        speed.table.get_children.return_value = ()
        self.addCleanup(self.close_active_csv)

    def close_active_csv(self):
        if self.engine.capture is not None:
            self.engine.capture.file.close()

    def reply(self, kind, token, **fields):
        self.rx._receive(json.dumps(dict(kind=kind, token=token, **fields)).encode())

    def acknowledge_off(self):
        token, _, op = self.engine.pending
        self.assertEqual(op, 'laser_off')
        self.engine.event(dict(event='laser', request_id=token, laser_gpio_level=0))

    def test_both_presets_use_the_connected_pi_and_plan_thirty_trials_per_rate(self):
        p, e = self.panel, self.engine
        for profile, periods, sample in [('reference', [100, 50, 30, 20, 15, 14, 12], 5),
                                         ('limit', [12, 10, 8, 6, 5], 2)]:
            with self.subTest(profile=profile):
                p.speed.start(profile)
                self.assertTrue(p.speed.running, p.speed.progress.get())
                self.assertTrue(p.owns_output)
                self.assertIs(e.arduino, self.rx)
                root = p.speed.last_root
                self.assertEqual(root.parent, self.captures / 'M6-4')
                plan = json.loads((root / 'experiment.json').read_text(encoding='utf-8'))
                self.assertEqual(plan['receiver'], PI_RECEIVER)
                self.assertEqual(Counter((t['settings']['bit_ms'], t['sample_ms']) for t in plan['trials']),
                                 {(ms, sample): 30 for ms in periods})
                p.start_trial('baseline')
                p.home()
                self.assertTrue(self.rx.commands.empty())
                self.assertIsNone(p.method)
                self.assertFalse(p.allows_pattern('other'))

                p.speed.stop()
                p.tick()
                self.assertTrue(p.speed.running)
                self.acknowledge_off()
                p.tick()
                self.assertFalse(p.owns_output)
                self.assertFalse(p.speed.running)
                self.assertIsNone(e.arduino)
                self.assertIs(p.rx, self.rx)
                self.assertFalse(self.rx.stopping.is_set())

    def test_single_consumer_records_raw_samples_and_only_authorizes_the_active_trial(self):
        p, e = self.panel, self.engine
        p.speed.start('limit')
        e.tick_experiment()
        self.assertIsInstance(e.capture, PiCapture)
        trial, csv_path = e.trial_id, e.capture.folder / 'rx.csv'
        self.acknowledge_off()
        self.assertEqual(self.rx.commands.get_nowait(), dict(cmd='start', token=trial, interval_ms=2))
        self.reply('started', trial, bus=3, address=0x40, config=SAMPLE_CONFIG, interval_ms=2)
        self.reply('sample', trial, seq=0, rx_monotonic_ns=1_000_000, bus_raw=500)
        p.tick()
        self.assertEqual(self.rx.events.qsize(), 2)
        self.assertFalse(p.allows_pattern(None))
        self.assertFalse(p.allows_pattern('other'))
        patterns = []

        def check_request(request):
            if request['cmd'] == 'laser_pattern':
                patterns.append(request['request_id'])
                self.assertTrue(p.allows_pattern(request['request_id']))
            return True

        e.app.send.side_effect = check_request
        e.tick_capture()
        self.assertTrue(self.rx.events.empty())
        self.assertEqual(e.capture.rows, 1)
        self.assertEqual(e.capture.config['confirmed_sample_ms'], 2)
        if e.rx_phase == 'launch_delay':
            e.rx_deadline = time.monotonic() - 1
            e.tick_capture()
        self.assertEqual(patterns, [trial])

        p.stop('operator')
        self.assertFalse(p.allows_pattern(trial))
        self.assertEqual(self.rx.commands.get_nowait(), dict(cmd='stop', token=trial))
        with csv_path.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([(r['sample_seq'], r['rx_monotonic_ns'], r['pv_V']) for r in rows],
                         [('0', '1000000', '2.000')])
        self.acknowledge_off()
        self.reply('sample', trial, seq=1, rx_monotonic_ns=3_000_000, bus_raw=510)
        self.reply('stopped', trial)
        p.tick()
        self.assertTrue(self.rx.events.empty())
        self.assertIsNone(p.logpath)
        self.assertIsNone(e.arduino)
        p.home()
        self.assertEqual(self.rx.commands.get_nowait(), dict(cmd='home', token=p.token))

    def test_complete_batch_displays_only_its_analysis_and_allows_a_point_trial(self):
        p, e = self.panel, self.engine
        p.speed.start('reference')
        root = p.speed.last_root
        analysis = root / 'analysis_fixture'
        e.last_analysis = root.parent / 'previous' / 'analysis'
        p.tick()
        self.assertIsNone(p.speed.shown)
        # Model the last successful acquisition, leaving the analysis queue and display path real.
        e.batch_index = len(e.experiment['trials']) - 1
        row = ('12', '83.33', '5', '30/30')
        with patch.object(e, 'start_analysis', side_effect=lambda _: e.analysis_events.put((True, analysis))) as analyze:
            e.trial_finished('completed')
            analyze.assert_called_once_with(root)
        with patch('Tx.Controller.final_speed_panel.load_speed_results', return_value=('30/30', [row])) as read:
            e.tick_experiment()
            p.tick()
            p.tick()
            read.assert_called_once_with(analysis)
        self.assertEqual(p.speed.headline.get(), '30/30')
        self.assertIn(str(analysis), p.speed.progress.get())
        p.speed.table.insert.assert_called_once_with('', 'end', values=row)
        self.assertFalse(p.speed.running)
        self.assertIsNone(e.arduino)
        p.station, p.run, p.inputs = {'point_id': 'P1'}, Mock(), Mock()
        p.inputs.snapshot.return_value = p.station
        p.start_trial('baseline')
        self.assertEqual(p.phase, 'bench_home', p.status.get())
        self.assertEqual(self.rx.commands.get_nowait(), dict(cmd='home', token=p.token))

    def test_leaving_cancels_plan_and_releases_receiver_with_off_pending(self):
        p, e = self.panel, self.engine
        p.speed.start('limit')
        p.leave()
        self.assertIsNone(e.experiment)
        self.assertIsNone(e.arduino)
        self.assertIsNone(p.rx)
        self.assertTrue(self.rx.stopping.is_set())
        self.assertFalse(p.speed.running)
        self.assertEqual(p.off_token, e.pending[0])
        self.assertFalse(p.allows_pattern('unrelated'))

    def test_failed_limit_setup_unlends_the_receiver_and_creates_no_run(self):
        p = self.panel
        p.speed.periods.set('0')
        p.speed.start('limit')
        self.assertFalse(p.speed.running)
        self.assertIsNone(self.engine.arduino)
        self.assertIsNone(p.phase)
        self.assertFalse(any((self.captures / 'M6-4').iterdir()))
        self.assertFalse(self.rx.stopping.is_set())
