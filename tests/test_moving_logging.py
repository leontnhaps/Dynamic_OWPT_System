import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np

from common.tx_tracking import Sample
from Tx.Controller.moving_logging import LOG_SCHEMA, MovingRunLog, generate_plots
from Tx.Controller.stationary_learning import RunLog
from tools.analysis import plot_moving_training as plots


def sample(target=None):
    return Sample(7, 10., 10.02, 1296, 972, target=target, count=int(target is not None),
                  seq=31, receive_unix_ns=1_800_000_000_000_000_000, inference_ms=20., simulated=False)


def summary(**extra):
    return dict(episode=1, mode='resume', reason='manual_episode_end',
                initial_command=[-20, 3], new_transitions=2, reward_sum=-.7,
                reward_mean=-.35, reward_count=2, rms_px=24., p95_px=30.,
                hit_ratio=.5, valid_ratio=2/3, checkpoint='checkpoints/episode_000001_test', **extra)


class MovingLogTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)/'captures'/'M5-3'/'run'
        self.log = MovingRunLog(self.folder, {'schema': 'test', 'period_s': .72})
        self.addCleanup(self.log.close)

    def test_episode_extension_preserves_stationary_fields_and_does_not_plot(self):
        self.assertNotIn('reward_sum', RunLog.fields['episodes'])
        self.assertEqual(MovingRunLog.fields['episodes'][:-3], RunLog.fields['episodes'])
        self.assertEqual(MovingRunLog.fields['steps'], RunLog.fields['steps'])
        self.assertIsNot(MovingRunLog.fields['steps'], RunLog.fields['steps'])
        config = json.loads((self.folder/'config.json').read_text())
        self.assertEqual(config['schema'], 'test')
        self.assertEqual(config['moving_log_schema'], LOG_SCHEMA)
        with patch('Tx.Controller.moving_logging.generate_plots') as plot:
            self.log.episode(summary())
            plot.assert_not_called()
        with (self.folder/'episodes.csv').open(newline='') as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row['reward_sum'], '-0.7')
        self.assertEqual(row['reward_count'], '2')
        self.assertEqual(row['checkpoint'], summary()['checkpoint'])
        self.assertTrue((self.folder/'summary.csv').is_file())

    def test_missing_and_duplicate_frame_ids_preserve_originals_and_processing_time(self):
        jpeg = b'\xff\xd8original bytes\xff\xd9'
        metadata = {'SensorTimestamp': 7_100_000_000,
                    'servo_at_capture_start': {'commanded': {'command_id': 'cmd-7', 'pan': -20}}}
        with patch('Tx.Controller.moving_logging.time.monotonic', side_effect=[20., 21., 22., 23.]):
            first = self.log.record_frame(1, 'tracking', sample(), metadata, jpeg, step=3,
                                          detection={'image': object(), 'target': None, 'count': 0})
        second = self.log.record_frame(1, 'tracking', sample(), metadata, jpeg, step=3)
        self.assertNotEqual(first['image'], second['image'])
        self.assertEqual((self.folder/first['image']).read_bytes(), jpeg)
        self.assertEqual(first['sha256'], hashlib.sha256(jpeg).hexdigest())
        self.assertEqual(first['storage_finished_s'], 23.)
        sidecar = json.loads((self.folder/first['sidecar']).read_text())
        self.assertEqual(sidecar['frame_metadata'], metadata)
        self.assertEqual(sidecar['sample']['processed'], 10.02)
        self.assertEqual(sidecar['status'], 'missing')
        self.assertEqual(sidecar['jpeg_saved_s'], 21.)
        self.assertNotIn('image', sidecar['detection'])
        indexed = [json.loads(line) for line in (self.folder/'observations.jsonl').read_text().splitlines()]
        self.assertEqual(len(indexed), 2)
        self.assertEqual(indexed[0]['artifacts_saved_s'], 22.)
        with (self.folder/'detections.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['status'], 'missing')
        self.assertEqual(rows[0]['u'], '')
        self.assertEqual(rows[0]['processed_s'], '10.02')

    def test_invalid_measurement_is_identified_and_raw_image_is_still_saved(self):
        target = {'center': [float('nan'), 5.], 'box': [0., 0., float('inf'), 10.], 'confidence': .9}
        result = self.log.record_frame(2, 'tracking', sample(target), {}, b'original',
                                      status='invalid_box', detection={'target': target})
        data = json.loads((self.folder/result['sidecar']).read_text())
        self.assertIsNone(data['sample']['target']['center'][0])
        self.assertTrue(any(row['path'] == '$.sample.target.center[0]' and row['value'] == 'nan'
                            for row in data['invalid_numeric_fields']))
        self.assertEqual((self.folder/result['image']).read_bytes(), b'original')
        with (self.folder/'detections.csv').open(newline='') as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row['status'], 'invalid_box')
        self.assertEqual(row['u'], '')
        self.assertEqual(row['x2'], '')

    def test_kalman_and_raw10_event_preserves_full_numeric_arrays(self):
        self.log.event('kalman_observation', episode=1, frame_id=7,
                       kalman={'P': np.eye(4), 'state': np.zeros(4)}, raw10=np.arange(10),
                       normalized10=np.arange(10)/10, exclusion_reason=None)
        event = json.loads((self.folder/'events.jsonl').read_text())
        self.assertEqual(event['kalman']['P'], np.eye(4).tolist())
        self.assertEqual(len(event['raw10']), 10)
        self.assertEqual(event['frame_id'], 7)

    def test_plot_failure_cannot_change_saved_episode_or_checkpoint_reference(self):
        self.log.episode(summary())
        before = (self.folder/'episodes.csv').read_bytes()
        with patch('Tx.Controller.moving_logging.generate_plots', side_effect=RuntimeError('renderer failed')):
            report = self.log.generate_plots()
        self.assertEqual(report['files'], [])
        self.assertIn('renderer failed', report['errors'][0])
        self.assertEqual((self.folder/'episodes.csv').read_bytes(), before)
        event = json.loads((self.folder/'events.jsonl').read_text())
        self.assertEqual(event['event'], 'plots_incomplete')

    def test_close_is_idempotent_and_late_frame_cannot_write(self):
        self.log.close()
        self.log.close()
        with self.assertRaisesRegex(ValueError, 'closed'):
            self.log.record_frame(1, 'tracking', sample(), {}, b'jpeg')
        self.assertFalse(list((self.folder/'frames').iterdir()))


class MovingPlotTests(unittest.TestCase):
    def test_missing_series_remains_gap_and_single_point_has_marker(self):
        rows = [{'episode': '1', 'reward_mean': '-.25'},
                {'episode': '2', 'reward_mean': ''},
                {'episode': '3', 'reward_mean': 'nan'}]
        x, y = plots._series(rows, 'episode', 'reward_mean')
        self.assertEqual(x, [1., 2., 3.])
        self.assertEqual(y[0], -.25)
        self.assertTrue(all(math.isnan(value) for value in y[1:]))
        axis = Mock()
        plots._draw(axis, rows, 'episode', [('reward_mean', 'Mean')], 'reward', 'test')
        args, kwargs = axis.plot.call_args
        self.assertEqual(args[0], x)
        self.assertTrue(math.isnan(args[1][1]))
        self.assertEqual(kwargs['marker'], 'o')
        low, high = axis.set_xlim.call_args.args
        self.assertLess(low, 1.)
        self.assertGreater(high, 3.)
        axis.xaxis.set_major_locator.assert_called_once()
        axis.text.assert_not_called()

    def test_missing_metrics_show_no_measurement_and_real_zero_is_not_missing(self):
        axis = Mock()
        plots._draw(axis, [{'episode': '1'}], 'episode', [('rms_px', 'RMS')], 'px', 'test')
        axis.plot.assert_not_called()
        axis.text.assert_called_once()
        axis = Mock()
        plots._draw(axis, [{'episode': '1', 'rms_px': '0'}], 'episode', [('rms_px', 'RMS')], 'px', 'test')
        axis.plot.assert_called_once()
        self.assertEqual(axis.plot.call_args.args[1], [0.])
        axis.set_xlim.assert_called_once_with(.5, 1.5)

    def test_graph_error_is_reported_and_later_graphs_still_attempted(self):
        with TemporaryDirectory() as name, patch.object(plots, '_save_figure') as save:
            save.side_effect = [OSError('disk full'), None, None, None]
            report = plots.generate_plots(name)
            self.assertEqual(save.call_count, 4)
            self.assertEqual(len(report['files']), 3)
            self.assertEqual(len(report['errors']), 1)
            self.assertIn('disk full', report['errors'][0])
            manifest = json.loads((Path(name)/'graphs'/'manifest.json').read_text())
            self.assertEqual(manifest['measured_counts']['reward_sum'], 0)
            self.assertEqual(manifest['errors'], report['errors'])

    def test_wrapper_handles_missing_plot_dependency(self):
        with patch.object(plots, 'generate_plots', side_effect=ImportError('matplotlib missing')):
            report = generate_plots('unused')
        self.assertEqual(report['files'], [])
        self.assertIn('matplotlib missing', report['errors'][0])

    def test_real_headless_graphs_save_without_modifying_csvs(self):
        from PIL import Image
        with TemporaryDirectory() as name:
            folder = Path(name)/'captures'/'M5-3'/'run'
            log = MovingRunLog(folder, {})
            try:
                log.episode(summary())
                # Blank performance measurements from no valid results stay blank.
                log.episode(dict(episode=2, mode='resume', reason='initial_target_lost',
                                 reward_sum=None, reward_mean=None, reward_count=0))
                log.row('updates', dict(episode=1, update=121, actor_loss=-1.2,
                                       critic_loss=.2, alpha=.9, alpha_loss=-.05))
                before = {filename: (folder/filename).read_bytes() for filename in ('episodes.csv', 'updates.csv')}
                with ThreadPoolExecutor(max_workers=1) as worker:
                    report = worker.submit(log.generate_plots).result(timeout=30)
                self.assertEqual(report['errors'], [])
                self.assertEqual(len(report['files']), 4)
                for filename in report['files']:
                    with Image.open(filename) as image:
                        self.assertEqual(image.format, 'PNG')
                        self.assertGreater(image.width, 100)
                for filename, data in before.items():
                    self.assertEqual((folder/filename).read_bytes(), data)
                manifest = json.loads((folder/'graphs'/'manifest.json').read_text())
                self.assertEqual(manifest['sources'], {'episodes.csv': 2, 'updates.csv': 1})
                self.assertEqual(manifest['measured_counts']['reward_sum'], 1)
                self.assertEqual(manifest['measured_counts']['rms_px'], 1)
                self.assertEqual(manifest['measured_counts']['actor_loss'], 1)
                self.assertFalse(list((folder/'graphs').glob('*.tmp')))
            finally:
                log.close()


if __name__ == '__main__':
    unittest.main()
