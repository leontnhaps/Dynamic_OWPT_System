import csv
import json
import tempfile
import unittest
from pathlib import Path

from common.angle_demo import encode_angles
from Rx.ina219_ook_stream import SAMPLE_CONFIG
from Tx.Controller.angle_waveform import AngleWaveform


class WaveformFiles(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.wave = AngleWaveform(Path(tmp.name) / 'M6-4', 'trial', 8, encode_angles(103, 24),
                                  context={'method': 'xyz', 'rx_ssh_target': 'user@pi'})
        self.addCleanup(self.wave.finish, False, 'cleanup')

    def event(self, kind, **data):
        return dict(kind=kind, token='trial', **data)

    def test_timeout_keeps_complete_waveform_and_sample_clock(self):
        w = self.wave
        w.tx_event(self.event('armed', bit_ms=8, sample_ms=1, interval_ms=1, bus=3,
                              address=0x40, config=SAMPLE_CONFIG))
        w.samples(self.event('angle_samples', rows=[[0, 100_000, 250], [1, 1_100_000, 500]],
                             pc_receive_unix_ns=98000, pc_receive_monotonic_ns=305))
        w.samples(self.event('angle_samples', rows=[[11, 11_100_000, 501]]))
        w.end_samples(self.event('angle_samples_end', count=3))
        w.finish(False, 'timeout')
        summary = json.loads((w.folder / 'summary.json').read_text(encoding='utf-8'))
        self.assertTrue(summary['waveform_complete'])
        self.assertFalse(summary['angle_verified'])
        self.assertEqual((summary['missing_sample_slots'], summary['gaps_longer_than_one_bit']), (9, 1))
        with (w.folder / 'rx.csv').open(encoding='utf-8') as stream:
            first, second, last = list(csv.DictReader(stream))
        self.assertEqual((first['pv_V'], last['pv_V'], last['sample_dt_ms']), ('1.000', '2.004', '10.0'))
        self.assertEqual(first['pc_receive_monotonic_ns'], second['pc_receive_monotonic_ns'])
        self.assertNotEqual(first['rx_monotonic_ns'], second['rx_monotonic_ns'])

    def test_partial_capture_keeps_diagnostics_from_before_valid_arm(self):
        w = self.wave
        w.samples(self.event('angle_samples', rows=[[0, 100_000, 8191]]))
        with self.assertRaises(ValueError):
            w.end_samples(self.event('angle_samples_end', count=2))
        w.finish(False, 'capture interrupted')
        summary = json.loads((w.folder / 'summary.json').read_text(encoding='utf-8'))
        self.assertFalse(summary['interval_confirmed'])
        self.assertFalse(summary['waveform_complete'])
        self.assertEqual((summary['clipped_rows'], summary['pc_rows'], summary['rx_reported_rows']), (1, 1, 2))

    def test_missing_trailer_sample_is_saved_alongside_observed_payload(self):
        w = self.wave
        check = dict(verified=False, header_verified=True, payload_bits=f'{103:016b}{24:016b}',
                     tail_bits='000?', origin_ms=60.0, reason='끝 신호 4번째 비트에 전압 표본이 없습니다.')
        w.tx_event(dict(event='angle_verification', token='trial', frame_check=check))
        w.finish(False, check['reason'])
        summary = json.loads((w.folder / 'summary.json').read_text(encoding='utf-8'))
        self.assertFalse(summary['angle_verified'])
        self.assertEqual(summary['frame_check'], check)

    def test_wrong_sampling_or_bitrate_never_confirms_settings(self):
        w = self.wave
        with self.assertRaises(ValueError):
            w.tx_event(self.event('armed', bit_ms=4, sample_ms=1, interval_ms=1,
                                  bus=3, address=0x40, config=SAMPLE_CONFIG))
        self.assertFalse(w.config['interval_confirmed'])
