"""Synthetic evidence and UI state tests; these are not optical measurements."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from common.ook_experiment import HEADER, RULES, make_plan, number_list
from Tx.Controller.ook_analysis import decode, score_payload, analyze_experiment, aggregate
from Tx.Controller.arduino_capture import RxCapture, parse_reading
from tests import test_arduino_capture as capture_fixture


class DecoderTests(unittest.TestCase):
    payload = '10110010'*4

    def signal(self, tb=30, sample=5, offset=100.7, payload=None):
        bits = HEADER+(payload or self.payload)+'0000'
        t = np.arange(0, len(bits)*tb+offset+100, sample, dtype=float)
        indexes = np.floor((t-offset)/tb).astype(int)
        y = np.array([400+32*int(bits[i]) if 0 <= i < len(bits) else 400 for i in indexes], dtype=float)
        return t, y

    def test_clean_frames_at_different_phases(self):
        for phase in [100.1, 102.4, 104.9]:
            t, y = self.signal(offset=phase)
            result = decode(t, y, 30)
            self.assertTrue(score_payload(result['decoded'], self.payload)['success'])
            self.assertLess(abs(result['origin_ms']-phase), 6)

    def test_wrong_bit_is_counted_not_fitted_away(self):
        damaged = self.payload[:17] + str(1-int(self.payload[17])) + self.payload[18:]
        t, y = self.signal(payload=damaged)
        result = decode(t, y, 30)
        score = score_payload(result['decoded'], self.payload)
        self.assertEqual(score['bit_errors'], 1)
        self.assertFalse(score['success'])
        self.assertTrue(score_payload(result['decoded'], damaged)['success'])

    def test_missing_sample_is_erasure_not_interpolated(self):
        t, y = self.signal()
        center = 100.7 + (len(HEADER)+10.5)*30
        keep = abs(t-center)>15
        result = decode(t[keep], y[keep], 30)
        self.assertTrue(result['sync'])
        self.assertIn('?', result['decoded'])
        self.assertGreater(score_payload(result['decoded'], self.payload)['erasures'], 0)

    def test_constant_noise_and_undersampling_fail_sync(self):
        t, y = self.signal()
        for signal in [np.full(len(t), 400.), np.random.default_rng(7).normal(400, 1, len(t))]:
            self.assertFalse(decode(t, signal, 30)['sync'])
        t, y = self.signal(tb=2, sample=10)
        self.assertFalse(decode(t, y, 2)['sync'])

    def test_plan_reproducible_and_short_bit_inputs_preserved(self):
        plan = make_plan([30, .5], [2, 10], 3, 7)
        self.assertEqual(plan, make_plan([30, .5], [2, 10], 3, 7))
        self.assertEqual(len(plan['trials']), 12)
        self.assertEqual({t['settings']['bit_ms'] for t in plan['trials']}, {30, .5})
        self.assertEqual(len({t['payload'] for t in plan['trials']}), 12)
        for value in ('2,0', '2.5', '1001'):
            with self.assertRaises(ValueError): number_list(value, integer=True)

    def test_sync_failure_not_hidden_by_conditional_ber(self):
        base = dict(sample_ms=5, bit_ms=30, mode='sweep', tx_complete=True,
                    end_reason='completed', sync=True, success=True, status='success',
                    bit_errors=0, decided_bits=32, erasures=0, delta_pv_v=.2, median_dt_ms=5)
        rows = [base, dict(base, sync=False, success=False, status='sync_failed',
                           decided_bits=0, erasures=32, delta_pv_v=None),
                dict(base, tx_complete=False, end_reason='tx_timing_overrun', success=False)]
        result = aggregate(rows)[0]
        self.assertEqual(result['block_success_pct'], 50)
        self.assertEqual(result['conditional_ber'], 0)
        self.assertEqual(result['sync_failures'], 1)
        self.assertEqual(result['incomplete'], 1)

    def test_complete_artifact_analysis(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); plan = make_plan([30], [5], 1, 9)
            (root/'experiment.json').write_text(json.dumps(plan))
            spec = dict(plan['trials'][0], rules_version=RULES['version'])
            folder = root/'trial_00001'; folder.mkdir()
            capture = RxCapture(folder, spec['settings'], 'TEST', 'id', 5, spec)
            capture.confirm_interval('OK RAW2 R1 baud=250000 interval_ms=5')
            t, y = self.signal(payload=spec['payload'])
            for seq, (ms, adc) in enumerate(zip(t,y)):
                line = f'R1,{seq},{int(ms)},{int(adc)}'
                capture.write(dict(line=line, pc_receive_unix_ns=1, pc_receive_monotonic_ns=1), parse_reading(line))
            capture.finish('completed', 'completed')
            trace = [dict(index=i, level=int(b), scheduled_ms=i*30, gpio_call_done_ms=i*30+.1)
                     for i,b in enumerate(spec['settings']['bits'])]
            (folder/'events.jsonl').write_text(json.dumps(dict(event='laser_pattern', request_id='id', state='completed', trace=trace))+'\n')
            out = analyze_experiment(root)
            result = json.loads((out/'analysis.json').read_text())
            self.assertEqual(result['conditions'][0]['block_success_pct'], 100)
            self.assertEqual(result['conditions'][0]['decided_bits'], 32)
            self.assertTrue((out/'rates.csv').exists())
            self.assertGreater((out/'block_success.png').stat().st_size, 1000)
            # Reanalysis creates a new output directory, preserving the prior result.
            self.assertNotEqual(out, analyze_experiment(root))


class BatchTests(unittest.TestCase):
    panel = capture_fixture.CaptureTests.panel
    arm = capture_fixture.CaptureTests.arm
    ready = capture_fixture.CaptureTests.ready

    def test_batch_completes_in_order_and_analyzes(self):
        p = self.panel(); p.sweep_bits.set('30'); p.sweep_samples.set('5'); p.sweep_count.set('2')
        p.start_experiment('sweep')
        root = p.experiment_root
        self.assertTrue(p.busy)
        with patch.object(p, 'start_analysis') as analyze:
            for i in range(2):
                p.next_trial_at = time.monotonic()-1
                p.tick()
                self.assertEqual(p.rx_phase, 'off')
                p.event(dict(event='laser', request_id=p.pending[0], laser_gpio_level=0))
                self.ready(p)
                self.assertEqual(p.rx_phase, 'launch_delay')
                p.rx_deadline = time.monotonic()-1; p.tick()
                token = p.run_id
                self.assertEqual(self.sent[-1]['cmd'], 'laser_pattern')
                p.event(dict(event='laser_pattern', state='started', request_id=token))
                p.event(dict(event='laser_pattern', state='completed', request_id=token))
                p.rx_deadline=time.monotonic()-1; p.tick()
            self.assertIsNone(p.experiment)
            analyze.assert_called_once_with(root)
        self.assertEqual(len(list(root.glob('trial_*/summary.json'))), 2)

    def test_cancel_between_trials_cannot_restart(self):
        p=self.panel(); p.start_experiment('sweep')
        p.off(); p.pending=None; p.tick()
        self.assertIsNone(p.experiment)
        self.assertFalse(any(x['cmd']=='laser_pattern' for x in self.sent))

    def test_cancel_launch_delay_and_late_records_cannot_restart(self):
        p=self.panel(); p.sweep_samples.set('5'); p.start_experiment('sweep'); p.tick()
        p.event(dict(event='laser', request_id=p.pending[0], laser_gpio_level=0))
        self.ready(p); self.assertEqual(p.rx_phase, 'launch_delay')
        p.off()
        p.arduino.emit(kind='line',line=capture_fixture.reading(1500)); p.tick()
        self.assertIsNone(p.experiment)
        self.assertFalse(any(x['cmd']=='laser_pattern' for x in self.sent))

    def test_batch_cannot_disable_capture_by_changing_checkbox(self):
        p=self.panel(); p.start_experiment('sweep'); p.rx_save.set(False); p.tick()
        self.assertIsNotNone(p.capture)
        self.assertFalse(any(x['cmd']=='laser_pattern' for x in self.sent))
        p.off()

    def test_configuration_mismatch_stops_before_transmission(self):
        p=self.panel(); self.arm(p)
        p.arduino.emit(kind='command_sent',command='raw 5',token=p.trial_id)
        p.arduino.emit(kind='line',line='OK RAW2 R1 baud=250000 interval_ms=10')
        p.tick()
        self.assertIsNone(p.capture)
        self.assertFalse(any(x['cmd']=='laser_pattern' for x in self.sent))

    def test_batch_disconnect_does_not_resume_on_reconnect(self):
        p=self.panel(); p.start_experiment('sweep')
        p.event(dict(event='network',port=7600,state='disconnected'))
        p.event(dict(event='hello',agent_state='connected')); p.tick()
        self.assertIsNone(p.experiment)
        self.assertFalse(any(x['cmd']=='laser_pattern' for x in self.sent))


if __name__ == '__main__': unittest.main()
