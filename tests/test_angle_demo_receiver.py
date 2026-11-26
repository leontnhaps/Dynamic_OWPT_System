import json
import unittest
from contextlib import contextmanager
from unittest.mock import Mock, patch

from common.angle_demo import (BIT_MS, DECODE_SAMPLE_MS, HOME_PAN, HOME_TILT, encode_angles,
                               payload_from_frame)
from common.ook_experiment import make_frame
from common.ook_frame_decode import decode_captured_frame
from Rx.angle_demo import ANGLE_BATCH_ROWS, CaptureConflict, LISTEN_NS, Receiver, ServoWrites, serve
from Rx.arc_bench import PanTilt
from Rx.ina219_ook_stream import BUS_VOLTAGE, CONFIG, INA219, SAMPLE_CONFIG


class Sensor(INA219):
    def __init__(self):
        self.registers = {CONFIG: 0x399F, BUS_VOLTAGE: 0}

    def transfer(self, register, value=None):
        if value is not None:
            self.registers[register] = value
        return self.registers[register]


class ReceiverTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.now = 9_000_000_000
        self.sensor = Sensor()
        self.bus = Mock()
        self.bus.write.side_effect = len
        self.servos = PanTilt(self.bus, ServoWrites(maxlen=2))
        self.sensor_open = False

        @contextmanager
        def sensor_context():
            self.sensor_open = True
            try:
                yield self.sensor
            finally:
                self.sensor_open = False

        self.rx = Receiver(self.servos, lambda **event: self.events.append(event),
                           sensor_context, lambda: self.now)
        self.addCleanup(self.rx.disarm)

    def samples(self, bits, bit_ms=BIT_MS):
        for bit in bits:
            self.sensor.registers[BUS_VOLTAGE] = (550 if bit == '1' else 220) << 3
            for _ in range(bit_ms // DECODE_SAMPLE_MS):
                self.now += DECODE_SAMPLE_MS * 1_000_000
                self.rx.tick()

    def arm(self, token='T'):
        self.rx.command(dict(cmd='arm', token=token))

    def jog(self, pan_step, tilt_step):
        self.rx.command(dict(cmd='jog', token='J', pan_step=pan_step, tilt_step=tilt_step))

    def test_jog_uses_both_last_commanded_axes_and_returns_write_records(self):
        with self.assertRaisesRegex(ValueError, '원위치'):
            self.jog(1, 0)
        self.bus.write.assert_not_called()
        self.rx.command(dict(cmd='home', token='home'))
        self.jog(5, -1)
        self.jog(0, -5)
        last = self.events[-1]
        self.assertEqual((last['kind'], last['token'], last['pan'], last['tilt']),
                         ('jogged', 'J', HOME_PAN + 5, HOME_TILT - 6))
        self.assertEqual([(r[2], r[3]) for r in last['servo_writes']],
                         [(0, HOME_PAN + 5), (1, HOME_TILT - 6)])
        self.rx.release_servos()
        self.bus.write.reset_mock()
        with self.assertRaisesRegex(ValueError, '원위치'):
            self.jog(0, 1)
        self.bus.write.assert_not_called()

    def test_jog_from_optical_position_checks_both_limits_before_writing(self):
        self.arm()
        self.samples('00' + encode_angles(175, 35) + '00')
        self.rx.command(dict(cmd='commit', token='T'))
        self.assertEqual(self.events[-1]['kind'], 'applied')
        self.bus.write.reset_mock()
        for delta in [(5, 1), (0, 0), (True, 0), (1, 2), (1, 1.0)]:
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                self.jog(*delta)
        self.bus.write.assert_not_called()
        self.jog(5, -1)
        self.assertEqual(self.rx.last_pose, (180, 34))
        self.bus.write.reset_mock()
        with self.assertRaises(ValueError):
            self.jog(1, -5)
        self.bus.write.assert_not_called()

    def test_ssh_xyz_moves_absolute_axes_without_home_sensor_or_ook(self):
        cmd = dict(cmd='move_xyz', token='point-3', pan=62, tilt=23)
        for bad in (dict(pan=-1), dict(pan=True), dict(tilt=36), dict(tilt=23.0), dict(bit_ms=8)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.rx.command(dict(cmd, **bad))
        self.bus.write.assert_not_called()
        self.rx.sensor_context = Mock(side_effect=AssertionError('unexpected ADC access'))
        self.rx.command(cmd)
        self.rx.sensor_context.assert_not_called()
        self.assertEqual(self.rx.last_pose, (62, 23))
        event, = self.events
        self.assertEqual((event['kind'], event['token'], event['pan'], event['tilt']),
                         ('xyz_applied', 'point-3', 62, 23))
        self.assertEqual([row[2:] for row in event['servo_writes']],
                         [[0, 62, '', 'xyz_applied'], [1, 23, '', 'xyz_applied']])
        self.assertNotIn('rx_payload_bits', event)
        with self.assertRaises(CaptureConflict):
            self.rx.command(dict(cmd='commit', token='point-3'))

    def test_jog_and_xyz_during_capture_preserve_each_capture_and_never_write(self):
        for op in ('arm', 'measure', 'start'):
            command = dict(cmd=op, token=op)
            if op == 'start':
                command['interval_ms'] = 1
            self.rx.command(command)
            with self.assertRaises(CaptureConflict):
                self.jog(1, -1)
            with self.assertRaises(CaptureConflict):
                self.rx.command(dict(cmd='move_xyz', token='other', pan=62, tilt=23))
            self.assertEqual(self.rx.token, op)
            self.assertTrue(self.sensor_open)
            self.rx.disarm()
        self.bus.write.assert_not_called()

    def test_raw_ina219_speed_capture_matches_stream_and_never_moves_servos(self):
        self.rx.command(dict(cmd='start', token='S', interval_ms=2))
        self.assertEqual(self.events, [dict(kind='started', token='S', interval_ms=2, bus=3,
                                           address=0x40, config=SAMPLE_CONFIG)])
        self.assertEqual(self.rx.wait_seconds(), .002)
        self.sensor.registers[BUS_VOLTAGE] = 515 << 3
        self.now += 2_000_000
        self.rx.tick()
        for op in ['stop', 'arm', 'measure', 'home', 'cancel']:
            with self.assertRaises(ValueError):
                self.rx.command(dict(cmd=op, token='different'))
        self.assertTrue(self.sensor_open)
        self.assertEqual(self.events[-1], dict(kind='sample', token='S', seq=0,
                                              rx_monotonic_ns=self.now, bus_raw=515))
        self.rx.command(dict(cmd='stop', token='S'))
        self.assertEqual(self.events[-1], dict(kind='stopped', token='S'))
        self.assertIsNone(self.rx.wait_seconds())
        self.assertEqual(self.sensor.registers[CONFIG], 0x399F)
        self.assertFalse(self.sensor_open)
        # A new trial gets its own clock/seq, rather than old samples.
        self.rx.command(dict(cmd='start', token='S2', interval_ms=5))
        self.now += 5_000_000
        self.rx.tick()
        self.assertEqual((self.events[-1]['token'], self.events[-1]['seq']), ('S2', 0))
        self.bus.write.assert_not_called()

    def test_speed_capture_rejects_stale_stop_and_restores_config_on_ssh_eof(self):
        output = Mock(pending=False)
        output.emit.side_effect = lambda **event: self.events.append(event)
        chunks = [b'{"cmd":"start","token":"S","interval_ms":2}\n',
                  b'{"cmd":"stop","token":"old"}\n', b'{"cmd":"home","token":"h"}\n', b'']
        with patch('Rx.angle_demo.select.select', return_value=([123], [], [])), patch(
                'Rx.angle_demo.os.read', side_effect=chunks):
            serve(self.rx, 123, output)
        self.assertEqual([r['kind'] for r in self.events], ['ready', 'started', 'error', 'error'])
        self.assertEqual(self.sensor.registers[CONFIG], 0x399F)
        self.assertFalse(self.sensor_open)
        self.bus.write.assert_not_called()

    def test_one_decoded_frame_waits_for_a_matching_commit_and_writes_once(self):
        self.arm()
        self.assertEqual(self.rx.wait_seconds(), 0.001)
        self.assertEqual(self.events, [])  # Wait for an actual ADC sample before arming Tx.
        self.assertFalse(self.bus.write.called)
        bits = encode_angles(104, 19)
        self.samples('00' + bits + '00' + bits)
        self.assertEqual([e['kind'] for e in self.events], ['armed', 'decoded'])
        payload = payload_from_frame(bits)
        self.assertEqual(self.events[-1], dict(kind='decoded', token='T', rx_payload_bits=payload,
                                              pan=104, tilt=19))
        self.assertFalse(self.sensor_open)
        self.bus.write.assert_not_called()
        with self.assertRaises(CaptureConflict):
            self.rx.command(dict(cmd='move_xyz', token='X', pan=62, tilt=23))
        self.assertEqual(self.rx.pending_payload, payload)
        with self.assertRaises(CaptureConflict):
            self.rx.command(dict(cmd='commit', token='old'))
        self.rx.command(dict(cmd='commit', token='T'))
        applied = self.events[-1]
        self.assertEqual((applied['kind'], applied['rx_payload_bits']), ('applied', payload))
        self.assertEqual((applied['token'], applied['pan'], applied['tilt']), ('T', 104, 19))
        self.assertEqual([(r[2], r[3]) for r in applied['servo_writes']], [(0, 104), (1, 19)])
        self.assertEqual(self.sensor.registers[CONFIG], 0x399F)
        self.assertFalse(self.sensor_open)
        with self.assertRaises(CaptureConflict):
            self.rx.command(dict(cmd='commit', token='T'))
        angle_writes = [c for c in self.bus.write.call_args_list if len(c.args[0]) == 5]
        self.assertEqual(len(angle_writes), 2)

    def test_selected_rate_reports_all_actual_sample_times_and_codes_before_decode(self):
        start = self.now
        token = '광' * 128
        bit_ms = 6
        bits = '000' + encode_angles(103, 24) + '000'
        self.rx.command(dict(cmd='arm', token=token, bit_ms=bit_ms))

        def decode_after_close(rows, bit_ms):
            self.assertFalse(self.sensor_open)
            self.assertEqual(self.events[-1]['kind'], 'angle_samples_end')
            return decode_captured_frame(rows, bit_ms)

        with patch('Rx.angle_demo.decode_captured_frame', side_effect=decode_after_close) as fit:
            self.samples(bits, bit_ms)
            self.assertIsNone(self.rx.decoder)
            fit.assert_not_called()
            self.assertFalse(any(e['kind'] == 'decoded' for e in self.events))
            self.rx.command(dict(cmd='finish_angle', token=token))
            self.assertTrue(self.sensor_open)
            self.samples('0', bit_ms=DECODE_SAMPLE_MS)
            fit.assert_not_called()
            self.samples('0', bit_ms=DECODE_SAMPLE_MS)
            fit.assert_called_once()
        self.assertEqual(self.events[0], dict(kind='armed', token=token, bit_ms=6, sample_ms=1,
                                              bus=3, address=0x40, config=SAMPLE_CONFIG, interval_ms=1,
                                              capture_mode='post_tx'))
        self.assertEqual([e['kind'] for e in self.events[-2:]], ['angle_samples_end', 'decoded'])
        batches = self.events[1:-2]
        self.assertTrue(all(b['kind'] == 'angle_samples' and 1 <= len(b['rows']) <= ANGLE_BATCH_ROWS
                            and b['token'] == token and len(json.dumps(b).encode()) < 4096 for b in batches))
        rows = [r for b in batches for r in b['rows']]
        self.assertGreaterEqual(len(rows), 65 * bit_ms)
        self.assertEqual(self.events[-2]['count'], len(rows))
        for i, (seq, ns, code) in enumerate(rows):
            self.assertEqual((seq, ns, code), (i, start+(i+1)*1_000_000, 550 if (bits+'0')[i//bit_ms]=='1' else 220))
        self.assertEqual(self.events[-1]['rx_payload_bits'], payload_from_frame(encode_angles(103, 24)))
        self.assertFalse(self.bus.write.called)
        self.rx.command(dict(cmd='commit', token=token))
        self.assertEqual(self.events[-1]['kind'], 'applied')
        self.assertEqual(sum(e['kind'] == 'angle_samples_end' for e in self.events), 1)

    def test_finish_requires_armed_matching_token_and_complete_transmission_time(self):
        self.rx.command(dict(cmd='arm', token='T', bit_ms=8))
        with self.assertRaises(CaptureConflict):
            self.rx.command(dict(cmd='finish_angle', token='T'))
        self.samples('0000')
        for cmd, token in [('finish_angle','old'), ('finish_angle','T'), ('cancel','old'), ('commit','T')]:
            with self.assertRaises(CaptureConflict):
                self.rx.command(dict(cmd=cmd, token=token))
        self.assertTrue(self.sensor_open)
        self.assertIsNone(self.rx.finish_after_ns)
        self.samples(encode_angles(103, 24))
        self.rx.command(dict(cmd='finish_angle', token='T'))
        cutoff = self.rx.finish_after_ns
        self.rx.command(dict(cmd='finish_angle', token='T'))
        self.assertEqual(self.rx.finish_after_ns, cutoff)
        self.rx.command(dict(cmd='cancel', token='T'))
        self.samples('00')
        self.assertFalse(any(e['kind'] == 'decoded' for e in self.events))
        self.assertFalse(self.sensor_open)
        self.bus.write.assert_not_called()

    def test_rejected_post_capture_frame_never_grants_a_commit(self):
        self.rx.command(dict(cmd='arm', token='T', bit_ms=8))
        self.samples('000' + encode_angles(103, 24)[:-1])
        self.now += BIT_MS * 1_000_000
        self.samples('000')
        self.rx.command(dict(cmd='finish_angle', token='T'))
        self.samples('0')
        rejection = self.events[-1]
        self.assertEqual(rejection['kind'], 'error')
        self.assertTrue(rejection['message'])
        self.assertFalse(rejection['frame_check']['verified'])
        self.assertEqual(rejection['frame_check']['payload_bits'], payload_from_frame(encode_angles(103, 24)))
        self.assertIn('?', rejection['frame_check']['tail_bits'])
        self.assertEqual(self.events[-2]['kind'], 'angle_samples_end')
        with self.assertRaises(CaptureConflict):
            self.rx.command(dict(cmd='commit', token='T'))
        self.bus.write.assert_not_called()

    def test_timeout_and_cancel_keep_partial_rows_and_acquisition_gaps(self):
        for op in ('timeout', 'cancel'):
            with self.subTest(op=op):
                self.events.clear()
                self.rx.command(dict(cmd='arm', token='capture', bit_ms=8))
                self.samples('00000000')
                self.now += 12_000_000
                gap_ns = self.now
                self.rx.tick()
                if op == 'timeout':
                    self.now += LISTEN_NS
                    self.rx.tick()
                else:
                    self.rx.command(dict(cmd='cancel', token='capture'))
                rows = [r for e in self.events if e['kind'] == 'angle_samples' for r in e['rows']]
                self.assertEqual(rows[-1], [75, gap_ns, 220])
                self.assertEqual(self.events[-2], dict(kind='angle_samples_end', token='capture', count=65))
                self.assertEqual(self.events[-1]['kind'], 'timeout' if op == 'timeout' else 'cancelled')
                self.assertFalse(self.bus.write.called)
                self.assertFalse(self.sensor_open)

    def test_slowest_selected_frame_has_listen_margin_and_invalid_rate_never_arms(self):
        for bad in (True, 7, 8.5):
            with self.assertRaises(ValueError):
                self.rx.command(dict(cmd='arm', token='bad', bit_ms=bad))
            self.assertFalse(self.sensor_open)
        self.rx.command(dict(cmd='arm', token='slow', bit_ms=100))
        self.assertGreaterEqual(self.rx.expires - self.now, (65 * 100 + 2000) * 1_000_000)
        self.assertEqual(self.rx.wait_seconds(), .001)
        self.assertFalse(self.bus.write.called)

    def test_sampler_error_preserves_first_saturated_sample_without_arming_tx(self):
        self.rx.command(dict(cmd='arm', token='T', bit_ms=8))
        self.sensor.registers[BUS_VOLTAGE] = 8191 << 3
        self.now += 1_000_000
        out = Mock(pending=False)
        out.emit.side_effect = lambda **event: self.events.append(event)
        with patch('Rx.angle_demo.select.select', side_effect=[([],[],[]), ([123],[],[])]), patch(
                'Rx.angle_demo.os.read', return_value=b''):
            serve(self.rx, 123, out)
        self.assertEqual([e['kind'] for e in self.events], ['ready', 'angle_samples', 'angle_samples_end', 'error'])
        self.assertEqual(self.events[1]['rows'], [[0, self.now, 8191]])
        self.assertEqual(self.events[-1]['token'], 'T')
        self.assertFalse(self.bus.write.called)

    def test_decoded_pose_cannot_move_after_cancel_or_expiry(self):
        for outcome in ('cancel', 'tick', 'late_commit', 'disconnect'):
            with self.subTest(outcome=outcome):
                self.arm()
                self.samples('00' + encode_angles(103, 20) + '00')
                self.assertEqual(self.events[-1]['kind'], 'decoded')
                self.assertGreater(self.rx.wait_seconds(), 0)
                if outcome == 'cancel':
                    self.rx.command(dict(cmd='cancel', token='T'))
                elif outcome == 'disconnect':
                    with patch('Rx.angle_demo.select.select', return_value=([123], [], [])), patch(
                            'Rx.angle_demo.os.read', return_value=b''):
                        serve(self.rx, 123, Mock(pending=False))
                else:
                    self.now += LISTEN_NS
                    if outcome == 'tick':
                        self.rx.tick()
                    else:
                        self.rx.command(dict(cmd='commit', token='T'))
                    self.assertEqual(self.events[-1]['kind'], 'timeout')
                self.assertIsNone(self.rx.wait_seconds())
                with self.assertRaises(CaptureConflict):
                    self.rx.command(dict(cmd='commit', token='T'))
        self.bus.write.assert_not_called()

    def test_decoded_out_of_range_payload_is_observable_but_cannot_move(self):
        self.arm()
        self.samples('00' + make_frame('1' * 32) + '00')
        self.assertEqual(self.events[-1]['kind'], 'decoded')
        self.assertEqual(self.events[-1]['pan'], 65535)
        with self.assertRaises(ValueError):
            self.rx.command(dict(cmd='commit', token='T'))
        self.assertIsNone(self.rx.pending_payload)
        self.bus.write.assert_not_called()

    def test_home_clears_partial_frame_and_works_without_light_or_sensor(self):
        self.arm()
        bits = encode_angles(15, 4)
        self.samples('00' + bits[:12])
        self.rx.command(dict(cmd='home', token='HOME'))
        self.samples(bits[12:] + '00')
        self.assertEqual([e['kind'] for e in self.events], ['armed', 'home'])
        home = self.events[-1]
        self.assertEqual((home['pan'], home['tilt'], home['token']), (HOME_PAN, HOME_TILT, 'HOME'))
        self.rx.sensor_context = Mock(side_effect=OSError('no INA219'))
        with self.assertRaisesRegex(OSError, 'no INA219'):
            self.arm()
        self.rx.command(dict(cmd='home', token='HOME2'))
        self.assertEqual(self.events[-1]['token'], 'HOME2')

    def test_measure_excludes_settling_and_keeps_actual_time_sequence_and_raw_codes(self):
        start = self.now
        self.rx.command(dict(cmd='measure', token='window1'))
        self.sensor.registers[BUS_VOLTAGE] = 120 << 3
        for i in range(1, 2101):
            self.now = start + i * 5_000_000
            if i == 100:
                self.sensor.registers[BUS_VOLTAGE] = 720 << 3
            self.rx.tick()
        samples = self.events[:-1]
        self.assertEqual(len(samples), 2000)
        self.assertTrue(all(ev['kind'] == 'voltage' and ev['bus_raw'] == 720 for ev in samples))
        self.assertEqual((samples[0]['rx_monotonic_ns'], samples[-1]['rx_monotonic_ns']),
                         (start + 500_000_000, start + 10_495_000_000))
        self.assertEqual((samples[0]['seq'], samples[-1]['seq']), (99, 2098))
        self.assertEqual(self.events[-1], dict(kind='measured', token='window1', count=2000,
                         window_start_ns=start + 500_000_000, window_end_ns=start + 10_500_000_000,
                         max_gap_ns=5_000_000))
        self.assertFalse(self.sensor_open)
        self.assertEqual(self.sensor.registers[CONFIG], 0x399F)
        self.assertFalse(self.bus.write.called)

    def test_measure_rejects_a_long_gap_or_sparse_coverage_and_can_restart(self):
        for exclude in (lambda i: 120 <= i < 131, lambda i: i >= 100 and i % 2 == 1):
            self.events.clear()
            start = self.now
            self.rx.command(dict(cmd='measure', token='M'))
            for i in range(1, 2101):
                self.now = start + i * 5_000_000
                if not exclude(i):
                    self.rx.tick()
            self.assertEqual(self.events[-1]['kind'], 'error')
            self.assertGreater(self.events[-1]['count'], 0)
            self.assertTrue(all(ev['kind'] == 'voltage' for ev in self.events[:-1]))
            self.assertFalse(self.sensor_open)

    def test_cancel_discards_partial_measurement_and_restores_sensor(self):
        self.rx.command(dict(cmd='measure', token='M'))
        self.now += 520_000_000
        self.rx.tick()
        self.rx.command(dict(cmd='cancel', token='M'))
        self.now += 3_000_000_000
        self.rx.tick()
        self.assertEqual([event['kind'] for event in self.events], ['voltage', 'cancelled'])
        self.assertEqual(self.sensor.registers[CONFIG], 0x399F)
        self.assertFalse(self.sensor_open)
        self.assertFalse(self.bus.write.called)

    def test_window_saturation_retains_raw_sample_and_fails(self):
        self.rx.command(dict(cmd='measure', token='M'))
        self.sensor.registers[BUS_VOLTAGE] = 8191 << 3
        self.now += 495_000_000
        self.rx.tick()
        self.assertFalse(self.events)
        self.now += 5_000_000
        with self.assertRaisesRegex(ValueError, '포화'):
            self.rx.tick()
        self.assertEqual(self.events[-1]['bus_raw'], 8191)
        self.assertFalse(self.bus.write.called)

    def test_cancel_timeout_and_invalid_ssh_angles_do_not_move(self):
        for cmd in [dict(cmd='move', token='T'), dict(cmd='home', token='T', tilt=25)]:
            with self.assertRaises(ValueError):
                self.rx.command(cmd)
        self.arm()
        self.samples('0')
        self.rx.command(dict(cmd='cancel', token='C'))
        self.samples(encode_angles(4, 12))
        self.arm('T2')
        self.now += LISTEN_NS
        self.rx.tick()
        self.assertEqual([e['kind'] for e in self.events], ['armed', 'cancelled', 'timeout'])
        self.assertEqual(self.events[-1]['token'], 'T2')
        self.assertIsNone(self.rx.wait_seconds())
        self.assertFalse(self.bus.write.called)

    def test_an_incomplete_servo_pair_is_released_without_success_ack(self):
        def fail_on_tilt(data):
            if len(data) == 5 and data[0] == 0x0A:
                raise OSError('tilt write failed')
            return len(data)

        self.rx.command(dict(cmd='home', token='H'))
        self.bus.write.side_effect = fail_on_tilt
        with self.assertRaisesRegex(OSError, 'tilt write failed'):
            self.jog(1, 1)
        self.assertEqual([e['kind'] for e in self.events], ['home'])
        self.assertEqual([c.args[0] for c in self.bus.write.call_args_list[-2:]],
                         [bytes((0x09, 0x10)), bytes((0x0D, 0x10))])
        self.bus.write.reset_mock()
        with self.assertRaisesRegex(ValueError, '원위치'):
            self.jog(1, -1)
        self.bus.write.assert_not_called()

    def test_failed_initialisation_releases_both_and_reinitialises_on_next_home(self):
        def fail_on_prescale(data):
            if data[0] == 0xFE:
                raise OSError('prescale write failed')
            return len(data)

        self.bus.write.side_effect = fail_on_prescale
        with self.assertRaisesRegex(OSError, 'prescale write failed'):
            self.rx.command(dict(cmd='home', token='H'))
        self.assertFalse(self.events)
        releases = [bytes((0x09, 0x10)), bytes((0x0D, 0x10))]
        self.assertEqual([c.args[0] for c in self.bus.write.call_args_list[-2:]], releases)
        self.bus.write.reset_mock()
        self.bus.write.side_effect = len
        self.rx.command(dict(cmd='home', token='again'))
        self.assertIn(bytes((0xFE, 121)), [c.args[0] for c in self.bus.write.call_args_list])
        self.assertEqual(self.events[-1]['kind'], 'home')
        self.rx.release_servos()
        self.assertEqual([c.args[0] for c in self.bus.write.call_args_list[-2:]], releases)

    def test_input_fragments_and_eof_disarm_and_restore_sensor(self):
        output = Mock(pending=False)
        output.emit.side_effect = lambda **event: self.events.append(event)
        with patch('Rx.angle_demo.select.select', return_value=([123], [], [])), patch(
            'Rx.angle_demo.os.read', side_effect=[b'{"cmd":"ar', b'm","token":"T"}\n', b'']
        ):
            serve(self.rx, 123, output)
        self.assertFalse(self.sensor_open)
        self.assertEqual(self.sensor.registers[CONFIG], 0x399F)
        self.assertEqual(self.events, [dict(kind='ready')])
        self.assertFalse(self.bus.write.called)


if __name__ == '__main__':
    unittest.main()
