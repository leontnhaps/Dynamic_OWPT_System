import unittest
from contextlib import contextmanager
from unittest.mock import Mock, patch

from common.angle_demo import BIT_MS, HOME_PAN, HOME_TILT, encode_angles
from Rx.angle_demo import LISTEN_NS, Receiver, ServoWrites, serve
from Rx.arc_bench import PanTilt, SAMPLE_MS
from Rx.ina219_ook_stream import BUS_VOLTAGE, CONFIG, INA219


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

    def samples(self, bits):
        for bit in bits:
            self.sensor.registers[BUS_VOLTAGE] = (550 if bit == '1' else 220) << 3
            for _ in range(BIT_MS // SAMPLE_MS):
                self.now += SAMPLE_MS * 1_000_000
                self.rx.tick()

    def arm(self, token='T'):
        self.rx.command(dict(cmd='arm', token=token))

    def test_one_real_decoder_frame_gives_one_ack_with_servo_writes(self):
        self.arm()
        self.assertEqual(self.events, [])  # Wait for an actual ADC sample before arming Tx.
        self.assertFalse(self.bus.write.called)
        bits = encode_angles(104, 19)
        self.samples('00' + bits + '00' + bits)
        self.assertEqual([e['kind'] for e in self.events], ['armed', 'applied'])
        applied = self.events[-1]
        self.assertEqual((applied['token'], applied['pan'], applied['tilt']), ('T', 104, 19))
        self.assertEqual([(r[2], r[3]) for r in applied['servo_writes']], [(0, 104), (1, 19)])
        self.assertEqual(self.sensor.registers[CONFIG], 0x399F)
        self.assertFalse(self.sensor_open)
        angle_writes = [c for c in self.bus.write.call_args_list if len(c.args[0]) == 5]
        self.assertEqual(len(angle_writes), 2)

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
        for i in range(1, 301):
            self.now = start + i * 5_000_000
            if i == 100:
                self.sensor.registers[BUS_VOLTAGE] = 720 << 3
            self.rx.tick()
        samples = self.events[:-1]
        self.assertEqual(len(samples), 200)
        self.assertTrue(all(ev['kind'] == 'voltage' and ev['bus_raw'] == 720 for ev in samples))
        self.assertEqual((samples[0]['rx_monotonic_ns'], samples[-1]['rx_monotonic_ns']),
                         (start + 500_000_000, start + 1_495_000_000))
        self.assertEqual((samples[0]['seq'], samples[-1]['seq']), (99, 298))
        self.assertEqual(self.events[-1], dict(kind='measured', token='window1', count=200,
                         window_start_ns=start + 500_000_000, window_end_ns=start + 1_500_000_000,
                         max_gap_ns=5_000_000))
        self.assertFalse(self.sensor_open)
        self.assertEqual(self.sensor.registers[CONFIG], 0x399F)
        self.assertFalse(self.bus.write.called)

    def test_measure_rejects_a_long_gap_or_sparse_coverage_and_can_restart(self):
        for exclude in (lambda i: 120 <= i < 131, lambda i: i >= 100 and i % 2 == 1):
            self.events.clear()
            start = self.now
            self.rx.command(dict(cmd='measure', token='M'))
            for i in range(1, 301):
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

        self.bus.write.side_effect = fail_on_tilt
        with self.assertRaisesRegex(OSError, 'tilt write failed'):
            self.rx.command(dict(cmd='home', token='H'))
        self.assertEqual(self.events, [])
        self.assertEqual([c.args[0] for c in self.bus.write.call_args_list[-2:]],
                         [bytes((0x09, 0x10)), bytes((0x0D, 0x10))])

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
