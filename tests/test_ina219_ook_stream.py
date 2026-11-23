import unittest

from Rx.ina219_ook_stream import BUS_VOLTAGE, CONFIG, SAMPLE_CONFIG, INA219, Sampler, swap_word


class MemoryINA219(INA219):
    def __init__(self):
        self.registers = {CONFIG: 0x399F, BUS_VOLTAGE: (1150 << 3) | 3}

    def transfer(self, register, value=None):
        if value is not None:
            self.registers[register] = value
        return self.registers[register]


class PiSamplerTests(unittest.TestCase):
    def setUp(self):
        self.sensor = MemoryINA219()
        self.events = []
        self.now = 3_400_000_000_000
        self.worker = Sampler(self.sensor, 3, 0x40, lambda **e: self.events.append(e), lambda: self.now)
        self.addCleanup(self.worker.stop)

    def start(self):
        self.worker.command(dict(cmd='start', token='trial', interval_ms=5))

    def test_byte_order_and_warmup_and_units(self):
        self.assertEqual(swap_word(0x1234), 0x3412)
        self.start()
        self.worker.sample_due()
        self.assertEqual([e['kind'] for e in self.events], ['started'])
        self.assertEqual(self.sensor.registers[CONFIG], SAMPLE_CONFIG)
        self.now += 5_000_000
        self.worker.sample_due()
        self.assertEqual(self.events[-1], dict(kind='sample', seq=0, token='trial',
                                              rx_monotonic_ns=self.now, bus_raw=1150))
        self.worker.command(dict(cmd='stop', token='trial'))
        self.assertEqual(self.sensor.registers[CONFIG], 0x399F)
        self.assertEqual(self.events[-1], dict(kind='stopped', token='trial'))

    def test_late_wakeup_leaves_missing_slots(self):
        self.start()
        self.now += 26_000_000
        self.worker.sample_due()
        self.worker.sample_due()
        self.assertEqual([e['seq'] for e in self.events if e['kind'] == 'sample'], [4])
        self.now += 4_000_000
        self.worker.sample_due()
        self.assertEqual(self.events[-1]['seq'], 5)

    def test_slow_read_skips_slots_without_catchup(self):
        self.start()
        transfer = self.sensor.transfer
        def delayed(register, value=None):
            if register == BUS_VOLTAGE:
                self.now += 12_000_000
            return transfer(register, value)
        self.sensor.transfer = delayed
        self.now += 5_000_000
        before = self.now
        self.worker.sample_due()
        self.assertEqual(self.events[-1]['rx_monotonic_ns'], before + 6_000_000)
        self.worker.sample_due()
        self.assertEqual(len(self.events), 2)
        self.now += 3_000_000
        self.worker.sample_due()
        self.assertEqual(self.events[-1]['seq'], 3)

    def test_stale_token_cannot_stop_active_trial(self):
        self.start()
        with self.assertRaises(ValueError):
            self.worker.command(dict(cmd='stop', token='old_trial'))
        self.assertEqual(self.worker.token, 'trial')
        self.worker.stop()
        self.assertEqual(self.sensor.registers[CONFIG], 0x399F)

    def test_late_sample_does_not_duplicate_adc_conversion_at_next_tick(self):
        self.start()
        self.now += 9_800_000
        self.worker.sample_due()
        self.now += 200_000
        self.worker.sample_due()
        self.assertEqual(len(self.events), 2)
        self.now += 5_000_000
        self.worker.sample_due()
        self.assertEqual(self.events[-1]['seq'], 2)

    def test_config_restore_on_setup_failure(self):
        transfer = self.sensor.transfer
        def bad_readback(register, value=None):
            result = transfer(register, value)
            if register == CONFIG and value is None and result == SAMPLE_CONFIG:
                return 0
            return result
        self.sensor.transfer = bad_readback
        with self.assertRaisesRegex(OSError, 'readback'):
            self.start()
        self.assertEqual(self.sensor.registers[CONFIG], 0x399F)
        self.assertEqual(self.events, [])


if __name__ == '__main__':
    unittest.main()
