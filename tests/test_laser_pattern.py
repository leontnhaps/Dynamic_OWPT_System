import json
import socket
import threading
import time
import unittest
from types import SimpleNamespace
from common.laser_pattern import LaserPattern, pattern_settings
from common.protocol import send_json
from Tx.RaspberryPi.Rasp_main import Agent


class PatternTests(unittest.TestCase):
    def test_invalid_requests_do_not_start(self):
        for change in ({'bits':'102'}, {'bits':''}, {'bit_ms':float('nan')},
                       {'bit_ms':0}, {'bit_ms':-1}, {'bit_ms':True},
                       {'bit_ms':float('inf')}, {'bit_ms':5001},
                       {'repeats':True}, {'repeats':100},
                       {'bit_ms':5000,'bits':'1'*30}):
            with self.assertRaises(ValueError):
                pattern_settings(dict(dict(bits='1010',bit_ms=100,repeats=1), **change))

    def test_short_bit_durations_are_preserved(self):
        for ms in (50, 20, 10, 1, 0.5, 0.01):
            with self.subTest(bit_ms=ms):
                settings = pattern_settings(dict(bits='1010', bit_ms=ms, repeats=1))
                self.assertEqual(settings['bit_ms'], ms)

    def engine(self, output=None):
        calls, reports = [], []
        done = threading.Event()
        def write(value):
            calls.append((value,time.monotonic()))
            if output: output(value)
        def finish(report):
            reports.append(report); done.set()
        return LaserPattern(write,finish), calls, reports, done

    def test_sequence_final_duration_and_off(self):
        engine,calls,reports,done=self.engine()
        engine.start(dict(request_id='test',bits='101',bit_ms=100,repeats=1))
        self.assertTrue(done.wait(2))
        self.assertEqual([c[0] for c in calls], [0,1,0,1,0])
        self.assertGreaterEqual(calls[-1][1]-calls[-2][1], .07)
        self.assertEqual(reports[0]['state'],'completed')
        self.assertFalse(engine.status()['pattern_active'])

    def test_off_cancels_without_later_on_and_rejects_overlap(self):
        engine,calls,reports,done=self.engine()
        cmd=dict(request_id='test',bits='111111',bit_ms=100,repeats=1)
        engine.start(cmd)
        with self.assertRaises(ValueError): engine.start(cmd)
        with self.assertRaises(ValueError): engine.manual(1)
        engine.manual(0)
        off_index=len(calls)-1
        self.assertTrue(done.wait(2))
        self.assertTrue(all(level==0 for level,_ in calls[off_index:]))
        self.assertEqual(reports[0]['state'],'stopped')

    def test_late_processing_aborts_instead_of_catching_up(self):
        def delay(value):
            if value: time.sleep(.17)
        engine,calls,reports,done=self.engine(delay)
        engine.start(dict(request_id='test',bits='1010',bit_ms=100,repeats=1))
        self.assertTrue(done.wait(2))
        self.assertEqual(reports[0]['state'],'timing_overrun')
        self.assertEqual([c[0] for c in calls],[0,1,0])

    def test_output_failure_attempts_off(self):
        def fail(value):
            if value: raise RuntimeError('GPIO failed')
        engine,calls,reports,done=self.engine(fail)
        engine.start(dict(request_id='test',bits='1',bit_ms=100,repeats=1))
        self.assertTrue(done.wait(2))
        self.assertEqual(reports[0]['state'],'output_error')
        self.assertEqual(calls[-1][0],0)

    def test_agent_stop_status_and_disconnect(self):
        client, peer=socket.socketpair()
        client.settimeout(3)
        agent=Agent(SimpleNamespace(simulate=True,servo_port=None,ir_pin=17,laser_pin=23))
        agent.ctrl=peer
        reader=client.makefile('r')
        worker=threading.Thread(target=agent.commands)
        worker.start()
        def receive(token):
            while True:
                obj=json.loads(reader.readline())
                if obj.get('request_id')==token: return obj
        try:
            send_json(client,dict(cmd='laser_pattern',request_id='run',bits='1010',bit_ms=500,repeats=1))
            self.assertEqual(receive('run')['state'],'started')
            send_json(client,dict(cmd='move',request_id='bad',pan=0,tilt=0))
            self.assertEqual(receive('bad')['event'],'error')
            send_json(client,dict(cmd='laser',request_id='off',value=0))
            self.assertEqual(receive('off')['laser_gpio_level'],0)
            send_json(client,dict(cmd='hardware_status',request_id='status'))
            self.assertIn('laser_pattern',receive('status')['capabilities'])
        finally:
            client.shutdown(socket.SHUT_RDWR); reader.close(); client.close()
            worker.join(2); peer.close()
        self.assertFalse(worker.is_alive())
        self.assertEqual(agent.gpio_state()['laser_gpio_level'],0)


if __name__=='__main__': unittest.main()
