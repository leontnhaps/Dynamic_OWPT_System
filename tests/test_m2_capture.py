"""M2 validation, real GPIO polarity via fake GPIO, and fresh-request API contract."""
import io
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from common.capture import camera_config,capture_request,save_capture,append_index,export_catalog
from Tx.RaspberryPi.Rasp_main import Agent


def command(**updates):
    result=dict(request_id=uuid.uuid4().hex,measurement=dict(distance_m=3,target_x_m=.2,target_y_m=0))
    result.update(updates)
    return result


class CaptureTests(unittest.TestCase):
    def test_bad_config_and_position_rejected(self):
        for update in [dict(width=float('nan')),dict(width=1),dict(width=32.5),
                       dict(shutter_speed=100),dict(analogue_gain=float('inf'),shutter_speed=100),dict(quality=101)]:
            with self.assertRaises(ValueError): camera_config(update)
        for z in [0,-1,'nan','inf']:
            with self.assertRaises(ValueError): capture_request(command(measurement=dict(distance_m=z,target_x_m=0,target_y_m=0)))
        with self.assertRaises(ValueError): capture_request(command(request_id='../bad'))

    def test_storage_identity_hash_and_index(self):
        with tempfile.TemporaryDirectory() as d:
            meta=dict(capture_request(command()),simulated=True,capture_done_unix_ns=123)
            p,m=save_capture(d,b'jpeg bytes',meta)
            self.assertEqual(p.read_bytes(),b'jpeg bytes')
            self.assertEqual(json.loads(p.with_name('metadata.json').read_text())['measurement']['distance_m'],3)
            self.assertTrue(p.with_name('complete.json').exists())
            p.with_name('annotations.json').write_text(json.dumps({'points':{'laser':{'u':101.5,'v':202.5}}}))
            catalog,count=export_catalog(d)
            self.assertEqual(count,1)
            self.assertIn('101.5',catalog.read_text(encoding='utf-8-sig'))
            with self.assertRaises(FileExistsError): save_capture(d,b'other',meta)
            append_index(d,p,m)
            self.assertIn(meta['request_id'],(Path(d)/'index.csv').read_text(encoding='utf-8-sig'))

    def test_gpio_pin_polarity_and_simulation_isolation(self):
        calls=[]
        gpio=SimpleNamespace(output=lambda pin,level:calls.append((pin,level)))
        args=SimpleNamespace(simulate=False,servo_port=None,ir_pin=17,laser_pin=23)
        agent=Agent(args);agent.gpio=gpio
        agent.set_output('ir_cut',1);agent.set_output('laser',1);agent.safe_off()
        self.assertEqual(calls,[(17,1),(23,1),(23,0)])
        self.assertEqual(agent.gpio_state()['laser_gpio_level'],0)
        args.simulate=True
        agent.set_output('laser',1)
        self.assertEqual(len(calls),3)

    def test_fresh_frame_metadata_and_release_on_save_error(self):
        args=SimpleNamespace(simulate=False,servo_port=None)
        agent=Agent(args)
        class Request:
            config={'main':{'size':(640,480)}}
            released=False
            def get_metadata(self): return {'SensorTimestamp':123,'ExposureTime':1000}
            def save(self,*args,**kwargs): raise OSError('encode failed')
            def release(self): self.released=True
        req=Request()
        class Camera:
            def capture_request(self,flush):
                assert flush is True
                return req
        with self.assertRaises(OSError): agent.acquire(Camera(),{})
        self.assertTrue(req.released)

if __name__=='__main__': unittest.main()
