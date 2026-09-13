"""Actual relay + simulated Pi + actual laptop Network capture round trip."""
import io
import json
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from PIL import Image
from common.protocol import send_json
from Tx.Controller.network_client import Network

ROOT=Path(__file__).resolve().parents[1]

class M2RoundTrip(unittest.TestCase):
    def test_still_metadata_preview_resume_and_disconnect_off(self):
        processes=[];net=None
        with tempfile.TemporaryDirectory() as d:
            try:
                for port in (7500,7501,7600,7601):
                    with socket.socket() as s: s.bind(('127.0.0.1',port))
                processes.append(subprocess.Popen([sys.executable,'Server/Server_main.py'],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE))
                net=Network('127.0.0.1')
                def event(predicate,network=None):
                    network=network or net
                    end=time.monotonic()+8
                    while time.monotonic()<end:
                        e=network.events.get(timeout=8)
                        if predicate(e): return e
                    self.fail('Event missing')
                event(lambda e:e.get('event')=='hello')
                processes.append(subprocess.Popen([sys.executable,'Tx/RaspberryPi/Rasp_main.py','--server','127.0.0.1',
                                                   '--simulate','--capture-dir',d],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE))
                event(lambda e:e.get('event')=='ready')
                def request(cmd):
                    token=uuid.uuid4().hex
                    net.send(dict(cmd,request_id=token))
                    return event(lambda e:e.get('request_id')==token)
                request(dict(cmd='servo_config',pan_min=-20,pan_max=20,tilt_min=-10,tilt_max=10))
                request(dict(cmd='move',pan=4,tilt=-2,speed=100,acc=1))
                request(dict(cmd='ir_cut',level=1))
                request(dict(cmd='laser',value=1))
                net.send(dict(cmd='preview',width=320,height=240,fps=30,quality=65))
                event(lambda e:e.get('event')=='preview' and e.get('state')=='started')
                ids=[]
                for quality in (95,80):
                    token=uuid.uuid4().hex;ids.append(token)
                    net.send(dict(cmd='snap',request_id=token,width=640,height=480,quality=quality,
                                  shutter_speed=1000,analogue_gain=1,awb=False,
                                  measurement=dict(distance_m=3,target_x_m=.1,target_y_m=.2,kind='laser_reference',sample_label='pair-1')))
                    event(lambda e:e.get('event')=='capture_accepted' and e.get('request_id')==token)
                    # Must reject a command that would corrupt the still's angle label.
                    rejected=request(dict(cmd='move',pan=5,tilt=-2,speed=100,acc=1))
                    self.assertEqual(rejected['event'],'error')
                    if quality==80:
                        off=request(dict(cmd='laser',value=0))
                        self.assertEqual(off['laser_gpio_level'],0)
                    data,meta,_,_=net.captures.get(timeout=8)
                    self.assertEqual(meta['request_id'],token)
                    self.assertEqual(Image.open(io.BytesIO(data)).size,(640,480))
                    self.assertEqual(meta['servo_at_capture_start']['commanded']['pan'],4)
                    self.assertEqual(meta['laser_gpio_level'],1 if quality==95 else 0)
                    self.assertEqual(meta['ir_gpio_level'],1)
                    self.assertFalse(meta['gpio_changed_during_capture'])
                    self.assertEqual(meta['measurement']['sample_label'],'pair-1')
                    self.assertEqual(meta['requested']['quality'],quality)
                    self.assertEqual((Path(d)/token/'image.jpg').read_bytes(),data)
                    event(lambda e:e.get('event')=='preview' and e.get('state')=='started')
                self.assertEqual(len(list(Path(d).glob('*/complete.json'))),2)
                end=time.monotonic()+3
                frame=None
                while time.monotonic()<end:
                    frame,_=net.pop()
                    if frame and frame[1]['seq']>2: break
                    time.sleep(.02)
                self.assertIsNotNone(frame)
                self.assertEqual(Image.open(io.BytesIO(frame[0])).size,(320,240))
                request(dict(cmd='laser',value=1))
                # Orderly GUI close is intentionally omitted: relay must turn laser off.
                net.close();net=None
                time.sleep(.15)
                net=Network('127.0.0.1')
                event(lambda e:e.get('event')=='hello')
                hw=request(dict(cmd='hardware_status'))
                self.assertEqual(hw['laser_gpio_level'],0)
                self.assertEqual(request(dict(cmd='servo_status'))['commanded']['pan'],4)
            finally:
                if net:net.close()
                for p in reversed(processes):
                    if p.poll() is None:p.terminate()
                    _,err=p.communicate(timeout=8)
                    if err: print(err.decode(),file=sys.stderr)

if __name__=='__main__':unittest.main()
