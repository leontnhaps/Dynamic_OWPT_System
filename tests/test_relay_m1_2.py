"""End-to-end command and preview test with the actual relay and simulated Pi.
Requires the project's four localhost ports to be unused.
"""
import json
import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path
from common.protocol import send_json, receive_frame

ROOT=Path(__file__).resolve().parents[1]

class RelayTest(unittest.TestCase):
    def test_preview_moves_rejection_and_reconnect(self):
        processes=[];sockets=[]
        try:
            for port in (7500,7501,7600,7601):
                with socket.socket() as s:
                    s.bind(('127.0.0.1',port))
            processes.append(subprocess.Popen([sys.executable,'Server/Server_main.py'],cwd=ROOT,
                                             stdout=subprocess.DEVNULL,stderr=subprocess.PIPE))
            def connect(port):
                end=time.monotonic()+5
                while True:
                    try:
                        s=socket.create_connection(('127.0.0.1',port),timeout=2)
                        s.settimeout(5);sockets.append(s);return s
                    except OSError:
                        if time.monotonic()>end:raise
                        time.sleep(.05)
            ctrl=connect(7600);images=connect(7601)
            stream=ctrl.makefile('rb')
            def until(predicate):
                for _ in range(30):
                    event=json.loads(stream.readline())
                    if predicate(event):return event
                self.fail('Expected event missing')
            until(lambda e:e.get('event')=='hello')
            processes.append(subprocess.Popen([sys.executable,'Tx/RaspberryPi/Rasp_main.py',
                                               '--server','127.0.0.1','--simulate'],cwd=ROOT,
                                               stdout=subprocess.DEVNULL,stderr=subprocess.PIPE))
            until(lambda e:e.get('event')=='ready')
            def request(cmd,token):
                send_json(ctrl,dict(cmd,request_id=token))
                return until(lambda e:e.get('request_id')==token)
            event=request(dict(cmd='move',pan=0,tilt=0,speed=100,acc=1),'before')
            self.assertEqual(event['event'],'error')
            event=request(dict(cmd='servo_config',pan_min=-10,pan_max=10,tilt_min=-5,tilt_max=5),'cfg')
            self.assertEqual(event['event'],'servo')
            send_json(ctrl,dict(cmd='preview',enable=True,width=320,height=240,fps=10,quality=70))
            name,jpeg=receive_frame(images)
            self.assertTrue(jpeg.startswith(b'\xff\xd8'))
            event=request(dict(cmd='move',pan=3,tilt=-1,speed=100,acc=1),'move')
            self.assertEqual(event['state'],'simulated')
            self.assertEqual(event['commanded']['pan'],3)
            self.assertFalse(event['arrival_verified'])
            for _ in range(20):
                name,jpeg=receive_frame(images)
                meta=json.loads(name.removeprefix('_preview_'))
                if meta['servo_at_capture_start']['commanded']:
                    break
            self.assertEqual(meta['servo_at_capture_start']['commanded']['command_id'],'move')
            event=request(dict(cmd='move',pan=11,tilt=0,speed=100,acc=1),'bad')
            self.assertEqual(event['event'],'error')
            event=request(dict(cmd='servo_status'),'status')
            self.assertEqual(event['commanded']['pan'],3)
            # New Pi session must require reapplication, never replay a movement.
            processes[-1].terminate();processes[-1].wait(timeout=5)
            until(lambda e:e.get('event')=='agent' and e.get('state')=='disconnected')
            processes.append(subprocess.Popen([sys.executable,'Tx/RaspberryPi/Rasp_main.py',
                                               '--server','127.0.0.1','--simulate'],cwd=ROOT,
                                               stdout=subprocess.DEVNULL,stderr=subprocess.PIPE))
            until(lambda e:e.get('event')=='ready')
            event=request(dict(cmd='servo_status'),'new')
            self.assertIsNone(event['commanded']);self.assertIsNone(event['limits'])
            stream.close()
        finally:
            for s in sockets:s.close()
            for p in reversed(processes):
                if p.poll() is None:p.terminate()
                p.communicate(timeout=5)

if __name__=='__main__':unittest.main()
