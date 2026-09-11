"""DLC Picamera2 JPEG acquisition and relay client, M1-1 camera-only revision."""
import argparse
import io
import json
import socket
import threading
import time
import uuid
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from common.protocol import messages, send_json, send_frame
from common.servo import Servo

class Agent:
    def __init__(self,args):
        self.args=args
        self.stop=threading.Event()
        self.lock=threading.Lock()
        self.output_lock=threading.Lock()
        self.cfg=None
        self.seq=0
        self.session=uuid.uuid4().hex
        self.gpio=None
        self.ir_level=None
        self.servo=Servo(getattr(args, "servo_port", None), args.simulate)
    def event(self,**obj):
        with self.output_lock:
            send_json(self.ctrl,obj)
    def commands(self):
        try:
            for cmd in messages(self.ctrl):
                try:
                    if cmd.get('cmd') in ('servo_status', 'servo_config', 'move', 'led_off'):
                        action=cmd['cmd']
                        with self.lock:
                            if action=='servo_config':
                                result=self.servo.configure(cmd)
                            elif action=='led_off':
                                result=self.servo.led_off()
                            elif action=='move':
                                result=self.servo.move(cmd)
                            else:
                                result=self.servo.status()
                        self.event(event='servo',operation=action,request_id=cmd.get('request_id'),**result)
                    elif cmd.get('cmd')=='preview':
                        cfg=None
                        if cmd.get('enable',True):
                            cfg={k:int(cmd.get(k,v)) for k,v in dict(width=640,height=480,fps=10,quality=80).items()}
                            if not (16<=cfg['width']<=2592 and 16<=cfg['height']<=1944 and 1<=cfg['fps']<=60 and 1<=cfg['quality']<=100):
                                raise ValueError('Invalid size/fps/quality')
                            exposure=cmd.get('shutter_speed')
                            gain=cmd.get('analogue_gain')
                            if (exposure is None)!=(gain is None):
                                raise ValueError('Specify both exposure and gain, or neither')
                            cfg.update(shutter_speed=int(exposure) if exposure is not None else None,
                                       analogue_gain=float(gain) if gain is not None else None)
                            if exposure is not None and (cfg['shutter_speed']<=0 or cfg['analogue_gain']<=0):
                                raise ValueError('Exposure/gain must be positive')
                        with self.lock:
                            self.cfg=cfg
                    elif cmd.get('cmd')=='ir_cut':
                        if self.gpio is None:
                            raise ValueError('IR disabled: verified wiring and --ir-pin required')
                        level=cmd.get('level')
                        if level not in (0,1):
                            raise ValueError('GPIO level must be 0 or 1')
                        self.gpio.output(self.args.ir_pin,level)
                        self.ir_level=level
                        self.event(event='ir_cut',level=level)
                    elif cmd.get('cmd')=='ping':
                        self.event(event='pong',token=cmd.get('token'))
                    else:
                        raise ValueError('Unsupported command')
                except (ValueError,TypeError,OSError,ImportError) as exc:
                    self.event(event='error',operation=cmd.get('cmd'),request_id=cmd.get('request_id'),message=str(exc))
        except (OSError,ValueError):
            pass
        finally:
            self.stop.set()
    def connection(self):
        self.cfg=None
        self.servo.limits=None
        self.servo.last=None
        self.stop.clear()
        with socket.create_connection((self.args.server,7500),timeout=5) as ctrl, socket.create_connection((self.args.server,7501),timeout=5) as images:
            self.ctrl=ctrl
            ctrl.settimeout(None)
            images.settimeout(2)
            worker=threading.Thread(target=self.commands,daemon=True)
            worker.start()
            camera=None
            active=None
            try:
                self.event(event='ready',simulated=self.args.simulate,capabilities=['servo_status','servo_config','move','led_off'])
                while not self.stop.is_set():
                    with self.lock:
                        cfg=self.cfg.copy() if self.cfg else None
                    if cfg!=active:
                        if camera:
                            camera.stop();camera.close();camera=None
                        if cfg and not self.args.simulate:
                            from picamera2 import Picamera2
                            camera=Picamera2()
                            controls={'AeEnable':cfg['shutter_speed'] is None}
                            if cfg['shutter_speed'] is not None:
                                controls.update(ExposureTime=cfg['shutter_speed'],AnalogueGain=cfg['analogue_gain'])
                            camera.configure(camera.create_video_configuration(main={'size':(cfg['width'],cfg['height'])},controls=controls))
                            camera.options['quality']=cfg['quality']
                            camera.start()
                        active=cfg
                        self.event(event='preview',state='started' if cfg else 'stopped',requested=cfg)
                    if cfg is None:
                        self.stop.wait(.05)
                        continue
                    begin=time.monotonic()
                    with self.lock:
                        servo_at_capture=self.servo.status()
                    with io.BytesIO() as bio:
                        if self.args.simulate:
                            from PIL import Image,ImageDraw
                            image=Image.new('RGB',(cfg['width'],cfg['height']),'#19364b')
                            ImageDraw.Draw(image).text((20,20),f'SIMULATION {self.seq}',fill='white')
                            image.save(bio,format='JPEG',quality=cfg['quality'])
                        else:
                            camera.capture_file(bio,format='jpeg')
                        jpeg=bio.getvalue()
                    self.seq+=1
                    meta=dict(session=self.session,seq=self.seq,simulated=self.args.simulate,requested=cfg,
                              capture_done_unix_ns=time.time_ns(),capture_call_ms=(time.monotonic()-begin)*1000,
                              ir_gpio_level=self.ir_level,servo_at_capture_start=servo_at_capture)
                    send_frame(images,'_preview_'+json.dumps(meta,separators=(',',':')),jpeg)
                    self.stop.wait(max(0,1/cfg['fps']-(time.monotonic()-begin)))
            except Exception as exc:
                try:
                    self.event(event='error',message=str(exc))
                except OSError:
                    pass
                raise
            finally:
                self.stop.set()
                try:
                    ctrl.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                worker.join(timeout=2)
                if camera:
                    try:
                        camera.stop()
                    finally:
                        camera.close()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--server',required=True)
    parser.add_argument('--simulate',action='store_true')
    parser.add_argument('--ir-pin',type=int)
    parser.add_argument('--servo-port',help='ESP32 serial device, e.g. /dev/ttyUSB0; no auto movement')
    args=parser.parse_args()
    if args.simulate and args.ir_pin is not None:
        parser.error('Simulation cannot operate GPIO')
    agent=Agent(args)
    try:
        if args.ir_pin is not None:
            import RPi.GPIO as GPIO
            agent.gpio=GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(args.ir_pin,GPIO.OUT,initial=GPIO.LOW)
            agent.ir_level=0
        while True:
            try:
                agent.connection()
            except (OSError,ValueError,RuntimeError,ImportError) as exc:
                print(f'Agent error: {exc}; reconnect in 2s',flush=True)
                time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        agent.servo.close()
        if agent.gpio:
            agent.gpio.cleanup(args.ir_pin)
if __name__=='__main__':
    main()
