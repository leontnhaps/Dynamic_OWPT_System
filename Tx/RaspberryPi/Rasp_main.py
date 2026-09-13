"""M2 Pi acquisition agent. Pins/polarity from DLC Raspberrypi/Rasp_main.py.

One camera owner handles preview and still capture. Angles are transmitted
commands, not encoder measurements. Camera SensorTimestamp is a Pi clock value.
"""
import argparse
import io
import json
import socket
import signal
import threading
import time
import uuid
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from common.protocol import messages, send_json, send_frame
from common.servo import Servo
from common.capture import camera_config, capture_request, save_capture

IR_CUT_PIN = 17  # BCM: day LOW / night HIGH, verified in DLC main.
LASER_PIN = 23   # BCM: ON HIGH / OFF LOW, verified in DLC main.


class Agent:
    def __init__(self,args):
        self.args=args
        self.stop=threading.Event()
        self.lock=threading.RLock()
        self.output_lock=threading.Lock()
        self.gpio_lock=threading.Lock()
        self.cfg=None
        self.pending_capture=None
        self.capture_busy=False
        self.seq=0
        self.session=uuid.uuid4().hex
        self.gpio=None
        self.ir_level=None
        self.laser_level=0 if args.simulate else None
        self.gpio_revision=0
        self.servo=Servo(getattr(args,'servo_port',None),args.simulate)

    def event(self,**obj):
        with self.output_lock: send_json(self.ctrl,obj)

    def gpio_state(self):
        with self.gpio_lock:
            return dict(ir_gpio_level=self.ir_level,laser_gpio_level=self.laser_level,
                        gpio_revision=self.gpio_revision)

    def set_output(self,kind,level):
        if type(level) is not int or level not in (0,1):
            raise ValueError('GPIO level must be integer 0 or 1')
        with self.gpio_lock:
            if not self.args.simulate:
                if self.gpio is None: raise ValueError('GPIO unavailable')
                self.gpio.output(self.args.ir_pin if kind=='ir_cut' else self.args.laser_pin,level)
            if kind=='ir_cut': self.ir_level=level
            else: self.laser_level=level
            self.gpio_revision+=1

    def safe_off(self):
        self.set_output('laser',0)

    def commands(self):
        try:
            for cmd in messages(self.ctrl):
                try:
                    action=cmd.get('cmd')
                    if action=='laser' and cmd.get('value')==0:
                        # OFF bypasses camera/motion locks, including during capture.
                        self.safe_off()
                        self.event(event='laser',request_id=cmd.get('request_id'),**self.gpio_state())
                        continue
                    if action=='outputs_off':
                        self.safe_off()
                        with self.lock:
                            self.cfg=None
                            if self.pending_capture is not None:
                                cancelled=self.pending_capture['request_id']
                                self.pending_capture=None
                                self.capture_busy=False
                                self.event(event='error',operation='snap',request_id=cancelled,message='Capture cancelled on controller disconnect')
                        self.event(event='outputs_off',**self.gpio_state())
                        continue
                    with self.lock:
                        if self.capture_busy and action not in ('ping','servo_status','hardware_status'):
                            raise ValueError('Still capture in progress; wait before changing settings or moving')
                        if action in ('servo_status','servo_config','move','led_off'):
                            if action=='servo_config': result=self.servo.configure(cmd)
                            elif action=='led_off': result=self.servo.led_off()
                            elif action=='move': result=self.servo.move(cmd)
                            else: result=self.servo.status()
                            self.event(event='servo',operation=action,request_id=cmd.get('request_id'),**result)
                        elif action=='preview':
                            self.cfg=camera_config(cmd) if cmd.get('enable',True) else None
                        elif action=='snap':
                            request=capture_request(cmd)
                            if self.servo.last is None:
                                raise ValueError('Send a known pan/tilt command before recording a calibration sample')
                            self.pending_capture=request
                            self.capture_busy=True
                            self.event(event='capture_accepted',request_id=request['request_id'])
                        elif action in ('ir_cut','laser'):
                            self.set_output(action,cmd.get('level') if action=='ir_cut' else cmd.get('value'))
                            self.event(event=action,request_id=cmd.get('request_id'),**self.gpio_state())
                        elif action=='hardware_status':
                            self.event(event='hardware',request_id=cmd.get('request_id'),simulated=self.args.simulate,
                                       ir_pin=self.args.ir_pin,laser_pin=self.args.laser_pin,**self.gpio_state())
                        elif action=='ping': self.event(event='pong',token=cmd.get('token'))
                        else: raise ValueError('Unsupported command')
                except (ValueError,TypeError,OSError,ImportError,RuntimeError) as exc:
                    self.event(event='error',operation=cmd.get('cmd'),request_id=cmd.get('request_id'),message=str(exc))
        except (OSError,ValueError):
            pass
        finally:
            self.stop.set()
            self.safe_off()

    def configure_camera(self,camera,cfg,still=False):
        if self.args.simulate: return None
        from picamera2 import Picamera2
        if camera is None: camera=Picamera2()
        else: camera.stop()
        try:
            controls={'AeEnable':cfg['shutter_speed'] is None,'AwbEnable':cfg['awb']}
            if cfg['shutter_speed'] is not None:
                controls.update(ExposureTime=cfg['shutter_speed'],AnalogueGain=cfg['analogue_gain'])
            if not cfg['awb']: controls['ColourGains']=(1.0,1.0)
            make=camera.create_still_configuration if still else camera.create_video_configuration
            camera.configure(make(queue=False,main={'size':(cfg['width'],cfg['height'])},controls=controls))
            camera.options['quality']=cfg['quality']
            camera.start()
            return camera
        except Exception:
            camera.close()
            raise

    def acquire(self,camera,cfg):
        with io.BytesIO() as bio:
            if self.args.simulate:
                from PIL import Image,ImageDraw
                image=Image.new('RGB',(cfg['width'],cfg['height']),'#19364b')
                draw=ImageDraw.Draw(image)
                draw.text((20,20),f'SIMULATION {self.seq}',fill='white')
                draw.ellipse((cfg['width']//2-8,cfg['height']//2-8,cfg['width']//2+8,cfg['height']//2+8),fill='white')
                image.save(bio,format='JPEG',quality=cfg['quality'])
                actual={'simulated':True}
            else:
                request=camera.capture_request(flush=True)
                try:
                    actual=dict(request.get_metadata(),stream_configuration=request.config.get('main'))
                    request.save('main',bio,format='jpeg')
                finally: request.release()
            return bio.getvalue(),actual

    def connection(self):
        self.cfg=None
        self.pending_capture=None
        self.capture_busy=False
        self.servo.limits=None
        self.servo.last=None
        self.safe_off()
        self.stop.clear()
        with socket.create_connection((self.args.server,7500),timeout=5) as ctrl, socket.create_connection((self.args.server,7501),timeout=5) as images:
            self.ctrl=ctrl
            ctrl.settimeout(None)
            images.settimeout(5)
            worker=threading.Thread(target=self.commands,daemon=True)
            worker.start()
            camera=None
            active=None
            try:
                self.event(event='ready',simulated=self.args.simulate,
                           capabilities=['servo_status','servo_config','move','led_off','snap','ir_cut','laser','hardware_status'])
                while not self.stop.is_set():
                    with self.lock:
                        cfg=self.cfg.copy() if self.cfg else None
                        snap=self.pending_capture
                        self.pending_capture=None
                    if snap:
                        try:
                            still=snap['requested']
                            camera=self.configure_camera(camera,still,still=True)
                            if self.stop.wait(.5 if still['shutter_speed'] is None else max(.2,still['shutter_speed']/1e6*3)):
                                break
                            with self.lock: servo_state=self.servo.status()
                            gpio=self.gpio_state()
                            begin=time.monotonic()
                            jpeg,actual=self.acquire(camera,still)
                            self.seq+=1
                            meta=dict(snap,session=self.session,seq=self.seq,simulated=self.args.simulate,
                                      capture_done_unix_ns=time.time_ns(),capture_call_ms=(time.monotonic()-begin)*1000,
                                      camera_metadata=actual,actual_image_size=list(camera.camera_configuration()['main']['size']) if camera else [still['width'],still['height']],
                                      servo_at_capture_start=servo_state,**gpio,
                                      gpio_changed_during_capture=self.gpio_state()['gpio_revision']!=gpio['gpio_revision'],
                                      note='Servo and GPIO values are commands, not optical/position feedback. Unix clocks are not synchronized.')
                            path,_=save_capture(self.args.capture_dir,jpeg,meta)
                            meta['pi_backup']=str(path)
                            send_frame(images,'_capture_'+json.dumps(meta,separators=(',',':'),default=str),jpeg)
                            self.event(event='capture_sent',request_id=snap['request_id'],pi_backup=str(path))
                        except Exception as exc:
                            self.event(event='error',operation='snap',request_id=snap['request_id'],message=str(exc))
                        finally:
                            with self.lock: self.capture_busy=False
                            active=None  # Reconfigure preview, including its exposure settings.
                            if camera:
                                camera.stop();camera.close();camera=None
                        continue
                    if cfg!=active:
                        if cfg: camera=self.configure_camera(camera,cfg)
                        elif camera: camera.stop();camera.close();camera=None
                        active=cfg
                        self.event(event='preview',state='started' if cfg else 'stopped',requested=cfg)
                    if cfg is None:
                        self.stop.wait(.05)
                        continue
                    begin=time.monotonic()
                    with self.lock: servo_state=self.servo.status()
                    gpio=self.gpio_state()
                    jpeg,actual=self.acquire(camera,cfg)
                    self.seq+=1
                    meta=dict(session=self.session,seq=self.seq,simulated=self.args.simulate,requested=cfg,
                              capture_done_unix_ns=time.time_ns(),capture_call_ms=(time.monotonic()-begin)*1000,
                              camera_metadata=actual,servo_at_capture_start=servo_state,**gpio)
                    send_frame(images,'_preview_'+json.dumps(meta,separators=(',',':'),default=str),jpeg)
                    self.stop.wait(max(0,1/cfg['fps']-(time.monotonic()-begin)))
            finally:
                self.stop.set()
                self.safe_off()
                try: ctrl.shutdown(socket.SHUT_RDWR)
                except OSError: pass
                worker.join(timeout=2)
                if camera:
                    try: camera.stop()
                    finally: camera.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--server',required=True)
    parser.add_argument('--simulate',action='store_true')
    parser.add_argument('--ir-pin',type=int,default=IR_CUT_PIN)
    parser.add_argument('--laser-pin',type=int,default=LASER_PIN)
    parser.add_argument('--servo-port',help='ESP32 serial device, e.g. /dev/ttyUSB0')
    parser.add_argument('--capture-dir',default='captures/m2_pi')
    args=parser.parse_args()
    if args.ir_pin==args.laser_pin: parser.error('IR and laser pins must differ')
    agent=Agent(args)
    def terminate(signum,frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,terminate)
    try:
        if not args.simulate:
            import RPi.GPIO as GPIO
            agent.gpio=GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(args.laser_pin,GPIO.OUT,initial=GPIO.LOW)
            GPIO.setup(args.ir_pin,GPIO.OUT,initial=GPIO.LOW)
        agent.ir_level=0
        agent.laser_level=0
        while True:
            try: agent.connection()
            except (OSError,ValueError,RuntimeError,ImportError) as exc:
                print(f'Agent error: {exc}; reconnect in 2s',flush=True)
                time.sleep(2)
    except KeyboardInterrupt: pass
    finally:
        agent.safe_off()
        agent.servo.close()
        if agent.gpio: agent.gpio.cleanup([args.ir_pin,args.laser_pin])

if __name__=='__main__': main()
