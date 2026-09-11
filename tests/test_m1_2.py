import json
import unittest
from common.servo import Servo, limits_from, move_from
from Tx.Controller.servo_panel import ServoPanel

LIMITS=dict(pan_min=-10,pan_max=10,tilt_min=-5,tilt_max=5)
MOVE=dict(pan=1.5,tilt=-2,speed=100,acc=1,request_id='trial')

class Serial:
    def __init__(self,fail=False): self.raw=None;self.fail=fail;self.closed=False
    def write(self,raw): self.raw=raw;return 2 if self.fail else len(raw)
    def close(self): self.closed=True

class Var:
    def __init__(self,value): self.value=value
    def get(self): return self.value
    def set(self,value): self.value=value

class Commands(unittest.TestCase):
    def test_requires_limits_and_no_startup_command(self):
        serial=Serial();servo=Servo(transport=serial)
        self.assertIsNone(serial.raw)
        with self.assertRaises(ValueError): servo.move(MOVE)
        self.assertIsNone(serial.raw)

    def test_exact_reference_protocol_without_actual_feedback(self):
        serial=Serial();servo=Servo(transport=serial);servo.configure(LIMITS)
        result=servo.move(MOVE)
        self.assertEqual(json.loads(serial.raw),dict(T=133,X=1.5,Y=-2,SPD=100,ACC=1))
        self.assertTrue(serial.raw.endswith(b'\n'))
        self.assertEqual(result['state'],'serial_written')
        self.assertIsNone(result['actual_angle'])
        self.assertFalse(result['arrival_verified'])

    def test_reject_bad_values_before_serial_write(self):
        serial=Serial();servo=Servo(transport=serial);servo.configure(LIMITS)
        for key,value in [('pan',11),('tilt',-6),('pan',float('nan')),
                          ('tilt',float('inf')),('speed',0),('speed',1.5),('acc',-1),('pan',True)]:
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                servo.move(dict(MOVE,**{key:value}))
        self.assertIsNone(serial.raw)

    def test_bad_limits_do_not_replace_applied_limits(self):
        servo=Servo(simulate=True);servo.configure(LIMITS)
        for update in [dict(pan_min=10),dict(tilt_max=91),dict(pan_min=-181),dict(pan_min='nan')]:
            with self.assertRaises(ValueError):servo.configure(dict(LIMITS,**update))
            self.assertEqual(servo.limits,LIMITS)

    def test_partial_write_invalidates_command(self):
        serial=Serial();servo=Servo(transport=serial);servo.configure(LIMITS);servo.move(MOVE)
        serial.fail=True
        with self.assertRaises(OSError):servo.move(MOVE)
        self.assertIsNone(servo.last);self.assertTrue(serial.closed)

    def test_simulation_and_disabled_port(self):
        servo=Servo(simulate=True);servo.configure(LIMITS)
        self.assertEqual(servo.move(MOVE)['state'],'simulated')
        servo=Servo();servo.configure(LIMITS)
        with self.assertRaises(ValueError):servo.move(MOVE)

    def test_led_off_without_limits_and_preserves_position(self):
        serial=Serial();servo=Servo(transport=serial)
        result=servo.led_off()
        self.assertEqual(json.loads(serial.raw),dict(T=132,IO4=0,IO5=0))
        self.assertTrue(serial.raw.endswith(b'\n'))
        self.assertIsNone(result['commanded'])
        servo.configure(LIMITS);servo.move(MOVE)
        before=dict(servo.last)
        servo.led_off()
        self.assertEqual(servo.last,before)
        self.assertEqual(servo.limits,LIMITS)

    def test_led_off_disabled_and_failed_write(self):
        with self.assertRaises(ValueError):Servo().led_off()
        serial=Serial(fail=True)
        with self.assertRaises(OSError):Servo(transport=serial).led_off()
        self.assertTrue(serial.closed)

    def panel(self):
        panel=ServoPanel.__new__(ServoPanel)
        panel.last=dict(pan=2,tilt=3)
        panel.step=Var('1')
        panel.calls=[]
        panel.move=lambda **kw:panel.calls.append(kw)
        return panel

    def test_jog_keeps_other_axis_and_uses_last_command(self):
        panel=self.panel();panel.jog('pan',1)
        self.assertEqual(panel.calls,[dict(pan=3,tilt=3)])
        panel.jog('tilt',-1)
        self.assertEqual(panel.calls[-1],dict(pan=2,tilt=2))

    def test_jog_without_known_command_or_invalid_step(self):
        panel=self.panel();panel.last=None
        with self.assertRaises(ValueError):panel.jog('pan',1)
        panel.last=dict(pan=2,tilt=3)
        for step in ('0','-1','nan'):
            panel.step=Var(step)
            with self.assertRaises(ValueError):panel.jog('pan',1)
        self.assertEqual(panel.calls,[])

if __name__=='__main__': unittest.main()
