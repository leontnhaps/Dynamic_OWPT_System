import unittest
from dataclasses import replace
from unittest.mock import Mock,patch
from types import SimpleNamespace
from PIL import Image
from simulation.config import Config
from Tx.Controller.live_core import fresh,target_from_boxes,validate_config,command_inside
from Tx.Controller.live_panel import LivePanel

class LiveTests(unittest.TestCase):
    def test_missing_and_ambiguous_target(self):
        b=[10,20,40,60,.9,0]
        self.assertEqual(target_from_boxes([],0,.5,(100,100)),(None,0))
        self.assertEqual(target_from_boxes([b,b],0,.5,(100,100)),(None,2))
        target,n=target_from_boxes([b,[0,0,10,10,.99,1]],0,.5,(100,100))
        self.assertEqual(n,1);self.assertEqual(target['center'],[25,40])
    def test_bad_box_rejected(self):
        self.assertEqual(target_from_boxes([[0,0,200,10,.9,0]],0,.5,(100,100)),(None,0))
    def test_age(self):
        self.assertTrue(fresh(10,10.4));self.assertFalse(fresh(10,10.6));self.assertFalse(fresh(11,10))
    def test_config_compatibility(self):
        validate_config(Config())
        for c in [replace(Config(),action_mode='absolute'),replace(Config(),command_step_deg=0),replace(Config(),delta_limit_deg=6)]:
            with self.assertRaises(ValueError):validate_config(c)
    def test_operating_bounds(self):
        limits=dict(pan_min=-10,pan_max=10,tilt_min=-5,tilt_max=5)
        self.assertTrue(command_inside([10,5],limits))
        self.assertFalse(command_inside([11,0],limits));self.assertFalse(command_inside([float('nan'),0],limits))
    def test_inference_coordinates_and_quantization(self):
        import io
        import numpy as np
        from Tx.Controller.live_core import LiveModels
        m=LiveModels.__new__(LiveModels);m.cfg=Config();m.device='cpu'
        boxes=Mock();boxes.data.detach.return_value.cpu.return_value.numpy.return_value=np.array([[746,354,846,454,.9,0]])
        m.detector=Mock();m.detector.predict.return_value=[SimpleNamespace(boxes=boxes)]
        m.policy=Mock();m.policy.predict.return_value=(np.array([.24,-.36]),None)
        raw=io.BytesIO();Image.new('RGB',(1296,972)).save(raw,format='JPEG')
        r=m.infer(raw.getvalue(),[0,0],None,.5,0)
        self.assertEqual(r['error'],[100,20]);self.assertEqual(r['command'],[1,-2])
        np.testing.assert_allclose(r['observation'][:4],[100/648,20/486,0,0],rtol=1e-6)
        m.policy.predict.assert_called_once()
        wrong=io.BytesIO();Image.new('RGB',(640,480)).save(wrong,format='JPEG')
        with self.assertRaises(ValueError):m.infer(wrong.getvalue(),[0,0],None,.5,0)

    def panel(self):
        p=LivePanel.__new__(LivePanel)
        p.models=SimpleNamespace(cfg=Config());p.running=True;p.detecting=True;p.generation=0
        p.previous_error=None;p.pending=None;p.log=None
        p.status=Mock();p.detail=Mock();p.canvas=Mock();p.write_row=Mock()
        s=SimpleNamespace(online=True,last={'pan':0,'tilt':0},limits=dict(pan_min=-10,pan_max=10,tilt_min=-5,tilt_max=5),values=lambda:dict(speed=100,acc=1))
        p.app=SimpleNamespace(servo=s,send=Mock(return_value=True),record=Mock())
        return p
    def result(self):
        return dict(image=Image.new('RGB',(1296,972)),target=dict(box=[700,300,800,400],center=[750,350]),count=1,error=[54,-34],raw_delta=[1.2,0.2],applied_delta=[1,0],command=[1,0])
    @patch('Tx.Controller.live_panel.ImageTk.PhotoImage',return_value=object())
    def test_preview_never_sends(self,_):
        p=self.panel();p.running=False;p.consume(self.result(),(b'',{},10,0),10,10.1)
        p.app.send.assert_not_called()
    @patch('Tx.Controller.live_panel.ImageTk.PhotoImage',return_value=object())
    def test_tracking_sends_one_absolute_command(self,_):
        p=self.panel();p.consume(self.result(),(b'',{},10,0),10,10.1)
        cmd=p.app.send.call_args.args[0]
        self.assertEqual(cmd['pan'],1);self.assertEqual(cmd['tilt'],0)
        self.assertIsNotNone(p.pending)
    def test_stale_result_stops_without_send(self):
        p=self.panel();p.consume(self.result(),(b'',{},10,0),10,11)
        self.assertFalse(p.running);p.app.send.assert_not_called()
    @patch('Tx.Controller.live_panel.ImageTk.PhotoImage',return_value=object())
    def test_lost_target_stops(self,_):
        p=self.panel();r=self.result();r.update(target=None,count=0)
        p.consume(r,(b'',{},10,0),10,10.1)
        self.assertFalse(p.running);p.app.send.assert_not_called()
    def test_stop_discards_inflight_generation(self):
        p=self.panel();p.stop();self.assertEqual(p.generation,1);self.assertFalse(p.detecting)

if __name__=='__main__':unittest.main()
