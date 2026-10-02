"""Inference and logging must not depend on visible Tk image updates."""
import csv
import io
import json
import queue
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from PIL import Image
from Tx.Controller.Com_main import App
from Tx.Controller.detection_panel import DetectionPanel
from Tx.Controller.processing_profile import ProcessingProfile


class VideoPipelineTests(unittest.TestCase):
    def app(self):
        a=App.__new__(App)
        a.root=Mock();a.tabs=Mock();a.preview=Mock();a.preview_frame=None
        a.closing=False;a.current=None;a.status=Mock();a.status_mark=0
        a.record=Mock();a.mark=10;a.previous=0;a.fps=30
        a.servo=Mock();a.capture=Mock()
        a.detection=Mock(name='detection');a.live=Mock(name='live');a.timing=Mock(name='timing')
        for p in (a.detection,a.live,a.timing):p.guard=lambda fn:fn()
        a.timing.profile.running=False
        return a

    @patch('Tx.Controller.Com_main.ImageTk.PhotoImage')
    def test_new_frame_reaches_inference_without_pixel_decode_or_render(self,photo):
        a=self.app();data=io.BytesIO();Image.new('RGB',(1296,972)).save(data,format='JPEG')
        frame=(data.getvalue(),dict(seq=1,simulated=False),10,123)
        a.net=SimpleNamespace(events=queue.Queue(),pop=lambda:(frame,1))
        seen=[];a.timing.poll=lambda:seen.append(a.current)
        with patch('Tx.Controller.Com_main.time.monotonic',return_value=10.01), \
             patch('PIL.JpegImagePlugin.JpegImageFile.load',side_effect=AssertionError('pixel decode in poll')):
            a.poll()
        self.assertEqual(seen,[frame]);self.assertEqual(a.size,(1296,972))
        photo.assert_not_called();a.preview.configure.assert_not_called()
        a.root.after.assert_called_once_with(5,a.poll)

    @patch('Tx.Controller.Com_main.Image.open',side_effect=AssertionError('hidden raw preview decoded'))
    def test_only_selected_panel_renders(self,_):
        a=self.app();a.current=(b'invalid jpeg',{},10,0)
        a.tabs.select.return_value=str(a.detection)
        a.render()
        a.detection.render.assert_called_once()
        a.live.render.assert_not_called();a.timing.render.assert_not_called()
        a.preview.configure.assert_not_called()
        a.root.after.assert_called_once()

    @patch('Tx.Controller.Com_main.Image.open',side_effect=AssertionError('duplicate raw decode'))
    def test_profile_owns_visible_timing_canvas(self,_):
        a=self.app();a.timing.profile.running=True
        a.tabs.select.return_value=str(a.timing);a.render()
        a.timing.profile.render.assert_called_once();a.timing.render.assert_not_called()
        a.preview.configure.assert_not_called()

    @patch('Tx.Controller.detection_panel.ImageTk.PhotoImage',return_value=object())
    def test_detection_records_before_render_and_keeps_full_resolution(self,photo):
        p=DetectionPanel.__new__(DetectionPanel)
        p.app=SimpleNamespace(fps=30);p.processed=0;p.started=9;p.log=None
        p.status=Mock();p.canvas=Mock();p.rendered_frame=None
        image=Image.new('RGB',(1296,972))
        result=dict(image=image,target=dict(box=[20,30,60,70],center=[40,50],confidence=.9),count=1,inference_ms=40)
        p.consume(result,(b'jpeg',{'seq':1},10,1),10.1)
        photo.assert_not_called();self.assertEqual(p.latest[2]['u'],40)
        p.render();p.render()
        photo.assert_called_once();self.assertEqual(image.size,(1296,972))
        self.assertEqual(p.latest[0].size,(1296,972))
        annotated=p.annotated(p.latest[0],p.latest[2])
        self.assertEqual(annotated.getpixel((20,30)),(0,255,0))
        self.assertEqual(image.getpixel((20,30)),(0,0,0))

    def profile(self):
        p=ProcessingProfile.__new__(ProcessingProfile)
        p.running=True;p.state='measure';p.index=1;p.mode='yolo'
        p.measure_start=9;p.deadline=39;p.rows=[];p.frames=[];p.events=[];p.receipts=[]
        p.rendered_frame=None;p.display=None;p.settings=(.5,0);p.model=object()
        p.panel=SimpleNamespace(canvas=Mock(),measure_status=Mock())
        p.app=SimpleNamespace(live=SimpleNamespace(detecting=False),detection=SimpleNamespace(running=False),
                              current=(b'jpeg',{'seq':2},10.06,2),received_count=2,record=Mock())
        p.job=(b'jpeg',{'seq':1},10,1);p.job_total=1;p.submitted=10.01;p.last=10
        p.future=Future()
        p.image=Image.new('RGB',(1296,972))
        p.future.set_result((dict(image=p.image,inference_ms=40,count=1),
                            dict(receive=10,submitted=10.01,worker_started=10.01,worker_finished=10.05)))
        p.pool=Mock();p.pool.submit.return_value=Future()
        return p

    @patch('Tx.Controller.processing_profile.ImageTk.PhotoImage',return_value=object())
    def test_result_logged_and_next_inference_submitted_before_render(self,photo):
        p=self.profile()
        with patch('Tx.Controller.processing_profile.time.monotonic',return_value=10.07):p._tick()
        photo.assert_not_called();p.pool.submit.assert_called_once()
        self.assertEqual(len(p.rows),1);self.assertIsNone(p.rows[0]['widget_updated'])
        self.assertAlmostEqual(p.rows[0]['receive_to_result_ms'],70)
        with patch('Tx.Controller.processing_profile.time.monotonic',side_effect=[10.08,10.09]):p.render()
        self.assertAlmostEqual(p.rows[0]['receive_to_widget_ms'],90)
        self.assertEqual(p.frames[0]['seq'],1)  # The rendered image is not the newer raw frame.
        self.assertEqual(p.frames[0]['total_received'],1)
        self.assertEqual(p.image.size,(1296,972));photo.assert_called_once()

    def test_unrendered_results_remain_in_summary_without_zero_latency(self):
        p=self.profile()
        with patch('Tx.Controller.processing_profile.time.monotonic',return_value=10.07):p._tick()
        p.config={'note':'test'};p.phases=[dict(index=1,mode='yolo')]
        p.receipts=[dict(phase=1,receive=10,total=1),dict(phase=1,receive=11,total=31)]
        with tempfile.TemporaryDirectory() as folder:
            p.folder=Path(folder);p.finish('측정 완료')
            summary=json.loads((p.folder/'summary.json').read_text())['phases'][0]
            self.assertEqual(summary['inference']['samples'],1)
            self.assertEqual(summary['inference']['rendered_samples'],0)
            self.assertEqual(summary['inference']['metrics']['render_ms'],{'count':0})
            self.assertEqual(summary['receive_fps'],30)
            with (p.folder/'processing.csv').open() as f:row=next(csv.DictReader(f))
            self.assertEqual(row['widget_updated'],'')

    def test_phase_boundary_discards_pending_display_and_late_result(self):
        p=self.profile();p.phases=[]
        p.app.current=(b'jpeg',{},39.01,3)
        p.future.result()[1]['worker_finished']=39.0
        p.submitted=38.99
        with patch('Tx.Controller.processing_profile.time.monotonic',return_value=39.02):p._tick()
        self.assertEqual(p.mode,'video');self.assertEqual(p.index,2)
        self.assertIsNone(p.display);self.assertEqual(p.rows,[])
        p.pool.submit.assert_not_called()

    @patch('Tx.Controller.processing_profile.ImageTk.PhotoImage',return_value=object())
    def test_stationary_run_completes_six_phases_and_saves_without_moves(self,_):
        p=ProcessingProfile.__new__(ProcessingProfile)
        setting=lambda value:SimpleNamespace(get=lambda:value,set=Mock())
        p.panel=SimpleNamespace(app=None,active=False,preparing=None,future=None,stop=Mock(),
                                conf=setting('.5'),class_id=setting('0'),device=setting('cpu'),
                                path=setting('test.pt'),canvas=Mock(),measure_status=Mock())
        p.app=SimpleNamespace(live=SimpleNamespace(stop=Mock(),pending=None,detecting=False,future=None),
                              detection=SimpleNamespace(stop=Mock(),running=False,future=None),
                              servo=SimpleNamespace(pending=None),send=Mock(return_value=True),
                              values={k:setting('') for k in ('width','height','fps','quality','shutter_speed','analogue_gain')},
                              current=None,received_count=0,record=Mock())
        raw=io.BytesIO();Image.new('RGB',(32,24)).save(raw,format='JPEG')
        model=SimpleNamespace(names={0:'PV'},infer=lambda *args:dict(image=Image.new('RGB',(32,24)),count=1,inference_ms=1))
        def submit(fn,*args):
            f=Future()
            f.set_result(model if len(args)==2 else fn(*args))
            return f
        p.pool=SimpleNamespace(submit=submit);p.running=False;p.future=None
        with tempfile.TemporaryDirectory() as folder, \
             patch('Tx.Controller.processing_profile.time.monotonic') as clock:
            p.app.stage_dir=lambda stage:Path(folder)
            clock.return_value=100;p.start()
            for i in range(1,1931):
                clock.return_value=100+i/10
                p.app.current=(raw.getvalue(),dict(seq=i,simulated=False,requested=p.config['camera']),clock.return_value,i)
                p.app.received_count=i
                p.received(p.app.current,i);p.tick();p.render()
                if not p.running:break
            self.assertFalse(p.running)
            summary=json.loads((p.folder/'summary.json').read_text())
            self.assertEqual(summary['reason'],'측정 완료');self.assertEqual(len(summary['phases']),6)
            for phase in summary['phases']:
                self.assertGreater(phase['video']['samples'],200)
                self.assertEqual(phase['inference']['samples']>0,phase['mode']=='yolo')
            self.assertEqual([c.args[0]['cmd'] for c in p.app.send.call_args_list],['preview'])


if __name__=='__main__':unittest.main()
