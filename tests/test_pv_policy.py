import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from common.pv_detection import TARGET_SELECTION, fresh, selected_observation_valid, target_from_boxes
from Tx.Controller.detection_panel import DetectionPanel
from Tx.Controller.timing_panel import TimingPanel


class PVPolicyTests(unittest.TestCase):
    def test_shared_selector_and_selected_coordinate_validation(self):
        target,count=target_from_boxes([[0,0,10,10,.7,0],[10,10,20,20,.9,0]],0,.5,(100,100))
        self.assertEqual(target['center'],[15,15]);self.assertEqual(count,2)
        row=dict(status='detected',count=count,u=15,v=15)
        self.assertTrue(selected_observation_valid(row))
        for change in (dict(count=0),dict(status='stale'),dict(u=None),dict(v=float('nan'))):
            self.assertFalse(selected_observation_valid(dict(row,**change)))

    @patch.object(DetectionPanel,'poll')
    @patch('Tx.Controller.timing_panel.time.monotonic',return_value=10)
    def test_multiple_candidates_can_begin_and_pass_baseline(self,clock,_):
        p=TimingPanel.__new__(TimingPanel)
        p.active=False;p.running=True;p.preparing=None;p.auto_measure=False
        row=dict(status='detected',count=2,u=100,v=200)
        p.latest=(None,(b'',{},10,0),row)
        p.profile=SimpleNamespace(tick=Mock(),running=False)
        p.fields={k:SimpleNamespace(get=lambda v=v:v) for k,v in dict(
            baseline='2',window='3',hold='.5',floor='2',missing_timeout='1',retries='2',steps='1',repeats='1').items()}
        p.loaded={'path':'test.pt','device':'cpu'};p.settings=(.5,0);p.measure_status=Mock()
        servo=SimpleNamespace(online=True,pending=False,last={'pan':0,'tilt':0},simulated=False,
            limits=dict(pan_min=-180,pan_max=180,tilt_min=-15,tilt_max=40),
            values=lambda:dict(pan=0,tilt=0,speed=100,acc=1),context=lambda:{})
        with tempfile.TemporaryDirectory() as folder:
            p.app=SimpleNamespace(servo=servo,values={},
                stage_dir=lambda stage:Path(folder),detection=SimpleNamespace(stop=Mock()),send=Mock(return_value=True))
            p.begin()
            try:
                self.assertTrue(p.active)
                session=json.loads((p.folder/'session.json').read_text())
                self.assertEqual(session['target_selection'],TARGET_SELECTION)
                p.samples=[dict(row,receive=10+i*.2) for i in range(10)]
                p.app.current=(b'',{},12,0);clock.return_value=12
                p.retry_trial=Mock();p.poll()
                p.retry_trial.assert_not_called();p.app.send.assert_called_once()
                self.assertEqual(p.phase,'response')
            finally:
                p.sample_file.close();p.event_file.close()


    def test_freshness_rejects_stale_and_future_frames(self):
        self.assertTrue(fresh(10,10.5))
        self.assertFalse(fresh(10,10.501))
        self.assertFalse(fresh(10.1,10))


if __name__=='__main__':unittest.main()
