"""Test response analysis independently of GUI, hardware and YOLO."""
import ast
import math
from pathlib import Path
import statistics
import unittest
from common.pv_detection import selected_observation_valid
source=Path(__file__).resolve().parents[1]/'Tx/Controller/timing_panel.py'
tree=ast.parse(source.read_text())
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='analyze')
ns={'math':math,'statistics':statistics,'selected_observation_valid':selected_observation_valid}
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns)
analyze=ns['analyze']

class TimingTests(unittest.TestCase):
    def rows(self):
        return [dict(receive=i/10,status='detected',count=1,u=0 if i<23 else 20,v=0) for i in range(10,51)]
    def test_response(self):
        r=analyze(self.rows(),2,1,.5,2)
        self.assertEqual(r['status'],'valid')
        self.assertAlmostEqual(r['onset_s'],.3)
        self.assertAlmostEqual(r['settled_s'],.3)
    def test_missing(self):
        rows=self.rows();rows[15]['status']='missing'
        self.assertEqual(analyze(rows,2,1,.5,2)['status'],'valid')
    def test_transition_missing_bounds(self):
        rows=self.rows()
        for row in rows:
            if 2.2<=row['receive']<=2.3:row['status']='missing'
        r=analyze(rows,2,1,.5,2)
        self.assertEqual(r['status'],'valid')
        self.assertEqual(r['missing_frames'],2)
        self.assertAlmostEqual(r['onset_lower_s'],.1)
        self.assertAlmostEqual(r['onset_upper_s'],.4)
        self.assertAlmostEqual(r['onset_uncertainty_s'],.3)
    def test_terminal_missing_rejected(self):
        rows=self.rows();rows[-2]['status']='missing'
        self.assertEqual(analyze(rows,2,1,.5,2)['status'],'invalid_detection')
    def test_long_missing_rejected(self):
        rows=self.rows()
        for row in rows:
            if 2.2<=row['receive']<=2.6:row['status']='missing'
        self.assertEqual(analyze(rows,2,1,.5,2)['status'],'invalid_detection')
    def test_multiple_uses_selected_pv_in_baseline_response_and_endpoint(self):
        rows=self.rows()
        for row in rows:row['count']=2
        result=analyze(rows,2,1,.5,2)
        self.assertEqual(result['status'],'valid')
        self.assertAlmostEqual(result['settled_s'],.3)
    def test_invalid_selected_coordinates_are_rejected(self):
        for value in (None,float('nan'),float('inf')):
            rows=self.rows();rows[15].update(count=2,u=value)
            self.assertEqual(analyze(rows,2,1,.5,2)['status'],'invalid_detection')
    def test_multiple_does_not_override_stale_or_missing(self):
        rows=self.rows();rows[15].update(count=2,status='stale')
        self.assertEqual(analyze(rows,2,1,.5,2)['status'],'invalid_detection')
        rows=self.rows();rows[-1].update(count=0,status='missing')
        self.assertEqual(analyze(rows,2,1,.5,2)['status'],'invalid_detection')
    def test_no_movement(self):
        rows=self.rows()
        for r in rows:r['u']=0
        self.assertEqual(analyze(rows,2,1,.5,2)['status'],'movement_too_small')
    def test_gap(self):
        rows=[r for r in self.rows() if not 2.4<r['receive']<3.5]
        self.assertEqual(analyze(rows,2,1,.5,2)['status'],'insufficient_sampling')
    def test_not_settled(self):
        rows=self.rows()
        for r in rows:
            if r['receive']>4.5:r['u']=40 if int(r['receive']*10)%2 else 20
        self.assertEqual(analyze(rows,2,1,.5,2)['status'],'unresolved')

class PreparationTests(unittest.TestCase):
    def panel(self):
        from types import SimpleNamespace
        method=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TimingPanel')
        method=next(n for n in method.body if isinstance(n,ast.FunctionDef) and n.name=='prepare_poll')
        import time
        env={'time':time}
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),env)
        state=SimpleNamespace(model=None,preparing='model',setup_deadline=time.monotonic()+100)
        return state,env['prepare_poll']
    def test_wait_for_loading(self):
        state,poll=self.panel()
        poll(state)
        self.assertEqual(state.preparing,'model')
    def test_apply_camera_and_limits(self):
        from types import SimpleNamespace
        state,poll=self.panel();state.model=object();sent=[];settings={}
        state.app=SimpleNamespace(values={k:SimpleNamespace(set=lambda v,k=k:settings.update({k:v})) for k in ('width','height','fps','quality','shutter_speed','analogue_gain')},send=lambda cmd,tracking:sent.append(cmd) or True)
        state.status=SimpleNamespace(set=lambda v:None)
        state.setup_command=lambda cmd,phase:sent.append(dict(cmd,phase=phase))
        poll(state)
        self.assertEqual(sent[0]['width'],1296)
        self.assertEqual(sent[0]['height'],972)
        self.assertIsNone(sent[0]['shutter_speed'])
        self.assertEqual(sent[1]['tilt_min'],-15)
        self.assertEqual(sent[1]['phase'],'limits')
    def test_timeout(self):
        state,poll=self.panel();state.setup_deadline=0
        with self.assertRaises(ValueError):poll(state)

class RetryTests(unittest.TestCase):
    def panel(self, index=0, attempt=0):
        import io,json,time,uuid
        from types import SimpleNamespace
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TimingPanel')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='retry_trial')
        env={'time':time,'json':json,'uuid':uuid,'move_from':lambda data,limits:data}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),env)
        sent=[];aborted=[]
        state=SimpleNamespace(index=index,attempt=attempt,cfg={'retries':2,'window':3},
            plan=[('pan',1,{'pan':1,'tilt':0}),('pan',-1,{'pan':0,'tilt':0})],
            origin={'pan':0,'tilt':0},results=[],event_file=io.StringIO(),
            measure_status=SimpleNamespace(set=lambda text:None),abort=aborted.append,
            app=SimpleNamespace(servo=SimpleNamespace(values=lambda:{'speed':100,'acc':1},limits={}),
                send=lambda cmd,tracking:sent.append(cmd) or True))
        return state,env['retry_trial'],sent,aborted
    def test_restore_start_then_retry(self):
        state,retry,sent,aborted=self.panel()
        retry(state,{'status':'invalid_detection'})
        self.assertEqual(state.phase,'retry_return')
        self.assertEqual(state.attempt,1)
        self.assertEqual(sent[0]['pan'],0)
        self.assertFalse(aborted)
        self.assertEqual(state.results[0]['attempt'],0)
    def test_return_trial_restores_prior_target(self):
        state,retry,sent,aborted=self.panel(index=1)
        retry(state,{'status':'invalid_detection'})
        self.assertEqual(sent[0]['pan'],1)
    def test_retry_exhausted(self):
        state,retry,sent,aborted=self.panel(attempt=2)
        retry(state,{'status':'invalid_detection'})
        self.assertFalse(sent)
        self.assertEqual(len(aborted),1)

class MissingTimeoutTests(unittest.TestCase):
    def test_continuous_missing_and_recovery(self):
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='missing_timeout')
        env={}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),env)
        update=env['missing_timeout']
        since,expired=update('missing',10,None,1)
        self.assertFalse(expired)
        since,expired=update('missing',10.5,since,1)
        self.assertFalse(expired)
        self.assertEqual(update('missing',11,since,1),(10,True))
        self.assertEqual(update('detected',10.6,since,1),(None,False))

if __name__=='__main__':unittest.main()
