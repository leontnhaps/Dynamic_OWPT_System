import ast
import csv
import io
import json
import math
from pathlib import Path
import statistics
import tempfile
import time
from types import SimpleNamespace
import unittest

source=Path(__file__).resolve().parents[1]/'Tx/Controller/processing_profile.py'
tree=ast.parse(source.read_text())
# Load the actual scheduler and writer without requiring a display or CUDA.
nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
ns=dict(math=math,statistics=statistics,time=time,csv=csv,json=json)
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
Profile=ns['ProcessingProfile']

class ProfileTests(unittest.TestCase):
    def test_six_phases(self):
        p=Profile.__new__(Profile);p.index=-1;p.phases=[];ended=[];p.finish=ended.append
        for i in range(6):
            p.next_phase(i*32)
            self.assertEqual(p.mode,'video' if i%2==0 else 'yolo')
            self.assertEqual(p.deadline-p.measure_start,30)
        p.next_phase(192)
        self.assertEqual(ended,['측정 완료'])
    def test_worker_timestamps(self):
        model=SimpleNamespace(infer=lambda jpeg,*settings:{'count':1})
        submitted=time.monotonic();frame=(b'x',{},submitted-.1,0)
        result,marks=ns['timed_infer'](model,frame,(.5,0),submitted)
        self.assertLessEqual(marks['submitted'],marks['worker_started'])
        self.assertLessEqual(marks['worker_started'],marks['worker_finished'])
    def test_summary_keeps_phase_groups(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Profile.__new__(Profile);p.running=True;p.folder=Path(folder)
            p.rows=[dict(phase=1,worker_ms=40),dict(phase=1,worker_ms=60)]
            p.frames=[dict(phase=0,gui_work_ms=10)]
            p.events=[dict(event='preview_timing',mean={'send_ms':.4})]
            p.phases=[dict(index=0,mode='video'),dict(index=1,mode='yolo')]
            p.config={'note':'separate clocks'}
            p.panel=SimpleNamespace(measure_status=SimpleNamespace(set=lambda x:None))
            p.app=SimpleNamespace(record=lambda x:None)
            p.finish('測定')
            summary=json.loads((p.folder/'summary.json').read_text())
            self.assertEqual(summary['phases'][0]['inference']['samples'],0)
            self.assertEqual(summary['phases'][1]['inference']['metrics']['worker_ms']['median'],50)
            self.assertEqual(len((p.folder/'pi_events.jsonl').read_text().splitlines()),1)
            p.finish()  # Double stop preserves files.

if __name__=='__main__':unittest.main()
