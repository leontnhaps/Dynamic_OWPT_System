"""Test response analysis independently of GUI, hardware and YOLO."""
import ast
import math
from pathlib import Path
import statistics
import unittest
source=Path(__file__).resolve().parents[1]/'Tx/Controller/timing_panel.py'
tree=ast.parse(source.read_text())
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='analyze')
ns={'math':math,'statistics':statistics}
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
        self.assertEqual(analyze(rows,2,1,.5,2)['status'],'invalid_detection')
    def test_multiple(self):
        rows=self.rows();rows[15]['count']=2
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

if __name__=='__main__':unittest.main()
