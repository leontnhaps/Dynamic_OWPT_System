import unittest
from analysis.fps_report import analyze

def event(t, **kw):return dict(gui_unix_ns=int(t*1e9),**kw)
def start(t):return event(t,event='preview',state='started',requested={'width':640,'height':480,'fps':30})
def stat(t,n):return event(t,event='receive_stats',total=n)

class FpsTests(unittest.TestCase):
    def test_weighted_and_zero(self):
        r=analyze([start(0),stat(1,10),stat(2,20),stat(4,20)],0)[0]
        self.assertAlmostEqual(r['mean_receive_fps'],10/3)
        self.assertEqual(r['zero_frame_seconds'],2)
    def test_warmup_stop(self):
        es=[start(0),stat(1,50),stat(5,100),stat(6,110),
            event(7,event='command',command={'cmd':'preview','enable':False}),stat(8,110)]
        self.assertEqual(analyze(es)[0]['received_frames'],10)
    def test_reconnect_reset(self):
        es=[start(0),stat(1,10),stat(2,20),event(3,event='network',state='disconnected'),
            stat(5,20),start(6),stat(7,0),stat(8,5)]
        self.assertEqual([r['mean_receive_fps'] for r in analyze(es,0)],[10,5])
