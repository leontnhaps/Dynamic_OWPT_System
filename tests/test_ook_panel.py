import unittest
from types import SimpleNamespace
from Tx.Controller.ook_panel import OokPanel


class Value:
    def __init__(self): self.value=''
    def set(self,value): self.value=value


class PanelStateTests(unittest.TestCase):
    def panel(self):
        p=OokPanel.__new__(OokPanel)
        p.app=SimpleNamespace(send=lambda cmd: True)
        p.online=True; p.hardware={}; p.pending=None; p.run_id=None
        p.deadline=None; p.output_requested=False; p.logpath=None; p.state=Value()
        return p

    def test_old_on_response_does_not_override_pending_off(self):
        p=self.panel(); p.output_requested=True; p.pending=('on',0,'laser')
        p.off(); off=p.pending[0]
        p.event(dict(event='laser',request_id='on',laser_gpio_level=1))
        self.assertEqual(p.pending[0],off)
        p.event(dict(event='laser',request_id=off,laser_gpio_level=0))
        self.assertFalse(p.busy)
        self.assertIsNone(p.pending)

    def test_failed_off_does_not_loop_or_claim_off(self):
        p=self.panel(); p.output_requested=True; p.off(); token=p.pending[0]
        p.event(dict(event='error',request_id=token,message='GPIO unavailable'))
        self.assertIsNone(p.pending)
        self.assertTrue(p.busy)

    def test_disconnect_invalidates_state_and_reconnect_never_starts(self):
        p=self.panel(); sent=[]
        p.app.send=lambda cmd: sent.append(cmd) or True
        p.output_requested=True; p.run_id='run'
        p.event(dict(event='network',port=7600,state='disconnected'))
        self.assertFalse(p.online)
        self.assertIsNone(p.hardware)
        p.event(dict(event='hello',agent_state='connected'))
        self.assertEqual([cmd['cmd'] for cmd in sent],['hardware_status'])


if __name__=='__main__': unittest.main()
