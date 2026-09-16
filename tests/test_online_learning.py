import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
from simulation.config import Config
from Tx.Controller.online_learning import Episode,EpisodeSettings,RealLearner

class EpisodeTests(unittest.TestCase):
    def result(self,err=100):
        return dict(observation=[err/648,0,0,0,0,-12.5/27.5],action=[.2,0],command=[1,0],applied_delta=[1,0],error=[err,0])
    def test_random_reset_bounds_and_validation(self):
        cfg=Config();limits=dict(pan_min=-5,pan_max=5,tilt_min=-3,tilt_max=3)
        s=EpisodeSettings();s.validate(cfg,limits)
        rng=np.random.default_rng(42)
        points=[s.sample(rng) for _ in range(100)]
        self.assertTrue(all(-3<=p<=3 and -3<=t<=3 for p,t in points))
        self.assertGreater(len(set(map(tuple,points))),20)
        with self.assertRaises(ValueError):replace(s,pan_max=6).validate(cfg,limits)
    def test_reward_is_after_action_and_loaded_weight(self):
        cfg=replace(Config(),command_weight=0)
        e=Episode(cfg,EpisodeSettings(),1);e.remember(self.result(200),1)
        e.observe(self.result(50),1.1)
        self.assertAlmostEqual(e.rows[0]['reward'],-10*(50/648)**2)
        self.assertEqual(e.rows[0]['action'],[.2,0])
        self.assertEqual(e.rows[0]['next_observation'],self.result(50)['observation'])
    def test_misses_hold_until_threshold_and_recover(self):
        e=Episode(Config(),EpisodeSettings(),1)
        self.assertFalse(e.missed(1));self.assertFalse(e.missed(1.1))
        e.recovered();self.assertFalse(e.missed(1.2));self.assertTrue(e.loss_expired(1.8))
        self.assertFalse(e.missed(1.3));self.assertTrue(e.missed(1.4))
    def test_loss_terminal_has_no_bootstrap_or_fake_error(self):
        e=Episode(Config(),EpisodeSettings(),1);e.remember(self.result(),1)
        e.finish('target_lost',1.5);r=e.rows[0]
        self.assertTrue(r['terminated']);self.assertFalse(r['truncated']);self.assertFalse(r['observed_next'])
        self.assertEqual(r['reward'],-10)
    def test_fault_drops_unobserved_transition(self):
        for reason in ('time_limit','manual_stop','network_error'):
            e=Episode(Config(),EpisodeSettings(),1);e.remember(self.result(),1)
            report=e.finish(reason,2)
            self.assertEqual(e.rows,[]);self.assertTrue(report['unpaired_action_dropped'])
    def test_no_cross_episode_transition_and_timeout(self):
        e=Episode(Config(),EpisodeSettings(),1);e.remember(self.result(),1);e.observe(self.result(),1.1)
        e.finish('time_limit',2)
        self.assertTrue(e.rows[-1]['truncated']);self.assertFalse(e.rows[-1]['terminated'])
        new=Episode(Config(),EpisodeSettings(),2);new.observe(self.result(),3)
        self.assertEqual(new.rows,[])

class OptimizerTest(unittest.TestCase):
    def test_real_sac_update_save_reload_and_timeout_masks(self):
        import torch,gymnasium as gym
        from stable_baselines3 import SAC
        torch.set_num_threads(1)
        class Dummy(gym.Env):
            observation_space=gym.spaces.Box(-np.inf,np.inf,(6,),np.float32)
            action_space=gym.spaces.Box(-1,1,(2,),np.float32)
        model=SAC('MlpPolicy',Dummy(),policy_kwargs=dict(net_arch=[16,16]),device='cpu',verbose=0)
        with tempfile.TemporaryDirectory() as d:
            learner=RealLearner(SimpleNamespace(policy=model,cfg=Config()),d,{})
            e=Episode(Config(),EpisodeSettings(),1)
            for i in range(64):
                e.rows.append(dict(observation=[.2,0,0,0,0,0],action=[.2,0],next_observation=[.1,0,0,0,0,0],reward=-.1,terminated=False,truncated=(i==63),elapsed_s=.1,observed_next=True))
            before=[x.detach().clone() for x in model.actor.parameters()]
            result=learner.update(e,dict(reason='time_limit'),2)
            self.assertEqual(result['gradient_updates'],2)
            self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,model.actor.parameters())))
            self.assertEqual(model.replay_buffer.timeouts[63,0],1)
            saved=SAC.load(Path(d)/'episode_0001/model.zip',device='cpu')
            self.assertEqual(saved.num_timesteps,64)
            self.assertTrue((Path(d)/'episode_0001/replay_buffer.pkl').is_file())
            e2=Episode(Config(),EpisodeSettings(),2);learner.cancel.set()
            result=learner.update(e2,dict(reason='manual_stop'),32)
            self.assertEqual(result['gradient_updates'],0)
            learner.close()


class LearningPanelTests(unittest.TestCase):
    def panel(self):
        from Tx.Controller.learning_panel import LearningPanel
        p=LearningPanel.__new__(LearningPanel)
        p.learning=True;p.running=True;p.detecting=True;p.phase='track';p.pending=None;p.generation=0
        p.models=SimpleNamespace(cfg=Config(),learning=True);p.episode_number=1
        p.episode=Episode(Config(),EpisodeSettings(),1);p.episode.started=10
        p.episode.remember(EpisodeTests().result(),10)
        p.previous_error=[100,0];p.log=None;p.status=Mock();p.learning_status=Mock();p.detail=Mock();p.canvas=Mock()
        p.learner=SimpleNamespace(cancel=Mock());p.initial_command=[0,0];p.seed=42;p.update_requested=None
        p.app=SimpleNamespace(record=Mock(),send=Mock(),servo=SimpleNamespace(last={'pan':0,'tilt':0},limits={},vars={'pan':Mock(),'tilt':Mock()}))
        return p
    @patch('Tx.Controller.live_panel.ImageTk.PhotoImage',return_value=object())
    def test_first_miss_holds_third_ends_without_any_move(self,_):
        from PIL import Image
        p=self.panel()
        for i in range(3):
            result=dict(image=Image.new('RGB',(1296,972)),target=None,count=0)
            p.consume(result,(b'',{},10+i*.1,0),10+i*.1,10.05+i*.1)
            p.app.send.assert_not_called()
            self.assertEqual(p.learning,i<2)
        self.assertEqual(p.update_requested[1]['reason'],'target_lost')
        self.assertTrue(p.episode.rows[-1]['terminated'])
    def test_reset_ack_waits_before_new_frame(self):
        p=self.panel();p.phase='reset';p.reset_id='reset';p.pending=('reset',time.monotonic())
        p.event(dict(event='servo',operation='move',request_id='reset',commanded={'pan':1,'tilt':2},limits={}))
        self.assertEqual(p.phase,'settle');self.assertGreater(p.not_before,time.monotonic())
        self.assertIsNone(p.pending)
    def test_user_stop_cancels_updates_and_drops_unknown_action(self):
        p=self.panel();p.stop('사용자 정지')
        self.assertFalse(p.learning);p.learner.cancel.set.assert_called_once()
        self.assertTrue(p.update_requested[1]['unpaired_action_dropped'])

if __name__=='__main__':unittest.main()
