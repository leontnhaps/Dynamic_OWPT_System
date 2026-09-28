import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from simulation.config import Config
from simulation.control import reward_terms
from Tx.Controller.online_learning import EpisodeBatch, Episode, EpisodeSettings, RealLearner
from Tx.Controller.learning_panel import LearningPanel


class AlignmentTests(unittest.TestCase):
    def test_after_action_reward_and_config_roundtrip(self):
        cfg=replace(Config(),reward_mode='alignment',alignment_scale_px=50,command_weight=0)
        rewards=[reward_terms([d,0],np.array([5,5]),np.zeros(2),cfg)[0] for d in (200,100,50,0)]
        np.testing.assert_allclose(rewards,[2.,10/3,5.,10.])
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'config.json';cfg.save(path)
            self.assertEqual(Config.load(path),cfg)
        e=Episode(cfg,EpisodeSettings(),1)
        e.remember(dict(observation=[1,0,0,0,0,0],action=[1,0],command=[1,0],applied_delta=[1,0]),0)
        e.observe(dict(observation=[0]*6,error=[0,50]),.2)
        self.assertEqual(e.rows[0]['reward'],5.)
        # Fresh physical next observation rather than previous error determines reward.
        self.assertEqual(e.rows[0]['next_observation'],[0]*6)

    def test_new_sac_updates_without_loading_old_weights(self):
        import torch
        from stable_baselines3 import SAC
        from Tx.Controller.live_core import new_policy
        torch.set_num_threads(1)
        cfg=replace(Config(),reward_mode='alignment',command_weight=0,delta_limit_deg=1)
        with patch.object(SAC,'load',side_effect=AssertionError('must not load old weights')):
            model=new_policy(cfg,'cpu',42)
        self.assertEqual(model.num_timesteps,0)
        self.assertEqual(model.replay_buffer.size(),0)
        with tempfile.TemporaryDirectory() as d:
            learner=RealLearner(SimpleNamespace(policy=model,cfg=cfg,from_scratch=True),d,{})
            ep=Episode(cfg,EpisodeSettings(),1)
            ep.rows=[dict(observation=[.2,0,0,0,0,0],next_observation=[.1,0,0,0,0,0],
                action=[.2,0],reward=5.,terminated=False,truncated=False,elapsed_s=.1,observed_next=True) for _ in range(64)]
            before=[p.detach().clone() for p in model.actor.parameters()]
            learner.update(ep,dict(reason='time_limit'),2)
            self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,model.actor.parameters())))
            restored=SAC.load(Path(d)/'episode_0001/model.zip',device='cpu')
            self.assertEqual(restored.num_timesteps,64)
            a,_=restored.predict(np.zeros(6,dtype=np.float32),deterministic=True)
            self.assertTrue(np.isfinite(a).all())
            learner.close()


class BatchTests(unittest.TestCase):
    def test_exact_count_and_normal_endings(self):
        b=EpisodeBatch(3)
        for i,reason in enumerate(('target_lost','time_limit','time_limit')):
            b.began();self.assertFalse(b.ready(100))
            b.saved(reason,10*i)
            if i<2:
                self.assertFalse(b.ready(10*i+.5));self.assertTrue(b.ready(10*i+1))
            else:self.assertFalse(b.active)
        self.assertEqual(b.started,3)
        for bad in (0,1001,1.5):
            with self.assertRaises(ValueError):EpisodeBatch(bad)

    def panel(self):
        p=LearningPanel.__new__(LearningPanel)
        p.batch=EpisodeBatch(3);p.batch.began();p.batch.saved('time_limit',0)
        p.batch_settings=EpisodeSettings();p.future=None;p.update_requested=None;p.pending=None
        p.learning=False;p.learner=None;p.running=False;p.detecting=False;p.log=None;p.generation=0
        p.status=Mock();p.learning_status=Mock();p.previous_error=None
        p.app=SimpleNamespace(servo=SimpleNamespace(pending=False),record=Mock())
        return p

    def test_wait_for_inflight_command_then_one_restart(self):
        p=self.panel()
        def start(settings):
            p.stop();p.learning=True  # parent.start invokes virtual stop
        p._start_episode=Mock(side_effect=start)
        p.pending=('move',0);p.advance_batch(10);p._start_episode.assert_not_called()
        p.pending=None;p.advance_batch(10)
        self.assertEqual(p.batch.started,2);self.assertTrue(p.batch.active)
        p.advance_batch(11);p._start_episode.assert_called_once()

    def test_stop_during_delay_prevents_restart(self):
        p=self.panel();p._start_episode=Mock()
        p.stop();p.advance_batch(10)
        p._start_episode.assert_not_called();self.assertFalse(p.batch.active)

    def test_fault_during_save_cancels_repeat_and_optimizer(self):
        p=self.panel();p.learner=SimpleNamespace(cancel=Mock());p.update_requested=(None,None,32)
        p.event(dict(event='network',state='disconnected'))
        self.assertFalse(p.batch.active);p.learner.cancel.set.assert_called_once()
        p.batch.saved('time_limit',5);self.assertFalse(p.batch.ready(10))

    def test_restart_validation_failure_cancels_batch(self):
        p=self.panel();p._start_episode=Mock(side_effect=ValueError('stale image'))
        p.advance_batch(10);self.assertFalse(p.batch.active)
        p.app.record.assert_called_once()

    def test_error_reason_never_restarts(self):
        for reason in ('network_error','user_stop','ambiguous','stale','save_error'):
            b=EpisodeBatch(3);b.began();b.saved(reason,0)
            self.assertFalse(b.ready(10));self.assertFalse(b.active)

if __name__=='__main__':unittest.main()
