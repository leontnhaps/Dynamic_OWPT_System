import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from simulation.config import Config
from simulation.control import reward_terms
from Tx.Controller.learning_panel import LearningPanel
from Tx.Controller.online_learning import Episode, EpisodeSettings

class BonusTests(unittest.TestCase):
    def test_values_continuity_and_bound(self):
        cfg=replace(Config(),reward_mode='alignment',command_weight=0,alignment_bonus=30)
        r=lambda d:reward_terms([d,0],np.zeros(2),np.zeros(2),cfg)[0]
        np.testing.assert_allclose([r(d) for d in [100,50,25,10,0]],[10/3,5,10/1.5+7.5,10/1.2+19.2,40])
        self.assertAlmostEqual(r(50-1e-7),r(50+1e-7),places=6)
        values=[r(d) for d in range(1000)]
        self.assertTrue(all(0<v<=40 for v in values))
        self.assertTrue(all(a>b for a,b in zip(values,values[1:])))

    def test_old_config_unchanged_and_new_roundtrip(self):
        cfg=replace(Config(),reward_mode='alignment',command_weight=0)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'config.json';cfg.save(p)
            data=json.loads(p.read_text());data.pop('alignment_bonus');data.pop('alignment_bonus_radius_px')
            p.write_text(json.dumps(data));old=Config.load(p)
            self.assertEqual(reward_terms([0,0],np.zeros(2),np.zeros(2),old)[0],10)
            new=replace(old,alignment_bonus=30);new.save(p);self.assertEqual(Config.load(p),new)
        for key,value in [('alignment_bonus',-1),('alignment_bonus',float('nan')),('alignment_bonus_radius_px',0)]:
            with self.assertRaises(ValueError):replace(cfg,**{key:value})

    def test_bonus_after_action_and_loss_penalty(self):
        cfg=replace(Config(),reward_mode='alignment',command_weight=0,alignment_bonus=30)
        ep=Episode(cfg,EpisodeSettings(),1)
        result=dict(observation=[0]*6,action=[0,0],command=[0,0],applied_delta=[0,0],error=[0,0])
        ep.remember(result,0);ep.observe(result,1)
        self.assertEqual(ep.rows[0]['reward'],40)
        ep.remember(result,1);ep.finish('target_lost',2)
        self.assertEqual(ep.rows[-1]['reward'],-10)

    def test_apply_preserves_policy_and_calibration_resets_old_buffer(self):
        p=LearningPanel.__new__(LearningPanel);p.busy=Mock();p.stop=Mock();p.future=None
        policy=Mock();p.models=SimpleNamespace(cfg=replace(Config(),laser_u=700,delta_limit_deg=3),policy=policy)
        p.learner=Mock();old_learner=p.learner;p.load_paths={};p.app=SimpleNamespace(record=Mock());p.learning_status=Mock()
        p.apply_alignment_bonus()
        self.assertIs(p.models.policy,policy);self.assertEqual(p.models.cfg.laser_u,700)
        self.assertEqual(p.models.cfg.delta_limit_deg,3);self.assertEqual(p.models.cfg.alignment_bonus,30)
        policy.replay_buffer.reset.assert_called_once();old_learner.close.assert_called_once()
        self.assertIsNone(p.learner)
        p.apply_alignment_bonus();policy.replay_buffer.reset.assert_called_once()

if __name__=='__main__':unittest.main()
