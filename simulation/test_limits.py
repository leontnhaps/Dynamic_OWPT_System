import unittest
import tempfile
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
from stable_baselines3.common.env_checker import check_env
from .config import Config
from .control import absolute_command, action_for_angles
from .env import TrackingEnv
from .evaluate import rollout

class LimitsTests(unittest.TestCase):
    def test_command_grid_without_slew(self):
        c=Config()
        self.assertIsNone(c.slew_deg_s)
        for target, expected in [([10.3,-3.8],[10,-4]),([180,40],[180,40]),([-180,-15],[-180,-15]),([.5,-.5],[1,-1])]:
            np.testing.assert_allclose(absolute_command(action_for_angles(target,c),np.zeros(2),c),expected)
        check_env(TrackingEnv(c),warn=True)

    def fixture(self, axis, side, sign=1, motion=1., outside=True, at_limit=True):
        c=replace(Config(),pan_sign=sign,tilt_sign=sign,measurement_noise_px=0.,focal_random_fraction=0.)
        env=TrackingEnv(c);env.reset(seed=7)
        angles=np.zeros(2);angles[axis]=(c.angle_low if side<0 else c.angle_high)[axis]
        if not at_limit: angles[axis]-=side*2
        pan,tilt=np.deg2rad(angles*np.array([sign,sign]));cp,sp,ct,st=np.cos(pan),np.sin(pan),np.cos(tilt),np.sin(tilt)
        R=np.array([[cp,-sp*st,sp*ct],[0,ct,st],[-sp,-cp*st,cp*ct]])
        direction=(sign if axis==0 else -sign)*side
        uv=np.array([c.width/2,c.height/2]);extent=(c.width,c.height)[axis]
        uv[axis]=(-10 if direction<0 else extent+10) if outside else extent/2
        start=np.array([(uv[0]-c.width/2)*5/env.focal[0],-(uv[1]-c.height/2)*5/env.focal[1],5.])
        velocity=np.zeros(3);velocity[axis]=direction*motion*(1 if axis==0 else -1)*.1
        env.target_position=lambda t:R@(start+velocity*t)
        return env,action_for_angles(angles,c)

    def test_all_boundaries_and_reversed_signs(self):
        for axis in [0,1]:
            for side in [-1,1]:
                for sign in [-1,1]:
                    env,a=self.fixture(axis,side,sign)
                    _,_,terminated,truncated,info=env.step(a)
                    self.assertTrue(terminated,(axis,side,sign))
                    self.assertFalse(truncated)
                    self.assertIn('outward_exit',info['termination_reason'])
                    with self.assertRaises(RuntimeError):env.step(a)

    def test_no_false_termination(self):
        for kwargs in [dict(motion=-1),dict(motion=0),dict(outside=False),dict(at_limit=False)]:
            env,a=self.fixture(1,-1,**kwargs)
            self.assertFalse(env.step(a)[2],kwargs)
        env=TrackingEnv();env.reset(seed=3)
        self.assertFalse(env.step([1,1])[2]) # pointing backwards is not a limit exit

    def test_legacy_config_preserved(self):
        # Self-contained old-format config.
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'config.json'
            path.write_text(json.dumps(dict(slew_deg_s=2.,command_reference_deg=.2)))
            c=Config.load(path)
        self.assertEqual(c.command_step_deg,0)
        self.assertFalse(c.end_on_limit_exit)
        np.testing.assert_allclose(absolute_command([1,1],np.zeros(2),c),[.2,.2])

    def test_evaluation_stops_and_reports_length(self):
        rows,m=rollout(Config(),10000,'B1')
        self.assertLess(len(rows),300)
        self.assertEqual(m['limit_exit'],1)
        self.assertEqual(m['episode_steps'],len(rows))
        self.assertTrue(rows[-1]['termination_reason'])

if __name__=='__main__':unittest.main()
