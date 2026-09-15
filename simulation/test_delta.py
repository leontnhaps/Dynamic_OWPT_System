import tempfile
from pathlib import Path
import unittest
import numpy as np
from stable_baselines3.common.env_checker import check_env
from .config import Config
from .control import absolute_command, encode_observation, proportional_action
from .env import TrackingEnv
from .evaluate import rollout

class DeltaTests(unittest.TestCase):
    def test_increment_rounding_hold_and_bounds(self):
        c=Config()
        for a,prev,want in [([1,-1],[20,10],[25,5]),([.2,-.2],[20,10],[21,9]),([0,0],[20,10],[20,10]),([.1,-.1],[20,10],[21,9]),([1,-1],[178,-13],[180,-15])]:
            np.testing.assert_allclose(absolute_command(a,np.array(prev),c),want)
        command=np.zeros(2)
        for _ in range(40):command=absolute_command([1,0],command,c)
        np.testing.assert_allclose(command,[180,0])
        np.testing.assert_allclose(absolute_command([-1,0],command,c),[175,0])

    def test_observation_stays_absolute_and_b1_direction(self):
        c=Config();e=np.array([100.,-100.]);prev=np.array([20.,10.])
        obs=encode_observation(e,e,prev,c)
        np.testing.assert_allclose(obs[-2:],[20/180,(10-12.5)/27.5],rtol=1e-6)
        a=proportional_action(obs,c,np.array([68.,70.]))
        self.assertTrue(np.all(a>0))
        np.testing.assert_allclose(absolute_command(a,prev,c),[21,11])

    def test_gym_and_evaluation(self):
        c=Config();check_env(TrackingEnv(c),warn=True)
        rows,_=rollout(c,10000,'B0')
        self.assertTrue(all(r['pan_cmd_deg']==r['tilt_cmd_deg']==0 for r in rows))
        rows,m=rollout(c,10000,'B1')
        self.assertLessEqual(m['max_command_step_deg'],5)
        self.assertEqual(m['limit_exit'],1)

    def test_saved_modes_are_not_reinterpreted(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'config.json';p.write_text('{}')
            self.assertEqual(Config.load(p).action_mode,'absolute')
            Config().save(p)
            self.assertEqual(Config.load(p).action_mode,'delta')

if __name__=='__main__':unittest.main()
