"""기하, M0 식, 제한/지연, 동일 평가 조건의 회귀 검증."""
from dataclasses import replace
import unittest
import numpy as np
from stable_baselines3.common.env_checker import check_env
from .calibration import fit_scale, SAMPLES
from .config import Config
from .control import absolute_command, encode_observation, reward_terms
from .env import TrackingEnv, project
from .evaluate import rollout


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = replace(Config(), focal_random_fraction=0., measurement_noise_px=0., episode_steps=50)

    def test_corrected_calibration_and_exclusion(self):
        self.assertEqual([r["x"] for r in SAMPLES if not r["use"]], [.225])
        self.assertEqual(SAMPLES[-1]["y"], .05)
        scale, _ = fit_scale()
        np.testing.assert_allclose(scale/100, [7.8320691358, 7.9886233422])

    def test_projection_matches_translation_and_rotation(self):
        focal = fit_scale()[0]*5
        cfg = self.cfg
        center = project([0, 0, 5], [0, 0], focal, cfg)
        np.testing.assert_allclose(center, [648, 486])
        shift = project([.01, .01, 5], [0, 0], focal, cfg)-center
        np.testing.assert_allclose(shift, fit_scale()[0]*[.01, -.01])
        epsilon = .0001
        pan_delta = (project([0, 0, 5], [epsilon, 0], focal, cfg)-center)/epsilon
        tilt_delta = (project([0, 0, 5], [0, epsilon], focal, cfg)-center)/epsilon
        np.testing.assert_allclose(pan_delta, [-focal[0]*np.pi/180, 0], atol=1e-6)
        np.testing.assert_allclose(tilt_delta, [0, focal[1]*np.pi/180], atol=1e-6)

    def test_absolute_action_and_slew(self):
        c = replace(self.cfg, slew_deg_s=10000)
        np.testing.assert_allclose(absolute_command([.5, -.5], np.array([2., 2.]), c), [3., -3.])
        command = np.zeros(2)
        for _ in range(100):
            next_command = absolute_command([1, -1], command, self.cfg)
            self.assertLessEqual(np.max(np.abs(next_command-command)), .200000001)
            self.assertLessEqual(np.max(np.abs(next_command)), 6.)
            command = next_command
        np.testing.assert_allclose(command, [6, -6])
        with self.assertRaises(ValueError):
            absolute_command([float("nan"), 0], command, c)

    def test_m0_observation_and_reward(self):
        obs = encode_observation(np.array([64.8, -48.6]), np.array([0., 0.]), np.array([3., -3.]), self.cfg)
        np.testing.assert_allclose(obs, [.1, -.1, .1, -.1, .5, -.5])
        reward, point, cost = reward_terms([64.8, -48.6], np.array([.2, 0]), np.zeros(2), self.cfg)
        self.assertAlmostEqual(point, -.02)
        self.assertAlmostEqual(cost, 1.)
        self.assertAlmostEqual(reward, -.22)

    def test_command_is_observed_not_lagged_angle(self):
        env = TrackingEnv(replace(self.cfg, command_delay_steps=1))
        env.reset(seed=4)
        obs, _, _, _, info = env.step([1, 1])
        np.testing.assert_allclose(obs[-2:]*self.cfg.angle_limit_deg, [.2, .2])
        np.testing.assert_allclose(info["actual_angles_deg"], [0, 0])
        _, _, _, _, info = env.step([1, 1])
        np.testing.assert_allclose(info["actual_angles_deg"], [.2, .2])

    def test_reproducibility_and_gym_contract(self):
        check_env(TrackingEnv(self.cfg), warn=True)
        a, b = TrackingEnv(), TrackingEnv()
        np.testing.assert_array_equal(a.reset(seed=14)[0], b.reset(seed=14)[0])
        for i in range(15):
            x, y = a.step([.1, -.2]), b.step([.1, -.2])
            np.testing.assert_array_equal(x[0], y[0])
            self.assertEqual(x[1], y[1])

    def test_lost_observation_holds_and_episode_has_fixed_length(self):
        env = TrackingEnv(self.cfg)
        env.reset(seed=11)
        held = env.last_measured_error.copy()
        env.actual_angles = np.array([15., 0.])
        _, visible, measured = env._measure()
        self.assertFalse(visible)
        np.testing.assert_array_equal(measured, held)
        env.reset(seed=11)
        for i in range(self.cfg.episode_steps):
            _, _, terminated, truncated, _ = env.step([1, 1])
            self.assertFalse(terminated)
            self.assertEqual(truncated, i == self.cfg.episode_steps-1)
        with self.assertRaises(RuntimeError):
            env.step([0, 0])

    def test_paired_baselines_and_proportional_reduces_error(self):
        fixed, fixed_metrics = rollout(self.cfg, 10000, "B0", scenario="stationary")
        tracking, tracking_metrics = rollout(self.cfg, 10000, "B1", scenario="stationary")
        np.testing.assert_array_equal([[r["target_x_m"], r["target_y_m"]] for r in fixed],
                                      [[r["target_x_m"], r["target_y_m"]] for r in tracking])
        self.assertLess(tracking_metrics["pointing_rms_px"], fixed_metrics["pointing_rms_px"])
        self.assertLess(tracking[-1]["error_px"], 1.)
        self.assertLessEqual(tracking_metrics["max_command_step_deg"], .200000001)


if __name__ == "__main__":
    unittest.main()
