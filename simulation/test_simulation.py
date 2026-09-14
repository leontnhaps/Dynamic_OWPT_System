"""기하, M0 식, 제한/지연, 동일 평가 조건의 회귀 검증."""
from dataclasses import replace
import unittest
import numpy as np
from stable_baselines3.common.env_checker import check_env
from .calibration import fit_scale, SAMPLES
from .config import Config
from .control import absolute_command, encode_observation, reward_terms, action_for_angles, angle_scale
from .env import TrackingEnv, project
from .evaluate import rollout


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = replace(Config(), angle_limit_deg=6., scenario="mixed", focal_random_fraction=0., measurement_noise_px=0., episode_steps=50)

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


    def test_straight_constant_velocity_random_direction_and_zero_reset(self):
        cfg = Config()
        velocities = []
        for seed in range(20):
            env = TrackingEnv(cfg)
            env.reset(seed=seed)
            self.assertEqual(env.scenario, "straight")
            np.testing.assert_array_equal(env.command, [0, 0])
            np.testing.assert_array_equal(env.actual_angles, [0, 0])
            points = np.array([env.target_position(t) for t in (0., 10., 20., 30.)])
            np.testing.assert_allclose(np.diff(points, axis=0), np.tile(np.r_[env.velocity*10, 0], (3, 1)))
            self.assertAlmostEqual(np.linalg.norm(env.velocity), cfg.target_speed_m_s)
            velocities.append(env.velocity)
        self.assertTrue(np.any(np.array(velocities) > 0))
        self.assertTrue(np.any(np.array(velocities) < 0))

    def test_wide_asymmetric_angles_and_fixed_baseline(self):
        cfg = replace(Config(), slew_deg_s=10000)
        np.testing.assert_allclose(absolute_command([-1, -1], np.zeros(2), cfg), [-180, -15])
        np.testing.assert_allclose(absolute_command([1, 1], np.zeros(2), cfg), [180, 40])
        np.testing.assert_allclose(absolute_command(action_for_angles([0, 0], cfg), np.zeros(2), cfg), [0, 0])
        rows, _ = rollout(Config(), 10000, "B0")
        np.testing.assert_allclose([[r["pan_cmd_deg"], r["tilt_cmd_deg"]] for r in rows], 0)
        self.assertFalse(rows[-1]["visible"])

    def test_sustained_tracking_beyond_old_limit_and_finite_lost_reward(self):
        cfg = replace(Config(), measurement_noise_px=0., focal_random_fraction=0.)
        rows, metrics = rollout(cfg, 10000, "B1")
        self.assertGreater(max(abs(r["pan_cmd_deg"]) for r in rows), 6)
        self.assertLessEqual(metrics["max_command_step_deg"], .20000001)
        self.assertLessEqual(max(r["tilt_cmd_deg"] for r in rows), 40)
        self.assertGreaterEqual(min(r["tilt_cmd_deg"] for r in rows), -15)
        env = TrackingEnv(replace(cfg, slew_deg_s=10000))
        env.reset(seed=2)
        obs, reward, _, _, info = env.step([1, 0])
        self.assertFalse(info["visible"])
        self.assertTrue(np.all(np.isfinite(obs)))
        self.assertTrue(np.isfinite(reward))


if __name__ == "__main__":
    unittest.main()
