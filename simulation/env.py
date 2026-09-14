"""5 m 평면에서 움직이는 PV와 동축에 가까운 고정 camera/laser Tx 모델."""
from collections import deque
import gymnasium as gym
import numpy as np
from .calibration import fit_scale
from .config import Config
from .control import absolute_command, encode_observation, reward_terms


def project(point_world, angles_deg, focal, cfg):
    """R=Ry(pan) Rx(-tilt), Pc=R.T Pw, u=cx+fx Xc/Zc, v=cy-fy Yc/Zc."""
    pan, tilt = np.deg2rad(np.asarray(angles_deg) * [cfg.pan_sign, cfg.tilt_sign])
    cp, sp, ct, st = np.cos(pan), np.sin(pan), np.cos(tilt), np.sin(tilt)
    rotation = np.array([[cp, -sp*st, sp*ct], [0., ct, st], [-sp, -cp*st, cp*ct]])
    x, y, z = rotation.T @ point_world
    if z <= 1e-9:
        return np.array([np.nan, np.nan])  # 카메라 뒤/투영면은 검출 불가
    return np.array([cfg.width/2 + focal[0]*x/z, cfg.height/2 - focal[1]*y/z])


class TrackingEnv(gym.Env):
    """SAC Box(2) → software-limited Tx command → projection → M0 reward."""
    metadata = {"render_modes": []}

    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        self.nominal_focal = fit_scale()[0] * 5.0 * [self.cfg.width/1296, self.cfg.height/972]
        self.action_space = gym.spaces.Box(-1., 1., (2,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (6,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        c = self.cfg
        # 궤적과 측정 노이즈 난수 분리: 제어기가 달라도 같은 seed는 같은 운동.
        seeds = self.np_random.integers(0, 2**32, size=3)
        trajectory_rng, model_rng, self.noise_rng = [np.random.default_rng(int(s)) for s in seeds]
        self.focal = self.nominal_focal * model_rng.uniform(1-c.focal_random_fraction,
                                                         1+c.focal_random_fraction, 2)
        self.center = np.array([(c.laser_u-c.width/2)*c.distance_m/self.nominal_focal[0],
                                -(c.laser_v-c.height/2)*c.distance_m/self.nominal_focal[1]])
        self.phase = trajectory_rng.uniform(-np.pi, np.pi, 2)
        self.amplitude = trajectory_rng.uniform(.3, 1., 2) * c.target_radius_m
        self.omega = trajectory_rng.uniform(.4, 1., 2) * c.target_speed_m_s / self.amplitude
        self.scenario = (options or {}).get("scenario", c.scenario)
        if self.scenario == "mixed":
            self.scenario = str(trajectory_rng.choice(["sine", "linear"]))
        if self.scenario not in ("straight", "sine", "linear", "stationary"):
            raise ValueError("Invalid scenario")
        self.velocity = trajectory_rng.uniform(-1, 1, 2)*c.target_speed_m_s
        self.start = trajectory_rng.uniform(-1, 1, 2)*c.target_radius_m
        if self.scenario == "straight":
            direction = trajectory_rng.uniform(-np.pi, np.pi)
            self.velocity = c.target_speed_m_s * np.array([np.cos(direction), np.sin(direction)])
        self.command = np.zeros(2)
        self.actual_angles = np.zeros(2)  # simulator internal; never in observation
        self.queue = deque([np.zeros(2) for _ in range(c.command_delay_steps)])
        self.t = 0
        self.done = False
        self.last_measured_error = np.zeros(2)
        uv, visible, error = self._measure()
        obs = encode_observation(error, error, self.command, c)
        return obs, self._info(uv, visible, error)

    def target_position(self, time_s):
        """5 m 고정 깊이, x-right/y-up 평면 궤적. 관측에는 위치/속도를 넣지 않는다."""
        if self.scenario == "sine":
            offset = self.amplitude*np.sin(self.omega*time_s+self.phase)
        elif self.scenario == "straight":
            offset = self.start + self.velocity*time_s
        elif self.scenario == "linear":
            radius = self.cfg.target_radius_m
            raw = (self.start + self.velocity*time_s + radius) % (4*radius)
            offset = radius - np.abs(raw-2*radius)  # 경계에서 반사하는 직선 왕복
        else:
            offset = self.start
        return np.r_[self.center+offset, self.cfg.distance_m]

    def _measure(self):
        c = self.cfg
        self.position = self.target_position(self.t*c.dt)
        uv = project(self.position, self.actual_angles, self.focal, c)
        visible = bool(0 <= uv[0] < c.width and 0 <= uv[1] < c.height)
        if visible:
            self.last_measured_error = uv + self.noise_rng.normal(0, c.measurement_noise_px, 2) - [c.laser_u, c.laser_v]
        # 미검출 시 마지막 관측 유지. GT 좌표를 정책에 주지 않는다.
        return uv, visible, self.last_measured_error.copy()

    def _info(self, uv, visible, error):
        return dict(time_s=self.t*self.cfg.dt, uv=uv.copy(), visible=visible,
                    true_error_px=uv-[self.cfg.laser_u, self.cfg.laser_v],
                    measured_error_px=error.copy(), command_deg=self.command.copy(),
                    actual_angles_deg=self.actual_angles.copy(), target_world_m=self.position.copy(),
                    scenario=self.scenario)

    def _limit_exit(self, uv, visible):
        """실제 축 한계 + 해당 화면 경계 이탈 + PV 자체의 바깥쪽 이동.

        이전 PV도 현재 카메라 자세로 투영해 카메라 회전으로 생긴 움직임을 제외.
        GT는 종료 판정에만 사용하며 policy observation에는 추가하지 않는다.
        후방을 향한 잘못된 카메라 명령은 이 조건으로 조기 종료시키지 않는다.
        """
        c = self.cfg
        if not c.end_on_limit_exit or visible or not np.all(np.isfinite(uv)):
            return None
        previous_uv = project(self.target_position((self.t-1)*c.dt), self.actual_angles, self.focal, c)
        if not np.all(np.isfinite(previous_uv)):
            return None
        motion = uv-previous_uv
        # 낮은/높은 명령각이 향하는 화면 방향: u는 오른쪽, v는 아래쪽.
        directions = (c.pan_sign, -c.tilt_sign)
        for axis, (name, extent, direction) in enumerate(zip(("pan", "tilt"), (c.width, c.height), directions)):
            for side, bound, command_direction in (("min", c.angle_low[axis], -1), ("max", c.angle_high[axis], 1)):
                outward = direction*command_direction
                at_limit = abs(self.actual_angles[axis]-bound) < 1e-6
                outside = uv[axis] < 0 if outward < 0 else uv[axis] >= extent
                if at_limit and outside and motion[axis]*outward > 1e-8:
                    return name+"_"+side+"_outward_exit"
        return None

    def step(self, action):
        if self.done:
            raise RuntimeError("Call reset() after episode end")
        c = self.cfg
        previous_command = self.command.copy()
        previous_error = self.last_measured_error.copy()
        self.command = absolute_command(action, previous_command, c)
        self.queue.append(self.command.copy())
        delayed_command = self.queue.popleft()
        alpha = 1. if c.actuator_tau_s == 0 else -np.expm1(-c.dt/c.actuator_tau_s)
        self.actual_angles += alpha*(delayed_command-self.actual_angles)
        self.t += 1
        uv, visible, error = self._measure()
        # 가시 영역 안에서는 카메라 측정 오차를 사용. 밖에서는 GT로 연속 비용을 유지한다.
        # 일반 미검출은 계속 진행하고, 한계 방향 이탈만 종료한다.
        reward_error = error if visible else np.clip(
            np.nan_to_num(uv-[c.laser_u, c.laser_v], nan=10*c.width,
                          posinf=10*c.width, neginf=-10*c.width),
            -10*c.width, 10*c.width)
        # 투영 특이점/후방에서도 유한한 이탈 비용. 정책에는 GT를 넣지 않는다.
        reward, pointing, command_cost = reward_terms(reward_error, self.command, previous_command, c)
        info = self._info(uv, visible, error)
        info.update(pointing_reward=pointing, command_cost=command_cost,
                    reward_uses_out_of_view_truth=not visible)
        reason = self._limit_exit(uv, visible)
        terminated = reason is not None
        truncated = self.t >= c.episode_steps and not terminated
        self.done = terminated or truncated
        info.update(termination_reason=reason or ("time_limit" if truncated else ""))
        return encode_observation(error, previous_error, self.command, c), reward, terminated, truncated, info

