"""Moving-target observation contract and causal, fixed-parameter bearing filter.

Calibration is supplied by an experiment, never inferred from these defaults.
Angles inside the filter are radians in the fixed Tx frame, not servo feedback.
"""
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from common.tx_tracking import TrackingConfig, observation

MOVING_SCHEMA = 'tx-moving-v2'
MOVING_TERMINAL_SCHEMA = 'tx-moving-v3'
MOVING_CONTROL_PERIODS_S = (.720, .800, 1., 1.2, 1.5, 2.)
KALMAN_SCHEMA = 'tx-bearing-kalman-v1'


class PredictionInputError(ValueError):
    """Keep a KF diagnostic, but exclude its invalid policy observation."""


def wrap_azimuth_rad(angle):
    """Signed shortest difference in [-pi, pi); a half turn maps to -pi."""
    return (angle+np.pi) % (2*np.pi)-np.pi


def prediction_displacement_deg(current_bearing_rad, predicted_bearing_rad):
    current = np.asarray(current_bearing_rad, dtype=float)
    predicted = np.asarray(predicted_bearing_rad, dtype=float)
    if (current.shape != (2,) or predicted.shape != (2,)
            or not np.isfinite(current).all() or not np.isfinite(predicted).all()):
        raise PredictionInputError('현재·예측 target bearing에 유한한 방위각·고도각 rad 값이 필요합니다.')
    if abs(current[1]) > np.pi/2 or abs(predicted[1]) > np.pi/2:
        raise PredictionInputError('현재·예측 고도각이 물리 범위 [-90,90]°를 벗어났습니다.')
    delta = predicted-current
    delta[0] = wrap_azimuth_rad(delta[0])
    # Elevation is not circular: two valid elevations can differ by ±180°.
    return np.degrees(delta)


@dataclass(frozen=True)
class MovingConfig(TrackingConfig):
    pan_min: int = -90
    pan_max: int = 90
    pan_delta_deg: float = 90.
    tilt_delta_deg: float = 5.
    visible_bonus: float = 0.
    loss_mode: str = 'exclude'
    loss_pan_weight: float = 10.
    loss_base: float = 0.

    @property
    def action_scale(self):
        return np.array([self.pan_delta_deg, self.tilt_delta_deg])

    @property
    def prediction_scale(self):
        # Angular feature ranges, independent of servo action limits.
        return np.array([180., 180.])

    @property
    def observation_size(self):
        return 10

    @property
    def schema(self):
        return MOVING_TERMINAL_SCHEMA if self.loss_mode == 'confirmed_missing_terminal' else MOVING_SCHEMA

    @property
    def learning_starts(self):
        return 0

    @property
    def normalization(self):
        return dict(base_divisors=super().normalization, command='affine operating limits to [-1,1]',
                    prediction='linear: prediction displacement in degrees / 180',
                    prediction_scale_deg=self.prediction_scale.tolist(),
                    prediction_azimuth='shortest signed difference in [-180,180); +180 maps to -180',
                    prediction_elevation='ordinary difference; current and predicted elevation in [-90,90]',
                    invalid_prediction='exclude replay observation; retain raw KF diagnostics; no clipping',
                    prediction_reference='posterior target bearing; not Tx pointing angle',
                    order=['e_u', 'e_v', 'delta_e_u', 'delta_e_v', 'previous_delta_pan',
                           'previous_delta_tilt', 'pan_command', 'tilt_command',
                           'predicted_delta_alpha_deg', 'predicted_delta_beta_deg'])

    def validate(self, *, evaluation=False):
        super().validate(evaluation=evaluation)
        if not -90 <= self.pan_min < self.pan_max <= 90:
            raise ValueError('이동 학습의 Pan 운용 범위는 -90~90° 안이어야 합니다.')
        if not -15 <= self.tilt_min < self.tilt_max <= 40:
            raise ValueError('이동 학습의 Tilt 운용 범위는 -15~40° 안이어야 합니다.')
        if (self.pan_delta_deg, self.tilt_delta_deg) != (90, 5):
            raise ValueError('이동 SAC의 요청 이동량은 Pan ±90° / Tilt ±5°입니다.')
        if self.loss_mode not in ('exclude', 'confirmed_missing_terminal'):
            raise ValueError('지원하지 않는 이동 표적 유실 보상 모드입니다.')
        for name in ('visible_bonus', 'loss_pan_weight', 'loss_base'):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name}: 유한한 0 이상의 보상 계수가 필요합니다.')

    def _validate_control_period(self, *, evaluation=False):
        if (not isinstance(self.dt, (int, float)) or isinstance(self.dt, bool)
                or not math.isfinite(self.dt) or self.dt not in MOVING_CONTROL_PERIODS_S):
            raise ValueError('이동 SAC 제어주기는 0.720 / 0.800 / 1.000 / 1.200 / 1.500 / 2.000 s입니다.')


def moving_run_config(saved_cfg, mode, period=None):
    """Change only execution timing, retaining the saved moving model contract.

    Historical replay remains at its original timing when learning continues;
    a changed period is an explicit mixed-period dynamics experiment.
    """
    if not isinstance(saved_cfg, MovingConfig):
        raise ValueError('이동 SAC에는 이동 checkpoint 설정이 필요합니다.')
    saved_cfg.validate()
    if mode not in ('resume', 'evaluate'):
        raise ValueError('Checkpoint 실행 모드를 확인하세요.')
    cfg = replace(saved_cfg, dt=period) if period is not None else saved_cfg
    cfg.validate(evaluation=mode == 'evaluate')
    return cfg


def moving_visible_reward(error_px, intensity_mean, cfg):
    """Keep the calibrated tracking terms and add the saved detection bonus."""
    if not math.isfinite(error_px) or error_px < 0 or not math.isfinite(intensity_mean):
        raise ValueError('검출 보상에 유한한 오차와 세기 값이 필요합니다.')
    return -cfg.error_weight*error_px/cfg.reward_distance+cfg.aim_weight*intensity_mean+cfg.visible_bonus


def moving_loss_reward(requested_pan_deg, cfg):
    """Penalize the requested relative Pan motion only on confirmed target loss."""
    if not math.isfinite(requested_pan_deg) or abs(requested_pan_deg) > cfg.pan_delta_deg+1e-6:
        raise ValueError('유실 보상의 Pan 요청 이동량이 유효 범위를 벗어났습니다.')
    return -cfg.loss_base-cfg.loss_pan_weight*abs(requested_pan_deg)/90.


def moving_observation(sample, command, previous_delta, previous_error,
                       current_bearing_rad, predicted_bearing_rad, cfg):
    raw, normalized = observation(sample, command, previous_delta, previous_error, cfg)
    displacement = prediction_displacement_deg(current_bearing_rad, predicted_bearing_rad)
    return np.r_[raw, displacement], np.r_[normalized, displacement/cfg.prediction_scale].astype(np.float32)


def _covariance(value, size, name, *, positive=False):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (size, size) or not np.isfinite(matrix).all():
        raise ValueError(f'{name}: {size}×{size} 유한 공분산 행렬이 필요합니다.')
    if not np.allclose(matrix, matrix.T, rtol=1e-9, atol=1e-14):
        raise ValueError(f'{name}: 대칭 행렬이어야 합니다.')
    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues.min() < -1e-14 or (positive and eigenvalues.min() <= 0):
        raise ValueError(f'{name}: 유효한 공분산 행렬이 아닙니다.')
    return matrix


def validate_kalman_config(value):
    if value.get('schema') != KALMAN_SCHEMA:
        raise ValueError(f'KF 설정 schema는 {KALMAN_SCHEMA}이어야 합니다.')
    if value.get('state_order') != ['alpha', 'beta', 'alpha_dot', 'beta_dot']:
        raise ValueError('KF 상태 순서는 alpha, beta, alpha_dot, beta_dot입니다.')
    if value.get('angle_unit') != 'rad':
        raise ValueError('KF 각도 단위는 rad, 시간 단위는 초입니다.')
    model = value.get('process_noise_model')
    if model not in ('per_observation', 'continuous_white_acceleration'):
        raise ValueError('Q의 시간 모델을 per_observation 또는 continuous_white_acceleration로 명시하세요.')
    _covariance(value.get('Q'), 4 if model == 'per_observation' else 2, 'Q')
    _covariance(value.get('R'), 2, 'R', positive=True)
    _covariance(value.get('P0'), 4, 'P0')
    return json.loads(json.dumps(value, allow_nan=False))


def load_kalman_config(path):
    data = Path(path).read_bytes()
    cfg = validate_kalman_config(json.loads(data))
    return cfg, dict(path=str(Path(path).resolve()), sha256=hashlib.sha256(data).hexdigest())


class BearingKalman:
    """CV filter with frozen Q/R/P0 settings; x and P still update normally.

Q is explicitly either a covariance added per observation (4×4), or a 2×2
continuous white angular-acceleration spectral density integrated over dt.
There are no guessed, optimized or automatically adapted noise parameters.
"""
    def __init__(self, config):
        self.config = validate_kalman_config(config)
        self.Q = np.asarray(config['Q'], dtype=float)
        self.R = np.asarray(config['R'], dtype=float)
        self.P0 = np.asarray(config['P0'], dtype=float)
        self.x = self.P = self.time = None

    def reset(self):
        self.x = self.P = self.time = None

    def step(self, measurement, timestamp, horizon):
        timestamp, horizon = float(timestamp), float(horizon)
        if not math.isfinite(timestamp) or not math.isfinite(horizon) or horizon <= 0:
            raise ValueError('KF 시각과 예측 시간을 확인하세요.')
        z = None if measurement is None else np.asarray(measurement, dtype=float)
        if z is not None and (z.shape != (2,) or not np.isfinite(z).all()):
            raise ValueError('KF 관측은 유효한 방위각·고도각 rad 값 두 개입니다.')
        if self.x is None:
            if z is None:
                return dict(initialized=False, measurement=None, timestamp_s=timestamp, horizon_s=horizon)
            self.x, self.P, self.time = np.r_[z, 0., 0.], self.P0.copy(), timestamp
            prior, prior_p, dt, q = self.x.copy(), self.P.copy(), 0., np.zeros((4, 4))
            innovation = innovation_covariance = gain = nis = None
            initialized_now = True
        else:
            dt = timestamp-self.time
            if dt <= 0:
                raise ValueError('KF 관측 시각은 편도 안에서 증가해야 합니다.')
            f = np.eye(4); f[:2, 2:] = np.eye(2)*dt
            q = (self.Q.copy() if self.config['process_noise_model'] == 'per_observation' else
                 np.block([[self.Q*(dt**3/3), self.Q*(dt**2/2)],
                           [self.Q*(dt**2/2), self.Q*dt]]))
            prior, prior_p = f@self.x, f@self.P@f.T+q
            self.x, self.P, self.time = prior.copy(), prior_p.copy(), timestamp
            innovation = innovation_covariance = gain = nis = None
            if z is not None:
                innovation = z-prior[:2]
                innovation[0] = wrap_azimuth_rad(innovation[0])
                innovation_covariance = prior_p[:2, :2]+self.R
                gain = np.linalg.solve(innovation_covariance, prior_p[:2, :]).T
                self.x = prior+gain@innovation
                residual = np.eye(4); residual[:, :2] -= gain
                self.P = residual@prior_p@residual.T+gain@self.R@gain.T
                self.P = (self.P+self.P.T)/2
                nis = float(innovation@np.linalg.solve(innovation_covariance, innovation))
            initialized_now = False
        displacement = self.x[2:]*horizon
        forecast = self.x[:2]+displacement
        if not np.isfinite(self.x).all() or not np.isfinite(self.P).all():
            raise ValueError('KF 상태 또는 공분산이 유효하지 않습니다.')
        return dict(initialized=True, initialized_now=initialized_now,
                    measurement=None if z is None else z.tolist(), timestamp_s=timestamp,
                    dt_s=dt, horizon_s=horizon, prediction_timestamp_s=timestamp+horizon,
                    prior_state=prior.tolist(), prior_P=prior_p.tolist(), Q_applied=q.tolist(),
                    posterior_state=self.x.tolist(), posterior_P=self.P.tolist(),
                    innovation=None if innovation is None else innovation.tolist(),
                    innovation_covariance=None if innovation_covariance is None else innovation_covariance.tolist(),
                    gain=None if gain is None else gain.tolist(), nis=nis,
                    predicted_bearing_rad=forecast.tolist(), predicted_displacement_rad=displacement.tolist(),
                    predicted_displacement_deg=np.degrees(displacement).tolist())
