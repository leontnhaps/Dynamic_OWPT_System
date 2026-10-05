"""M2/M3 camera observations, commands and episode metrics (no hardware I/O)."""
from dataclasses import asdict, dataclass
import math
import numpy as np
from common.control_timing import PRIMARY_CONTROL_PERIOD_S, CONTROL_PERIODS_S
from common.servo import limits_from
from common.tx_setup import TX_OPERATING_LIMITS

SCHEMA = 'tx-stationary-v1'
RUN_SCHEMA = 'tx-stationary-run-v2'


@dataclass(frozen=True)
class TrackingConfig:
    dt: float = PRIMARY_CONTROL_PERIOD_S
    pan_min: int = TX_OPERATING_LIMITS['pan_min']
    pan_max: int = TX_OPERATING_LIMITS['pan_max']
    tilt_min: int = TX_OPERATING_LIMITS['tilt_min']
    tilt_max: int = TX_OPERATING_LIMITS['tilt_max']
    width: int = 1296
    height: int = 972
    laser_u: float = 696.
    laser_v: float = 384.
    beam_wx: float = 59.07
    beam_wy: float = 62.255
    delta_deg: float = 5.
    reward_distance: float = 1000.
    error_weight: float = 1.
    aim_weight: float = 1.
    hit_px: float = 23.
    confidence: float = .5
    class_id: int = 0
    speed: int = 100
    acc: float = 1.

    @property
    def low(self):
        return np.array([self.pan_min, self.tilt_min], dtype=float)

    @property
    def high(self):
        return np.array([self.pan_max, self.tilt_max], dtype=float)

    @property
    def limits(self):
        return {k: getattr(self, k) for k in ('pan_min', 'pan_max', 'tilt_min', 'tilt_max')}

    def validate(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('설정은 유한한 수치여야 합니다.')
        limits_from(self.limits)
        if not all(float(v).is_integer() for v in self.limits.values()):
            raise ValueError('운용 한계각은 1° 단위로 적용하세요.')
        if self.dt not in CONTROL_PERIODS_S:
            raise ValueError('제어주기는 0.720 또는 0.800 s입니다.')
        fixed = TrackingConfig()
        # M3-1 geometry, action and detector settings are one persisted contract.
        for key in ('width', 'height', 'laser_u', 'laser_v', 'beam_wx', 'beam_wy',
                    'delta_deg', 'hit_px', 'confidence', 'class_id', 'speed', 'acc'):
            if getattr(self, key) != getattr(fixed, key):
                raise ValueError(f'M3-1 고정 설정 불일치: {key}')
        if min(self.reward_distance, self.error_weight, self.aim_weight) <= 0:
            raise ValueError('보상 정규화와 계수는 양수여야 합니다.')


@dataclass(frozen=True)
class RunSettings:
    max_steps: int = 100  # Editable initial UI value, not a convergence criterion.
    episodes: int = 10
    settle_s: float = 2.
    acquire_s: float = 3.
    missing_s: float = 3.
    pan_low: int = -27
    pan_high: int = -7
    tilt_low: int = -10
    tilt_high: int = 5
    updates_per_transition: float = 1.

    def validate(self, cfg):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('실행 설정은 유한한 수치여야 합니다.')
        for key in ('max_steps', 'episodes', 'pan_low', 'pan_high', 'tilt_low', 'tilt_high'):
            if not isinstance(getattr(self, key), int):
                raise ValueError(f'{key}: 정수 필요')
        if self.max_steps < 1 or self.episodes < 1 or self.settle_s < 0 or self.updates_per_transition < 0:
            raise ValueError('최대 step·반복은 양수, 대기·업데이트 비율은 0 이상이어야 합니다.')
        if self.acquire_s != 3 or self.missing_s != 3:
            raise ValueError('초기 검출·연속 미검출 제한은 3초입니다.')
        for i, axis in enumerate(('pan', 'tilt')):
            lo, hi = getattr(self, axis+'_low'), getattr(self, axis+'_high')
            if not cfg.low[i] <= lo <= hi <= cfg.high[i]:
                raise ValueError('초기각 범위를 적용된 운용 한계 안에 설정하세요.')


@dataclass(frozen=True)
class Sample:
    frame_id: int
    received: float
    processed: float
    width: int
    height: int
    target: object = None
    count: int = 0
    seq: object = None
    receive_unix_ns: object = None
    inference_ms: object = None
    simulated: bool = False


def invalid_reason(sample, now, cfg, after=-math.inf, previous=-1):
    if sample is None:
        return 'no_processed_frame'
    if sample.simulated or (sample.width, sample.height) != (cfg.width, cfg.height):
        return 'camera_configuration'
    if not all(math.isfinite(x) for x in (sample.received, sample.processed, now)):
        return 'invalid_timestamp'
    if sample.frame_id <= previous:
        return 'no_new_frame'
    if sample.received <= after:
        return 'before_command_or_reset_wait'
    if not 0 <= now-sample.received <= cfg.dt:
        return 'stale'
    t = sample.target
    if t is None:
        return 'missing'
    try:
        x1, y1, x2, y2 = t['box']
        u, v = t['center']
        if (not all(math.isfinite(x) for x in (x1, y1, x2, y2, u, v, t['confidence']))
                or not 0 <= x1 < x2 <= cfg.width or not 0 <= y1 < y2 <= cfg.height
                or not cfg.confidence <= t['confidence'] <= 1
                or not np.allclose([u, v], [(x1+x2)/2, (y1+y2)/2])):
            return 'invalid_box'
    except (KeyError, TypeError, ValueError):
        return 'invalid_box'
    return None


class BeamMap:
    def __init__(self, cfg):
        self.cfg = cfg
        u, v = np.arange(cfg.width), np.arange(cfg.height)
        beam = np.exp(-2*((u[None, :]-cfg.laser_u)/cfg.beam_wx)**2
                      -2*((v[:, None]-cfg.laser_v)/cfg.beam_wy)**2)
        self.integral = np.pad(beam.cumsum(0).cumsum(1), ((1, 0), (1, 0)))

    def mean(self, box):
        # Integer pixel centers inside [x1,x2) × [y1,y2), clipped to image.
        x1, y1, x2, y2 = (math.ceil(v) for v in box)
        x1, x2 = np.clip([x1, x2], 0, self.cfg.width)
        y1, y2 = np.clip([y1, y2], 0, self.cfg.height)
        n = int((x2-x1)*(y2-y1))
        if n <= 0:
            raise ValueError('empty_pixel_region')
        s = self.integral
        return float(np.clip((s[y2, x2]-s[y1, x2]-s[y2, x1]+s[y1, x1])/n, 0, 1)), n


def observation(sample, command, previous_delta, previous_error, cfg):
    error = np.asarray(sample.target['center'])-[cfg.laser_u, cfg.laser_v]
    change = np.zeros(2) if previous_error is None else error-previous_error
    raw = np.r_[error, change, previous_delta, command].astype(float)
    normalized = raw.copy()
    normalized[:6] /= [696, 588, 1296, 972, 5, 5]
    normalized[6:] = 2*(raw[6:]-cfg.low)/(cfg.high-cfg.low)-1
    return raw, normalized.astype(np.float32)


def action_command(action, command, cfg):
    action = np.asarray(action, dtype=float)
    if action.shape != (2,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1.000001):
        raise ValueError('SAC action must be finite normalized 2D values in [-1,1]')
    degrees = np.clip(action, -1, 1)*cfg.delta_deg
    quantized = np.sign(degrees)*np.floor(np.abs(degrees)+.5)
    target = np.clip(np.asarray(command)+quantized, cfg.low, cfg.high)
    return degrees, target, target-command


class EpisodeMetrics:
    def __init__(self, cfg, started, initial_error):
        self.cfg, self.started = cfg, started
        self.errors, self.intensities, self.deltas = [], [], []
        self.first_hit_s = 0. if initial_error <= cfg.hit_px else None
        self.unavailable_s = 0.
        self.unavailable_count = 0
        self.missing_since = None

    def command(self, delta):
        self.deltas.append(np.asarray(delta).copy())

    def result(self, error, intensity, now):
        self.errors.append(error)
        self.intensities.append(intensity)
        if self.first_hit_s is None and error <= self.cfg.hit_px:
            self.first_hit_s = now-self.started

    def missing(self, now):
        if self.missing_since is None:
            self.missing_since = now
            self.unavailable_count += 1

    def recovered(self, now):
        if self.missing_since is not None:
            self.unavailable_s += now-self.missing_since
            self.missing_since = None

    def summary(self, now):
        self.recovered(now)
        e = np.array(self.errors)
        n, k = len(e), len(self.deltas)
        delta = np.array(self.deltas).reshape(-1, 2)
        total = np.abs(delta).sum(0)
        variation = ((delta/(self.cfg.high-self.cfg.low))**2).mean(0) if k else [None, None]
        result = dict(executed_actions=k, valid_results=n, valid_ratio=n/k if k else None,
                      rms_px=float(np.sqrt(np.mean(e**2))) if n else None,
                      p95_px=float(np.percentile(e, 95)) if n else None,
                      hit_ratio=float(np.mean(e <= self.cfg.hit_px)) if n else None,
                      intensity_mean=float(np.mean(self.intensities)) if n else None,
                      first_hit_s=self.first_hit_s, reached=self.first_hit_s is not None,
                      control_elapsed_s=now-self.started, unavailable_s=self.unavailable_s,
                      unavailable_count=self.unavailable_count,
                      command_variation=sum(variation) if k else None)
        for i, axis in enumerate(('pan', 'tilt')):
            result.update({axis+'_total_deg': float(total[i]), axis+'_mean_deg': float(total[i]/k) if k else None,
                           axis+'_variation': float(variation[i]) if k else None})
        return result
