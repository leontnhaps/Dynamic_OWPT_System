"""실측값과 미확인 모델 가정을 분리한 실행 설정."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import math


@dataclass(frozen=True)
class Config:
    width: int = 1296
    height: int = 972
    laser_u: float = 696.
    laser_v: float = 384.
    distance_m: float = 5.0
    dt: float = .1                 # 설계 주기 10 Hz; 실측 수신 FPS와 같다는 뜻은 아님
    episode_steps: int = 300
    pan_min_deg: float = -180.0
    pan_max_deg: float = 180.0
    tilt_min_deg: float = -15.0
    tilt_max_deg: float = 40.0
    scenario: str = "straight"  # episode마다 랜덤 방향, 반사 없는 등속 직선
    angle_limit_deg: float | None = None  # 이전 config/model의 ±6° 표현 호환용
    slew_deg_s: float | None = None        # 소프트웨어 명령 제한, 물리 서보 속도 추정값 아님
    action_mode: str = "delta"
    delta_limit_deg: float = 5.0
    command_step_deg: float = 1.0
    end_on_limit_exit: bool = True
    pan_sign: int = 1              # 양의 명령이 카메라를 오른쪽으로: 실제 부호 미확인
    tilt_sign: int = 1             # 양의 명령이 카메라를 위로: 실제 부호 미확인
    focal_random_fraction: float = .10
    measurement_noise_px: float = 1.0
    actuator_tau_s: float = 0.0    # 0=이상적 응답; 0보다 크면 가정한 1차 지연
    command_delay_steps: int = 0
    target_speed_m_s: float = .12
    target_radius_m: float = .30
    pointing_weight: float = 10.0
    command_weight: float = .02
    command_reference_deg: float = 1.0
    p_gain: float = .45            # Δθ = p_gain * e / nominal_px_per_deg
    engineering_threshold_px: float = 20.0  # 비교 표시용; PV 충전 성공 판정값 아님

    def __post_init__(self):
        for name in ("width", "height", "distance_m", "dt", "episode_steps",
                     "command_reference_deg",
                     "target_speed_m_s", "target_radius_m", "engineering_threshold_px"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.action_mode not in ("absolute", "delta"):
            raise ValueError("action_mode must be absolute or delta")
        if not math.isfinite(self.delta_limit_deg) or self.delta_limit_deg <= 0:
            raise ValueError("delta_limit_deg must be finite and positive")
        if self.action_mode == "delta" and self.command_step_deg:
            if not math.isclose(self.delta_limit_deg/self.command_step_deg, round(self.delta_limit_deg/self.command_step_deg)):
                raise ValueError("delta limit must lie on the command grid")
        if self.slew_deg_s is not None and (not math.isfinite(self.slew_deg_s) or self.slew_deg_s <= 0):
            raise ValueError("slew_deg_s must be None or positive")
        if not math.isfinite(self.command_step_deg) or self.command_step_deg < 0:
            raise ValueError("Invalid command_step_deg")
        if self.command_step_deg and self.slew_deg_s is not None:
            raise ValueError("Quantized commands cannot use the legacy slew limiter")
        if self.command_step_deg:
            for limit in (*self.angle_low, *self.angle_high):
                if not math.isclose(limit/self.command_step_deg, round(limit/self.command_step_deg)):
                    raise ValueError("Angle bounds must lie on the command grid")
        for name in ("width", "height", "episode_steps", "command_delay_steps"):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f"{name} must be an integer")
        if self.command_delay_steps < 0 or self.pan_sign not in (-1, 1) or self.tilt_sign not in (-1, 1):
            raise ValueError("Invalid command delay or axis sign")
        if not 0 <= self.focal_random_fraction < .5:
            raise ValueError("focal_random_fraction must be in [0, .5)")
        for name in ("measurement_noise_px", "actuator_tau_s", "pointing_weight", "command_weight", "p_gain"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"Invalid {name}")
        if not (0 <= self.laser_u < self.width and 0 <= self.laser_v < self.height):
            raise ValueError("Laser reference must be inside the image")
        if self.angle_limit_deg is not None:
            if not math.isfinite(self.angle_limit_deg) or not 0 < self.angle_limit_deg <= 15:
                raise ValueError("Invalid legacy angle_limit_deg")
        for low, high in zip(self.angle_low, self.angle_high):
            if not (math.isfinite(low) and math.isfinite(high) and low < 0 < high):
                raise ValueError("Angle limits must be finite and contain zero")
        if self.scenario not in ("straight", "sine", "linear", "stationary", "mixed"):
            raise ValueError("Invalid scenario")

    @property
    def angle_low(self):
        return (-self.angle_limit_deg,)*2 if self.angle_limit_deg is not None else (self.pan_min_deg, self.tilt_min_deg)

    @property
    def angle_high(self):
        return (self.angle_limit_deg,)*2 if self.angle_limit_deg is not None else (self.pan_max_deg, self.tilt_max_deg)

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path=None):
        if not path:
            return cls()
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if "angle_limit_deg" in values and "scenario" not in values:
            values["scenario"] = "mixed"  # 구 모델을 새 action 의미로 조용히 바꾸지 않는다.
        # 기존 학습 config는 원래의 연속 명령/고정 길이 동작을 보존한다.
        values.setdefault("command_step_deg", 0.0)
        values.setdefault("end_on_limit_exit", False)
        values.setdefault("action_mode", "absolute")
        return cls(**values)

