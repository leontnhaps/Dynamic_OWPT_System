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
    angle_limit_deg: float = 6.0   # 0° 중심의 국소 시뮬레이션 범위; 하드웨어 한계 아님
    slew_deg_s: float = 2.0        # 소프트웨어 명령 제한, 물리 서보 속도 추정값 아님
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
    command_reference_deg: float = .20
    p_gain: float = .45            # Δθ = p_gain * e / nominal_px_per_deg
    engineering_threshold_px: float = 20.0  # 비교 표시용; PV 충전 성공 판정값 아님

    def __post_init__(self):
        for name in ("width", "height", "distance_m", "dt", "episode_steps",
                     "angle_limit_deg", "slew_deg_s", "command_reference_deg",
                     "target_speed_m_s", "target_radius_m", "engineering_threshold_px"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
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
        if self.angle_limit_deg > 15:
            raise ValueError("This local geometry model supports at most +/-15 degrees")

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path=None):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8"))) if path else cls()
