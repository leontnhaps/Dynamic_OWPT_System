"""Live inference adapter. No network, GPIO, training, or implicit movement."""
import io
import math
from pathlib import Path


def fresh(received, now, maximum_age=.5):
    return 0 <= now-received <= maximum_age


def validate_config(cfg):
    if cfg.angle_limit_deg is not None:
        raise ValueError('구형 ±6° 모델 config는 실전 모드에서 지원하지 않습니다.')
    if cfg.action_mode != 'delta' or cfg.command_step_deg != 1 or cfg.delta_limit_deg > 5:
        raise ValueError('실전 모드는 Δ각도 모델, 1° 단위, 최대 ±5° config가 필요합니다.')
    if cfg.slew_deg_s is not None or cfg.command_delay_steps or cfg.actuator_tau_s:
        raise ValueError('현재 실전 어댑터는 지연 없는 Δ각도 학습 config를 지원합니다.')
    if cfg.pan_min_deg < -180 or cfg.pan_max_deg > 180 or cfg.tilt_min_deg < -15 or cfg.tilt_max_deg > 40:
        raise ValueError('모델 각도 범위가 실전 범위 Pan ±180°, Tilt -15~40° 밖입니다.')


def target_from_boxes(boxes, class_id, confidence, size):
    """Require exactly one matching PV; never switch arbitrarily between receivers."""
    w,h=size
    candidates=[]
    for x1,y1,x2,y2,score,cls in boxes:
        if int(cls)!=class_id or score < confidence:
            continue
        if not all(math.isfinite(float(x)) for x in (x1,y1,x2,y2,score)):
            continue
        if 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h:
            candidates.append(dict(box=[x1,y1,x2,y2],confidence=score,
                                   center=[(x1+x2)/2,(y1+y2)/2]))
    return candidates[0] if len(candidates)==1 else None, len(candidates)


def command_inside(command, limits):
    return bool(limits) and all(math.isfinite(float(command[i])) and
        limits[a+'_min'] <= command[i] <= limits[a+'_max'] for i,a in enumerate(('pan','tilt')))


class LiveModels:
    def __init__(self, yolo_path, model_path, config_path, device, new_config=None, seed=42):
        import numpy as np
        from ultralytics import YOLO
        from stable_baselines3 import SAC
        from simulation.config import Config
        required=(yolo_path,) if new_config is not None else (yolo_path, model_path, config_path)
        if not all(Path(p).is_file() for p in required):
            raise ValueError('YOLO .pt, SAC .zip, 해당 모델 config.json을 모두 선택하세요.')
        self.cfg=new_config if new_config is not None else Config.load(config_path)
        validate_config(self.cfg)
        self.detector=YOLO(yolo_path)
        self.policy=(new_policy(self.cfg,device,seed) if new_config is not None
                     else SAC.load(model_path,device=device))
        self.from_scratch=new_config is not None
        if self.policy.observation_space.shape!=(6,) or self.policy.action_space.shape!=(2,):
            raise ValueError('6차원 관측 / 2차원 행동 SAC 모델이 아닙니다.')
        self.device=device
        self.names=self.detector.names

    def infer(self, jpeg, command, previous_error, confidence, class_id):
        import numpy as np
        from PIL import Image
        from simulation.control import encode_observation, absolute_command
        image=Image.open(io.BytesIO(jpeg)).convert('RGB')
        c=self.cfg
        if image.size!=(c.width,c.height):
            raise ValueError(f'영상 해상도 {image.size} ≠ 모델 해상도 {(c.width,c.height)}. Camera 탭에서 맞추세요.')
        # PIL input: Ultralytics restores xyxy coordinates to original frame size.
        prediction=self.detector.predict(source=image,conf=confidence,classes=[class_id],
                                         device=self.device,verbose=False)[0]
        boxes=prediction.boxes.data.detach().cpu().numpy().tolist() if prediction.boxes is not None else []
        target,count=target_from_boxes(boxes,class_id,confidence,image.size)
        result=dict(image=image,count=count,target=target)
        if target is None:
            return result
        error=np.asarray(target['center'])-[c.laser_u,c.laser_v]
        obs=encode_observation(error,error if previous_error is None else previous_error,command,c)
        action,_=self.policy.predict(obs,deterministic=not getattr(self,'learning',False))
        if not np.all(np.isfinite(action)):
            raise ValueError('SAC 출력에 비정상 값이 있습니다.')
        applied=absolute_command(action,np.asarray(command,dtype=float),c)
        result.update(error=error.tolist(),observation=obs.tolist(),action=action.tolist(),
                      raw_delta=(action*c.delta_limit_deg).tolist(),command=applied.tolist(),
                      applied_delta=(applied-command).tolist())
        return result


def new_policy(cfg, device, seed):
    """Initialize SAC without loading weights or stepping a simulated environment."""
    import gymnasium as gym
    import numpy as np
    from stable_baselines3 import SAC

    class SpacesOnly(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (6,), np.float32)
        action_space = gym.spaces.Box(-1., 1., (2,), np.float32)

        def reset(self, **kwargs):
            raise RuntimeError('Live SAC uses camera observations, not env.reset()')

        def step(self, action):
            raise RuntimeError('Live SAC uses physical commands, not env.step()')

    return SAC('MlpPolicy', SpacesOnly(), device=device, seed=seed,
               learning_rate=3e-4, buffer_size=50000, batch_size=64,
               learning_starts=64, policy_kwargs=dict(net_arch=[128, 128]), verbose=0)
