"""정책 입력과 명령 변환. 실제 연결 시에도 같은 정규화/제한이 필요하다."""
import numpy as np


def angle_scale(cfg):
    low, high = np.array(cfg.angle_low), np.array(cfg.angle_high)
    return (high-low)/2, (high+low)/2


def action_for_angles(angles, cfg):
    scale, midpoint = angle_scale(cfg)
    return np.clip((np.asarray(angles)-midpoint)/scale, -1, 1)


def encode_observation(error, previous_error, previous_command, cfg):
    """M0의 Tx 관측 6개. Δe는 프레임 차분이며 시간 미분이 아니다."""
    scale = np.array([cfg.width/2, cfg.height/2])
    return np.concatenate((np.asarray(error)/scale,
                           (np.asarray(error)-previous_error)/scale,
                           action_for_angles(previous_command, cfg))).astype(np.float32)


def absolute_command(action, previous_command, cfg):
    """정책 출력을 해석해 실제 전달할 절대 명령각을 반환한다."""
    action = np.asarray(action, dtype=float)
    if action.shape != (2,) or not np.all(np.isfinite(action)):
        raise ValueError("action must contain two finite values")
    scale, midpoint = angle_scale(cfg)
    if cfg.action_mode == "delta":
        desired = np.clip(action, -1, 1) * cfg.delta_limit_deg
    else:
        desired = np.clip(action, -1, 1) * scale + midpoint
    if cfg.command_step_deg:
        # 가장 가까운 각도 격자, 정확한 절반은 0에서 먼 방향으로 반올림.
        units = desired / cfg.command_step_deg
        desired = np.sign(units)*np.floor(np.abs(units)+.5+1e-10)*cfg.command_step_deg
    if cfg.action_mode == "delta":
        desired = np.asarray(previous_command) + desired
    if cfg.slew_deg_s is not None:  # 구 모델 평가/재개 호환만을 위한 경로
        maximum_delta = cfg.slew_deg_s * cfg.dt
        desired = previous_command + np.clip(desired-previous_command, -maximum_delta, maximum_delta)
    return np.clip(desired, cfg.angle_low, cfg.angle_high)


def proportional_action(observation, cfg, pixels_per_degree):
    """M0 B1: 영상 오차에서 목표 절대각을 생성. simulator GT를 읽지 않는다."""
    e = observation[:2] * np.array([cfg.width/2, cfg.height/2])
    scale, midpoint = angle_scale(cfg)
    previous = observation[4:6] * scale + midpoint
    signed_gain = np.array([cfg.pan_sign, -cfg.tilt_sign]) * cfg.p_gain / pixels_per_degree
    if cfg.action_mode == "delta":
        return np.clip(signed_gain * e / cfg.delta_limit_deg, -1, 1)
    return action_for_angles(previous + signed_gain * e, cfg)


def reward_terms(next_error, command, previous_command, cfg):
    """M0 r_point와 r_cmd. 보상에는 제어 후 오차와 실제 보낸 명령을 사용."""
    e = np.asarray(next_error) / np.array([cfg.width/2, cfg.height/2])
    pointing = -float(e @ e)
    delta = (command-previous_command) / cfg.command_reference_deg
    command_cost = float(delta @ delta)
    return cfg.pointing_weight*pointing - cfg.command_weight*command_cost, pointing, command_cost

