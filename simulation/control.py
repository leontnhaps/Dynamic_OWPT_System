"""정책 입력과 명령 변환. 실제 연결 시에도 같은 정규화/제한이 필요하다."""
import numpy as np


def encode_observation(error, previous_error, previous_command, cfg):
    """M0의 Tx 관측 6개. Δe는 프레임 차분이며 시간 미분이 아니다."""
    scale = np.array([cfg.width/2, cfg.height/2])
    return np.concatenate((np.asarray(error)/scale,
                           (np.asarray(error)-previous_error)/scale,
                           np.asarray(previous_command)/cfg.angle_limit_deg)).astype(np.float32)


def absolute_command(action, previous_command, cfg):
    """[-1,1]을 절대 목표각으로 변환한 뒤 각도/변화율 제한을 적용한다."""
    action = np.asarray(action, dtype=float)
    if action.shape != (2,) or not np.all(np.isfinite(action)):
        raise ValueError("action must contain two finite values")
    desired = np.clip(action, -1, 1) * cfg.angle_limit_deg
    maximum_delta = cfg.slew_deg_s * cfg.dt
    return np.clip(previous_command + np.clip(desired-previous_command, -maximum_delta, maximum_delta),
                   -cfg.angle_limit_deg, cfg.angle_limit_deg)


def proportional_action(observation, cfg, pixels_per_degree):
    """M0 B1: 영상 오차에서 목표 절대각을 생성. simulator GT를 읽지 않는다."""
    e = observation[:2] * np.array([cfg.width/2, cfg.height/2])
    previous = observation[4:6] * cfg.angle_limit_deg
    signed_gain = np.array([cfg.pan_sign, -cfg.tilt_sign]) * cfg.p_gain / pixels_per_degree
    return np.clip((previous + signed_gain * e) / cfg.angle_limit_deg, -1, 1)


def reward_terms(next_error, command, previous_command, cfg):
    """M0 r_point와 r_cmd. 보상에는 제어 후 오차와 실제 보낸 명령을 사용."""
    e = np.asarray(next_error) / np.array([cfg.width/2, cfg.height/2])
    pointing = -float(e @ e)
    delta = (command-previous_command) / cfg.command_reference_deg
    command_cost = float(delta @ delta)
    return cfg.pointing_weight*pointing - cfg.command_weight*command_cost, pointing, command_cost
