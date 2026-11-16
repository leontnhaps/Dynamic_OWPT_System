"""SAC adapter and paired checkpoints for M3-2 real camera transitions."""
import csv
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import threading
import time
import uuid
import zipfile
import numpy as np
from common.tx_tracking import SCHEMA, TrackingConfig, checkpoint_run_config

SAC_SETTINGS = dict(learning_rate=3e-4, buffer_size=1_000_000, batch_size=256,
                    gamma=.99, tau=.005, learning_starts=1000, ent_coef='auto',
                    target_entropy=-2., target_update_interval=1)


def sac_settings(cfg):
    return dict(SAC_SETTINGS, learning_starts=cfg.learning_starts)


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def dump(path, value):
    Path(path).write_text(json.dumps(value, default=json_value, ensure_ascii=False,
                                    indent=2, allow_nan=False), encoding='utf-8')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


class RunLog:
    fields = {
        'steps': ('episode step behavior observation_frame_id observation normalized action action_degrees '
                  'applied_delta command command_time command_interval_s interval_excess_s send_duration_s '
                  'policy_duration_s frame_id seq receive_unix_ns receive_s processed_s result_s result_age_s '
                  'u v x1 y1 x2 y2 confidence next_observation next_normalized error_px intensity_mean pixel_count '
                  'reward_error reward_aim reward valid stored exclusion_reason terminated truncated').split(),
        'detections': ('episode phase frame_id seq receive_unix_ns receive_s processed_s width height simulated '
                       'status count confidence u v x1 y1 x2 y2 inference_ms').split(),
        'episodes': ('episode mode reason terminated truncated initial_command initial_observation reset_elapsed_s '
                     'executed_actions valid_results valid_ratio new_transitions excluded_transitions control_elapsed_s '
                     'rms_px p95_px first_hit_s reached hit_ratio intensity_mean unavailable_s unavailable_count '
                     'pan_total_deg tilt_total_deg pan_mean_deg tilt_mean_deg command_variation pan_variation tilt_variation '
                     'planned_updates completed_updates total_updates total_transitions total_actions update_s checkpoint').split(),
        'updates': ('episode update actor_loss critic_loss alpha alpha_loss').split(),
    }

    def __init__(self, folder, config):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        self.lock = threading.Lock()
        self.files, self.writers = {}, {}
        self.episodes = []
        dump(self.folder/'config.json', config)
        self.events = (self.folder/'events.jsonl').open('w', encoding='utf-8')
        for name, fields in self.fields.items():
            f = (self.folder/(name+'.csv')).open('w', encoding='utf-8', newline='')
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='raise')
            writer.writeheader()
            f.flush()
            self.files[name], self.writers[name] = f, writer

    def event(self, event, **values):
        with self.lock:
            self.events.write(json.dumps(dict(event=event, **values), default=json_value,
                                         ensure_ascii=False, allow_nan=False)+'\n')
            self.events.flush()

    def row(self, name, row):
        row = {k: (json.dumps(v, default=json_value, allow_nan=False) if isinstance(v, (dict, list, np.ndarray)) else v)
               for k, v in row.items()}
        with self.lock:
            self.writers[name].writerow(row)
            self.files[name].flush()

    def episode(self, row):
        self.row('episodes', row)
        self.episodes.append(row)
        metrics = ('rms_px', 'p95_px', 'hit_ratio', 'first_hit_s', 'intensity_mean', 'valid_ratio',
                   'unavailable_s', 'unavailable_count', 'pan_total_deg', 'tilt_total_deg',
                   'pan_mean_deg', 'tilt_mean_deg', 'command_variation')
        fields = ('mode', 'reason', 'episodes', 'reached', 'not_reached', 'metric', 'n', 'mean', 'sample_std')
        temp = self.folder/'summary.tmp'
        with temp.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for mode, reason in sorted({(r['mode'], r['reason']) for r in self.episodes}):
                rows = [r for r in self.episodes if (r['mode'], r['reason']) == (mode, reason)]
                for metric in metrics:
                    values = [float(r[metric]) for r in rows if r.get(metric) is not None]
                    writer.writerow(dict(mode=mode, reason=reason, episodes=len(rows),
                                         reached=sum(r.get('reached') is True for r in rows),
                                         not_reached=sum(r.get('reached') is False for r in rows),
                                         metric=metric, n=len(values), mean=statistics.mean(values) if values else None,
                                         sample_std=statistics.stdev(values) if len(values)>1 else None))
        os.replace(temp, self.folder/'summary.csv')

    def close(self):
        with self.lock:
            for f in self.files.values():
                f.close()
            self.events.close()


def make_policy(cfg, device, seed):
    import gymnasium as gym
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.logger import configure

    class CameraSpaces(gym.Env):
        observation_space = gym.spaces.Box(-1., 1., (cfg.observation_size,), np.float32)
        action_space = gym.spaces.Box(-1., 1., (2,), np.float32)

        def reset(self, **kwargs):
            raise RuntimeError('Reset comes from the actual Tx and camera')

        def step(self, action):
            raise RuntimeError('Transitions come from real post-command observations')

    policy = SAC('MlpPolicy', CameraSpaces(), device=device, seed=seed, **sac_settings(cfg),
                 policy_kwargs=dict(net_arch=[256, 256], activation_fn=torch.nn.ReLU, n_critics=2),
                 train_freq=1, gradient_steps=1, verbose=0)
    policy.set_logger(configure(folder=None, format_strings=[]))
    return policy


def checkpoint_info(path):
    path = Path(path)
    if path.is_file():
        path = path.parent
    manifest = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    from common.tx_moving import MOVING_SCHEMA, MOVING_TERMINAL_SCHEMA, MovingConfig
    if manifest.get('schema') == 'tx-moving-v1':
        raise ValueError('이전 tanh 정규화 이동 모델입니다. 원본 수집 데이터를 재처리하여 초기 모델을 다시 학습하세요.')
    if manifest.get('schema') not in (SCHEMA, MOVING_SCHEMA, MOVING_TERMINAL_SCHEMA):
        raise ValueError('지원되는 정지 8차원 / 이동 10차원 checkpoint가 아닙니다.')
    required = {'model.zip', 'replay.npz', 'state.json', 'rng.npz'}
    if set(manifest.get('files', {})) != required:
        raise ValueError('완전한 checkpoint 구성 파일이 필요합니다.')
    for name, expected in manifest['files'].items():
        if digest(path/name) != expected:
            raise ValueError('Checkpoint 무결성 불일치: '+name)
    state = json.loads((path/'state.json').read_text(encoding='utf-8'))
    config_class = MovingConfig if manifest['schema'] in (MOVING_SCHEMA, MOVING_TERMINAL_SCHEMA) else TrackingConfig
    cfg = config_class(**state['tracking'])
    cfg.validate()
    # Older v2 fixtures omitted the redundant state schema. An explicit
    # disagreement is never accepted; the terminal-capable contract requires it.
    if (state.get('schema', manifest['schema']) != manifest['schema']
            or cfg.schema != manifest['schema']
            or (manifest['schema'] == MOVING_TERMINAL_SCHEMA and 'schema' not in state)):
        raise ValueError('Checkpoint schema와 보상·종료 계약이 일치하지 않습니다.')
    if state['checkpoint_id'] != manifest['checkpoint_id'] or state['sac'] != sac_settings(cfg):
        raise ValueError('모델·버퍼·설정 checkpoint 식별 불일치')
    if cfg.observation_size == 10 and state.get('normalization') != cfg.normalization:
        raise ValueError('이동 모델의 입력 정규화 정의가 일치하지 않습니다.')
    if manifest['schema'] == MOVING_TERMINAL_SCHEMA:
        for key in ('total_transitions', 'updates', 'executed_actions', 'episodes'):
            if type(state.get(key)) is not int or state[key] < 0:
                raise ValueError('Checkpoint 누적 카운터가 올바르지 않습니다: '+key)
        with np.load(path/'replay.npz', allow_pickle=False) as data:
            validate_replay(data, cfg, state['total_transitions'], SAC_SETTINGS['buffer_size'])
    return path, state, cfg


def validate_replay(data, cfg, total_transitions, capacity):
    """Check stored SB3 ring-buffer semantics before mutating a live learner."""
    fields = ('observations', 'next_observations', 'actions', 'rewards', 'dones', 'timeouts')
    if not set(fields+('capacity', 'total_transitions', 'pos', 'full')).issubset(data.files):
        raise ValueError('Replay 구성 파일에 필요한 배열·메타데이터가 없습니다.')

    def integer(name):
        value = data[name]
        if value.shape != () or value.dtype.kind not in 'iu' or int(value) < 0:
            raise ValueError('Replay 정수 메타데이터 불일치: '+name)
        return int(value)

    if integer('capacity') != capacity or integer('total_transitions') != total_transitions:
        raise ValueError('Replay 용량 또는 누적 transition 불일치')
    full = data['full']
    if full.shape != () or full.dtype.kind not in 'biu' or int(full) not in (0, 1):
        raise ValueError('Replay full 플래그가 올바르지 않습니다.')
    if integer('pos') != total_transitions % capacity or bool(full) != (total_transitions >= capacity):
        raise ValueError('Replay 위치 불일치')
    n = min(total_transitions, capacity)
    shapes = dict(observations=(n, 1, cfg.observation_size),
                  next_observations=(n, 1, cfg.observation_size), actions=(n, 1, 2),
                  rewards=(n, 1), dones=(n, 1), timeouts=(n, 1))
    for key, shape in shapes.items():
        values = data[key]
        if values.shape != shape or values.dtype.kind not in 'fbiu' or not np.isfinite(values).all():
            raise ValueError('Replay 데이터 불일치: '+key)
    if np.any(np.abs(data['actions']) > 1.+1e-6):
        raise ValueError('Replay 정규화 action이 [-1,1] 범위를 벗어났습니다.')
    dones, timeouts = data['dones'], data['timeouts']
    if not np.isin(dones, (0, 1)).all() or not np.isin(timeouts, (0, 1)).all() or np.any(timeouts > dones):
        raise ValueError('Replay 종료·timeout 플래그가 올바르지 않습니다.')
    if np.any((dones == 1) & (timeouts == 0)) and getattr(cfg, 'loss_mode', 'exclude') != 'confirmed_missing_terminal':
        raise ValueError('이전 보상 계약에는 실제 terminal transition을 넣을 수 없습니다.')
    return n


class SACLearner:
    def __init__(self, cfg, mode='new', checkpoint=None, device='cpu', seed=42):
        if mode not in ('new', 'resume', 'evaluate'):
            raise ValueError('Unknown run mode')
        cfg.validate(evaluation=mode == 'evaluate')
        self.mode, self.cfg = mode, cfg
        self.cancel = threading.Event()
        self.rng = np.random.default_rng(seed)
        self.total_transitions = self.updates = self.executed_actions = self.episodes = 0
        self.source = None
        self.experiment_context = None
        if mode == 'new':
            self.policy = make_policy(cfg, device, seed)
        else:
            source, state, saved_cfg = checkpoint_info(checkpoint)
            self.experiment_context = state.get('experiment_context')
            if cfg != checkpoint_run_config(saved_cfg, mode, cfg.dt):
                raise ValueError('저장된 정규화·보상·각도 설정과 학습 재개의 제어주기를 유지해야 합니다.')
            from stable_baselines3 import SAC
            from stable_baselines3.common.logger import configure
            self.source = str(source.resolve())
            # SB3 pickles its derived constant-schedule closure. Python code
            # objects cannot reliably be unpickled across Python versions.
            # Verify the authoritative numeric rate before rebuilding only
            # that derived object; preserve every trained tensor/optimizer.
            learning_rate = state['sac']['learning_rate']
            with zipfile.ZipFile(source/'model.zip') as archive:
                model_data = json.loads(archive.read('data'))
            saved_rate = model_data.get('learning_rate')
            if type(saved_rate) not in (int, float) or saved_rate != learning_rate:
                raise ValueError('모델의 고정 학습률이 checkpoint 설정과 일치하지 않습니다.')
            self.policy = SAC.load(source/'model.zip', device=device,
                                   custom_objects={'lr_schedule': lambda _remaining: learning_rate})
            if self.policy.observation_space.shape != (cfg.observation_size,) or self.policy.action_space.shape != (2,):
                raise ValueError(f'{cfg.observation_size}차원 관측 / 2차원 action 모델 필요')
            self.policy.set_logger(configure(folder=None, format_strings=[]))
            for key in ('total_transitions', 'updates', 'executed_actions', 'episodes'):
                setattr(self, key, int(state[key]))
            if self.policy.num_timesteps != self.total_transitions or self.policy._n_updates != self.updates:
                raise ValueError('저장 모델과 학습 진행 상태 불일치')
            if mode == 'resume':
                self.restore_replay(source/'replay.npz')
                self.restore_rng(source, state)
        self.policy.policy.set_training_mode(False)

    def action(self, obs, warmup):
        if warmup and self.mode != 'evaluate':
            return self.rng.uniform(-1, 1, 2).astype(np.float32)
        return self.policy.predict(obs, deterministic=self.mode == 'evaluate')[0]

    def add(self, obs, action, reward, next_obs, truncated, *, terminated=False):
        if self.mode == 'evaluate':
            return False
        if not isinstance(truncated, (bool, np.bool_)) or not isinstance(terminated, (bool, np.bool_)):
            raise ValueError('Transition 종료 플래그는 bool이어야 합니다.')
        if terminated and truncated:
            raise ValueError('Transition은 terminal과 timeout을 동시에 가질 수 없습니다.')
        if terminated and getattr(self.cfg, 'loss_mode', 'exclude') != 'confirmed_missing_terminal':
            raise ValueError('확정 유실 보상 계약에서만 terminal transition을 저장할 수 있습니다.')
        obs, next_obs, action = (np.asarray(value, dtype=np.float32) for value in (obs, next_obs, action))
        if (obs.shape != (self.cfg.observation_size,) or next_obs.shape != obs.shape or action.shape != (2,)
                or not all(np.isfinite(value).all() for value in (obs, next_obs, action))
                or not math.isfinite(reward) or np.any(np.abs(action) > 1.+1e-6)):
            raise ValueError('Transition의 관측·action·보상 값이 올바르지 않습니다.')
        self.policy.replay_buffer.add(obs[None], next_obs[None], action[None],
                                     np.array([reward]), np.array([terminated or truncated]),
                                     [{'TimeLimit.truncated': bool(truncated)}])
        self.total_transitions += 1
        self.policy.num_timesteps = self.total_transitions
        return True

    def finish_episode(self, summary, log, ratio):
        summary = dict(summary, mode=self.mode)
        cfg = getattr(self, 'cfg', TrackingConfig())
        reasons = ('step_limit', 'target_lost', 'initial_target_lost')
        if cfg.observation_size == 10:
            reasons += ('range_complete', 'manual_leg_end', 'manual_episode_end', 'initial_dataset')
        allowed = summary['reason'] in reasons
        planned = (math.floor(summary['new_transitions']*ratio)
                   if self.mode != 'evaluate' and allowed and self.total_transitions >= cfg.learning_starts else 0)
        completed, started = 0, time.monotonic()
        log.event('update_start', episode=summary['episode'], planned=planned)
        error = None
        try:
            for _ in range(planned):
                if self.cancel.is_set():
                    break
                self.policy.train(gradient_steps=1, batch_size=SAC_SETTINGS['batch_size'])
                completed += 1
                self.updates = int(self.policy._n_updates)
                values = self.policy.logger.name_to_value
                log.row('updates', dict(episode=summary['episode'], update=self.updates,
                                       actor_loss=float(values['train/actor_loss']), critic_loss=float(values['train/critic_loss']),
                                       alpha=float(values['train/ent_coef']), alpha_loss=float(values['train/ent_coef_loss'])))
        except Exception as exc:
            self.cancel.set()
            error = exc
            log.event('update_error', message=str(exc), episode=summary['episode'])
        finally:
            self.policy.policy.set_training_mode(False)
            self.episodes += 1
            summary.update(planned_updates=planned, completed_updates=completed, total_updates=self.updates,
                           total_transitions=self.total_transitions, total_actions=self.executed_actions,
                           update_s=time.monotonic()-started, checkpoint=None)
            if self.mode != 'evaluate':
                summary['checkpoint'] = str(self.save_checkpoint(log.folder))
                log.event('checkpoint_saved', path=summary['checkpoint'], episode=summary['episode'])
            log.episode(summary)
        if error is not None:
            raise error
        return summary

    def save_checkpoint(self, folder):
        import torch
        checkpoint_id = uuid.uuid4().hex
        root = Path(folder)/'checkpoints'
        root.mkdir(exist_ok=True)
        name = f'episode_{self.episodes:06d}_{checkpoint_id[:8]}'
        temp = root/('.pending_'+checkpoint_id)
        temp.mkdir()
        # Recreated by SB3 from the saved numeric learning_rate on load.
        self.policy.save(temp/'model.zip', exclude=['lr_schedule'])
        buffer = self.policy.replay_buffer
        n = buffer.size()
        fields = ('observations', 'next_observations', 'actions', 'rewards', 'dones', 'timeouts')
        np.savez_compressed(temp/'replay.npz', **{key: getattr(buffer, key)[:n] for key in fields},
                            pos=buffer.pos, full=buffer.full, capacity=buffer.buffer_size,
                            total_transitions=self.total_transitions)
        numpy_rng = np.random.get_state()
        torch_rng = {'cpu': torch.get_rng_state().cpu().numpy()}
        if torch.cuda.is_available():
            torch_rng.update({f'cuda_{i}': v.cpu().numpy() for i, v in enumerate(torch.cuda.get_rng_state_all())})
        np.savez_compressed(temp/'rng.npz', **torch_rng)
        state = dict(schema=self.cfg.schema, checkpoint_id=checkpoint_id, tracking=asdict(self.cfg), sac=sac_settings(self.cfg),
                     total_transitions=self.total_transitions, updates=self.updates,
                     executed_actions=self.executed_actions, episodes=self.episodes,
                     reset_action_rng=self.rng.bit_generator.state,
                     numpy_rng=[numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]],
                     python_rng=random.getstate(), source=self.source,
                     experiment_context=getattr(self, 'experiment_context', None),
                     normalization=self.cfg.normalization, action_grid='round_half_away_from_zero',
                     versions=dict({p: importlib.metadata.version(p)
                                    for p in ('numpy', 'torch', 'gymnasium', 'stable-baselines3', 'cloudpickle')},
                                   python=platform.python_version()))
        dump(temp/'state.json', state)
        files = {name: digest(temp/name) for name in ('model.zip', 'replay.npz', 'state.json', 'rng.npz')}
        dump(temp/'manifest.json', dict(schema=self.cfg.schema, checkpoint_id=checkpoint_id, files=files))
        final = root/name
        os.replace(temp, final)
        latest = Path(folder)/'latest.tmp'
        dump(latest, dict(checkpoint=str(final.relative_to(folder)), checkpoint_id=checkpoint_id))
        os.replace(latest, Path(folder)/'latest.json')
        return final

    def restore_replay(self, path):
        buffer = self.policy.replay_buffer
        with np.load(path, allow_pickle=False) as data:
            n = validate_replay(data, getattr(self, 'cfg', TrackingConfig()), self.total_transitions, buffer.buffer_size)
            for key in ('observations', 'next_observations', 'actions', 'rewards', 'dones', 'timeouts'):
                target = getattr(buffer, key)
                target[:n] = data[key]
            buffer.pos, buffer.full = int(data['pos']), bool(data['full'])

    def restore_rng(self, source, state):
        import torch
        self.rng.bit_generator.state = state['reset_action_rng']
        nr = state['numpy_rng']
        np.random.set_state((nr[0], np.array(nr[1], dtype=np.uint32), *nr[2:]))
        def tuples(value):
            return tuple(tuples(v) for v in value) if isinstance(value, list) else value
        random.setstate(tuples(state['python_rng']))
        with np.load(source/'rng.npz', allow_pickle=False) as rng:
            torch.set_rng_state(torch.from_numpy(rng['cpu'].copy()))
            saved_cuda = sorted(k for k in rng.files if k.startswith('cuda_'))
            if torch.cuda.is_available() and len(saved_cuda) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all([torch.from_numpy(rng[k].copy()) for k in saved_cuda])
