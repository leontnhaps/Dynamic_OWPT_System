"""SAC adapter and paired checkpoints for M3-2 real camera transitions."""
import csv
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import statistics
import threading
import time
import uuid
import numpy as np
from common.tx_tracking import SCHEMA, TrackingConfig

SAC_SETTINGS = dict(learning_rate=3e-4, buffer_size=1_000_000, batch_size=256,
                    gamma=.99, tau=.005, learning_starts=1000, ent_coef='auto',
                    target_entropy=-2., target_update_interval=1)


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
        observation_space = gym.spaces.Box(-1., 1., (8,), np.float32)
        action_space = gym.spaces.Box(-1., 1., (2,), np.float32)

        def reset(self, **kwargs):
            raise RuntimeError('Reset comes from the actual Tx and camera')

        def step(self, action):
            raise RuntimeError('Transitions come from real post-command observations')

    policy = SAC('MlpPolicy', CameraSpaces(), device=device, seed=seed, **SAC_SETTINGS,
                 policy_kwargs=dict(net_arch=[256, 256], activation_fn=torch.nn.ReLU, n_critics=2),
                 train_freq=1, gradient_steps=1, verbose=0)
    policy.set_logger(configure(folder=None, format_strings=[]))
    return policy


def checkpoint_info(path):
    path = Path(path)
    if path.is_file():
        path = path.parent
    manifest = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != SCHEMA:
        raise ValueError('M3-2 8차원 checkpoint를 선택하세요. 구형 6차원 모델은 호환되지 않습니다.')
    required = {'model.zip', 'replay.npz', 'state.json', 'rng.npz'}
    if set(manifest.get('files', {})) != required:
        raise ValueError('완전한 checkpoint 구성 파일이 필요합니다.')
    for name, expected in manifest['files'].items():
        if digest(path/name) != expected:
            raise ValueError('Checkpoint 무결성 불일치: '+name)
    state = json.loads((path/'state.json').read_text(encoding='utf-8'))
    cfg = TrackingConfig(**state['tracking'])
    cfg.validate()
    if state['checkpoint_id'] != manifest['checkpoint_id'] or state['sac'] != SAC_SETTINGS:
        raise ValueError('모델·버퍼·설정 checkpoint 식별 불일치')
    return path, state, cfg


class SACLearner:
    def __init__(self, cfg, mode='new', checkpoint=None, device='cpu', seed=42):
        from stable_baselines3 import SAC
        from stable_baselines3.common.logger import configure
        import torch
        if mode not in ('new', 'resume', 'evaluate'):
            raise ValueError('Unknown run mode')
        self.mode, self.cfg = mode, cfg
        self.cancel = threading.Event()
        self.rng = np.random.default_rng(seed)
        self.total_transitions = self.updates = self.executed_actions = self.episodes = 0
        self.source = None
        if mode == 'new':
            cfg.validate()
            self.policy = make_policy(cfg, device, seed)
        else:
            source, state, saved_cfg = checkpoint_info(checkpoint)
            if cfg != saved_cfg:
                raise ValueError('저장된 제어주기·정규화·보상·각도 설정을 유지해야 합니다.')
            self.source = str(source.resolve())
            self.policy = SAC.load(source/'model.zip', device=device)
            if self.policy.observation_space.shape != (8,) or self.policy.action_space.shape != (2,):
                raise ValueError('8차원 관측 / 2차원 action 모델 필요')
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

    def add(self, obs, action, reward, next_obs, truncated):
        if self.mode == 'evaluate':
            return False
        self.policy.replay_buffer.add(np.array(obs, dtype=np.float32)[None],
                                     np.array(next_obs, dtype=np.float32)[None],
                                     np.array(action, dtype=np.float32)[None],
                                     np.array([reward]), np.array([truncated]),
                                     [{'TimeLimit.truncated': bool(truncated)}])
        self.total_transitions += 1
        self.policy.num_timesteps = self.total_transitions
        return True

    def finish_episode(self, summary, log, ratio):
        summary = dict(summary, mode=self.mode)
        allowed = summary['reason'] in ('step_limit', 'target_lost')
        planned = (math.floor(summary['new_transitions']*ratio)
                   if self.mode != 'evaluate' and allowed and self.total_transitions >= SAC_SETTINGS['learning_starts'] else 0)
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
        self.policy.save(temp/'model.zip')
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
        state = dict(schema=SCHEMA, checkpoint_id=checkpoint_id, tracking=asdict(self.cfg), sac=SAC_SETTINGS,
                     total_transitions=self.total_transitions, updates=self.updates,
                     executed_actions=self.executed_actions, episodes=self.episodes,
                     reset_action_rng=self.rng.bit_generator.state,
                     numpy_rng=[numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]],
                     python_rng=random.getstate(), source=self.source,
                     normalization=[696, 588, 1296, 972, 5, 5], action_grid='round_half_away_from_zero',
                     versions={p: importlib.metadata.version(p) for p in ('numpy', 'torch', 'gymnasium', 'stable-baselines3')})
        dump(temp/'state.json', state)
        files = {name: digest(temp/name) for name in ('model.zip', 'replay.npz', 'state.json', 'rng.npz')}
        dump(temp/'manifest.json', dict(schema=SCHEMA, checkpoint_id=checkpoint_id, files=files))
        final = root/name
        os.replace(temp, final)
        latest = Path(folder)/'latest.tmp'
        dump(latest, dict(checkpoint=str(final.relative_to(folder)), checkpoint_id=checkpoint_id))
        os.replace(latest, Path(folder)/'latest.json')
        return final

    def restore_replay(self, path):
        buffer = self.policy.replay_buffer
        with np.load(path, allow_pickle=False) as data:
            if int(data['capacity']) != buffer.buffer_size or int(data['total_transitions']) != self.total_transitions:
                raise ValueError('Replay 용량 또는 누적 transition 불일치')
            n = min(self.total_transitions, buffer.buffer_size)
            expected_pos = self.total_transitions % buffer.buffer_size
            if int(data['pos']) != expected_pos or bool(data['full']) != (self.total_transitions >= buffer.buffer_size):
                raise ValueError('Replay 위치 불일치')
            for key in ('observations', 'next_observations', 'actions', 'rewards', 'dones', 'timeouts'):
                target = getattr(buffer, key)
                if data[key].shape != target[:n].shape or not np.isfinite(data[key]).all():
                    raise ValueError('Replay 데이터 불일치: '+key)
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
