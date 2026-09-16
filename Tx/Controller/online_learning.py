"""Measured real-world transitions and episode-boundary SAC updates.

No actuator I/O. Missing observations never produce nonterminal training targets.
"""
import json
import math
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class EpisodeSettings:
    pan_min: int = -3
    pan_max: int = 3
    tilt_min: int = -3
    tilt_max: int = 3
    seconds: float = 20.
    settle_seconds: float = 1.
    misses: int = 3
    loss_seconds: float = .5
    loss_penalty: float = 10.
    updates: int = 32

    def validate(self, cfg, limits):
        for i,axis in enumerate(('pan','tilt')):
            low,high=getattr(self,axis+'_min'),getattr(self,axis+'_max')
            if not all(isinstance(x,int) for x in (low,high)) or not (
                max(cfg.angle_low[i],limits[axis+'_min']) <= low <= high <= min(cfg.angle_high[i],limits[axis+'_max'])):
                raise ValueError('랜덤 시작 범위는 적용한 운용 범위/모델 범위 안의 정수 각도여야 합니다.')
        if not all(math.isfinite(x) for x in (self.seconds,self.settle_seconds,self.loss_seconds,self.loss_penalty)):
            raise ValueError('시간/벌점은 유한한 값이어야 합니다.')
        if not 1<=self.seconds<=300 or not 0<=self.settle_seconds<=30 or not .1<=self.loss_seconds<=10 or self.loss_penalty<0:
            raise ValueError('시간 1~300초, 대기 0~30초, 미검출 0.1~10초, 벌점 0 이상')
        if not isinstance(self.misses,int) or not 1<=self.misses<=100 or not isinstance(self.updates,int) or not 1<=self.updates<=1000:
            raise ValueError('미검출 횟수 1~100, 업데이트 1~1000 정수')

    def sample(self,rng):
        return [int(rng.integers(self.pan_min,self.pan_max+1)),int(rng.integers(self.tilt_min,self.tilt_max+1))]


class Episode:
    def __init__(self,cfg,settings,number):
        self.cfg=cfg;self.settings=settings;self.number=number
        self.pending=None;self.rows=[];self.started=None
        self.missing_since=None;self.missing_count=0

    def missed(self,now):
        if self.missing_since is None:self.missing_since=now
        self.missing_count+=1
        return self.missing_count>=self.settings.misses or self.loss_expired(now)

    def loss_expired(self,now):
        return self.missing_since is not None and now-self.missing_since>=self.settings.loss_seconds

    def recovered(self):
        self.missing_since=None;self.missing_count=0

    def remember(self,result,now):
        """Only call once the command has been sent successfully (or a real hold)."""
        if self.pending is not None:raise RuntimeError('Unpaired previous action')
        self.pending=dict(obs=list(result['observation']),action=list(result['action']),
                          command=list(result['command']),delta=list(result['applied_delta']),at=now)

    def observe(self,result,now,truncated=False):
        from simulation.control import reward_terms
        self.recovered()
        if self.pending is None:return
        p=self.pending
        reward,_,_=reward_terms(result['error'],np.asarray(p['command']),np.asarray(p['command'])-p['delta'],self.cfg)
        self.rows.append(dict(observation=p['obs'],action=p['action'],next_observation=list(result['observation']),
            reward=reward,terminated=False,truncated=bool(truncated),elapsed_s=now-p['at'],observed_next=True))
        self.pending=None

    def finish(self,reason,now):
        # No fake error/reward on stale video, unknown command execution, or a manual stop.
        if reason=='target_lost' and self.pending is not None:
            p=self.pending
            self.rows.append(dict(observation=p['obs'],action=p['action'],next_observation=p['obs'],
                reward=-self.settings.loss_penalty,terminated=True,truncated=False,
                elapsed_s=now-p['at'],observed_next=False))
        dropped=self.pending is not None and reason!='target_lost'
        self.pending=None
        if self.rows and not self.rows[-1]['terminated']:
            self.rows[-1]['truncated']=True
        return dict(episode=self.number,reason=reason,transitions=len(self.rows),
            reward_sum=sum(r['reward'] for r in self.rows),unpaired_action_dropped=dropped,
            duration_s=0 if self.started is None else now-self.started,settings=asdict(self.settings))


class RealLearner:
    def __init__(self,models,folder,source):
        from stable_baselines3.common.buffers import ReplayBuffer
        from stable_baselines3.common.logger import configure
        self.model=models.policy;self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.cancel=threading.Event();self.total=0
        # Start a new REAL buffer. Do not silently mix simulation and live experiences.
        self.model.replay_buffer=ReplayBuffer(50000,self.model.observation_space,self.model.action_space,
                                             device=self.model.device,n_envs=1,handle_timeout_termination=True)
        self.model.set_logger(configure(str(self.folder/'optimizer'),['csv']))
        self.model._current_progress_remaining=1.
        models.cfg.save(self.folder/'config.json')
        (self.folder/'session.json').write_text(json.dumps(dict(source=source,mode='real_from_scratch' if getattr(models,'from_scratch',False) else 'real_finetuning',
            initial_total_timesteps=self.model.num_timesteps,batch_size=64,buffer_size=50000,
            reward=dict(mode=models.cfg.reward_mode,pointing_weight=models.cfg.pointing_weight,
                        alignment_scale_px=models.cfg.alignment_scale_px,command_weight=models.cfg.command_weight),
            real_angle_feedback=False),ensure_ascii=False,indent=2),encoding='utf-8')

    def update(self,episode,summary,steps):
        """Runs on the SAME single worker as inference, never concurrently with it."""
        epdir=self.folder/f'episode_{episode.number:04d}';epdir.mkdir()
        with (epdir/'transitions.jsonl').open('w',encoding='utf-8') as f:
            for r in episode.rows:
                f.write(json.dumps(r)+'\n')
                done=r['terminated'] or r['truncated']
                self.model.replay_buffer.add(np.asarray(r['observation'],dtype=np.float32)[None],
                    np.asarray(r['next_observation'],dtype=np.float32)[None],np.asarray(r['action'],dtype=np.float32)[None],
                    np.array([r['reward']]),np.array([done]),[{'TimeLimit.truncated':r['truncated'] and not r['terminated']}])
        self.total+=len(episode.rows);self.model.num_timesteps+=len(episode.rows)
        count=0
        if self.model.replay_buffer.size()>=64:
            for _ in range(steps):
                if self.cancel.is_set():break
                self.model.train(gradient_steps=1,batch_size=64);count+=1
        self.model.logger.dump(step=self.total)
        summary.update(gradient_updates=count,buffer_samples=self.model.replay_buffer.size(),real_transitions_total=self.total)
        # Save each episode separately, including a cancelled partial update. Never overwrite source.
        self.model.save(epdir/'model.zip');self.model.save_replay_buffer(epdir/'replay_buffer.pkl')
        self.model.policy.set_training_mode(False)
        # Config next to checkpoint makes existing evaluate/live loaders work unchanged.
        (epdir/'config.json').write_text((self.folder/'config.json').read_text(),encoding='utf-8')
        (epdir/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        return dict(path=str(epdir),**summary)

    def close(self):
        self.model.logger.close()


class EpisodeBatch:
    """One bounded batch; only normal episode endings can schedule another reset."""
    def __init__(self, count):
        if not isinstance(count, int) or not 1 <= count <= 1000:
            raise ValueError('반복 횟수는 1~1000 정수')
        self.total=count;self.started=0;self.active=True;self.ready_at=None

    def began(self):
        self.started+=1;self.ready_at=None

    def saved(self, reason, now):
        if not self.active:return
        if reason not in ('time_limit','target_lost') or self.started>=self.total:
            self.cancel()
        else:
            self.ready_at=now+1.

    def ready(self, now):
        return self.active and self.ready_at is not None and now>=self.ready_at

    def cancel(self):
        self.active=False;self.ready_at=None
