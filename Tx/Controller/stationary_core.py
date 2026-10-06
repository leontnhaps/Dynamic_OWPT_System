"""Nonblocking real Tx episode state machine; clocks and I/O are injectable."""
import math
import uuid
import numpy as np
from common.tx_tracking import BeamMap, EpisodeMetrics, invalid_reason, observation, action_command


class StationaryRun:
    def __init__(self, cfg, settings, learner, log, send, clock):
        cfg.validate(evaluation=learner.mode == 'evaluate')
        settings.validate(cfg)
        self.cfg, self.settings = cfg, settings
        self.learner, self.log, self.send, self.clock = learner, log, send, clock
        self.beam = BeamMap(cfg)
        self.phase = 'idle'
        self.latest = None
        self.pending = None
        self.episode = 0
        self.stop_requested = False
        self.save_request = None
        self.last_command_time = None
        self.command = np.zeros(2)
        self.inflight = None

    @property
    def active(self):
        return self.phase not in ('idle', 'complete', 'error')

    def event(self, name, **fields):
        self.log.event(name, episode=self.episode, phase=self.phase, monotonic_s=self.clock(), **fields)

    def start(self):
        if self.active:
            raise ValueError('이미 실행 중입니다.')
        self.reset()

    def reset(self):
        self.episode += 1
        self.steps = self.new_transitions = self.excluded = 0
        self.metrics = None
        self.inflight = None
        self.initial = None
        self.previous_error = None
        self.previous_frame = -1
        self.previous_delta = np.zeros(2)
        self.missing_since = None
        self.reset_started = self.clock()
        self.warmup = self.learner.updates == 0 and self.learner.mode != 'evaluate'
        r, s = self.learner.rng, self.settings
        target = np.array([r.integers(s.pan_low, s.pan_high+1), r.integers(s.tilt_low, s.tilt_high+1)])
        self.initial_command = target.tolist()
        self.phase = 'reset'
        self.event('reset', command=self.initial_command)
        self.issue(target, 'reset')

    def issue(self, target, purpose):
        token = 'm3-2-'+uuid.uuid4().hex
        command = dict(cmd='move', request_id=token, pan=float(target[0]), tilt=float(target[1]),
                       speed=self.cfg.speed, acc=self.cfg.acc)
        before = self.clock()
        # No timer ever replays missed command slots. Every wait restarts here.
        try:
            if not self.send(command):
                raise OSError('명령 전송 실패')
        except Exception as exc:
            self.abort('send_error', str(exc))
            return
        sent = self.clock()
        interval = None if self.last_command_time is None else sent-self.last_command_time
        self.last_command_time = sent
        self.pending = dict(token=token, sent=sent, purpose=purpose, target=np.asarray(target),
                            interval=interval, send_duration_s=sent-before)
        self.phase = purpose+'_reply'
        self.event('command_sent', request_id=token, command=target.tolist(), purpose=purpose,
                   command_time=sent, interval_s=interval)

    def reply(self, event):
        if not self.pending or event.get('request_id') != self.pending['token']:
            return False
        if event.get('event') not in ('servo', 'error'):
            return False
        pending, self.pending = self.pending, None
        if (event.get('event') == 'error' or event.get('simulated') is not False
                or not event.get('available') or event.get('operation') != 'move'):
            self.abort('command_error', event.get('message', '실제 서보 전송을 확인하지 못함'))
            return True
        actual = event.get('commanded') or {}
        if any(actual.get(a) != pending['target'][i] for i, a in enumerate(('pan', 'tilt'))):
            self.abort('command_mismatch', '응답 목표각이 요청과 다릅니다.')
            return True
        if event.get('limits') != self.cfg.limits:
            self.abort('limits_changed', '운용 범위가 학습 설정과 다릅니다.')
            return True
        self.command = pending['target']
        self.event('command_confirmed', request_id=pending['token'], commanded=self.command.tolist())
        if self.stop_requested or self.phase in ('saving', 'complete', 'error'):
            return True
        if pending['purpose'] == 'reset':
            self.wait_until = pending['sent']+self.settings.settle_s
            self.phase = 'settle'
        else:
            self.steps += 1
            self.learner.executed_actions += 1
            self.metrics.command(self.previous_delta)
            self.inflight.update(command_time=pending['sent'], command_interval_s=pending['interval'],
                                 send_duration_s=pending['send_duration_s'],
                                 interval_excess_s=max(0, pending['interval']-self.cfg.dt)
                                 if pending['interval'] is not None else None,
                                 step=self.steps)
            self.result_after = pending['sent']
            self.wait_until = pending['sent']+self.cfg.dt
            self.phase = 'result'
        return True

    def observe(self, sample, reset_history=False):
        raw, normalized = observation(sample, self.command, self.previous_delta,
                                      None if reset_history else self.previous_error, self.cfg)
        self.raw, self.obs = raw, normalized
        self.previous_error = raw[:2].copy()
        self.previous_frame = sample.frame_id
        self.observation_sample = sample

    def row_sample(self, sample):
        row = dict(frame_id=sample.frame_id, seq=sample.seq, receive_s=sample.received,
                   processed_s=sample.processed, receive_unix_ns=sample.receive_unix_ns,
                   confidence=sample.target['confidence'])
        row.update(zip(('u', 'v'), sample.target['center']))
        row.update(zip(('x1', 'y1', 'x2', 'y2'), sample.target['box']))
        return row

    def tick(self):
        now = self.clock()
        if not self.active or self.phase == 'saving':
            return
        if self.pending:
            if now-self.pending['sent'] >= 5:
                self.pending = None
                self.abort('ack_timeout', '전송 명령 실행 여부 불명; 자동 재전송 없음')
            return
        if self.phase == 'settle':
            if now < self.wait_until:
                return
            self.acquire_after = self.wait_until
            self.phase = 'acquire'
        if self.phase == 'acquire':
            reason = invalid_reason(self.latest, now, self.cfg, after=self.acquire_after)
            if not reason:
                try:
                    self.beam.mean(self.latest.target['box'])
                except ValueError:
                    reason = 'empty_pixel_region'
            if not reason:
                self.observe(self.latest, True)
                self.initial = dict(self.row_sample(self.latest), error_px=float(np.linalg.norm(self.raw[:2])))
                self.metrics = EpisodeMetrics(self.cfg, now, self.initial['error_px'])
                self.event('initial_observation', observation=self.raw.tolist(), normalized=self.obs.tolist(), **self.initial)
                self.phase = 'ready'
            elif now >= self.acquire_after+self.settings.acquire_s:
                if reason in ('missing', 'invalid_box', 'empty_pixel_region'):
                    self.event('initial_observation_unavailable', reason=reason)
                    self.finish('initial_target_lost')
                else:
                    self.abort('initialization_failed', reason)
            return
        if self.phase == 'ready':
            # A newer missing result invalidates the old success even before the next action.
            reason = invalid_reason(self.latest, now, self.cfg)
            if reason or now-self.observation_sample.received > self.cfg.dt:
                self.begin_missing(reason or 'stale_control_observation', now)
                return
            action_started = self.clock()
            action = self.learner.action(self.obs, self.warmup)
            # Policy computation can take time: recheck freshness before issuing a command.
            now = self.clock()
            if now-self.observation_sample.received > self.cfg.dt:
                self.begin_missing('policy_computation_delay', now)
                return
            degrees, target, delta = action_command(action, self.command, self.cfg)
            self.previous_delta = delta
            self.inflight = dict(episode=self.episode, observation=self.raw.tolist(), normalized=self.obs.tolist(),
                                 observation_frame_id=self.observation_sample.frame_id,
                                 action=np.asarray(action).tolist(), action_degrees=degrees.tolist(),
                                 applied_delta=delta.tolist(), command=target.tolist(),
                                 behavior='uniform' if self.warmup else ('deterministic' if self.learner.mode == 'evaluate' else 'policy'),
                                 policy_duration_s=now-action_started)
            self.issue(target, 'action')
            return
        if self.phase == 'result':
            if now < self.wait_until:
                return
            reason = invalid_reason(self.latest, now, self.cfg, after=self.result_after, previous=self.previous_frame)
            if not reason:
                try:
                    intensity, pixels = self.beam.mean(self.latest.target['box'])
                except ValueError:
                    reason = 'empty_pixel_region'
            if reason:
                self.discard(reason, now)
                self.begin_missing(reason, now)
                return
            before = self.obs.copy()
            self.observe(self.latest)
            error = float(np.linalg.norm(self.raw[:2]))
            r_error, r_aim = -error/self.cfg.reward_distance, intensity
            reward = self.cfg.error_weight*r_error+self.cfg.aim_weight*r_aim
            truncated = self.steps >= self.settings.max_steps
            stored = self.learner.add(before, self.inflight['action'], reward, self.obs, truncated)
            self.new_transitions += int(stored)
            self.metrics.result(error, intensity, now)
            row = dict(self.inflight, **self.row_sample(self.latest), result_s=now,
                       result_age_s=now-self.latest.received, next_observation=self.raw.tolist(),
                       next_normalized=self.obs.tolist(), error_px=error, intensity_mean=intensity, pixel_count=pixels,
                       reward_error=r_error, reward_aim=r_aim, reward=reward,
                       valid=True, stored=stored, exclusion_reason=None, terminated=False, truncated=truncated)
            self.log.row('steps', row)
            self.inflight = None
            self.phase = 'ready'
            if truncated:
                self.finish('step_limit')
            return
        if self.phase == 'missing':
            if now >= self.missing_since+self.settings.missing_s:
                self.finish('target_lost')
                return
            reason = invalid_reason(self.latest, now, self.cfg, previous=self.previous_frame)
            if not reason:
                try:
                    self.beam.mean(self.latest.target['box'])
                except ValueError:
                    return
                self.observe(self.latest, True)
                self.metrics.recovered(now)
                self.event('reacquired', observation=self.raw.tolist(), normalized=self.obs.tolist(), **self.row_sample(self.latest))
                self.missing_since = None
                if self.steps >= self.settings.max_steps:
                    self.finish('step_limit')
                else:
                    self.phase = 'ready'

    def begin_missing(self, reason, now):
        self.missing_since = now
        self.metrics.missing(now)
        self.phase = 'missing'
        self.event('observation_unavailable', reason=reason)

    def discard(self, reason, now):
        if self.inflight is None:
            return
        # Only confirmed commands are action steps. Unconfirmed commands remain events.
        if 'step' in self.inflight:
            self.excluded += 1
            marks = {} if self.latest is None else dict(frame_id=self.latest.frame_id, seq=self.latest.seq,
                receive_s=self.latest.received, processed_s=self.latest.processed, receive_unix_ns=self.latest.receive_unix_ns)
            self.log.row('steps', dict(self.inflight, **marks, result_s=now, valid=False, stored=False,
                                      exclusion_reason=reason, terminated=False, truncated=False))
        else:
            self.event('unconfirmed_action', reason=reason, action=self.inflight)
        self.inflight = None

    def finish(self, reason):
        if self.phase in ('saving', 'complete', 'error'):
            return
        now = self.clock()
        self.discard(reason, now)
        summary = self.metrics.summary(now) if self.metrics else dict(executed_actions=0, valid_results=0)
        summary.update(episode=self.episode, reason=reason, terminated=reason in ('target_lost', 'initial_target_lost'),
                       truncated=reason == 'step_limit', new_transitions=self.new_transitions,
                       excluded_transitions=self.excluded, initial_command=self.initial_command,
                       initial_observation=self.initial, reset_elapsed_s=(self.metrics.started if self.metrics else now)-self.reset_started)
        self.event('episode_end', **{k:v for k,v in summary.items() if k != 'episode'})
        self.save_request = summary
        self.phase = 'saving'

    def saved(self):
        if self.stop_requested or self.episode >= self.settings.episodes:
            self.phase = 'complete'
        else:
            self.reset()

    def abort(self, reason='user_stop', detail=''):
        self.stop_requested = True
        self.learner.cancel.set()
        self.event('run_stop', reason=reason, detail=detail)
        if self.phase not in ('idle', 'complete', 'error', 'saving'):
            self.finish(reason)
