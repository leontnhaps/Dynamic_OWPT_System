"""Manual moving-target episodes; injectable clock, transport and learner."""
from dataclasses import asdict, dataclass
import math
import uuid
import numpy as np

from common.tx_bearing import commanded_bearing
from common.tx_moving import BearingKalman, PredictionInputError, moving_observation
from common.tx_tracking import BeamMap, EpisodeMetrics, action_command, invalid_reason


@dataclass(frozen=True)
class MovingRunSettings:
    start_pan: float = 0.
    start_tilt: float = 0.
    settle_s: float = 2.
    acquire_s: float = 3.
    missing_s: float = 3.
    updates_per_transition: float = 1.

    def validate(self, cfg):
        if not all(math.isfinite(value) for value in asdict(self).values()):
            raise ValueError('이동 실행 설정은 유한한 수치여야 합니다.')
        if self.settle_s < 0 or self.updates_per_transition < 0:
            raise ValueError('대기 시간과 업데이트 비율은 0 이상이어야 합니다.')
        if self.acquire_s != 3 or self.missing_s != 3:
            raise ValueError('초기 검출·연속 미검출 제한은 3초입니다.')
        target = np.array([self.start_pan, self.start_tilt])
        if np.any(target < cfg.low) or np.any(target > cfg.high) or np.any(target != np.round(target)):
            raise ValueError('시작점은 운용 범위 안의 1° 단위 명령각이어야 합니다.')


class MovingRun:
    """One explicit Start produces one episode, one save, and one return.

    A normal end drains the issued action; emergency abort never issues a return.
    The latest valid transition is held until its boundary is known, so ending
    an episode cannot silently leave the last replay item unmarked.
    """
    def __init__(self, cfg, settings, learner, log, send, clock, kalman_config):
        cfg.validate(evaluation=learner.mode == 'evaluate')
        if cfg.observation_size != 10:
            raise ValueError('이동 추적에는 10차원 모델이 필요합니다.')
        settings.validate(cfg)
        self.cfg, self.settings, self.learner = cfg, settings, learner
        self.log, self.send, self.clock = log, send, clock
        self.beam, self.kalman = BeamMap(cfg), BearingKalman(kalman_config)
        self.start_target = np.array([settings.start_pan, settings.start_tilt], dtype=float)
        self.phase = 'idle'
        self.episode = int(getattr(learner, 'episodes', 0))
        self.pending = self.save_request = self.latest = self.latest_metadata = None
        self.command = np.zeros(2)
        self.command_id = self.command_ack_s = self.last_command_time = None
        self.stop_requested = False
        self.end_requested = None
        self.inflight = self.buffered = None
        self.steps = self.new_transitions = self.excluded = 0
        self.metrics = self.initial = None
        self.rewards = []
        self.episode_saved = False

    @property
    def active(self):
        return self.pending is not None or self.phase not in ('idle', 'complete', 'error')

    @property
    def inference_allowed(self):
        return not self.stop_requested and self.phase in ('settle', 'acquire', 'ready', 'result', 'missing')

    def event(self, name, **fields):
        self.log.event(name, episode=self.episode, phase=self.phase, monotonic_s=self.clock(), **fields)

    def set_start(self, pan, tilt):
        candidate = MovingRunSettings(pan, tilt, self.settings.settle_s, self.settings.acquire_s,
                                      self.settings.missing_s, self.settings.updates_per_transition)
        candidate.validate(self.cfg)
        self.start_target = np.array([pan, tilt], dtype=float)
        self.event('start_point_changed', command=self.start_target.tolist(), applies_to='next_return_or_start')

    def start(self):
        if self.phase != 'idle' or self.pending is not None:
            raise ValueError('현재 episode의 저장·복귀를 마치거나 오류 후 다시 준비하세요.')
        self.episode = max(self.episode, int(getattr(self.learner, 'episodes', 0)))+1
        self.steps = self.new_transitions = self.excluded = 0
        self.metrics = self.initial = self.inflight = self.buffered = None
        self.previous_error = None
        self.previous_delta = np.zeros(2)
        self.previous_frame = self.last_kf_frame = -1
        self.missing_since = None
        self.latest = self.latest_metadata = None
        self.rewards = []
        self.stop_requested = self.episode_saved = False
        self.end_requested = self.save_request = None
        self.learner.cancel.clear()
        self.kalman.reset()
        self.reset_started = self.clock()
        self.initial_command = self.start_target.tolist()
        # Moving policies have no 1,000-transition uniform-action warmup.
        self.warmup = False
        self.phase = 'reset'
        self.event('episode_start', command=self.initial_command)
        self.issue(self.start_target.copy(), 'reset')

    def supply(self, sample, frame_metadata):
        if not self.inference_allowed:
            return False
        if self.latest is not None and sample.frame_id <= self.latest.frame_id:
            return False
        self.latest, self.latest_metadata = sample, frame_metadata
        return True

    def issue(self, target, purpose):
        if self.stop_requested or self.pending is not None:
            return
        target = np.asarray(target, dtype=float)
        token = 'm5-1-moving-'+uuid.uuid4().hex
        command = dict(cmd='move', request_id=token, pan=float(target[0]), tilt=float(target[1]),
                       speed=self.cfg.speed, acc=self.cfg.acc)
        before = self.clock()
        try:
            if not self.send(command):
                raise OSError('명령 전송 실패')
        except Exception as exc:
            self.abort('send_error', str(exc))
            return
        sent = self.clock()
        interval = None if self.last_command_time is None else sent-self.last_command_time
        self.last_command_time = sent
        self.pending = dict(token=token, sent=sent, purpose=purpose, target=target.copy(),
                            interval=interval, send_duration_s=sent-before)
        self.phase = purpose+'_reply'
        self.event('command_sent', request_id=token, command=target.tolist(), purpose=purpose,
                   command_time=sent, interval_s=interval)

    def reply(self, event):
        if (self.pending is None or event.get('request_id') != self.pending['token']
                or event.get('event') not in ('servo', 'error')):
            return False
        pending, self.pending = self.pending, None
        actual = event.get('commanded') or {}
        if self.clock()-pending['sent'] >= 5:
            self.abort('ack_timeout', '전송 명령 실행 여부 불명; 자동 재전송 없음')
            return True
        if (event.get('event') != 'servo' or event.get('operation') != 'move'
                or event.get('simulated') is not False or event.get('available') is not True
                or event.get('limits') != self.cfg.limits
                or actual.get('command_id') != pending['token']
                or any(actual.get(axis) != pending['target'][i] for i, axis in enumerate(('pan', 'tilt')))
                or actual.get('speed') != self.cfg.speed or actual.get('acc') != self.cfg.acc):
            self.abort('command_error', '실제 서보 응답의 명령 ID·목표각·범위·속도를 확인하지 못했습니다.')
            return True
        self.command, self.command_id = pending['target'], pending['token']
        self.command_ack_s = self.clock()
        self.event('command_confirmed', request_id=self.command_id, commanded=self.command.tolist())
        if self.stop_requested or self.phase in ('saving', 'error', 'idle'):
            return True
        purpose = pending['purpose']
        self.wait_until = pending['sent']+(self.cfg.dt if purpose == 'action' else self.settings.settle_s)
        if purpose == 'return':
            self.phase = 'return_settle'
        elif purpose == 'reset':
            if self.end_requested:
                self._finish_now(self.end_requested)
            else:
                self.phase = 'settle'
        else:
            self.steps += 1
            self.learner.executed_actions += 1
            self.metrics.command(self.previous_delta)
            self.inflight.update(command_time=pending['sent'], command_interval_s=pending['interval'],
                                 send_duration_s=pending['send_duration_s'],
                                 interval_excess_s=max(0, pending['interval']-self.cfg.dt)
                                 if pending['interval'] is not None else None, step=self.steps)
            self.result_deadline = max(self.wait_until, self.command_ack_s)+self.cfg.dt
            self.phase = 'result'
        return True

    def sample_reason(self, now, *, after=-math.inf, new=True):
        sample = self.latest
        reason = invalid_reason(sample, now, self.cfg, after=after,
                                previous=self.previous_frame if new else -1)
        if sample is None:
            return reason
        meta = self.latest_metadata or {}
        servo = meta.get('servo_at_capture_start') or {}
        command = servo.get('commanded') or {}
        if (servo.get('simulated') is not False or servo.get('available') is not True
                or servo.get('limits') != self.cfg.limits
                or command.get('command_id') != self.command_id
                or command.get('speed') != self.cfg.speed or command.get('acc') != self.cfg.acc
                or any(command.get(axis) != self.command[i] for i, axis in enumerate(('pan', 'tilt')))):
            return 'frame_command_mismatch'
        if sample.processed < sample.received or sample.processed > now:
            return 'invalid_timestamp'
        if self.command_ack_s is None or sample.received <= self.command_ack_s:
            return 'before_command_ack'
        return reason

    def row_sample(self, sample):
        row = dict(frame_id=sample.frame_id, seq=sample.seq, receive_s=sample.received,
                   processed_s=sample.processed, receive_unix_ns=sample.receive_unix_ns)
        if sample.target is not None:
            row['confidence'] = sample.target['confidence']
            row.update(zip(('u', 'v'), sample.target['center']))
            row.update(zip(('x1', 'y1', 'x2', 'y2'), sample.target['box']))
        return row

    def filter_missing(self, reason):
        sample = self.latest
        if (reason not in ('missing', 'invalid_box', 'empty_pixel_region') or sample is None
                or sample.frame_id <= self.last_kf_frame):
            return
        if self.kalman.time is not None and sample.received <= self.kalman.time:
            return
        kf = self.kalman.step(None, sample.received, self.cfg.dt)
        self.last_kf_frame = sample.frame_id
        self.event('kalman_observation', frame_id=sample.frame_id, request_id=self.command_id,
                   bearing_rad=None, kalman=kf, raw10=None, normalized10=None, exclusion_reason=reason)

    def observe(self, *, reset_history=False):
        sample = self.latest
        if sample.frame_id <= self.last_kf_frame or (self.kalman.time is not None and sample.received <= self.kalman.time):
            return 'nonincreasing_observation'
        try:
            self.beam.mean(sample.target['box'])
        except ValueError:
            self.filter_missing('empty_pixel_region')
            return 'empty_pixel_region'
        bearing = commanded_bearing(*sample.target['center'], *self.command)
        kf = self.kalman.step(bearing, sample.received, self.cfg.dt)
        self.last_kf_frame = sample.frame_id
        self.previous_frame = sample.frame_id
        try:
            raw, obs = moving_observation(sample, self.command, self.previous_delta,
                None if reset_history else self.previous_error, kf['posterior_state'][:2],
                kf['predicted_bearing_rad'], self.cfg)
        except PredictionInputError as exc:
            self.event('kalman_observation', frame_id=sample.frame_id, request_id=self.command_id,
                       bearing_rad=bearing, kalman=kf, raw10=None, normalized10=None,
                       exclusion_reason='invalid_kalman_prediction', detail=str(exc))
            return 'invalid_kalman_prediction'
        self.raw, self.obs, self.observation_sample = raw, obs, sample
        self.previous_error = raw[:2].copy()
        self.event('kalman_observation', frame_id=sample.frame_id, request_id=self.command_id,
                   bearing_rad=bearing, kalman=kf, raw10=raw.tolist(), normalized10=obs.tolist(), exclusion_reason=None)
        return None

    def tick(self):
        now = self.clock()
        if self.pending is not None:
            if now-self.pending['sent'] >= 5:
                self.pending = None
                self.abort('ack_timeout', '전송 명령 실행 여부 불명; 자동 재전송 없음')
            return
        if self.phase in ('idle', 'complete', 'error', 'saving'):
            return
        if self.phase == 'return_settle':
            if now >= self.wait_until:
                self.phase = 'idle'
                self.event('return_complete', commanded=self.command.tolist(), actual_arrival_verified=False)
            return
        if self.phase == 'settle':
            if now < self.wait_until:
                return
            self.acquire_after = max(self.wait_until, self.command_ack_s)
            self.phase = 'acquire'
        if self.phase == 'acquire':
            reason = self.sample_reason(now, after=self.acquire_after)
            deadline = self.acquire_after+self.settings.acquire_s
            # A result processed just after the deadline is usable only when
            # its actual receive timestamp precedes that deadline.
            if now >= deadline and (reason or self.latest.received >= deadline):
                if reason in (None, 'missing', 'invalid_box', 'empty_pixel_region', 'invalid_kalman_prediction'):
                    self._finish_now('initial_target_lost')
                else:
                    self.abort('initialization_failed', reason)
                return
            if not reason:
                reason = self.observe(reset_history=True)
            if not reason:
                self.initial = dict(self.row_sample(self.latest), error_px=float(np.linalg.norm(self.raw[:2])))
                self.metrics = EpisodeMetrics(self.cfg, now, self.initial['error_px'])
                self.event('initial_observation', observation=self.raw.tolist(), normalized=self.obs.tolist(), **self.initial)
                self.phase = 'ready'
            else:
                self.filter_missing(reason)
                if now >= self.acquire_after+self.settings.acquire_s:
                    if reason in ('missing', 'invalid_box', 'empty_pixel_region', 'invalid_kalman_prediction'):
                        self._finish_now('initial_target_lost')
                    else:
                        self.abort('initialization_failed', reason)
            return
        if self.phase == 'ready':
            if self.end_requested:
                self._finish_now(self.end_requested)
                return
            reason = self.sample_reason(now, new=False)
            if reason or not 0 <= now-self.observation_sample.received <= self.cfg.dt:
                self.begin_missing(reason or 'stale_control_observation', now)
                return
            if self.last_command_time is not None and now-self.last_command_time < self.cfg.dt:
                return
            action_started = self.clock()
            action = self.learner.action(self.obs, self.warmup)
            now = self.clock()
            if not 0 <= now-self.observation_sample.received <= self.cfg.dt:
                self.begin_missing('policy_computation_delay', now)
                return
            degrees, target, delta = action_command(action, self.command, self.cfg)
            self.previous_delta = delta
            self.inflight = dict(episode=self.episode, observation=self.raw.tolist(), normalized=self.obs.tolist(),
                observation_frame_id=self.observation_sample.frame_id, action=np.asarray(action).tolist(),
                action_degrees=degrees.tolist(), applied_delta=delta.tolist(), command=target.tolist(),
                behavior='deterministic' if self.learner.mode == 'evaluate' else 'policy',
                policy_duration_s=now-action_started)
            self.issue(target, 'action')
            return
        if self.phase == 'result':
            if now < self.wait_until:
                return
            reason = self.sample_reason(now, after=max(self.wait_until, self.command_ack_s))
            awaiting_frame = ('no_processed_frame', 'no_new_frame', 'before_command_or_reset_wait',
                              'before_command_ack', 'frame_command_mismatch')
            if reason in awaiting_frame and now < self.result_deadline:
                return
            before = self.obs.copy()
            if not reason:
                reason = self.observe()
            if reason:
                self.flush_transition(truncated=True)
                self.discard(reason, now)
                if self.end_requested:
                    self._finish_now(self.end_requested)
                else:
                    self.begin_missing(reason, now)
                return
            intensity, pixels = self.beam.mean(self.latest.target['box'])
            error = float(np.linalg.norm(self.raw[:2]))
            r_error, r_aim = -error/self.cfg.reward_distance, intensity
            reward = self.cfg.error_weight*r_error+self.cfg.aim_weight*r_aim
            self.flush_transition(truncated=False)
            row = dict(self.inflight, **self.row_sample(self.latest), result_s=now,
                result_age_s=now-self.latest.received, next_observation=self.raw.tolist(), next_normalized=self.obs.tolist(),
                error_px=error, intensity_mean=intensity, pixel_count=pixels, reward_error=r_error,
                reward_aim=r_aim, reward=reward, valid=True, exclusion_reason=None, terminated=False)
            self.buffered = (before, self.inflight['action'], reward, self.obs.copy(), row)
            self.metrics.result(error, intensity, now)
            self.rewards.append(reward)
            self.inflight = None
            self.phase = 'ready'
            if self.end_requested:
                self._finish_now(self.end_requested)
            return
        if self.phase == 'missing':
            reason = self.sample_reason(now)
            deadline = self.missing_since+self.settings.missing_s
            if now >= deadline and (reason or self.latest.received >= deadline):
                if reason in (None, 'missing', 'invalid_box', 'empty_pixel_region', 'invalid_kalman_prediction'):
                    self._finish_now('target_lost')
                else:
                    self.abort('observation_timeout', reason)
                return
            if not reason:
                reason = self.observe(reset_history=True)
            if not reason:
                self.metrics.recovered(now)
                self.missing_since = None
                self.event('reacquired', observation=self.raw.tolist(), normalized=self.obs.tolist(), **self.row_sample(self.latest))
                self.phase = 'ready'
            else:
                self.filter_missing(reason)
                if now >= self.missing_since+self.settings.missing_s:
                    if reason in ('missing', 'invalid_box', 'empty_pixel_region', 'invalid_kalman_prediction'):
                        self._finish_now('target_lost')
                    else:
                        self.abort('observation_timeout', reason)

    def begin_missing(self, reason, now):
        self.flush_transition(truncated=True)
        self.previous_error = None
        self.filter_missing(reason)
        if self.missing_since is None:
            self.missing_since = now
            self.metrics.missing(now)
        self.phase = 'missing'
        self.event('observation_unavailable', reason=reason)

    def flush_transition(self, truncated):
        if self.buffered is None:
            return
        before, action, reward, after, row = self.buffered
        stored = self.learner.add(before, action, reward, after, truncated)
        self.new_transitions += int(stored)
        # A failed disk write must not cause abort() to insert the same replay
        # item again while attempting to preserve the episode summary.
        self.buffered = None
        self.log.row('steps', dict(row, stored=stored, truncated=truncated))

    def discard(self, reason, now):
        if self.inflight is None:
            return
        inflight, self.inflight = self.inflight, None
        if 'step' in inflight:
            self.excluded += 1
            marks = {} if self.latest is None else {key:value for key,value in self.row_sample(self.latest).items()
                if key in ('frame_id', 'seq', 'receive_s', 'processed_s', 'receive_unix_ns')}
            self.log.row('steps', dict(inflight, **marks, result_s=now, valid=False, stored=False,
                                      exclusion_reason=reason, terminated=False, truncated=False))
        else:
            self.event('unconfirmed_action', reason=reason, action=inflight)

    def manual_end(self):
        if self.phase in ('idle', 'saving', 'return_reply', 'return_settle', 'error'):
            return
        self.end_requested = 'manual_episode_end'
        self.event('manual_episode_end_requested')
        # No next action is permitted. Drain the already issued action only.
        if self.pending is None and self.phase != 'result':
            self._finish_now(self.end_requested)

    def finish(self, reason='manual_episode_end'):
        if reason == 'manual_episode_end':
            self.manual_end()
        else:
            self._finish_now(reason)

    def _finish_now(self, reason):
        if self.phase in ('idle', 'saving', 'error') or self.episode_saved:
            return
        now = self.clock()
        self.flush_transition(truncated=True)
        self.discard(reason, now)
        summary = self.metrics.summary(now) if self.metrics else dict(executed_actions=0, valid_results=0)
        summary.update(episode=self.episode, reason=reason, terminated=False, truncated=True,
            new_transitions=self.new_transitions, excluded_transitions=self.excluded,
            initial_command=self.initial_command, initial_observation=self.initial,
            reset_elapsed_s=(self.metrics.started if self.metrics else now)-self.reset_started,
            reward_sum=float(sum(self.rewards)) if self.rewards else None,
            reward_mean=float(np.mean(self.rewards)) if self.rewards else None, reward_count=len(self.rewards))
        self.save_request = summary
        self.phase = 'saving'
        self.event('episode_end', **{key:value for key,value in summary.items() if key != 'episode'})

    def saved(self, success=True):
        if self.phase != 'saving':
            return
        self.save_request = None
        self.episode_saved = True
        if not success or self.stop_requested or self.pending is not None:
            self.phase = 'error'
            return
        self.issue(self.start_target.copy(), 'return')

    def updated(self, success=True):
        self.saved(success)

    def abort(self, reason='user_stop', detail=''):
        self.stop_requested = True
        self.learner.cancel.set()
        self.event('run_stop', reason=reason, detail=detail)
        if self.phase in ('return_reply', 'return_settle') or self.episode_saved:
            self.phase = 'error'
        elif self.phase not in ('idle', 'saving', 'error'):
            self._finish_now(reason)
        elif self.phase == 'idle':
            self.phase = 'error'
