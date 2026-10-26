"""M5-1 one-way data collection; no drone commands or learned controller."""
from dataclasses import dataclass, asdict
import time
import uuid
from common.control_timing import PRIMARY_CONTROL_PERIOD_S, CONTROL_PERIODS_S
from common.servo import limits_from, number
from common.tx_setup import TX_OPERATING_LIMITS, TX_CAMERA_SETTINGS
from common.tx_tracking import TrackingConfig

SCHEMA = 'm5-1-sweep-v3'
KF_MODE = '칼만필터 최적화용 수집 + YOLO 야외 Test'
YOLO_MODE = 'YOLO 야외 Test (이미지 수집)'
LEARNING_MODE = '이동환경 SAC 제어 (연결 예정)'


@dataclass(frozen=True)
class SweepSettings:
    pan: float = 0
    tilt: float = 0
    pan_min: float = TX_OPERATING_LIMITS['pan_min']
    pan_max: float = TX_OPERATING_LIMITS['pan_max']
    tilt_min: float = TX_OPERATING_LIMITS['tilt_min']
    tilt_max: float = TX_OPERATING_LIMITS['tilt_max']
    pan_step: float = -2
    tilt_step: float = 0
    tilt_motion: str = 'linear'
    tilt_osc_min: float = -2
    tilt_osc_max: float = 2
    dt: float = PRIMARY_CONTROL_PERIOD_S
    settle_s: float = 2
    confidence: float = TrackingConfig().confidence
    class_id: int = TrackingConfig().class_id
    speed: int = TrackingConfig().speed
    acc: int = TrackingConfig().acc

    @property
    def limits(self):
        return {k: getattr(self, k) for k in TX_OPERATING_LIMITS}

    @classmethod
    def from_text(cls, values):
        defaults = asdict(cls())
        result = cls(**{k: (str(values.get(k, '')).strip() or default) if isinstance(default, str)
                        else number(str(values.get(k, '')).strip() or default, k)
                        for k, default in defaults.items()})
        result.validate()
        return result

    def validate(self):
        for key, value in asdict(self).items():
            if key != 'tilt_motion':
                number(value, key)
        if self.tilt_motion not in ('linear', 'oscillate'):
            raise ValueError('Tilt 이동 방식은 linear / oscillate입니다.')
        limits_from(self.limits)
        for axis in ('pan', 'tilt'):
            for key in (axis, axis+'_min', axis+'_max', axis+'_step'):
                if not float(getattr(self, key)).is_integer():
                    raise ValueError('Pan/Tilt 각도·범위·이동각은 1° 단위 정수입니다.')
            if not self.limits[axis+'_min'] <= getattr(self, axis) <= self.limits[axis+'_max']:
                raise ValueError(f'최초 {axis}가 입력한 운용 범위를 벗어납니다.')
        if self.pan_step == self.tilt_step == 0:
            raise ValueError('편도 이동을 위해 적어도 한 축의 이동각은 0이 아니어야 합니다.')
        if self.tilt_motion == 'oscillate':
            if (not self.pan_step or not self.tilt_step
                    or not self.tilt_min <= self.tilt_osc_min <= self.tilt <= self.tilt_osc_max <= self.tilt_max
                    or self.tilt_osc_min >= self.tilt_osc_max
                    or not all(float(v).is_integer() for v in (self.tilt_osc_min, self.tilt_osc_max))):
                raise ValueError('Tilt 왕복은 Pan 이동과 0이 아닌 Tilt 이동각, 초기각을 포함하는 정수 왕복 범위가 필요합니다.')
        if self.dt not in CONTROL_PERIODS_S or self.settle_s < self.dt:
            raise ValueError('주기는 0.720 / 0.800 s, 초기 대기는 주기 이상이어야 합니다.')
        if not 0 < self.confidence <= 1 or self.class_id < 0 or not float(self.class_id).is_integer():
            raise ValueError('Confidence와 PV class ID를 확인하세요.')
        if (self.speed, self.acc) != (TrackingConfig().speed, TrackingConfig().acc):
            raise ValueError('확정된 speed/acc 100/1을 사용하세요.')
        if (self.tilt_motion == 'oscillate' and self.pan == (self.pan_max if self.pan_step > 0 else self.pan_min)
                or self.tilt_motion == 'linear' and self.next_pose((self.pan, self.tilt), 1) == (self.pan, self.tilt)):
            raise ValueError('최초 위치가 이미 진행 방향 끝점입니다. 이동 부호나 최초각을 바꾸세요.')

    def next_pose(self, pose, direction):
        return tuple(max(self.limits[a+'_min'], min(self.limits[a+'_max'],
                     value+direction*getattr(self, a+'_step')))
                     for a, value in zip(('pan', 'tilt'), pose))


def camera_matches(frame, size):
    return (size == (TX_CAMERA_SETTINGS['width'], TX_CAMERA_SETTINGS['height'])
            and frame[1].get('simulated') is False
            and all(k in frame[1].get('requested', {}) and frame[1]['requested'][k] == v
                    for k, v in TX_CAMERA_SETTINGS.items()))


class SweepRun:
    """One command -> ACK -> dwell -> fresh frame -> infer/save -> next command.

    Never schedules catch-up moves. A completed leg needs a new user click.
    Missing PV is a saved observation gap, not a motion stop condition.
    """
    def __init__(self, settings, log, send, clock=time.monotonic):
        settings.validate()
        self.cfg, self.log, self.send, self.clock = settings, log, send, clock
        self.active = False
        self.pending = None
        self.phase = 'idle'
        self.test = 0
        self.pose = (settings.pan, settings.tilt)
        self.last_sent = None
        self.last_frame = -1
        self.finalized = True

    def start_leg(self, note=''):
        if self.active or self.pending or self.phase == 'aborted':
            raise ValueError('진행 중인 편도를 마치거나 중단 후 자동 준비를 다시 실행하세요.')
        self.finalize()
        self.test += 1
        self.direction = 1 if self.test % 2 else -1
        self.tilt_direction = 1 if self.direction*self.cfg.tilt_step > 0 else -1
        self.step = self.saved = self.detected = self.valid = 0
        self.started = self.clock()
        self.start_pose = self.pose
        self.note = note
        self.finalized = False
        self.active = True
        self.log.event('leg_start', test=self.test, direction=self.direction, note=note,
                       pan_step=self.direction*self.cfg.pan_step,
                       tilt_step=self.direction*self.cfg.tilt_step, start_pose=self.pose)
        # Re-issue the endpoint on later legs to identify frames by command_id.
        # This does not return to the initial pose.
        self.issue(self.pose, self.cfg.settle_s if self.test == 1 else self.cfg.dt)

    def issue(self, pose, dwell):
        now = self.clock()
        token = 'm5-1-move-'+uuid.uuid4().hex
        cmd = dict(cmd='move', request_id=token, pan=pose[0], tilt=pose[1],
                   speed=self.cfg.speed, acc=self.cfg.acc)
        interval = None if self.last_sent is None else now-self.last_sent
        self.context = dict(test=self.test, step=self.step, direction=self.direction,
            request_id=token, pan_command_deg=pose[0], tilt_command_deg=pose[1],
            delta_pan_deg=pose[0]-self.pose[0], delta_tilt_deg=pose[1]-self.pose[1],
            command_sent_s=now, command_unix_ns=time.time_ns(), command_interval_s=interval,
            interval_excess_s=None if interval is None else interval-self.cfg.dt,
            dwell_s=dwell, note=self.note)
        self.log.event('command_attempt', **self.context, command=cmd)
        self.pending = dict(token=token, sent=now, pose=pose, dwell=dwell)
        self.phase = 'ack'
        if not self.send(cmd):
            self.pending = None
            self.abort('command_send_failed')
            return
        self.last_sent = now

    def reply(self, event):
        if not self.pending or event.get('request_id') != self.pending['token']:
            return False
        if event.get('event') not in ('servo', 'error'):
            return False
        pending, self.pending = self.pending, None
        self.last_reply_valid = False
        self.log.event('command_reply', test=self.test, reply=event)
        if self.clock()-pending['sent'] > 5:
            self.abort('servo_ack_timeout')
            return True
        commanded = event.get('commanded') or {}
        if (event.get('event') != 'servo' or event.get('operation') != 'move'
                or event.get('simulated') is not False or event.get('available') is not True
                or event.get('limits') != self.cfg.limits
                or tuple(commanded.get(a) for a in ('pan', 'tilt')) != pending['pose']
                or commanded.get('command_id') != pending['token']
                or (commanded.get('speed'), commanded.get('acc')) != (self.cfg.speed, self.cfg.acc)):
            self.abort('invalid_servo_reply')
            return True
        self.pose = pending['pose']
        self.last_reply_valid = True
        self.context['ack_s'] = self.clock()
        self.frame_after = max(pending['sent']+pending['dwell'], self.clock())
        if self.active:
            self.phase = 'dwell'
        return True

    def tick(self):
        now = self.clock()
        if self.pending and now-self.pending['sent'] > 5:
            self.pending = None
            self.abort('servo_ack_timeout')
        if not self.active:
            return
        if self.phase == 'dwell' and now >= self.frame_after:
            self.phase = 'capture'
            self.deadline = now+3
        if self.phase in ('capture', 'processing') and now > self.deadline:
            self.abort('camera_timeout' if self.phase == 'capture' else 'inference_or_save_timeout')

    def capture_request(self, frame, frame_id, size):
        if not self.active or self.phase != 'capture' or not frame:
            return None
        if not camera_matches(frame, size):
            self.abort('camera_configuration')
            return None
        if (frame_id <= self.last_frame or frame[2] <= self.frame_after
                or not 0 <= self.clock()-frame[2] <= self.cfg.dt):
            return None
        # Receive time alone cannot reject an old image delayed by transport.
        servo = frame[1].get('servo_at_capture_start') or {}
        commanded = servo.get('commanded') or {}
        if commanded.get('command_id') != self.context['request_id']:
            return None
        if (servo.get('simulated') is not False or servo.get('available') is not True
                or servo.get('limits') != self.cfg.limits
                or tuple(commanded.get(a) for a in ('pan', 'tilt')) != self.pose):
            self.abort('frame_servo_mismatch')
            return None
        self.last_frame = frame_id
        self.phase = 'processing'
        self.deadline = self.clock()+30
        return dict(self.context, frame_id=frame_id, submitted_s=self.clock())

    def captured(self, row):
        self.saved += 1
        self.detected += int(row['status'] == 'detected')
        self.valid += int(row['kf_eligible'])
        self.tick()
        if not self.active:
            return
        if row['status'] == 'error':
            self.abort('inference_error')
            return
        if self.cfg.tilt_motion == 'oscillate':
            pan = self.cfg.next_pose(self.pose, self.direction)[0]
            if pan == self.pose[0]:
                self.finish('range_complete')
                return
            lo, hi = self.cfg.tilt_osc_min, self.cfg.tilt_osc_max
            # Reverse at an endpoint without repeating it. Clamp a nondivisible
            # final increment; the actual applied delta is logged for learning.
            if self.pose[1] >= hi:
                self.tilt_direction = -1
            elif self.pose[1] <= lo:
                self.tilt_direction = 1
            tilt = max(lo, min(hi, self.pose[1]+self.tilt_direction*abs(self.cfg.tilt_step)))
            target = (pan, tilt)
        else:
            target = self.cfg.next_pose(self.pose, self.direction)
        if target == self.pose:
            self.finish('range_complete')
            return
        self.step += 1
        self.issue(target, self.cfg.dt)

    def finish(self, reason):
        self.active = False
        self.phase = 'complete' if reason == 'range_complete' else 'aborted'
        self.reason, self.ended = reason, self.clock()
        self.log.event('leg_stop', test=self.test, reason=reason, pose=self.pose,
                       pending_request=None if self.pending is None else self.pending['token'])

    def abort(self, reason):
        if self.active:
            self.finish(reason)

    def finalize(self):
        # Caller drains in-flight capture/save and ACK first, including on stop.
        if self.finalized or self.active or self.pending:
            return
        self.log.leg(dict(test=self.test, reason=self.reason, direction=self.direction,
            start_s=self.started, stop_s=self.ended, start_pan=self.start_pose[0],
            start_tilt=self.start_pose[1], end_pan=self.pose[0], end_tilt=self.pose[1],
            images=self.saved, detected=self.detected, kf_eligible=self.valid, note=self.note))
        self.finalized = True
