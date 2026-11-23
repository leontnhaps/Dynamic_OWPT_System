"""Fixed-source optical stimulus for a receiver bench. Time slots use the PC clock;
the existing Tx Pi pattern engine times the individual bits and reports GPIO calls.
"""
import math
import time
import uuid

from common.arc_protocol import VERSION, stimulus_bits, timing
from common.laser_pattern import pattern_settings


MAX_DISPATCH_LATE_S = 0.10
OFF_TIMEOUT_S = 5.0


class ArcSender:
    """One operator-started run. All output goes through the GUI's checked send."""

    def __init__(self, send, record, status, bit_ms, period_s, steps, *, pair_id, mode, clock=time.monotonic):
        timing(bit_ms, period_s)
        if type(steps) is not int or not 1 <= steps <= 180:
            raise ValueError('마지막 단계는 1..180 정수입니다.')
        if not pair_id.strip() or len(pair_id) > 80 or not pair_id.isprintable() or mode not in ('A', 'B'):
            raise ValueError('Rx와 같은 비교 ID와 조건(A/B)을 입력하세요.')
        self.pair_id, self.mode = pair_id.strip(), mode
        self.bit_ms, self.period_s, self.steps = bit_ms, period_s, steps
        self.send, self.record, self.status, self.clock = send, record, status, clock
        self.run_id = uuid.uuid4().hex
        self.phase = 'idle'
        self.seq = 0
        self.reason = ''
        self.off_confirmed = False
        self.token = None
        self.epoch = None
        self.deadline = None

    @property
    def active(self):
        return self.phase not in ('idle', 'done')

    def log(self, kind, **details):
        self.record(dict(event=kind, pc_monotonic_s=self.clock(), seq=self.seq, **details))

    def start(self):
        self.log('arc_start', run_id=self.run_id, bit_ms=self.bit_ms, period_s=self.period_s,
                 steps=self.steps, pair_id=self.pair_id, mode=self.mode, protocol_version=VERSION,
                 max_dispatch_late_s=MAX_DISPATCH_LATE_S)
        self._off('arming')

    def _off(self, phase):
        self.phase = phase
        self.token = self.run_id + '-' + phase
        self.deadline = self.clock() + OFF_TIMEOUT_S
        self.off_confirmed = False
        if not self.send(dict(cmd='laser', value=0, request_id=self.token)):
            self.complete(False, self.reason or 'off_send_failed')
            return
        self.status('초기 OFF 응답 대기' if phase == 'arming' else '종료 OFF 응답 대기')

    def stop(self, reason):
        if self.active and self.phase != 'stopping':
            self.reason = reason
            self._off('stopping')

    def complete(self, confirmed, reason):
        if not confirmed and reason == 'completed':
            reason = 'final_off_unconfirmed'
        self.phase = 'done'
        self.token = None
        self.off_confirmed = confirmed
        self.reason = reason
        self.status(f'반원 송신 종료 · {reason} · GPIO OFF ' + ('확인' if confirmed else '미확인'))
        self.log('arc_end', reason=reason, gpio_off_confirmed=confirmed)

    def handle_event(self, event):
        if not self.active or not self.token or event.get('request_id') != self.token:
            return
        kind = event.get('event')
        if self.phase in ('arming', 'stopping'):
            if kind == 'error':
                self.complete(False, self.reason or 'off_error')
            elif kind == 'laser':
                if event.get('laser_gpio_level') != 0:
                    self.complete(False, self.reason or 'off_unconfirmed')
                elif self.phase == 'stopping':
                    self.complete(True, self.reason)
                else:
                    self.epoch = self.clock() + 1.0
                    self.phase = 'waiting'
                    self.token = None
                    self.log('arc_armed', first_due_s=self.epoch)
                    self.status('시작 프레임 대기')
            return
        if self.phase != 'sending' or kind not in ('laser_pattern', 'error'):
            return
        self.log('arc_response', response=event)
        if kind == 'laser_pattern' and event.get('state') == 'started':
            return
        if kind == 'error' or event.get('state') != 'completed':
            self.stop('tx_' + str(event.get('state') or event.get('message') or 'error'))
            return
        if self.clock() > self.deadline:
            self.stop('pattern_ack_timeout')
            return
        off = event.get('off_call_done_ms')
        sent = self.pattern['bits']
        trace = event.get('trace')
        correct_trace = (isinstance(trace, list) and len(trace) == len(sent)
                         and all(isinstance(row, dict) and row.get('index') == i
                                 and row.get('level') == int(sent[i]) for i, row in enumerate(trace)))
        if (type(off) not in (int, float) or not math.isfinite(off) or off < 0
                or not correct_trace or event.get('settings') != self.pattern):
            self.stop('tx_incomplete_trace')
            return
        if self.seq == self.steps:
            self.stop('completed')
        else:
            self.seq += 1
            self.token = None
            self.phase = 'waiting'

    def tick(self, ready):
        if not self.active:
            return
        now = self.clock()
        if self.phase in ('arming', 'stopping'):
            if now > self.deadline:
                self.complete(False, self.reason or 'off_ack_timeout')
            return
        if not ready:
            self.stop('tx_not_ready')
            return
        if self.phase == 'sending':
            if now > self.deadline:
                self.stop('pattern_ack_timeout')
            return
        due = self.epoch + self.seq * self.period_s
        if now < due:
            return
        self.pattern = pattern_settings(dict(bits=stimulus_bits(self.seq, self.bit_ms),
                                             bit_ms=self.bit_ms, repeats=1))
        self.token = f'{self.run_id}-frame-{self.seq}'
        # Disk writes can block (for example on a synced captures directory).
        # Timestamp and check immediately before entering the existing GUI send.
        dispatch_at = self.clock()
        if dispatch_at - due > MAX_DISPATCH_LATE_S:
            self.stop('missed_slot')
            return
        # Never start a compressed successor to recover a missed slot.
        self.deadline = due + self.period_s + MAX_DISPATCH_LATE_S
        self.phase = 'sending'
        if not self.send(dict(cmd='laser_pattern', request_id=self.token, **self.pattern)):
            self.stop('pattern_send_failed')
            return
        self.log('arc_dispatch', request_id=self.token, due_s=due, dispatch_s=dispatch_at,
                 late_s=dispatch_at-due, settings=self.pattern)
        self.status(f'반원 광프레임 {self.seq}/{self.steps}')
