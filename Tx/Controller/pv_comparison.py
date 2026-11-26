"""Measurements and paired voltage summaries for one stationary Rx bench point."""
import csv
import json
import statistics
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from common.angle_demo import (MAX_VOLTAGE_GAP_NS, MIN_VOLTAGE_SAMPLES, VOLTAGE_SAMPLE_MS,
                               VOLTAGE_SETTLE_NS, VOLTAGE_WINDOW_NS)
from Rx.arc_bench import write_manifest


RAW_COLUMNS = ('seq', 'rx_monotonic_ns', 'bus_raw', 'pv_v', 'pc_receive_monotonic_ns',
               'pc_receive_unix_ns')
RESULT_COLUMNS = ('point_id', 'trial_id', 'method', 'laser_on', 'rx_pan_deg', 'rx_tilt_deg',
                  'status', 'success', 'reason',
                  'started_pc_unix_ns', 'end_pc_unix_ns', 'n', 'mean_v', 'sd_v',
                  'baseline_trial_id', 'delta_v', 'window_start_ns', 'window_end_ns', 'pose_transport')


class ComparisonRun:
    """Each instance owns a new point folder. Results from earlier points never mix."""

    def __init__(self, root_stage_dir, point_metadata):
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.folder = Path(root_stage_dir) / f'point_{stamp}_{uuid4().hex[:8]}'
        self.folder.mkdir(parents=True)
        self.point_id = point_metadata['point_id']
        write_manifest(self.folder, dict(schema_version=3, point=point_metadata,
                                         voltage_lsb_v=.004, sample_ms=VOLTAGE_SAMPLE_MS,
                                         settle_ns=VOLTAGE_SETTLE_NS, window_ns=VOLTAGE_WINDOW_NS,
                                         sd_definition='sample standard deviation (n-1)'))
        self.results = []
        self.baselines = {}
        self.trial_folder = None

    def _publish(self, rows):
        temporary = self.folder / 'summary.tmp'
        with temporary.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=RESULT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(self.folder / 'summary.csv')

    def start(self, method, metadata, *, laser_on):
        if method not in ('baseline', 'yolo', 'xyz') or self.trial_folder is not None:
            raise ValueError('측정 방식을 확인하고 현재 측정을 먼저 종료하세요.')
        if type(laser_on) is not bool:
            raise ValueError('Tx에서 확인한 레이저 ON/OFF 상태가 필요합니다.')
        illumination = 'on' if laser_on else 'off'
        trial_id = f'trial_{len(self.results)+1:03d}_{method}_{illumination}_{uuid4().hex[:12]}'
        trial_folder = self.folder / trial_id
        trial_folder.mkdir()
        pose_source = metadata.get('pose_source') or {}
        row = dict(point_id=self.point_id, trial_id=trial_id, method=method, laser_on=laser_on,
                   pose_transport='ssh' if method == 'baseline' else pose_source.get('transport'),
                   rx_pan_deg=metadata.get('rx_pan_deg'), rx_tilt_deg=metadata.get('rx_tilt_deg'),
                   status='running',
                   started_pc_unix_ns=time.time_ns())
        write_manifest(trial_folder, dict(schema_version=3, **row, settings=metadata))
        raw = (trial_folder / 'voltage.csv').open('x', encoding='utf-8', newline='')
        try:
            self.writer = csv.writer(raw)
            self.writer.writerow(RAW_COLUMNS)
            raw.flush()
            self._publish([*self.results, row])
        except (OSError, ValueError):
            raw.close()
            raise
        self.results.append(row)
        self.trial_folder, self.raw = trial_folder, raw
        self.volts = []
        self.sample_times = []
        self.last_seq = None
        self.token = self.window = self.invalid = None
        return trial_id

    @staticmethod
    def _write_event(folder, event):
        line = json.dumps(dict(event, recorded_pc_unix_ns=time.time_ns()),
                          ensure_ascii=False, allow_nan=False)
        with (folder / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(line + '\n')

    def log_station(self, event):
        self._write_event(self.folder, event)

    def log(self, event):
        if self.trial_folder is None:
            raise ValueError('진행 중인 전압 측정이 없습니다.')
        self._write_event(self.trial_folder, event)

    def save_frame(self, frame_bytes):
        if self.trial_folder is None:
            raise ValueError('진행 중인 전압 측정이 없습니다.')
        with (self.trial_folder / 'frame.jpg').open('xb') as stream:
            stream.write(frame_bytes)

    def _reject(self, message):
        self.invalid = message
        raise ValueError(message)

    def sample(self, event):
        if self.trial_folder is None:
            raise ValueError('진행 중인 전압 측정이 없습니다.')
        token = event.get('token')
        seq, ns, code = (event.get(key) for key in RAW_COLUMNS[:3])
        if (event.get('kind') != 'voltage' or self.window is not None
                or not isinstance(token, str) or not token or self.token not in (None, token)
                or any(type(v) is not int for v in (seq, ns, code)) or min(seq, ns) < 0
                or not 0 <= code <= 8191
                or (self.last_seq is not None and (seq <= self.last_seq or ns <= self.sample_times[-1]))):
            self._reject('INA219 표본의 범위, 순서 또는 측정 ID가 잘못되었습니다.')
        value = code * .004
        self.writer.writerow((seq, ns, code, f'{value:.3f}',
                              event.get('pc_receive_monotonic_ns'), event.get('pc_receive_unix_ns')))
        self.volts.append(value)
        self.sample_times.append(ns)
        self.token, self.last_seq = token, seq
        if code == 8191:
            self._reject('INA219 전압 표본이 포화됐습니다.')
        if len(self.volts) % 32 == 0:
            self.raw.flush()

    def measured(self, event):
        """Verify the Rx window against the raw samples already delivered to the PC."""
        if self.trial_folder is None:
            raise ValueError('진행 중인 전압 측정이 없습니다.')
        self.log(event)
        count, start, end = (event.get(k) for k in ('count', 'window_start_ns', 'window_end_ns'))
        if (self.window is not None or event.get('kind') != 'measured' or not self.token
                or event.get('token') != self.token or any(type(v) is not int for v in (count, start, end))
                or start < 0 or end - start != VOLTAGE_WINDOW_NS or count != len(self.volts)
                or count < MIN_VOLTAGE_SAMPLES):
            self._reject('Rx 측정 완료 응답과 PC에 저장한 표본 개수·측정 구간이 일치하지 않습니다.')
        times = [start, *self.sample_times, end]
        gap = max(b - a for a, b in zip(times, times[1:]))
        if (self.sample_times[0] < start or self.sample_times[-1] >= end or gap > MAX_VOLTAGE_GAP_NS
                or ('max_gap_ns' in event and event['max_gap_ns'] != gap)):
            self._reject('전압 측정 구간에 속하지 않는 표본 또는 허용치를 넘는 공백이 있습니다.')
        self.window = (start, end)

    def finish(self, success, reason):
        if self.trial_folder is None:
            raise ValueError('진행 중인 전압 측정이 없습니다.')
        self.raw.close()
        if success and (self.invalid or self.window is None):
            success, reason = False, self.invalid or 'Rx 전압 수집 완료 응답이 없습니다.'
        row = dict(self.results[-1], end_pc_unix_ns=time.time_ns(), n=len(self.volts),
                   status='complete' if success else 'failed', success=bool(success), reason=reason,
                   mean_v=None, sd_v=None, delta_v=None, baseline_trial_id=None)
        if self.window:
            row['window_start_ns'], row['window_end_ns'] = self.window
        if success:
            row['mean_v'] = statistics.mean(self.volts)
            row['sd_v'] = statistics.stdev(self.volts)
            baseline = row if row['method'] == 'baseline' else self.baselines.get(row['laser_on'])
            if baseline is not None:
                row.update(baseline_trial_id=baseline['trial_id'], delta_v=row['mean_v'] - baseline['mean_v'])
        self._publish([*self.results[:-1], row])
        self.results[-1] = row
        if success and row['method'] == 'baseline':
            self.baselines[row['laser_on']] = row
        self.trial_folder = None
        return row
