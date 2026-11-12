"""M5 live-run records; plotting is explicitly called from the save worker."""
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import threading
import time
import uuid

import numpy as np

from Tx.Controller.stationary_learning import RunLog, json_value


LOG_SCHEMA = 'm5-3-moving-live-v1'


def _finite_json(value, invalid, path='$'):
    """Keep invalid measurements identifiable without emitting nonstandard JSON."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        invalid.append(dict(path=path, value=str(value)))
        return None
    if isinstance(value, dict):
        return {key: _finite_json(item, invalid, f'{path}.{key}') for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item, invalid, f'{path}[{i}]') for i, item in enumerate(value)]
    return value


def generate_plots(folder):
    """Best-effort derived graphs. Call in a worker after CSV/checkpoint saving.

    A missing plotting dependency or a graph write failure cannot undo successful
    training/checkpoint saving. Raw CSV files are never changed by this function.
    """
    try:
        from tools.analysis.plot_moving_training import generate_plots as plot
        return plot(folder)
    except Exception as exc:
        return dict(files=[], errors=[f'{type(exc).__name__}: {exc}'])


class MovingRunLog(RunLog):
    # Copy each list: M3's schema and API must not change.
    fields = {name: list(fields) for name, fields in RunLog.fields.items()}
    fields['episodes'] += ['reward_sum', 'reward_mean', 'reward_count']

    def __init__(self, folder, config):
        super().__init__(folder, dict(config, moving_log_schema=LOG_SCHEMA))
        self._frame_lock = threading.Lock()
        self._closed = False
        self._frame_folder = self.folder/'frames'
        self._frame_folder.mkdir()
        self._observations = (self.folder/'observations.jsonl').open('w', encoding='utf-8')

    def record_frame(self, episode, phase, sample, frame_metadata, jpeg, *, step=None,
                     detection=None, status=None):
        """Save one inferred original JPEG, sidecar, JSONL and detection CSV row.

        Run in the inference worker. ``sample.processed`` must be captured before
        this call, so storage time never makes a stale observation appear fresh.
        ``status`` is optional processing status; absent it means detector presence
        only. Control/KF eligibility and raw10 are recorded separately in events.
        Returned paths are relative to this run. Frame ID + episode joins them to
        steps/events; observation_id distinguishes repeated IDs without overwrite.
        """
        if not isinstance(jpeg, (bytes, bytearray, memoryview)):
            raise TypeError('jpeg must be the original frame bytes')
        raw = asdict(sample) if is_dataclass(sample) else dict(vars(sample))
        target = raw.get('target')
        status = status or ('detected' if target is not None else 'missing')
        observation_id = uuid.uuid4().hex
        image_name = f'frames/E{int(episode):06d}_{observation_id}_{status if status in ("missing", "detected") else "invalid"}.jpeg'
        identity = dict(observation_id=observation_id, frame_id=raw.get('frame_id'),
                        image=image_name, sidecar=str(Path(image_name).with_suffix('.json')),
                        sha256=hashlib.sha256(jpeg).hexdigest())
        # A distinct frame lock keeps long JPEG writes out of the shared event lock.
        with self._frame_lock:
            if self._closed:
                raise ValueError('Moving run log is closed')
            storage_started_s = time.monotonic()
            with (self.folder/image_name).open('xb') as stream:
                stream.write(jpeg)
            jpeg_saved_s = time.monotonic()
            row = dict(episode=episode, phase=phase, frame_id=raw.get('frame_id'),
                       seq=raw.get('seq'), receive_unix_ns=raw.get('receive_unix_ns'),
                       receive_s=raw.get('received'), processed_s=raw.get('processed'),
                       width=raw.get('width'), height=raw.get('height'),
                       simulated=raw.get('simulated'), status=status,
                       count=raw.get('count'), inference_ms=raw.get('inference_ms'))
            if isinstance(target, dict):
                row['confidence'] = target.get('confidence')
                center, box = target.get('center'), target.get('box')
                if isinstance(center, (list, tuple, np.ndarray)) and len(center) == 2:
                    row.update(zip(('u', 'v'), center))
                if isinstance(box, (list, tuple, np.ndarray)) and len(box) == 4:
                    row.update(zip(('x1', 'y1', 'x2', 'y2'), box))
            record = dict(schema=LOG_SCHEMA, **identity, episode=episode, step=step,
                          phase=phase, status=status, sample=raw, frame_metadata=frame_metadata,
                          detection={key: value for key, value in (detection or {}).items() if key != 'image'},
                          storage_started_s=storage_started_s, jpeg_saved_s=jpeg_saved_s,
                          note='Original JPEG; detector outputs are not ground-truth labels. '
                               'Control eligibility and KF state are in events.jsonl. '
                               'Commanded servo angles do not verify actual angles or arrival.')
            invalid = []
            record = _finite_json(record, invalid)
            record['invalid_numeric_fields'] = invalid
            with (self.folder/identity['sidecar']).open('x', encoding='utf-8') as stream:
                json.dump(record, stream, default=json_value, ensure_ascii=False, indent=2, allow_nan=False)
            # This timestamp is after both raw artifacts, before appending the index.
            record['artifacts_saved_s'] = time.monotonic()
            encoded = json.dumps(record, default=json_value, ensure_ascii=False, allow_nan=False)
            row = _finite_json(row, [])
            with self.lock:
                self._observations.write(encoded+'\n')
                self._observations.flush()
                self.writers['detections'].writerow(row)
                self.files['detections'].flush()
            return dict(identity, storage_finished_s=time.monotonic())

    def generate_plots(self):
        """Flush and plot from the save worker; return errors without raising."""
        try:
            with self.lock:
                for stream in self.files.values():
                    if not stream.closed:
                        stream.flush()
            report = generate_plots(self.folder)
        except Exception as exc:
            report = dict(files=[], errors=[f'{type(exc).__name__}: {exc}'])
        try:
            self.event('plots_saved' if not report['errors'] else 'plots_incomplete', **report)
        except Exception:
            # A closed/full event log must not turn a saved checkpoint into failure.
            pass
        return report

    def close(self):
        with self._frame_lock:
            if self._closed:
                return
            self._closed = True
            with self.lock:
                self._observations.close()
            super().close()
