"""Lossless JPEG/metadata capture and explicit missing-observation timeline."""
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from PIL import Image
import io
from common.tx_bearing import commanded_bearing

OBSERVATION_FIELDS = (
    'test', 'step', 'direction', 'image', 'status', 'kf_eligible', 'error',
    'frame_id', 'pi_session', 'seq', 'receive_unix_ns', 'receive_s', 'submitted_s',
    'processed_s', 'saved_s', 'inference_ms', 'receive_to_result_s',
    'request_id', 'command_sent_s', 'command_unix_ns', 'ack_s', 'command_interval_s',
    'interval_excess_s', 'dwell_s', 'pan_command_deg', 'tilt_command_deg',
    'delta_pan_deg', 'delta_tilt_deg', 'actual_pan_deg', 'actual_tilt_deg',
    'arrival_verified', 'width', 'height', 'count', 'confidence',
    'u', 'v', 'x1', 'y1', 'x2', 'y2', 'alpha_rad', 'beta_rad', 'alpha_deg', 'beta_deg',
    'jpeg_sha256', 'note')
OBSERVATION_FIELDS += ('training_ready', 'training_exclusion', 'reward', 'error_px', 'intensity_mean',
                       'hit_23px', 'kf_nis', 'predicted_delta_alpha_deg', 'predicted_delta_beta_deg')
LEG_FIELDS = ('test', 'reason', 'direction', 'start_s', 'stop_s', 'start_pan',
              'start_tilt', 'end_pan', 'end_tilt', 'images', 'detected', 'kf_eligible', 'note')


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


class SweepLog:
    def __init__(self, folder, config):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        write_json(self.folder/'config.json', config)
        self.experience = None
        if config.get('moving_training'):
            from common.tx_moving import MovingConfig
            from Tx.Controller.sweep_training import SweepExperience
            self.experience = SweepExperience(MovingConfig(**config['moving_training']['tracking']),
                                               config.get('fixed_kalman'))
            for name in ('rl_samples.jsonl', 'rl_transitions.jsonl'):
                (self.folder/name).touch(exist_ok=False)
        for name, fields in [('observations.csv', OBSERVATION_FIELDS),
                             ('kalman_observations.csv', OBSERVATION_FIELDS), ('legs.csv', LEG_FIELDS)]:
            with (self.folder/name).open('x', newline='', encoding='utf-8') as stream:
                csv.DictWriter(stream, fieldnames=fields).writeheader()

    def event(self, name, **data):
        with (self.folder/'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(event=name, host_s=time.monotonic(),
                host_unix_ns=time.time_ns(), **data), ensure_ascii=False, allow_nan=False)+'\n')

    def append(self, name, fields, row):
        with (self.folder/name).open('a', newline='', encoding='utf-8') as stream:
            csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore').writerow(row)

    def leg(self, row):
        self.append('legs.csv', LEG_FIELDS, row)

    def capture(self, detector, cfg, frame, context, use_kalman):
        """Single bounded worker job. Original JPEG survives detector failures too."""
        row = dict(context, receive_s=frame[2], receive_unix_ns=frame[3],
                   seq=frame[1].get('seq'), pi_session=frame[1].get('session'),
                   actual_pan_deg=None, actual_tilt_deg=None, arrival_verified=False,
                   kf_eligible=False, error=None)
        result = None
        try:
            result = detector.infer(frame[0], cfg.confidence, int(cfg.class_id))
            image = result['image']
            row.update(width=image.width, height=image.height,
                       count=result['count'], inference_ms=result['inference_ms'])
            if image.size != (1296, 972):
                raise ValueError('Unexpected image dimensions')
            target = result['target']
            row['status'] = 'detected' if target else 'missing'
            if target:
                u, v = target['center']
                alpha, beta = commanded_bearing(u, v, row['pan_command_deg'], row['tilt_command_deg'])
                row.update(u=u, v=v, confidence=target['confidence'],
                           alpha_rad=alpha, beta_rad=beta, alpha_deg=math.degrees(alpha),
                           beta_deg=math.degrees(beta), kf_eligible=use_kalman)
                row.update(zip(('x1', 'y1', 'x2', 'y2'), target['box']))
        except Exception as exc:
            row.update(status='error', kf_eligible=False, error=str(exc))
        row.update(processed_s=time.monotonic(), jpeg_sha256=hashlib.sha256(frame[0]).hexdigest())
        row['receive_to_result_s'] = row['processed_s']-frame[2]
        name = (f"T{row['test']:03d}_S{row['step']:04d}"
                f"_pan{row['pan_command_deg']:+g}_tilt{row['tilt_command_deg']:+g}_{row['status']}.jpeg")
        row['image'] = name
        training = None
        if self.experience:
            try:
                training = self.experience.process(row)
                transition = training['transition']
                row.update(training_ready=bool(transition and transition['ready_for_replay']),
                           training_exclusion=training['exclusion_reason'])
                for key in ('reward', 'error_px', 'intensity_mean', 'hit_23px'):
                    row[key] = training[key]
                kf = training.get('kalman')
                if kf and kf['initialized']:
                    row.update(kf_nis=kf['nis'], predicted_delta_alpha_deg=kf['predicted_displacement_deg'][0],
                               predicted_delta_beta_deg=kf['predicted_displacement_deg'][1])
            except Exception as exc:
                row.update(status='error', error='training_metadata: '+str(exc), training_ready=False)
                name = name.replace('_detected.jpeg', '_error.jpeg').replace('_missing.jpeg', '_error.jpeg')
                row['image'] = name
        # Exclusive creation prevents a duplicate test/step from silently overwriting evidence.
        with (self.folder/name).open('xb') as stream:
            stream.write(frame[0])
        row['saved_s'] = time.monotonic()
        write_json((self.folder/name).with_suffix('.json'), dict(observation=row, frame_metadata=frame[1], training=training,
            clock_note='Host monotonic timestamps share one clock. Pi wall/sensor clocks are not synchronized.',
            annotation_note='YOLO predictions are not manually verified ground-truth labels.'))
        self.append('observations.csv', OBSERVATION_FIELDS, row)
        if row['kf_eligible']:
            self.append('kalman_observations.csv', OBSERVATION_FIELDS, row)
        if training is not None:
            with (self.folder/'rl_samples.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(training, ensure_ascii=False, allow_nan=False)+'\n')
            if training['transition'] is not None:
                with (self.folder/'rl_transitions.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(training['transition'], ensure_ascii=False, allow_nan=False)+'\n')
        if result is None:
            # A failed inference must still display the saved raw frame when decodable.
            try:
                image = Image.open(io.BytesIO(frame[0])).convert('RGB')
            except (ValueError, OSError):
                image = None
        else:
            image = result['image']
        return row, image
