"""Reuse sweep observations as causal moving-SAC experience, without hardware I/O."""
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from common.tx_bearing import BEARING_MODEL
from common.tx_moving import MovingConfig, BearingKalman, load_kalman_config, moving_observation
from common.tx_tracking import BeamMap, observation
from Tx.Controller.stationary_learning import SACLearner, RunLog, digest, dump


class SweepExperience:
    """One state per saved observation; never bridge missing samples or legs.

Without a calibrated filter the 8 raw features and reward are still saved.
Final 10D observations are later reconstructed by the same causal code with
one frozen KF configuration. No future-frame smoothing is used as input.
"""
    def __init__(self, cfg, kalman_config=None):
        cfg.validate()
        self.cfg, self.beam = cfg, BeamMap(cfg)
        self.kalman = BearingKalman(kalman_config) if kalman_config is not None else None
        self.test = self.previous = self.previous_row = None

    def process(self, row):
        if self.test != row['test']:
            self.test, self.previous, self.previous_row = row['test'], None, None
            if self.kalman:
                self.kalman.reset()
        detected = row['status'] == 'detected'
        kf = (self.kalman.step([row['alpha_rad'], row['beta_rad']] if detected else None,
                              row['receive_s'], self.cfg.dt) if self.kalman else None)
        record = dict(test=row['test'], step=row['step'], frame_id=row.get('frame_id'),
                      image=row.get('image'), receive_s=row['receive_s'],
                      raw8=None, normalized8=None, raw10=None, normalized10=None,
                      reward=None, error_px=None, intensity_mean=None, hit_23px=None,
                      kalman=kf, transition=None, exclusion_reason='initial_observation')
        if not detected:
            record['exclusion_reason'] = row['status']
            self.previous, self.previous_row = None, row.copy()
            return record
        command = np.array([row['pan_command_deg'], row['tilt_command_deg']], dtype=float)
        delta = np.array([row.get('delta_pan_deg', 0), row.get('delta_tilt_deg', 0)], dtype=float)
        if row['step'] == 0:
            delta[:] = 0
        target = dict(center=[row['u'], row['v']], box=[row[k] for k in ('x1', 'y1', 'x2', 'y2')])
        sample = SimpleNamespace(target=target)
        consecutive = (self.previous is not None and self.previous_row['step']+1 == row['step'])
        previous_error = self.previous['raw8'][:2] if consecutive else None
        raw, normalized = observation(sample, command, delta, previous_error, self.cfg)
        intensity, _ = self.beam.mean(target['box'])
        error = float(np.linalg.norm(raw[:2]))
        record.update(raw8=raw.tolist(), normalized8=normalized.tolist(), error_px=error,
                      intensity_mean=intensity, hit_23px=error <= self.cfg.hit_px,
                      reward=-self.cfg.error_weight*error/self.cfg.reward_distance+self.cfg.aim_weight*intensity)
        state_valid = (np.isfinite(raw).all() and np.all(command >= self.cfg.low)
                       and np.all(command <= self.cfg.high) and np.all(np.abs(delta) <= self.cfg.action_scale)
                       and row.get('width') == self.cfg.width and row.get('height') == self.cfg.height
                       and row.get('confidence', 0) >= self.cfg.confidence)
        if state_valid and kf and kf['initialized']:
            raw10, normalized10 = moving_observation(sample, command, delta, previous_error,
                                                    kf['predicted_displacement_deg'], self.cfg)
            record.update(raw10=raw10.tolist(), normalized10=normalized10.tolist())
        if not state_valid:
            record['exclusion_reason'] = 'outside_moving_contract'
        elif consecutive:
            sent = row.get('command_sent_s')
            previous = self.previous_row
            age = None if sent is None else sent-previous['receive_s']
            if age is None or not 0 <= age <= self.cfg.dt:
                record['exclusion_reason'] = 'stale_or_unknown_pre_action_observation'
            elif (sent < previous.get('processed_s', previous['receive_s'])
                  or row['receive_s'] <= row.get('ack_s', sent)
                  or row['receive_s']-sent < self.cfg.dt):
                record['exclusion_reason'] = 'invalid_action_observation_order'
            elif not np.allclose(delta, command-[previous['pan_command_deg'], previous['tilt_command_deg']], atol=1e-9):
                record['exclusion_reason'] = 'command_delta_mismatch'
            else:
                # Fixed sweep controller: the exact applied integer command is
                # represented by its inverse scale, including endpoint clamping.
                action = delta/self.cfg.action_scale
                record['transition'] = dict(episode=row['test'], step=row['step'],
                    before_image=previous.get('image'), after_image=row.get('image'),
                    observation8=self.previous['raw8'], next_observation8=raw.tolist(),
                    observation=self.previous['normalized10'], next_observation=record['normalized10'],
                    action=action.tolist(), applied_delta_deg=delta.tolist(), reward=record['reward'],
                    ready_for_replay=record['normalized10'] is not None and self.previous['normalized10'] is not None,
                    terminated=False, truncated=False)
                record['exclusion_reason'] = None if record['transition']['ready_for_replay'] else 'needs_fixed_kalman_reprocessing'
        elif row['step'] != 0:
            record['exclusion_reason'] = 'missing_or_nonconsecutive_pre_observation'
        self.previous = record if state_valid else None
        self.previous_row = row.copy()
        return record


def source_rows(folder):
    """Read source metadata in CSV order, never filesystem filename order."""
    folder = Path(folder)
    with (folder/'observations.csv').open(encoding='utf-8', newline='') as stream:
        for index in csv.DictReader(stream):
            name = index['image']
            if Path(name).name != name:
                raise ValueError('수집 이미지 경로가 결과 폴더 밖을 가리킵니다.')
            data = (folder/name).with_suffix('.json').read_bytes()
            meta = json.loads(data)
            row = meta['observation']
            if row['image'] != name or str(row['test']) != index['test'] or str(row['step']) != index['step']:
                raise ValueError('수집 CSV와 이미지 메타데이터가 일치하지 않습니다.')
            servo = meta['frame_metadata'].get('servo_at_capture_start') or {}
            command = servo.get('commanded') or {}
            if (servo.get('simulated') is not False or servo.get('available') is not True
                    or not row.get('request_id') or command.get('command_id') != row['request_id']
                    or command.get('pan') != row['pan_command_deg'] or command.get('tilt') != row['tilt_command_deg']):
                raise ValueError('촬영 프레임과 서보 명령의 연결 정보를 확인할 수 없습니다.')
            row['_metadata_sha256'] = hashlib.sha256(data).hexdigest()
            yield row


def build_dataset(folders, cfg, kalman_config, cancel=None):
    """Combine compatible sessions, reset KF per leg and reject duplicate inputs."""
    paths = sorted({Path(p).resolve() for p in folders})
    transitions, sources, excluded, fingerprints = [], [], Counter(), set()
    source_episode_count = 0
    for folder in paths:
        if cancel is not None and cancel.is_set():
            raise ValueError('초기 데이터 재처리가 취소되었습니다.')
        config = json.loads((folder/'config.json').read_text(encoding='utf-8'))
        if config.get('schema') not in ('m5-1-sweep-v1', 'm5-1-sweep-v2'):
            raise ValueError(f'편도 수집 결과 폴더가 아닙니다: {folder}')
        settings = config['settings']
        if settings['dt'] != cfg.dt or config.get('bearing_model') != BEARING_MODEL:
            raise ValueError(f'제어주기 또는 각도 보정값이 다른 수집 결과입니다: {folder}')
        for key in ('confidence', 'class_id', 'speed', 'acc'):
            if settings[key] != getattr(cfg, key):
                raise ValueError(f'수집 설정 불일치 {key}: {folder}')
        fingerprints_for_source = (digest(folder/'config.json'), digest(folder/'observations.csv'))
        if fingerprints_for_source in fingerprints:
            raise ValueError(f'같은 수집 결과의 복사본이 중복 선택되었습니다: {folder}')
        fingerprints.add(fingerprints_for_source)
        processor, current_episode, last_step, last_time = SweepExperience(cfg, kalman_config), None, None, None
        closed_episodes = set()
        counts = Counter()
        metadata_digest = hashlib.sha256()
        source_start = len(transitions)
        for row in source_rows(folder):
            metadata_digest.update(row['_metadata_sha256'].encode('ascii'))
            if cancel is not None and cancel.is_set():
                raise ValueError('초기 데이터 재처리가 취소되었습니다.')
            if row['test'] != current_episode:
                if row['test'] in closed_episodes:
                    raise ValueError('수집 파일에 편도 순서가 뒤섞였습니다.')
                if current_episode is not None:
                    closed_episodes.add(current_episode)
                    if len(transitions) > source_start and transitions[-1]['source_episode'] == current_episode:
                        transitions[-1]['truncated'] = True
                current_episode, last_step, last_time = row['test'], None, None
            if (last_step is not None and row['step'] <= last_step
                    or last_time is not None and row['receive_s'] <= last_time):
                raise ValueError('편도 관측의 step 또는 시각이 증가하지 않습니다.')
            last_step, last_time = row['step'], row['receive_s']
            if row['step'] > 0:
                counts['observed_action_results'] += 1
            record = processor.process(row)
            transition = record['transition']
            if transition and transition['ready_for_replay']:
                transition.update(source=str(folder), source_episode=row['test'])
                transitions.append(transition); counts['valid'] += 1
            else:
                counts[record['exclusion_reason']] += 1
                excluded[record['exclusion_reason']] += 1
        if len(transitions) > source_start:
            transitions[-1]['truncated'] = True
        episodes_in_source = len(closed_episodes)+(current_episode is not None)
        source_episode_count += episodes_in_source
        sources.append(dict(folder=str(folder), config_sha256=fingerprints_for_source[0],
                            observations_sha256=fingerprints_for_source[1],
                            metadata_sha256=metadata_digest.hexdigest(), episodes=episodes_in_source, counts=dict(counts)))
    if not transitions:
        raise ValueError('사용 가능한 유효 transition이 없습니다. 범위·검출·명령 전후 시각을 확인하세요.')
    return transitions, dict(sources=sources, excluded=dict(excluded), valid=len(transitions),
                             source_episode_count=source_episode_count,
                             observed_action_results=sum(s['counts'].get('observed_action_results', 0) for s in sources),
                             tracking=asdict(cfg), normalization=cfg.normalization, kalman=kalman_config,
                             reconstruction='causal filtering, reset each leg, no future smoothing')


def pretrain(folders, kalman_config, cfg, output, device='cpu', cancel=None, updates_per_transition=1.):
    transitions, report = build_dataset(folders, cfg, kalman_config, cancel)
    learner = SACLearner(cfg, device=device)
    learner.experiment_context = dict(kalman=kalman_config, bearing_model=BEARING_MODEL,
                                     imported_sources=report['sources'], normalization=cfg.normalization)
    if cancel is not None:
        learner.cancel = cancel
    log = RunLog(output, dict(schema='moving-sweep-pretraining-v1', **report))
    try:
        for transition in transitions:
            if learner.cancel.is_set():
                raise ValueError('초기 데이터 학습이 취소되었습니다.')
            learner.add(transition['observation'], transition['action'], transition['reward'],
                        transition['next_observation'], transition['truncated'])
        # Keep the exact imported experience and provenance beside the paired checkpoint.
        with (log.folder/'imported_transitions.jsonl').open('x', encoding='utf-8') as stream:
            for transition in transitions:
                stream.write(json.dumps(transition, ensure_ascii=False, allow_nan=False)+'\n')
        learner.executed_actions = report['observed_action_results']
        # This is one offline update batch for several real one-way episodes.
        # Do not report the entire imported set as a single flight episode.
        learner.episodes = report['source_episode_count']-1
        result = learner.finish_episode(dict(episode=report['source_episode_count'], reason='initial_dataset',
                                              new_transitions=len(transitions)), log, updates_per_transition)
        dump(log.folder/'import_report.json', dict(report, result=result))
        return result
    finally:
        log.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='편도 수집 결과를 고정 KF로 재처리하여 이동 SAC 초기 모델 학습')
    parser.add_argument('--source', type=Path, action='append', required=True, help='수집 세션 폴더; 여러 번 지정 가능')
    parser.add_argument('--kalman', type=Path, required=True, help='사전 실험에서 확정한 KF 설정 JSON')
    parser.add_argument('--period', type=float, default=.720)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    cfg = MovingConfig(dt=args.period); cfg.validate()
    kalman, _ = load_kalman_config(args.kalman)
    output = args.output or Path('captures/M5-1')/('pretraining_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    result = pretrain(args.source, kalman, cfg, output, args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
