"""Independently audit a saved SAC bundle's live export without hardware I/O.

The selected checkpoint is read-only. Synthetic transitions and gradient updates
are confined to a new audit directory, never to the selected model. This script
deliberately does not import or reuse the converter's preservation checks.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
import zipfile
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

REPLAY_FIELDS = ('observations', 'next_observations', 'actions', 'rewards', 'dones', 'timeouts')
EXPECTED_COUNTS = {
    308: {'rows': 308, 'true_terminals': 0, 'timeouts': 43},
    10309: {'rows': 10309, 'true_terminals': 112, 'timeouts': 1421},
    20316: {'rows': 20316, 'true_terminals': 113, 'timeouts': 2871},
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def snapshot_files(folder):
    return {str(p.relative_to(folder)): sha256(p.read_bytes())
            for p in sorted(Path(folder).rglob('*')) if p.is_file()}


def tuple_tree(value):
    return tuple(tuple_tree(v) for v in value) if isinstance(value, list) else value


def jsonable(value):
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def equal_tree(left, right, name):
    import torch
    if isinstance(left, torch.Tensor):
        require(isinstance(right, torch.Tensor), name + ': tensor type')
        require(left.dtype == right.dtype and left.shape == right.shape
                and torch.equal(left.cpu(), right.cpu()), name + ': tensor differs')
    elif isinstance(left, np.ndarray):
        require(isinstance(right, np.ndarray), name + ': array type')
        require(left.dtype == right.dtype and left.shape == right.shape
                and np.array_equal(left, right), name + ': array differs')
    elif isinstance(left, dict):
        require(isinstance(right, dict) and set(left) == set(right), name + ': keys')
        for key in left:
            equal_tree(left[key], right[key], name + '/' + str(key))
    elif isinstance(left, (tuple, list)):
        require(isinstance(right, (tuple, list)) and len(left) == len(right), name + ': length')
        for i, (a, b) in enumerate(zip(left, right)):
            equal_tree(a, b, name + '/' + str(i))
    else:
        require(left == right, name + ': value differs')


def finite_tree(value, name):
    import torch
    if isinstance(value, torch.Tensor):
        require(bool(torch.isfinite(value).all()), name + ': nonfinite tensor')
    elif isinstance(value, np.ndarray):
        require(bool(np.isfinite(value).all()), name + ': nonfinite array')
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, name + '/' + str(key))
    elif isinstance(value, (tuple, list)):
        for i, item in enumerate(value):
            finite_tree(item, name + '/' + str(i))
    elif isinstance(value, (float, np.floating)):
        require(bool(np.isfinite(value)), name + ': nonfinite number')


def model_trees(model):
    return {
        'policy_including_critic_target': model.policy.state_dict(),
        'actor_optimizer': model.actor.optimizer.state_dict(),
        'critic_optimizer': model.critic.optimizer.state_dict(),
        'entropy_optimizer': model.ent_coef_optimizer.state_dict(),
        'log_entropy_coefficient': model.log_ent_coef,
    }


def load_source(path):
    with zipfile.ZipFile(path) as archive:
        require(archive.testzip() is None, 'Source ZIP CRC failure')
        names = archive.namelist()

        def entry(suffix):
            matches = [name for name in names if name.endswith('/' + suffix)]
            require(len(matches) == 1, 'Source must contain one ' + suffix)
            return archive.read(matches[0])

        state = json.loads(entry('checkpoint/state.json'))
        manifest = json.loads(entry('checkpoint/manifest.json'))
        for filename, digest in manifest['files'].items():
            require(sha256(entry('checkpoint/' + filename)) == digest,
                    'Source manifest mismatch: ' + filename)
        replay = dict(np.load(io.BytesIO(entry('checkpoint/replay.npz')), allow_pickle=False))
        rng = dict(np.load(io.BytesIO(entry('checkpoint/rng.npz')), allow_pickle=False))
        runtime_names = [n for n in names if n.endswith('/checkpoint/experiment_runtime.json')]
        runtime = json.loads(archive.read(runtime_names[0])) if runtime_names else None
        numpy_state = state['numpy_rng']
        if isinstance(numpy_state, dict):
            numpy_state = [numpy_state['algorithm'], rng['numpy_keys'].tolist(),
                           numpy_state['pos'], numpy_state['has_gauss'], numpy_state['cached_gaussian']]
        return {
            'state': state, 'manifest': manifest, 'replay': replay,
            'model_bytes': entry('checkpoint/model.zip'), 'kalman_bytes': entry('kf_receive_time.json'),
            'torch_rng': rng['cpu'] if 'cpu' in rng else rng['torch_cpu'],
            'numpy_rng': numpy_state, 'python_rng': state['python_rng'],
            'reset_rng': state.get('reset_action_rng'),
            'source_seed': json.loads(entry('source_config.json'))['seed'],
            'runtime': runtime,
        }


def restore_global_rng(source):
    import torch
    torch.set_rng_state(torch.from_numpy(source['torch_rng'].copy()))
    nr = source['numpy_rng']
    np.random.set_state((nr[0], np.array(nr[1], dtype=np.uint32), *nr[2:]))
    random.setstate(tuple_tree(source['python_rng']))


def check_restored_rng(source, state, learner):
    import torch
    equal_tree(source['torch_rng'], torch.get_rng_state().cpu().numpy(), 'restored/torch_rng')
    equal_tree(source['numpy_rng'], jsonable(np.random.get_state()), 'restored/numpy_rng')
    equal_tree(source['python_rng'], jsonable(random.getstate()), 'restored/python_rng')
    equal_tree(state['reset_action_rng'], learner.rng.bit_generator.state, 'restored/reset_rng')


def action_sequence(model, observations, deterministic):
    return np.stack([model.predict(obs, deterministic=deterministic)[0] for obs in observations])


class AuditLog:
    """No hardware and no production log-schema assumptions; records learner calls."""
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        self.events, self.rows, self.episodes = [], [], []

    def event(self, kind, **values):
        self.events.append(dict(kind=kind, **values))

    def row(self, kind, values):
        finite_tree(values, 'update_log')
        self.rows.append(dict(kind=kind, **values))

    def episode(self, values):
        self.episodes.append(values)


def train_clone(learner, observations, source_replay, output):
    from Tx.Controller.stationary_learning import SACLearner, SAC_SETTINGS, checkpoint_info
    from common.moving_paths import load_moving_inputs
    from common.tx_moving import moving_loss_reward

    original_context = json.loads(json.dumps(learner.experiment_context))
    base_transitions, base_updates, base_episodes = learner.total_transitions, learner.updates, learner.episodes
    position = learner.policy.replay_buffer.pos
    request = np.asarray(source_replay['actions'][0, 0], dtype=np.float32)
    reward = moving_loss_reward(float(request[0]) * learner.cfg.pan_delta_deg, learner.cfg)
    require(learner.add(observations[0], request, reward, np.zeros(10, dtype=np.float32),
                        truncated=False, terminated=True), 'Synthetic terminal add failed')
    replay = learner.policy.replay_buffer
    require(float(replay.dones[position, 0]) == 1 and float(replay.timeouts[position, 0]) == 0,
            'Synthetic loss must be a true terminal, never a timeout')
    learner.executed_actions += 1
    log = AuditLog(output / 'synthetic_resume')
    summary = learner.finish_episode(dict(episode=base_episodes + 1, reason='target_lost', new_transitions=1),
                                     log, ratio=1.)
    require(summary['completed_updates'] == 1 and learner.updates == base_updates + 1,
            'First resumed gradient update did not advance by exactly one')
    require(learner.total_transitions == base_transitions + 1, 'Synthetic transition count mismatch')
    finite_tree(model_trees(learner.policy), 'after_first_update')
    first_checkpoint = Path(summary['checkpoint'])
    first_inputs = load_moving_inputs(first_checkpoint, output / 'kf_receive_time.json', mode='resume')
    reloaded = SACLearner(first_inputs.cfg, mode='resume', checkpoint=first_checkpoint, device='cpu')
    equal_tree(model_trees(learner.policy), model_trees(reloaded.policy), 'first_save_reload')
    require(reloaded.experiment_context == original_context, 'Provenance lost after first reload')
    require(reloaded.episodes == base_episodes + 1 and reloaded.updates == base_updates + 1,
            'First reload counters mismatch')
    # A second gradient update on the same saved replay proves training can resume
    # across process boundaries without inventing a second experience.
    reloaded.policy.train(gradient_steps=1, batch_size=SAC_SETTINGS['batch_size'])
    reloaded.updates = int(reloaded.policy._n_updates)
    reloaded.policy.policy.set_training_mode(False)
    require(reloaded.updates == base_updates + 2, 'Second resumed update did not advance')
    require(reloaded.total_transitions == base_transitions + 1, 'Second update fabricated experience')
    finite_tree(model_trees(reloaded.policy), 'after_second_update')
    second_metrics = dict(reloaded.policy.logger.name_to_value)
    finite_tree(second_metrics, 'second_update_metrics')
    second_checkpoint = reloaded.save_checkpoint(log.folder)
    _, second_state, _ = checkpoint_info(second_checkpoint)
    require(second_state['experiment_context'] == original_context, 'Provenance lost after second save')
    require(second_state['updates'] == base_updates + 2, 'Second save lost update counter')
    training_log = dict(events=log.events, updates=log.rows, episodes=log.episodes,
                        second_update_metrics=second_metrics)
    (output / 'synthetic_training_log.json').write_text(json.dumps(jsonable(training_log), indent=2), encoding='utf-8')
    return dict(synthetic_true_terminals_added=1, synthetic_timeout=False, gradient_updates=2,
                transitions_before=base_transitions, transitions_after=reloaded.total_transitions,
                updates_before=base_updates, updates_after=reloaded.updates,
                checkpoint_save_reload_verified=True, optimizer_tensors_finite=True,
                provenance_preserved=True, first_checkpoint=str(first_checkpoint),
                second_checkpoint=str(second_checkpoint))


def audit(source_zip, checkpoint, kalman, output):
    import torch
    from stable_baselines3 import SAC
    from common.moving_paths import load_moving_inputs
    from common.tx_bearing import BEARING_MODEL
    from Tx.Controller.stationary_learning import SACLearner

    source_zip, checkpoint, kalman, output = map(lambda p: Path(p).resolve(),
                                                (source_zip, checkpoint, kalman, output))
    require(not output.exists(), 'Audit output directory must be new: ' + str(output))
    require(not output.is_relative_to(checkpoint), 'Audit output cannot be inside selected checkpoint')
    output.mkdir(parents=True)
    before_source_hash = sha256(source_zip.read_bytes())
    before_export = snapshot_files(checkpoint)
    source = load_source(source_zip)
    state = read_json(checkpoint / 'state.json')
    source_state = source['state']
    n = int(source_state['total_transitions'])
    require(n in EXPECTED_COUNTS, 'Unexpected source training budget')
    require(source['model_bytes'] == (checkpoint / 'model.zip').read_bytes(), 'Model ZIP bytes changed')
    require(source['kalman_bytes'] == kalman.read_bytes(), 'Selected KF file bytes changed')
    (output / 'kf_receive_time.json').write_bytes(source['kalman_bytes'])
    exported_replay = dict(np.load(checkpoint / 'replay.npz', allow_pickle=False))
    for key in REPLAY_FIELDS:
        equal_tree(source['replay'][key], exported_replay[key], 'replay/' + key)
        finite_tree(exported_replay[key], 'replay/' + key)
    counts = dict(rows=len(exported_replay['observations']),
                  true_terminals=int(np.count_nonzero(exported_replay['dones'] * (1-exported_replay['timeouts']))),
                  timeouts=int(np.count_nonzero(exported_replay['timeouts'])))
    require(counts == EXPECTED_COUNTS[n], 'Source replay semantics/counts changed: ' + str(counts))
    require(state['total_transitions'] == n and state['updates'] == source_state['updates'],
            'Export changed experience/update counters')
    equal_tree(source['numpy_rng'], state['numpy_rng'], 'export/numpy_rng')
    equal_tree(source['python_rng'], state['python_rng'], 'export/python_rng')
    if source['runtime'] is None:
        equal_tree(source['reset_rng'], state['reset_action_rng'], 'export/reset_rng')
        reset_rng_policy = 'preserved source live reset-action RNG'
    else:
        equal_tree(source['runtime'], state['experiment_context']['live_export']['source_runtime'],
                   'provenance/simulation_runtime_and_rng')
        equal_tree(np.random.default_rng(source['source_seed']).bit_generator.state,
                   state['reset_action_rng'], 'export/new_live_reset_rng')
        reset_rng_policy = 'new seed-initialized live reset generator; source simulation generator retained in provenance'
    exported_rng = dict(np.load(checkpoint / 'rng.npz', allow_pickle=False))
    equal_tree(source['torch_rng'], exported_rng['cpu'], 'export/torch_rng')

    torch.set_num_threads(1)
    inputs = load_moving_inputs(checkpoint, kalman, mode='resume')
    cfg = inputs.cfg
    require(cfg.schema == 'tx-moving-v3' and cfg.loss_mode == 'confirmed_missing_terminal', 'Wrong live schema')
    require(cfg.visible_bonus == 1. and cfg.loss_pan_weight == 10. and cfg.loss_base == 0., 'Wrong reward contract')
    equal_tree(source_state['normalization'], cfg.normalization, 'normalization')
    for key, value in source_state['tracking'].items():
        require(getattr(cfg, key) == value, 'Original tracking setting changed: ' + key)
    require(inputs.state['experiment_context']['bearing_model'] == BEARING_MODEL, 'Bearing model changed')
    require(inputs.kalman == json.loads(source['kalman_bytes']), 'KF configuration changed')
    learner = SACLearner(cfg, mode='resume', checkpoint=checkpoint, device='cpu')
    check_restored_rng(source, state, learner)
    for key in REPLAY_FIELDS:
        equal_tree(source['replay'][key], getattr(learner.policy.replay_buffer, key)[:n], 'loaded_replay/' + key)
    original_model = SAC.load(io.BytesIO(source['model_bytes']), device='cpu')
    original_model.policy.set_training_mode(False)
    equal_tree(model_trees(original_model), model_trees(learner.policy), 'model_and_optimizers')
    finite_tree(model_trees(learner.policy), 'before_update')
    indices = np.linspace(0, n-1, 20, dtype=int)
    observations = np.asarray(source['replay']['observations'][indices, 0], dtype=np.float32)
    expected_deterministic = action_sequence(original_model, observations, deterministic=True)
    actual_deterministic = action_sequence(learner.policy, observations, deterministic=True)
    equal_tree(expected_deterministic, actual_deterministic, 'deterministic_actions')
    restore_global_rng(source)
    expected_stochastic = action_sequence(original_model, observations, deterministic=False)
    restore_global_rng(source)
    actual_stochastic = np.stack([learner.action(obs, warmup=False) for obs in observations])
    equal_tree(expected_stochastic, actual_stochastic, 'stochastic_actions')
    evaluation_inputs = load_moving_inputs(checkpoint, kalman, mode='evaluate')
    evaluator = SACLearner(evaluation_inputs.cfg, mode='evaluate', checkpoint=checkpoint, device='cpu')
    evaluated = np.stack([evaluator.action(obs, warmup=False) for obs in observations])
    equal_tree(expected_deterministic, evaluated, 'actual_evaluate_mode')
    np.savez_compressed(output / 'action_equivalence.npz', replay_indices=indices, observations=observations,
                        source_deterministic=expected_deterministic, export_deterministic=actual_deterministic,
                        source_stochastic=expected_stochastic, export_stochastic=actual_stochastic,
                        evaluate_mode=evaluated)
    # Restart the training RNG after action/evaluation comparisons have consumed it.
    restore_global_rng(source)
    smoke = train_clone(learner, observations, source['replay'], output)
    require(sha256(source_zip.read_bytes()) == before_source_hash, 'Source ZIP was modified')
    require(snapshot_files(checkpoint) == before_export, 'Selected export was modified by audit')
    report = dict(
        status='passed', audit_schema='moving-live-export-independent-audit-v1',
        source_zip=str(source_zip), selected_checkpoint=str(checkpoint), selected_kalman=str(kalman),
        source_sha256=before_source_hash, model_sha256=sha256(source['model_bytes']),
        selected_files_sha256=before_export, model_zip_byte_identical=True,
        policy_and_target_tensors_identical=True, all_three_optimizer_states_identical=True,
        entropy_coefficient_identical=True, replay_all_arrays_identical=True, replay_counts=counts,
        rng=dict(torch_cpu_identical=True, numpy_identical=True, python_identical=True,
                 reset_generator_policy=reset_rng_policy, source_simulation_runtime_preserved=source['runtime'] is not None,
                 actual_resume_restoration_verified=True),
        action_equivalence=dict(deterministic_count=20, stochastic_count=20, max_abs_difference=0.,
                                replay_indices=indices.tolist(), actual_resume_and_evaluate_verified=True),
        configuration=dict(schema=cfg.schema, dt=cfg.dt, normalization_identical=True,
                           source_tracking_fields_identical=True, kalman_file_byte_identical=True,
                           pan_delta_deg=cfg.pan_delta_deg, tilt_delta_deg=cfg.tilt_delta_deg,
                           visible_bonus=cfg.visible_bonus, loss_mode=cfg.loss_mode,
                           loss_pan_weight=cfg.loss_pan_weight, loss_base=cfg.loss_base),
        synthetic_training=smoke, source_and_selected_export_unchanged=True,
        scope='CPU offline checkpoint/resume audit only; no camera, actuator or hardware commands',
        provenance_context_keys=sorted(inputs.state['experiment_context']),
    )
    (output / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--kalman', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.source, args.checkpoint, args.kalman, args.output)
    print(json.dumps(dict(status=report['status'], replay=report['replay_counts'],
                          updates=report['synthetic_training']['updates_after'],
                          report=str(args.output / 'audit.json')), ensure_ascii=False))


if __name__ == '__main__':
    main()
