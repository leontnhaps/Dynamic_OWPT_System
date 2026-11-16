"""Strict, offline export of the selected real/simulation SAC checkpoints.

The policy archive is copied verbatim. Replay rewards/terminal flags are never
relabelled: the new contract applies only to future live transitions.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path, PurePosixPath
import random
import shutil
import stat
import sys
import tempfile
import uuid
import zipfile

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from common.tx_bearing import BEARING_MODEL
from common.tx_moving import MovingConfig, load_kalman_config
from Tx.Controller.stationary_learning import checkpoint_info, digest, dump, sac_settings

SIM_SCHEMA = 'tx-moving-geometric-sim-v1'
LIVE_SCHEMA = 'tx-moving-v3'
ARRAY_FIELDS = ('observations', 'next_observations', 'actions', 'rewards', 'dones', 'timeouts')
META_FIELDS = ('pos', 'full', 'capacity', 'total_transitions')
LIVE_FILES = {'model.zip', 'replay.npz', 'state.json', 'rng.npz'}
SIM_FILES = LIVE_FILES | {'experiment_runtime.json'}
REWARD = dict(visible_bonus=1., loss_pan_weight=10., loss_base=0.)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _integer(value, name, minimum=0):
    _require(type(value) is int and value >= minimum, 'Invalid integer: ' + name)
    return value


def _tuples(value):
    return tuple(_tuples(v) for v in value) if isinstance(value, list) else value


@contextmanager
def _source_bundle(source):
    source = Path(source).expanduser().resolve()
    if source.is_dir():
        bundle = source.parent if source.name == 'checkpoint' else source
        _require((bundle/'checkpoint/manifest.json').is_file(), 'Source bundle needs checkpoint/manifest.json')
        yield bundle
        return
    _require(source.is_file() and zipfile.is_zipfile(source), 'Source must be a checkpoint bundle folder or ZIP')
    with tempfile.TemporaryDirectory(prefix='moving_checkpoint_source_') as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(source) as archive:
            names = set()
            total_size = 0
            for info in archive.infolist():
                path = PurePosixPath(info.filename)
                _require(info.filename not in names, 'Duplicate archive member')
                names.add(info.filename)
                _require(not path.is_absolute() and '..' not in path.parts
                         and '\\' not in info.filename and ':' not in info.filename,
                         'Unsafe archive member: ' + info.filename)
                _require(not stat.S_ISLNK(info.external_attr >> 16), 'Archive symlinks are not supported')
                total_size += info.file_size
                _require(total_size <= 2_000_000_000, 'Source archive exceeds supported size')
            manifests = [name for name in names if name.endswith('/checkpoint/manifest.json')]
            _require(len(manifests) == 1, 'ZIP must contain exactly one checkpoint bundle')
            bundle_path = PurePosixPath(manifests[0]).parent.parent
            _require(len(bundle_path.parts) == 1, 'ZIP must contain one top-level bundle folder')
            _require(all(PurePosixPath(name).parts[0] == bundle_path.parts[0] for name in names),
                     'Unexpected ZIP root')
            archive.extractall(root)
        yield root/bundle_path


def _manifest(checkpoint):
    manifest = _json(checkpoint/'manifest.json')
    schema = manifest.get('schema')
    _require(schema in ('tx-moving-v2', SIM_SCHEMA), 'Only the selected v2 or geometric simulation checkpoint is supported')
    expected = SIM_FILES if schema == SIM_SCHEMA else LIVE_FILES
    _require(set(manifest.get('files', {})) == expected, 'Source manifest file set is incomplete or unexpected')
    for name, sha256 in manifest['files'].items():
        _require((checkpoint/name).is_file() and digest(checkpoint/name) == sha256,
                 'Source manifest hash mismatch: ' + name)
    state = _json(checkpoint/'state.json')
    _require(state.get('schema') == schema, 'Source state/manifest schema mismatch')
    if schema == 'tx-moving-v2':
        _require(bool(state.get('checkpoint_id')) and state.get('checkpoint_id') == manifest.get('checkpoint_id'),
                 'Source checkpoint ID mismatch')
    else:
        _require(state.get('simulation_only') is True and manifest.get('simulation_only') is True,
                 'Simulation source provenance mismatch')
    return manifest, state


def _configuration(bundle, state):
    source_config = _json(bundle/'source_config.json')
    _require(source_config.get('schema') == 'tx-moving-solution-study-v1', 'Missing solution-study provenance')
    _require(source_config.get('reward') == REWARD, 'Source reward differs from the selected candidate')
    seed = _integer(source_config.get('seed'), 'source seed')
    simulated = state['schema'] == SIM_SCHEMA
    _require(source_config.get('normalization') == state.get('normalization'), 'Source configuration normalization mismatch')
    if simulated:
        _require(state.get('config') == source_config, 'Simulation configuration disagrees with source_config.json')
        _require(source_config.get('tracking') == state.get('tracking'), 'Simulation tracking configuration mismatch')
    else:
        tracking = dict(state['tracking'])
        _require(tracking.pop('visible_bonus', 0) == REWARD['visible_bonus'], 'Real source visible reward mismatch')
        _require(tracking == source_config.get('tracking'), 'Real tracking configuration differs from its provenance')
        _require(state.get('action_grid') == 'round_half_away_from_zero', 'Unsupported action quantization')
        _require(state.get('simulation_only') is False and state.get('live_controller_compatible') is True,
                 'Real source compatibility flags mismatch')
    tracking = dict(state['tracking'])
    tracking.update(REWARD, loss_mode='confirmed_missing_terminal')
    cfg = MovingConfig(**tracking)
    cfg.validate()
    _require(cfg.schema == LIVE_SCHEMA, 'The current code lacks confirmed-missing loss support')
    _require(cfg.normalization == state.get('normalization'), 'Moving observation normalization mismatch')
    _require(source_config.get('sac') == sac_settings(cfg), 'Source SAC settings mismatch')
    if not simulated:
        _require(state.get('sac') == sac_settings(cfg), 'Real source SAC settings mismatch')
    kalman, identity = load_kalman_config(bundle/'kf_receive_time.json')
    _require(source_config.get('kalman', {}).get('sha256') == identity['sha256'], 'KF provenance hash mismatch')
    _require(kalman.get('time_source', 'receive_s') == 'receive_s', 'Live KF must use receive_s')
    _require(kalman.get('forecast_horizon_s') == cfg.dt, 'KF forecast horizon differs from the saved control period')
    context = state if simulated else state.get('experiment_context', {})
    _require(context.get('kalman') == kalman, 'Source context differs from the bundled KF')
    _require(context.get('bearing_model') == BEARING_MODEL, 'Source bearing calibration mismatch')
    _require(context.get('normalization') == cfg.normalization, 'Source context normalization mismatch')
    return cfg, source_config, kalman, identity, seed


def _model_metadata(checkpoint, state, cfg, seed):
    # Read JSON only; exporting never executes a pickle, trains or reserializes a model.
    with zipfile.ZipFile(checkpoint/'model.zip') as archive:
        required = {'data', 'pytorch_variables.pth', 'policy.pth', 'actor.optimizer.pth',
                    'critic.optimizer.pth', 'ent_coef_optimizer.pth'}
        names = archive.namelist()
        _require(len(names) == len(set(names)) and required <= set(names), 'Model archive is incomplete')
        _require(archive.testzip() is None, 'Model archive CRC mismatch')
        model = json.loads(archive.read('data'))
    for key, value in sac_settings(cfg).items():
        _require(model.get(key) == value, 'Model SAC setting mismatch: ' + key)
    _require(model.get('num_timesteps') == state['total_transitions'], 'Model transition counter mismatch')
    _require(model.get('_n_updates') == state['updates'], 'Model update counter mismatch')
    _require(model.get('seed') == seed, 'Model seed mismatch')
    _require(model.get('n_envs') == 1 and model.get('optimize_memory_usage') is False,
             'Unsupported replay layout')
    _require(model.get('use_sde') is False, 'Unsupported stochastic policy configuration')
    policy = model.get('policy_kwargs', {})
    _require(policy.get('net_arch') == [256, 256] and policy.get('n_critics') == 2
             and policy.get('activation_fn') == "<class 'torch.nn.modules.activation.ReLU'>",
             'Model architecture mismatch')
    for name, shape in (('observation_space', [10]), ('action_space', [2])):
        space = model.get(name, {})
        _require(space.get('_shape') == shape and space.get('dtype') == 'float32'
                 and space.get('low_repr') == '-1.0' and space.get('high_repr') == '1.0',
                 'Model space mismatch: ' + name)
    return model


def _replay(checkpoint, state, cfg):
    count = _integer(state.get('total_transitions'), 'total_transitions', 1)
    _integer(state.get('updates'), 'updates')
    capacity = sac_settings(cfg)['buffer_size']
    n = min(count, capacity)
    simulated = state['schema'] == SIM_SCHEMA
    with np.load(checkpoint/'replay.npz', allow_pickle=False) as saved:
        expected = set(ARRAY_FIELDS) if simulated else set(ARRAY_FIELDS+META_FIELDS)
        _require(set(saved.files) == expected, 'Unexpected replay fields')
        arrays = {key: saved[key].copy() for key in saved.files}
    shapes = dict(observations=(n, 1, 10), next_observations=(n, 1, 10), actions=(n, 1, 2),
                  rewards=(n, 1), dones=(n, 1), timeouts=(n, 1))
    for key in ARRAY_FIELDS:
        _require(arrays[key].shape == shapes[key] and arrays[key].dtype == np.float32
                 and np.isfinite(arrays[key]).all(), 'Invalid replay array: ' + key)
    _require((np.abs(arrays['actions']) <= 1.).all(), 'Replay actions exceed normalized bounds')
    for key in ('dones', 'timeouts'):
        _require(np.isin(arrays[key], [0., 1.]).all(), 'Invalid replay flag: ' + key)
    _require((arrays['timeouts'] <= arrays['dones']).all(), 'Timeout without done flag')
    if simulated:
        _require(count < capacity, 'Simulation conversion requires an unwrapped replay buffer')
        runtime = _json(checkpoint/'experiment_runtime.json')
        _require(runtime.get('replay_position') == count and runtime.get('replay_capacity') == capacity
                 and runtime.get('replay_full') is False and runtime.get('environment_at_episode_boundary') is True,
                 'Simulation replay runtime mismatch')
        _require(runtime.get('next_episode') == _integer(state.get('simulated_episodes'), 'simulated_episodes')+1,
                 'Simulation episode counter mismatch')
        _require(0 <= _integer(state.get('simulated_steps'), 'simulated_steps') <= count,
                 'Simulation transition counter mismatch')
    else:
        runtime = None
        for key in META_FIELDS:
            _require(arrays[key].shape == (), 'Replay metadata must be scalar: ' + key)
            allowed_kinds = 'biu' if key == 'full' else 'iu'
            _require(arrays[key].dtype.kind in allowed_kinds, 'Invalid replay metadata type: ' + key)
        _require(arrays['full'].item() in (0, 1), 'Invalid replay full flag')
        _require(arrays['pos'].item() == count % capacity and arrays['capacity'].item() == capacity
                 and arrays['total_transitions'].item() == count
                 and arrays['full'].item() == (count >= capacity), 'Source replay metadata mismatch')
        _require(not (arrays['dones'] > arrays['timeouts']).any(), 'Legacy real replay contains unsupported terminals')
    arrays.update(pos=np.array(count % capacity, dtype=np.int64), full=np.array(count >= capacity),
                  capacity=np.array(capacity, dtype=np.int64), total_transitions=np.array(count, dtype=np.int64))
    return arrays, runtime


def _rng(checkpoint, state, seed):
    with np.load(checkpoint/'rng.npz', allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    if state['schema'] == SIM_SCHEMA:
        _require(set(arrays) == {'torch_cpu', 'numpy_keys'}, 'Unexpected simulation RNG layout')
        nr = state['numpy_rng']
        _require(set(nr) == {'algorithm', 'pos', 'has_gauss', 'cached_gaussian'}, 'Invalid simulation NumPy RNG metadata')
        keys = arrays.pop('numpy_keys')
        _require(keys.shape == (624,) and keys.dtype == np.uint32, 'Invalid NumPy RNG keys')
        numpy_rng = [nr['algorithm'], keys.tolist(), nr['pos'], nr['has_gauss'], nr['cached_gaussian']]
        arrays = {'cpu': arrays.pop('torch_cpu')}
        reset_rng = np.random.default_rng(seed).bit_generator.state
    else:
        _require('cpu' in arrays and all(k == 'cpu' or (k.startswith('cuda_') and k[5:].isdigit()) for k in arrays),
                 'Unexpected live torch RNG layout')
        numpy_rng = deepcopy(state['numpy_rng'])
        reset_rng = deepcopy(state['reset_action_rng'])
    _require(isinstance(numpy_rng, list) and len(numpy_rng) == 5, 'Invalid NumPy RNG state')
    nr = numpy_rng
    np.random.RandomState().set_state((nr[0], np.array(nr[1], dtype=np.uint32), *nr[2:]))
    random.Random().setstate(_tuples(state['python_rng']))
    generator = np.random.default_rng()
    generator.bit_generator.state = reset_rng
    for name, value in arrays.items():
        _require(value.ndim == 1 and value.dtype == np.uint8 and value.size > 0, 'Invalid torch RNG: ' + name)
    # A local generator validates the CPU payload without changing process RNG.
    import torch
    torch.Generator(device='cpu').set_state(torch.from_numpy(arrays['cpu']))
    return arrays, numpy_rng, reset_rng


def export_checkpoint(source, output):
    """Return a new v3 bundle; preserve source model/replay/RNG and all update counts."""
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser()
    output = (REPOSITORY_ROOT/output if not output.is_absolute() else output).resolve()
    allowed_root = (REPOSITORY_ROOT/'captures/M5-3').resolve()
    _require(output != allowed_root and allowed_root in output.parents, 'Output must be a new folder under captures/M5-3')
    _require(not output.exists(), 'Refusing to overwrite an existing output folder')
    _require(not source.is_dir() or (source != output and source not in output.parents),
             'Output must not be inside the source bundle')
    with _source_bundle(source) as bundle:
        checkpoint = bundle/'checkpoint'
        manifest, state = _manifest(checkpoint)
        cfg, source_config, kalman, identity, seed = _configuration(bundle, state)
        arrays, runtime = _replay(checkpoint, state, cfg)
        _model_metadata(checkpoint, state, cfg, seed)
        torch_rng, numpy_rng, reset_rng = _rng(checkpoint, state, seed)
        simulated = state['schema'] == SIM_SCHEMA
        terminal_count = int(np.count_nonzero(arrays['dones'] > arrays['timeouts']))
        provenance = dict(
            schema='moving-live-export-v1', source=str(source), source_schema=state['schema'],
            source_manifest_sha256=digest(checkpoint/'manifest.json'), source_files=manifest['files'],
            source_config=source_config, source_runtime=runtime,
            source_simulated_steps=state.get('simulated_steps', 0),
            source_simulated_episodes=state.get('simulated_episodes', 0),
            imported_real_transitions=state['total_transitions']-state.get('simulated_steps', 0),
            source_live_counters={key: state.get(key) for key in ('executed_actions', 'episodes')},
            source_bundle_sha256=digest(source) if source.is_file() else None,
            model_archive_byte_identical=True, replay_arrays_unchanged=list(ARRAY_FIELDS),
            weights_reset=False, critic_reset=False, replay_reset=False, extra_gradient_updates=0,
            total_transitions=state['total_transitions'], updates=state['updates'],
            imported_true_terminals=terminal_count, imported_timeouts=int(arrays['timeouts'].sum()),
            source_terminal_policy=('immediate geometric loss terminal; source end bootstrapped truncation'
                                    if simulated else 'missing transition excluded; preceding valid transition truncated'),
            live_terminal_policy='confirmed fresh post-command missing for 3 seconds; terminal requested-Pan loss',
            replay_terminal_flags_reinterpreted=False, live_reacquisition_s=3.,
            reward=REWARD, loss_mode=cfg.loss_mode,
            terminal_domain_shift=bool(simulated),
            reset_action_rng=('new seed-initialized live-only generator, unused by manual starts; original simulation environment RNG retained unchanged in source_runtime'
                              if simulated else 'preserved source reset-action RNG'),
            hardware_validation_performed=False,
        )
        context = deepcopy(state.get('experiment_context', {}))
        context.update(kalman=kalman, bearing_model=BEARING_MODEL, normalization=cfg.normalization,
                       live_export=provenance)
        checkpoint_id = uuid.uuid4().hex
        exported_state = dict(
            schema=cfg.schema, checkpoint_id=checkpoint_id, tracking=asdict(cfg), sac=sac_settings(cfg),
            total_transitions=state['total_transitions'], updates=state['updates'],
            executed_actions=0 if simulated else _integer(state.get('executed_actions'), 'executed_actions'),
            episodes=0 if simulated else _integer(state.get('episodes'), 'episodes'),
            reset_action_rng=reset_rng, numpy_rng=numpy_rng, python_rng=state['python_rng'],
            source=str(source), experiment_context=context, normalization=cfg.normalization,
            action_grid='round_half_away_from_zero',
            versions=source_config['versions'] if simulated else state['versions'],
            simulation_only=False, live_controller_compatible=True,
            requires_code_support='moving-confirmed-missing-terminal-v1',
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix='.pending_live_export_', dir=output.parent))
        try:
            target = temporary/'checkpoint'
            target.mkdir()
            shutil.copyfile(checkpoint/'model.zip', target/'model.zip')
            np.savez_compressed(target/'replay.npz', **arrays)
            if simulated:
                np.savez_compressed(target/'rng.npz', **torch_rng)
            else:
                shutil.copyfile(checkpoint/'rng.npz', target/'rng.npz')
            dump(target/'state.json', exported_state)
            files = {name: digest(target/name) for name in sorted(LIVE_FILES)}
            _require(files['model.zip'] == manifest['files']['model.zip'], 'Export changed the policy archive')
            dump(target/'manifest.json', dict(schema=cfg.schema, checkpoint_id=checkpoint_id, files=files))
            shutil.copyfile(bundle/'kf_receive_time.json', temporary/'kf_receive_time.json')
            dump(temporary/'conversion.json', dict(provenance, checkpoint='checkpoint',
                                                   kalman='kf_receive_time.json', kalman_sha256=identity['sha256'],
                                                   checkpoint_id=checkpoint_id, output_manifest_sha256=digest(target/'manifest.json')))
            # Verify through the exact loader used by the live UI before publishing.
            checkpoint_info(target)
            with np.load(target/'replay.npz', allow_pickle=False) as written:
                for key in ARRAY_FIELDS:
                    _require(np.array_equal(written[key], arrays[key]), 'Export changed replay array: ' + key)
            _require(not output.exists(), 'Output was created concurrently; refusing overwrite')
            os.rename(temporary, output)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='Saved model bundle ZIP or folder containing checkpoint/')
    parser.add_argument('--output', required=True, type=Path, help='New bundle folder under captures/M5-3/')
    args = parser.parse_args(argv)
    result = export_checkpoint(args.source, args.output)
    print(json.dumps(dict(bundle=str(result), checkpoint=str(result/'checkpoint'),
                          kalman=str(result/'kf_receive_time.json')), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
