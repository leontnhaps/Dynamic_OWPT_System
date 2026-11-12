"""Approved moving experiment inputs and their paired-checkpoint contract."""
from dataclasses import dataclass
from pathlib import Path

from common.model_paths import REPOSITORY_ROOT
from common.tx_bearing import BEARING_MODEL
from common.tx_moving import MovingConfig, load_kalman_config
from Tx.Controller.stationary_learning import checkpoint_info


DEFAULT_KALMAN_PATH = (REPOSITORY_ROOT / 'captures' / 'M5-3' / 'kf_tuning_20261111'
                       / 'kf_candidate_receive_time.json')
DEFAULT_MOVING_CHECKPOINT_PATH = (REPOSITORY_ROOT / 'captures' / 'M5-3'
                                  / 'pretraining_20261112_011543_363660' / 'checkpoints'
                                  / 'episode_000013_318791cd')


def _resolve(value, default):
    value = str(value).strip() if value is not None else ''
    selected = Path(value).expanduser() if value else default
    return selected if selected.is_absolute() else REPOSITORY_ROOT / selected


def resolve_kalman_path(value=None):
    """Resolve only the selected file; never substitute another tuned filter."""
    return _resolve(value, DEFAULT_KALMAN_PATH)


def resolve_moving_checkpoint_path(value=None):
    """Resolve only the selected checkpoint; never guess the latest run."""
    return _resolve(value, DEFAULT_MOVING_CHECKPOINT_PATH)


@dataclass(frozen=True)
class MovingInputs:
    checkpoint: Path
    state: dict
    cfg: MovingConfig
    kalman: dict
    kalman_identity: dict


def load_moving_inputs(checkpoint=None, kalman_path=None, mode='resume'):
    """Validate saved 10D inputs before loading a model or issuing hardware I/O.

Both resume and evaluation retain the checkpoint's period, geometry and
normalization. A separately selected KF must match the one used to construct
the saved replay. A model.zip selection is accepted only with its complete
paired checkpoint beside it, as enforced by checkpoint_info.
"""
    if mode not in ('resume', 'evaluate'):
        raise ValueError('이동 모델 실행은 이어서 학습 또는 고정 모델 평가를 선택하세요.')
    source = resolve_moving_checkpoint_path(checkpoint)
    selected_kalman = resolve_kalman_path(kalman_path)
    if not source.exists():
        raise ValueError(f'이동 SAC checkpoint가 없습니다. 저장된 episode 폴더를 선택하세요: {source}')
    folder = source.parent if source.is_file() else source
    required = ('manifest.json', 'model.zip', 'replay.npz', 'state.json', 'rng.npz')
    missing = [name for name in required if not (folder / name).is_file()]
    if missing:
        raise ValueError('이동 SAC checkpoint 구성 파일이 없습니다: '
                         + ', '.join(missing) + f' · 폴더: {folder}')
    if not selected_kalman.is_file():
        raise ValueError(f'고정 KF 설정 JSON이 없습니다. 확정한 파일을 선택하세요: {selected_kalman}')
    # This checks all four hashes, the schema, SAC settings and saved normalization.
    folder, state, cfg = checkpoint_info(folder)
    if not isinstance(cfg, MovingConfig) or cfg.observation_size != 10:
        raise ValueError('이동 SAC에는 10차원 이동 checkpoint가 필요합니다. 정지 모델은 호환되지 않습니다.')
    context = state.get('experiment_context')
    if not isinstance(context, dict):
        raise ValueError('Checkpoint에 이동 학습의 KF·각도 보정 정보가 없습니다.')
    if context.get('bearing_model') != BEARING_MODEL:
        raise ValueError('Checkpoint의 각도 보정 모델이 현재 이동 실험과 다릅니다.')
    if context.get('normalization') != cfg.normalization:
        raise ValueError('Checkpoint 학습 문맥의 이동 입력 정규화가 일치하지 않습니다.')
    kalman, identity = load_kalman_config(selected_kalman)
    if context.get('kalman') != kalman:
        raise ValueError('선택한 KF 설정이 checkpoint를 학습할 때 사용한 KF와 다릅니다. 같은 JSON을 선택하세요.')
    if kalman.get('time_source', 'receive_s') != 'receive_s':
        raise ValueError('이동 실시간 KF는 수신 시각 receive_s 기준의 설정이 필요합니다.')
    cfg.validate(evaluation=mode == 'evaluate')
    return MovingInputs(folder, state, cfg, kalman, identity)
