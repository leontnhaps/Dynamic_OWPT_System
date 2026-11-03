"""Fine-tune existing PV detector weights and compare before/after on the same validation split."""
import argparse
from copy import deepcopy
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys


if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.model_paths import DEFAULT_YOLO_PATH, REPOSITORY_ROOT, resolve_yolo_path


DEFAULT_DATA = REPOSITORY_ROOT / 'captures/M5-1/yolo_outdoor_dataset/data.yaml'
DEFAULT_OUTPUT = REPOSITORY_ROOT / 'captures/M5-1/yolo_training'
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else _digest_stream(stream)


def _digest_stream(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        value.update(block)
    return value.hexdigest()


def repository_path(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else REPOSITORY_ROOT / path).resolve()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def inspect_dataset(path):
    """Require explicit labels (empty for reviewed negatives) and disjoint train/val images."""
    import yaml
    from PIL import Image

    data = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    if not isinstance(data, dict) or data.get('names') not in ({0: 'pv_cell'}, ['pv_cell']):
        raise ValueError('data.yaml의 클래스는 0: pv_cell이어야 합니다.')
    if data.get('download'):
        raise ValueError('자동 다운로드 대신 로컬 이미지와 라벨을 준비하세요.')
    root = Path(data.get('path') or path.parent).expanduser()
    root = (root if root.is_absolute() else path.parent / root).resolve()
    resolved = {'names': {0: 'pv_cell'}}
    rows, summary, hashes, label_paths = [], {}, {}, set()
    for split in ('train', 'val'):
        value = data.get(split)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{split}에는 images/{split} 폴더 경로가 필요합니다.')
        folder = (root / value).resolve()
        if not folder.is_dir() or folder.parent.name != 'images':
            raise ValueError(f'{split}: images/<분할명> 형식의 로컬 폴더가 필요합니다: {folder}')
        resolved[split] = str(folder)
        label_folder = folder.parent.parent / 'labels' / folder.name
        images = sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise ValueError(f'{split}에 이미지가 없습니다: {folder}')
        positive = boxes = 0
        for image in images:
            label = (label_folder / image.relative_to(folder)).with_suffix('.txt')
            if not label.is_file():
                raise ValueError(f'라벨 파일이 없습니다: {label}\nPV가 없는 사진도 확인 후 빈 .txt 라벨을 준비하세요.')
            if label.resolve() in label_paths:
                raise ValueError(f'이미지 여러 개가 같은 라벨을 사용합니다: {label}')
            label_paths.add(label.resolve())
            image_hash = digest(image)
            if image_hash in hashes:
                raise ValueError(f'중복 이미지 또는 train/val 중복: {image} / {hashes[image_hash]}')
            hashes[image_hash] = str(image)
            with Image.open(image) as opened:
                width, height = opened.size
                opened.verify()
            lines = [line.split() for line in label.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
            seen_boxes = set()
            for fields in lines:
                if len(fields) != 5:
                    raise ValueError(f'Detection 라벨은 class x y w h의 5열이어야 합니다: {label}')
                cls, x, y, w, h = map(float, fields)
                box = (cls, x, y, w, h)
                if box in seen_boxes:
                    raise ValueError(f'중복 박스 라벨: {label}')
                seen_boxes.add(box)
                if (not all(math.isfinite(n) for n in (cls, x, y, w, h)) or cls != 0
                        or not 0 < w <= 1 or not 0 < h <= 1
                        or min(x-w/2, y-h/2) < -1e-6 or max(x+w/2, y+h/2) > 1+1e-6):
                    raise ValueError(f'클래스 또는 정규화 박스 좌표 오류: {label}')
            positive += bool(lines)
            boxes += len(lines)
            rows.append(dict(split=split, image=str(image), label=str(label), width=width, height=height,
                             boxes=len(lines), image_sha256=image_hash, label_sha256=digest(label)))
        if not positive:
            raise ValueError(f'{split}에 PV 정답 박스가 한 개 이상 필요합니다.')
        summary[split] = dict(images=len(images), positive=positive, negative=len(images)-positive, boxes=boxes)
    return resolved, rows, summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--model', type=Path, default=DEFAULT_YOLO_PATH, help='기존 PV .pt (기본값: 공통 M1-3 best.pt)')
    result.add_argument('--data', type=Path, default=DEFAULT_DATA, help='images/train, images/val을 지정한 YOLO YAML')
    result.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help='실행별 새 폴더를 만들 상위 경로')
    result.add_argument('--epochs', type=int, default=500)
    result.add_argument('--patience', type=int, default=20)
    result.add_argument('--imgsz', type=int, default=960)
    result.add_argument('--batch', type=int, default=8)
    result.add_argument('--lr0', type=float, default=0.0001, help='AdamW 초기 학습률; 추가학습 시작값')
    result.add_argument('--device', default='0', help='단일 CUDA 장치 번호 또는 cpu')
    result.add_argument('--workers', type=int, default=0, help='Windows 기본값 0')
    result.add_argument('--seed', type=int, default=0)
    result.add_argument('--check-only', action='store_true', help='파일·라벨·중복 검사만 수행; GPU/학습/결과폴더 생성 없음')
    return result


def prepare(args):
    model = Path(resolve_yolo_path(args.model)).resolve()
    data, output = repository_path(args.data), repository_path(args.output)
    if model.suffix.lower() != '.pt' or not model.is_file() or not model.stat().st_size:
        raise ValueError(f'기존 학습된 PV 모델 .pt가 필요합니다: {model}')
    if not data.is_file():
        raise ValueError(f'data.yaml이 없습니다: {data}\n학습용 ZIP을 captures/M5-1/ 아래에 풀었는지 확인하세요.')
    if (args.epochs < 1 or args.patience < 0 or args.imgsz < 32 or args.imgsz % 32
            or args.batch < 1 or args.workers < 0 or args.seed < 0
            or not math.isfinite(args.lr0) or not 0 < args.lr0 < 1):
        raise ValueError('epochs/batch는 양수, imgsz는 32의 양의 배수, patience/workers/seed는 0 이상, lr0는 0~1 사이여야 합니다.')
    if args.device != 'cpu' and not args.device.isdecimal():
        raise ValueError('--device는 단일 GPU 번호(예: 0) 또는 cpu를 사용하세요.')
    resolved, rows, summary = inspect_dataset(data)
    for split in ('train', 'val'):
        folder = Path(resolved[split])
        if output == folder or folder in output.parents:
            raise ValueError('결과 폴더를 학습/검증 이미지 폴더 안에 둘 수 없습니다.')
    return model, data, output, resolved, rows, summary


def assert_pv_model(model):
    if model.task != 'detect' or model.names not in ({0: 'pv_cell'}, ['pv_cell']):
        raise ValueError('입력 모델은 0: pv_cell 단일 클래스 Detection 모델이어야 합니다.')


def check_training_compatibility(model, version):
    """Rebuild the checkpoint architecture on CPU before trusting validation or training."""
    from ultralytics.nn.tasks import DetectionModel

    try:
        # Training reconstructs from YAML; loading a .pt for inference does not test this.
        # Never edit checkpoint arguments to make an older SPPF implementation accept them.
        DetectionModel(deepcopy(model.model.yaml), nc=1, ch=3, verbose=False)
    except (TypeError, AttributeError, KeyError) as exc:
        raise ValueError(
            f'Ultralytics {version}에서 입력 모델 구조(SPPF 등)를 재구성하지 못했습니다.\n'
            'YOLO26에는 Ultralytics 8.4.0 이상이 필요합니다. 같은 Python 환경에서 업데이트하세요:\n'
            'python -m pip install --upgrade "ultralytics>=8.4.0,<9" "PyYAML>=6" '
            '"numpy>=1.26,<2" "opencv-python>=4.7,<4.12"\n'
            f'실행 Python: {sys.executable}\n'
            f'원래 오류: {type(exc).__name__}: {exc}\n'
            '평가·학습을 시작하지 않았으며 입력 가중치는 수정하지 않았습니다.'
        ) from exc


def metric_values(metrics):
    values = dict(precision=float(metrics.box.mp), recall=float(metrics.box.mr),
                  mAP50=float(metrics.box.map50), mAP50_95=float(metrics.box.map))
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError('검증 지표에 유효하지 않은 값이 있습니다.')
    return values


def run(args):
    model_path, data_path, output, dataset, rows, summary = prepare(args)
    print(json.dumps(dict(model=str(model_path), data=str(data_path), dataset=summary), ensure_ascii=False, indent=2))
    if args.check_only:
        print('파일/라벨 검사 통과. 모델 내부 구조·CUDA·실제 학습은 검사하지 않았습니다.')
        return None

    import torch
    import ultralytics
    import yaml
    from ultralytics import YOLO

    if args.device != 'cpu' and (not torch.cuda.is_available() or int(args.device) >= torch.cuda.device_count()):
        raise ValueError('선택한 CUDA 장치를 사용할 수 없습니다. GPU용 PyTorch 또는 --device cpu를 확인하세요.')
    original_hash = digest(model_path)
    baseline_model = YOLO(str(model_path))
    assert_pv_model(baseline_model)
    check_training_compatibility(baseline_model, ultralytics.__version__)
    print('CPU 모델 구조 재구성 통과. 학습 전 평가를 시작합니다.', flush=True)
    output.mkdir(parents=True, exist_ok=True)
    run_dir = output / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_dir.mkdir(exist_ok=False)
    # Absolute paths make this run independent of cwd and Ultralytics datasets_dir settings.
    run_data = run_dir / 'data.yaml'
    run_data.write_text(yaml.safe_dump(dataset, allow_unicode=True, sort_keys=False), encoding='utf-8')
    with (run_dir / 'dataset_manifest.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    validation = dict(data=str(run_data), split='val', imgsz=args.imgsz, batch=args.batch,
                      device=args.device, workers=args.workers, conf=0.001, iou=0.7, max_det=300,
                      augment=False, rect=True, half=False, single_cls=False, plots=True, save_json=False,
                      project=str(run_dir), exist_ok=False)
    training = dict(data=str(run_data), epochs=args.epochs, patience=args.patience, imgsz=args.imgsz,
                    batch=args.batch, device=args.device, workers=args.workers, seed=args.seed,
                    optimizer='AdamW', lr0=args.lr0, resume=False, cache=False, single_cls=False, val=True,
                    save=True, plots=True, project=str(run_dir), name='train', exist_ok=False)
    config = dict(schema='pv-yolo-finetune-v1', input_model=str(model_path), input_model_sha256=original_hash,
                  source_data=str(data_path), source_yaml_sha256=digest(data_path), dataset=summary,
                  versions=dict(python=sys.version, ultralytics=ultralytics.__version__, torch=torch.__version__),
                  compatibility=dict(architecture_reconstructed=True, device='cpu'),
                  training=training, validation=validation,
                  metric_note='Metrics are fractions. Precision/recall use the Ultralytics best-F1 threshold for each model, not the GUI confidence threshold. mAP is the primary before/after comparison. This is validation, not an independent test.',
                  status='running')
    write_json(run_dir / 'config.json', config)
    print(f'결과 저장: {run_dir}', flush=True)
    try:
        before = metric_values(baseline_model.val(name='before', **validation))
        write_json(run_dir / 'before.json', before)
        # Validation may fuse inference layers. Reload the untouched weights for training.
        del baseline_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = YOLO(str(model_path))
        model.train(**training)
        best = Path(model.trainer.best).resolve()
        if not best.is_file() or run_dir not in best.parents:
            raise ValueError('이번 실행의 best.pt가 생성되지 않았습니다.')
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        after_model = YOLO(str(best))
        after = metric_values(after_model.val(name='after', **validation))
        delta = {key: after[key] - before[key] for key in before}
        comparison = dict(before=before, after=after, delta=delta, best_model=str(best),
                          best_model_sha256=digest(best), metric_note=config['metric_note'])
        write_json(run_dir / 'comparison.json', comparison)
        with (run_dir / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['stage', *before])
            writer.writeheader()
            for stage, values in [('before', before), ('after', after), ('delta', delta)]:
                writer.writerow(dict(stage=stage, **values))
        if digest(model_path) != original_hash:
            raise ValueError('실행 중 입력 모델 파일이 변경되었습니다. 원본을 확인하세요.')
        config.update(status='completed', best_model=str(best), original_model_unchanged=True)
        write_json(run_dir / 'config.json', config)
        print(json.dumps(comparison, ensure_ascii=False, indent=2))
        print(f'새 모델: {best}\n기존 기본 모델 경로는 유지됩니다.')
        return run_dir
    except BaseException as exc:
        config.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                      error=f'{type(exc).__name__}: {exc}', original_model_unchanged=digest(model_path) == original_hash if model_path.is_file() else False)
        write_json(run_dir / 'config.json', config)
        raise


def main():
    cli = parser()
    try:
        run(cli.parse_args())
    except (ValueError, OSError, ImportError) as exc:
        cli.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
