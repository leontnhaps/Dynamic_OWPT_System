"""Shared model defaults, anchored to the repository rather than the working directory."""
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YOLO_PATH = REPOSITORY_ROOT / 'captures' / 'M1-3' / 'pv_detection' / 'runs' / 'y26n_v1' / 'weights' / 'best.pt'


def resolve_yolo_path(path=None):
    """Use the approved PV weights by default; resolve relative selections from repo root."""
    value = str(path).strip() if path is not None else ''
    selected = Path(value).expanduser() if value else DEFAULT_YOLO_PATH
    return str(selected if selected.is_absolute() else REPOSITORY_ROOT / selected)
