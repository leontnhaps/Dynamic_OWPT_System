"""Copy missing JPEG captures into one folder for manual CVAT annotation."""
import argparse
import csv
import filecmp
import hashlib
import os
from pathlib import Path
import shutil


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPOSITORY_ROOT / 'captures'
DEFAULT_OUTPUT = DEFAULT_SOURCE / 'M5-3' / 'yolo_missing'


def missing_images(source, output):
    """Skip the export folder and symlinks so exports cannot collect themselves."""
    def walk_error(error):
        raise error

    for folder, dirs, files in os.walk(source, onerror=walk_error):
        folder = Path(folder)
        dirs[:] = sorted(name for name in dirs
                         if not (folder/name).is_symlink()
                         and (folder/name).resolve() != output)
        for name in sorted(files):
            path = folder/name
            if (not path.is_symlink() and path.is_file()
                    and path.suffix.lower() in ('.jpg', '.jpeg')
                    and 'missing' in path.stem.lower()):
                yield path


def collect(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir():
        raise ValueError(f'Source folder does not exist: {source}')
    if output == source or output in source.parents:
        raise ValueError('Output must not be the source folder or its parent.')

    # Finish discovery before writing; fail visibly if a source directory is unreadable.
    images = list(missing_images(source, output))
    if not images:
        print(f'No missing .jpg/.jpeg images found in: {source}')
        return 0, 0

    output.mkdir(parents=True, exist_ok=True)
    manifest = output/'manifest.csv'
    has_header = manifest.exists() and manifest.stat().st_size > 0
    copied = skipped = 0
    with manifest.open('a', newline='', encoding='utf-8-sig') as log:
        writer = csv.writer(log)
        if not has_header:
            writer.writerow(['source', 'exported_name', 'status'])
        for path in images:
            relative = path.relative_to(source).as_posix()
            identifier = hashlib.sha256(relative.encode('utf-8')).hexdigest()[:16]
            name = f'{identifier}_{path.stem[:100]}{path.suffix.lower()}'
            destination = output/name
            if destination.exists():
                if destination.is_symlink() or not filecmp.cmp(path, destination, shallow=False):
                    raise ValueError(f'Existing export differs; use another --output folder: {destination}')
                skipped += 1
                status = 'already_copied'
            else:
                # Exclusive creation prevents overwriting an existing file.
                with path.open('rb') as original, destination.open('xb') as target:
                    try:
                        shutil.copyfileobj(original, target)
                    except BaseException:
                        target.close()
                        destination.unlink()
                        raise
                copied += 1
                status = 'copied'
            writer.writerow([str(path), name, status])
            log.flush()

    print(f'Matched: {len(images)} | Copied: {copied} | Already copied: {skipped}')
    print(f'Output: {output}')
    print(f'Source mapping: {manifest}')
    return copied, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE,
                        help='Search folder (default: repository captures/)')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT,
                        help='Flat export folder (default: captures/M5-3/yolo_missing/)')
    args = parser.parse_args()
    try:
        collect(args.source, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
