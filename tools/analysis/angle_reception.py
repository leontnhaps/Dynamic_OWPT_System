"""Verify one stationary Rx capture against its complete fixed-source command train."""
import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))
from common.arc_protocol import FRAME_LEN, VERSION, stimulus_bits, timing
from tools.analysis.arc_voltage import audit_tx


TRIALS = 30
SLOT_TOLERANCE = 0.15


def sample_coverage(folder, meta):
    first = last = previous_index = None
    missing_total = count = 0
    gaps = []
    with (folder / 'samples.csv').open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not {'monotonic_s', 'sample_index', 'missing_before', 'voltage_v'}.issubset(reader.fieldnames or []):
            raise ValueError('Invalid Rx sample CSV header')
        for row in reader:
            stamp = float(row['monotonic_s'])
            index, missed = int(row['sample_index']), int(row['missing_before'])
            volts = float(row['voltage_v'])
            if not (math.isfinite(stamp) and math.isfinite(volts) and index >= 0 and missed >= 0):
                raise ValueError('Invalid Rx sample')
            if last is None:
                first = stamp
            else:
                if stamp <= last or index - previous_index != missed + 1:
                    raise ValueError('Inconsistent Rx sampling order or gap count')
                if missed or stamp - last > 1.8 * meta['sample_ms'] / 1000:
                    gaps.append((last, stamp))
            count += 1
            missing_total += missed
            last, previous_index = stamp, index
    if (not count or count != meta['sample_count'] or missing_total != meta['missing_samples']
            or not math.isclose(first, meta['first_sample_s'], rel_tol=0, abs_tol=2e-5)
            or not math.isclose(last, meta['last_sample_s'], rel_tol=0, abs_tol=2e-5)):
        raise ValueError('Rx sample count/time does not match its manifest')
    required = TRIALS * meta['period_s'] + len(stimulus_bits(TRIALS, meta['bit_ms'])) * meta['bit_ms'] / 1000
    if last - first + 2e-5 < required:
        raise ValueError('Rx capture is shorter than the transmitted burst')
    return first, last, gaps


def frame_events(folder, first, last, bit_s):
    events = []
    previous = -math.inf
    with (folder / 'frames.csv').open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not {'sequence', 'frame_start_s', 'decoded_at_s'}.issubset(reader.fieldnames or []):
            raise ValueError('Invalid Rx frame CSV header')
        for row in reader:
            seq = int(row['sequence'])
            start, decoded = float(row['frame_start_s']), float(row['decoded_at_s'])
            if (not math.isfinite(start) or not math.isfinite(decoded)
                    or start < first - bit_s or decoded > last + 2e-5 or decoded < previous
                    or not (FRAME_LEN - 1) * bit_s <= decoded - start <= (FRAME_LEN + 1) * bit_s):
                raise ValueError('Decoded event timing is inconsistent with Rx recording')
            previous = decoded
            events.append((seq, start))
    return events


def clock_origin(events, first, last, period, frame_duration):
    """Choose the unique strongest sequence/timing consensus in the receiver's clock."""
    points = [(seq, start - seq * period) for seq, start in events if 0 <= seq <= TRIALS]
    markers = [offset for seq, offset in points if seq == 0]
    points_to_test = markers or [offset for _, offset in points]
    if not points_to_test:
        return None, 'operator_bracket'
    candidates = []
    for offset in points_to_test:
        if offset < first - SLOT_TOLERANCE or offset + TRIALS * period + frame_duration > last + SLOT_TOLERANCE:
            continue
        # Count distinct expected sequence numbers. Duplicated pulses do not vote twice.
        differences = {}
        for seq, value in points:
            if abs(value - offset) <= SLOT_TOLERANCE:
                if seq not in differences or abs(value - offset) < abs(differences[seq] - offset):
                    differences[seq] = value
        candidates.append((len(differences), offset if markers else median(differences.values())))
    if not candidates:
        return None, 'ambiguous'
    best = max(score for score, _ in candidates)
    offsets = [offset for score, offset in candidates if score == best]
    if max(offsets) - min(offsets) > SLOT_TOLERANCE:
        return None, 'ambiguous'
    return median(offsets), 'optical_marker' if markers else 'inferred_sequence'


def analyze(rx_folder, tx_path):
    folder, tx_path = Path(rx_folder), Path(tx_path)
    meta = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    if (meta.get('schema_version') != 1 or meta.get('kind') != 'static_angle_reception'
            or meta.get('profile') != VERSION or meta.get('steps') != TRIALS
            or meta.get('sample_ms') != 5 or meta.get('status') != 'completed'
            or meta.get('operator_bracketed') is not True):
        raise ValueError('Expected a completed, operator-bracketed 30-command static angle capture')
    timing(meta['bit_ms'], meta['period_s'])
    angle, pan = meta['angle_deg'], meta['pan_deg']
    if type(angle) is not int or not 0 <= angle <= 180 or pan != 180 - angle:
        raise ValueError('Inconsistent fixture angle and commanded Pan')
    audit = audit_tx(tx_path, meta, 'A')
    first, last, gaps = sample_coverage(folder, meta)
    bit_s = meta['bit_ms'] / 1000
    events = frame_events(folder, first, last, bit_s)
    period = meta['period_s']
    duration = FRAME_LEN * bit_s
    origin, basis = clock_origin(events, first, last, period, duration)
    counts = dict(duplicate_events=0, wrong_slot_events=0, unexpected_events=0)
    accepted = {}
    for seq, start in events:
        if not 0 <= seq <= TRIALS:
            counts['unexpected_events'] += 1
        elif origin is None:
            continue
        elif abs(start - (origin + seq * period)) > SLOT_TOLERANCE:
            counts['wrong_slot_events'] += 1
        elif seq in accepted:
            counts['duplicate_events'] += 1
        else:
            accepted[seq] = start
    rows = []
    for seq in range(1, TRIALS + 1):
        due = origin + seq * period if origin is not None else None
        if seq in accepted:
            status = 'received'
        elif due is None:
            status = 'indeterminate' if gaps or basis == 'ambiguous' else 'missed'
        else:
            # Conservative union: a valid arrival may shift anywhere inside the slot tolerance.
            lo, hi = due - SLOT_TOLERANCE, due + duration + SLOT_TOLERANCE
            broken = lo < first or hi > last or any(a < hi and b > lo for a, b in gaps)
            status = 'indeterminate' if broken else 'missed'
        rows.append(dict(sequence=seq, status=status, frame_start_s=accepted.get(seq),
                         expected_start_s=due))
    received = sum(r['status'] == 'received' for r in rows)
    missed = sum(r['status'] == 'missed' for r in rows)
    unknown = TRIALS - received - missed
    summary = dict(schema_version=1, pair=meta['pair'], angle_deg=angle, pan_deg=pan,
                   bit_ms=meta['bit_ms'], period_s=period, planned=TRIALS, received=received,
                   missed=missed, indeterminate=unknown, marker_received=0 in accepted,
                   timing_basis=basis, marker_or_sequence_origin_rx_s=origin,
                   success_percent=100*received/TRIALS if not unknown else None,
                   valid_shot_percent=100*received/(received+missed) if received+missed else None,
                   sample_gaps=len(gaps), operator_bracketed=True, tx_audit=audit,
                   rx_folder=str(folder.resolve()), tx_path=str(tx_path.resolve()), **counts)
    return rows, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rx_folder', type=Path)
    parser.add_argument('tx_log', type=Path)
    parser.add_argument('--out-root', type=Path, default=ROOT / 'captures' / 'M6-4')
    args = parser.parse_args(argv)
    try:
        rows, summary = analyze(args.rx_folder, args.tx_log)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        dest = args.out_root / f'angle_score_{stamp}_{uuid4().hex[:8]}'
        dest.mkdir(parents=True)
        (dest / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                                     allow_nan=False), encoding='utf-8')
        with (dest / 'commands.csv').open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"{dest}\n{summary['received']}/{TRIALS} received · {summary['missed']} missed · "
              f"{summary['indeterminate']} indeterminate ({summary['timing_basis']})")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(1, f'Angle analysis failed: {exc}\n')


if __name__ == '__main__':
    main()
