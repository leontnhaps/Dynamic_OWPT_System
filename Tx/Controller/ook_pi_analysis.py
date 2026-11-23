"""Offline OOK speed measurements from a Pi INA219 receiver, in the Rx clock domain."""
import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np

from common.ook_experiment import HEADER, PAYLOAD_BITS, PI_DECODER, PI_RECEIVER, PRESENTATION_REFERENCE, RULES
from Tx.Controller.ook_analysis import read_json, score_payload, write_csv, write_json


HISTORICAL_RESULT = 'https://github.com/leontnhaps/Dynamic_OWPT_System/issues/39#issuecomment-6651888535'


def find_header(t, y, bit_ms):
    """Fit two levels using only the known header. Times are relative milliseconds."""
    limit = t[-1] - (len(HEADER) + PAYLOAD_BITS) * bit_ms
    if limit < 0:
        return None
    # SSE/SST can change only when a sample crosses the header's ends or a 0/1 transition.
    edges = np.array([0, *[i for i in range(1, len(HEADER)) if HEADER[i] != HEADER[i - 1]], len(HEADER)])
    offsets = edges * bit_ms
    cuts = np.unique(np.clip((t[:, None] - offsets).ravel(), 0, limit))
    origins = (cuts[:-1] + cuts[1:]) / 2 if len(cuts) > 1 else cuts
    y = y - y[0]
    prefix = np.r_[0., np.cumsum(y)]
    squares = np.r_[0., np.cumsum(y * y)]
    on = np.array([HEADER[i] == '1' for i in edges[:-1]])
    scores = np.full(len(origins), np.inf)
    for start in range(0, len(origins), 4096):
        candidates = origins[start:start + 4096]
        indices = np.searchsorted(t, candidates[:, None] + offsets)
        total = indices[:, -1] - indices[:, 0]
        n1 = np.diff(indices, axis=1)[:, on].sum(axis=1)
        n0 = total - n1
        sums = prefix[indices]
        total_y = sums[:, -1] - sums[:, 0]
        y1 = np.diff(sums, axis=1)[:, on].sum(axis=1)
        y0 = total_y - y1
        sum_squares = squares[indices[:, -1]] - squares[indices[:, 0]]
        with np.errstate(divide='ignore', invalid='ignore'):
            lo, hi = y0 / n0, y1 / n1
            sst = sum_squares - total_y ** 2 / total
            sse = sum_squares - y0 * lo - y1 * hi
            valid = (n0 > 0) & (n1 > 0) & (sst > 0) & (hi > lo)
            scores[start:start + len(candidates)] = np.where(valid, np.maximum(0, sse) / sst, np.inf)
    best_score = np.min(scores)
    if not np.isfinite(best_score):
        return None
    matched = np.flatnonzero(np.isclose(scores, best_score, rtol=1e-12, atol=1e-12))
    first = matched[0]
    gaps = np.flatnonzero(np.diff(matched) != 1)
    last = matched[gaps[0]] if len(gaps) else matched[-1]
    origin = (cuts[first] + cuts[min(last + 1, len(cuts) - 1)]) / 2
    return float(origin), float(best_score)


def decode(t, volts, bit_ms):
    """Return header fit and per-payload bit decisions, without seeing the payload truth."""
    if len(t) < 2:
        return None
    fit = find_header(t, volts, bit_ms)
    if fit is None:
        return None
    origin, score = fit
    indices = np.searchsorted(t, origin + np.arange(len(HEADER) + PAYLOAD_BITS + 1) * bit_ms)
    counts = np.diff(indices)
    sums = np.diff(np.r_[0., np.cumsum(volts)][indices])
    is_on = np.array([b == '1' for b in HEADER])
    header = slice(0, len(HEADER))
    low = sums[header][~is_on].sum() / counts[header][~is_on].sum()
    high = sums[header][is_on].sum() / counts[header][is_on].sum()
    threshold = (low + high) / 2
    decisions = []
    for count, value in zip(counts[len(HEADER):], sums[len(HEADER):]):
        mean = float(value / count) if count else None
        decisions.append(dict(samples=int(count), mean_v=mean,
                              decision='?' if mean is None else str(int(mean >= threshold))))
    return dict(origin_ms=origin, header_sse_sst=score, threshold_v=float(threshold),
                header_low_v=float(low), header_high_v=float(high), bits=decisions)


def read_pi_csv(path):
    times, counts, sequences = [], [], []
    with path.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            timestamp, code, seq = int(row['rx_monotonic_ns']), int(row['bus_raw']), int(row['sample_seq'])
            if (timestamp < 0 or seq < 0 or not 0 <= code <= 8191 or
                    (times and (timestamp <= times[-1] or seq <= sequences[-1]))):
                raise ValueError('Non-monotonic Pi timestamps/sequence or invalid INA219 sample.')
            if not math.isclose(float(row['pv_V']), code * 0.004, abs_tol=0.000001):
                raise ValueError('Sample voltage disagrees with INA219 raw count.')
            times.append(timestamp)
            counts.append(code)
            sequences.append(seq)
    # Subtract in the integer domain before converting the elapsed times to float.
    t = (np.asarray(times, dtype=np.int64) - times[0]) / 1e6 if times else np.array([])
    steps = np.diff(sequences)
    record_dt = np.diff(t)
    consecutive_dt = record_dt[steps == 1]
    timing = dict(missing_slots=(sequences[0] if sequences else 0) + int(np.sum(steps - 1)),
                  clipped_rows=counts.count(8191), consecutive_intervals=len(consecutive_dt),
                  median_dt_ms=float(np.median(consecutive_dt)) if len(consecutive_dt) else None,
                  p95_dt_ms=float(np.percentile(consecutive_dt, 95)) if len(consecutive_dt) else None,
                  max_record_dt_ms=float(np.max(record_dt)) if len(record_dt) else None)
    return t, np.asarray(counts, dtype=float) * 0.004, timing


def tx_completed(folder, config):
    request_id = config['request_id']
    settings = config['settings']
    sent = settings['bits'] * settings['repeats']
    with (folder / 'events.jsonl').open(encoding='utf-8') as f:
        for line in f:
            event = json.loads(line)
            if (event.get('event') == 'laser_pattern' and event.get('state') == 'completed'
                    and event.get('request_id') == request_id):
                trace = event.get('trace', [])
                return len(trace) == len(sent) and all(
                    point.get('index') == i and point.get('level') == int(bit)
                    for i, (bit, point) in enumerate(zip(sent, trace)))
    return False


def analyze_trial(folder, spec):
    mode, settings = spec['mode'], spec['settings']
    bit_ms = settings['bit_ms']
    row = dict(trial=folder.name, mode=mode, sample_ms=spec['sample_ms'], bit_ms=bit_ms,
               bitrate_bps=1000 / bit_ms, status='not_started', tx_complete=False,
               success=False, decided_bits=0, erasures=0, bit_errors=0, sample_count=0)
    if not folder.is_dir():
        return row, [], None
    row['status'] = 'incomplete_tx'
    try:
        config = read_json(folder / 'config.json')
        if config.get('receiver') != PI_RECEIVER or config['settings'] != settings:
            raise ValueError('Receiver or recorded Tx settings differ from the plan.')
        if not tx_completed(folder, config):
            return row, [], None
        row['tx_complete'] = True
        row['erasures'] = PAYLOAD_BITS if mode == 'sweep' else 0
        row['status'] = 'rx_invalid'
        summary = read_json(folder / 'summary.json')
        row['end_reason'] = summary['end_reason']
        saved_spec = config['experiment']
        if (any(saved_spec[key] != value for key, value in spec.items())
                or saved_spec.get('decoder_version') != PI_DECODER
                or saved_spec.get('rules_version') != RULES['version']):
            raise ValueError('Trial metadata disagrees with experiment.json.')
        if (summary['end_reason'] != 'completed' or not config.get('interval_confirmed')
                or config['confirmed_sample_ms'] != spec['sample_ms']
                or summary.get('receiver') != PI_RECEIVER or summary['tx_state'] != 'completed'):
            return row, [], None
        t, volts, timing = read_pi_csv(folder / 'rx.csv')
        row.update(timing, sample_count=len(t), invalid_lines=summary['invalid_lines'])
        if (len(t) < 2 or len(t) != summary['rows'] or row['invalid_lines'] or row['clipped_rows']
                or row['missing_slots'] != summary['missing_records'] or row['clipped_rows'] != summary['clipped_rows']):
            return row, [], None
        waveform = (t, volts)
        if mode == 'step':
            row['status'] = 'waveform_only'
            return row, [], waveform
        expected = spec['payload']
        if len(expected) != PAYLOAD_BITS or settings['bits'] != HEADER + expected + '0000' or settings['repeats'] != 1:
            raise ValueError('Unexpected OOK test frame in trial plan.')
        decoded = decode(t, volts, bit_ms)
        if decoded is None:
            row['status'] = 'sync_failed'
            return row, [], waveform
        bits = decoded.pop('bits')
        received = ''.join(b['decision'] for b in bits)
        row.update(decoded, **score_payload(received, expected))
        row['status'] = 'success' if row['success'] else ('erasure' if row['erasures'] else 'bit_error')
        for i, b in enumerate(bits):
            b.update(trial=folder.name, bit_index=i, expected=expected[i])
        return row, bits, waveform
    except (KeyError, TypeError, ValueError, OSError) as exc:
        row['detail'] = str(exc)
        return row, [], None


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        if row['mode'] == 'sweep':
            groups[row['sample_ms'], row['bit_ms']].append(row)
    summary = []
    for (sample_ms, bit_ms), group in sorted(groups.items(), key=lambda pair: (pair[0][0], -pair[0][1])):
        eligible = [row for row in group if row['tx_complete']]
        decided = sum(r['decided_bits'] for r in eligible)
        errors = sum(r['bit_errors'] for r in eligible)
        correct = sum(r['success'] for r in eligible)
        summary.append(dict(sample_ms=sample_ms, bit_ms=bit_ms, bitrate_bps=1000 / bit_ms, planned=len(group),
                            not_started=sum(r['status'] == 'not_started' for r in group),
                            incomplete_tx=sum(r['status'] == 'incomplete_tx' for r in group),
                            evaluated_blocks=len(eligible), correct_blocks=correct, decided_bits=decided,
                            erased_bits=sum(r['erasures'] for r in eligible), error_bits=errors,
                            rx_invalid=sum(r['status'] == 'rx_invalid' for r in eligible),
                            sync_failed=sum(r['status'] == 'sync_failed' for r in eligible),
                            trials_with_record_gaps=sum(r.get('missing_slots', 0) > 0 for r in eligible),
                            missing_slots=sum(r.get('missing_slots', 0) for r in eligible),
                            ber_percent=100 * errors / decided if decided else None,
                            block_success_percent=100 * correct / len(eligible) if eligible else None))
    return summary


def plot_rates(out, summary, show_reference=False):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(9, 6), layout='constrained')
    axes = fig.subplots(2, 1, sharex=True)
    if show_reference:
        x = [1000 / period for period in PRESENTATION_REFERENCE]
        old = list(PRESENTATION_REFERENCE.values())
        old_ber = [100 * row.bit_errors / (row.blocks * PAYLOAD_BITS) for row in old]
        old_blocks = [100 * row.correct_blocks / row.blocks for row in old]
        for ax, y in zip(axes, [old_ber, old_blocks]):
            ax.plot(x, y, '--s', color='0.48', fillstyle='none',
                    label='Arduino (archived, 5 ms)')
    for interval in sorted({r['sample_ms'] for r in summary}):
        group = [r for r in summary if r['sample_ms'] == interval]
        for ax, field in zip(axes, ['ber_percent', 'block_success_percent']):
            ax.plot([r['bitrate_bps'] for r in group],
                    [float('nan') if r[field] is None else r[field] for r in group],
                    marker='o', label=f'Pi + INA219 ({interval} ms)')
    completed = sum(row['evaluated_blocks'] for row in summary)
    planned = sum(row['planned'] for row in summary)
    axes[0].set(title=f'OOK bitrate sweep  |  Pi run: {completed}/{planned} Tx frames completed',
                ylabel='BER of decided bits (%)')
    axes[1].set(ylabel='Correct 32-bit blocks (%)', xlabel='Requested bitrate (bit/s)', ylim=(-3, 103))
    axes[0].legend()
    for ax in axes:
        ax.grid(alpha=.25)
    fig.savefig(out / 'ook_bitrate_results.png', dpi=180)


def analyze_experiment(root):
    """Write a new analysis directory, keeping previous results and raw captures intact."""
    from matplotlib.figure import Figure
    root = Path(root)
    manifest = read_json(root / 'experiment.json')
    if manifest.get('receiver') != PI_RECEIVER:
        raise ValueError('Select a Raspberry Pi INA219 SSH experiment.')
    if manifest.get('decoder_version') != PI_DECODER or manifest.get('rules') != RULES:
        raise ValueError('Unsupported OOK frame or decoder version in experiment.json.')
    rows, bits, waveforms = [], [], {}
    for i, spec in enumerate(manifest['trials'], 1):
        row, decisions, wave = analyze_trial(root / f'trial_{i:05d}', spec)
        rows.append(row)
        if row['tx_complete'] and spec['mode'] == 'sweep' and not decisions:
            decisions = [dict(trial=row['trial'], bit_index=i, expected=bit, decision='?',
                              samples=None, mean_v=None) for i, bit in enumerate(spec['payload'])]
        bits.extend(decisions)
        if wave is not None:
            waveforms.setdefault((row['sample_ms'], row['bit_ms']), (row['trial'], wave))
    if not any(r['status'] != 'not_started' for r in rows):
        raise ValueError('No saved trials in this experiment.')
    summary = aggregate(rows)
    name = 'analysis_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + uuid4().hex[:6]
    with TemporaryDirectory(dir=root) as temporary:
        out = Path(temporary) / name
        out.mkdir()
        write_csv(out / 'trials.csv', rows)
        write_csv(out / 'bits.csv', bits)
        write_csv(out / 'summary.csv', summary)
        write_json(out / 'provenance.json', dict(decoder_version=PI_DECODER, receiver=PI_RECEIVER,
            source_manifest='../experiment.json', historical_method=HISTORICAL_RESULT,
            time_source='Rx monotonic ns; PC arrival times unused', voltage_lsb=0.004,
            ber_denominator='decided payload bits', block_denominator='confirmed complete Tx frames',
            alignment='SSE/SST over known header; midpoint of earliest minimum plateau',
            source_rules=manifest.get('rules'), planned_trials=len(rows),
            saved_trials=sum(r['status'] != 'not_started' for r in rows)))
        if summary:
            plot_rates(out, summary, show_reference=manifest.get('preset') == 'm4-4-presentation-150')
        for (sample_ms, bit_ms), (trial, (t, volts)) in waveforms.items():
            fig = Figure(figsize=(8, 3), layout='constrained')
            ax = fig.subplots()
            ax.plot(t / 1000, volts, '.', markersize=2)
            ax.set(xlabel='Rx elapsed time (s)', ylabel='Cell voltage (V)',
                   title=f'{trial} | {bit_ms:g} ms per bit, {sample_ms:g} ms sampling target')
            fig.savefig(out / f'waveform_{sample_ms:g}ms_{bit_ms:g}ms.png', dpi=150)
        out.rename(root / name)
    return root / name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder')
    print(analyze_experiment(parser.parse_args().folder))
