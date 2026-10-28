"""Offline test-frame analysis. Never fit timing/threshold to known payload bits."""
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import uuid
import numpy as np
from common.ook_experiment import HEADER, PAYLOAD_BITS, RULES, VERSION


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def read_samples(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    t = np.asarray([int(r['arduino_ms']) for r in rows], dtype=float)
    # R1 timestamps have 1ms resolution; correctly unwrap millis().
    if len(t):
        t = np.r_[0., np.cumsum(np.diff(t) % 2**32)]
    adc = np.asarray([int(r['adc']) for r in rows], dtype=float)
    return t, adc


def nearest(t, y, points, bit_ms):
    if not len(t):
        return np.full(np.shape(points), np.nan)
    right = np.clip(np.searchsorted(t, points), 0, len(t)-1)
    left = np.maximum(right-1, 0)
    idx = np.where(abs(t[left]-points) <= abs(t[right]-points), left, right)
    values = y[idx].astype(float)
    return np.where(abs(t[idx]-points) <= bit_ms * RULES['nearest_sample_window_bits'], values, np.nan)


def decode(t, y, bit_ms):
    """Use HEADER only for acquisition. Expected payload is not an argument."""
    result = dict(status='sync_failed', sync=False, origin_ms=None, threshold_adc=None,
                  contrast_adc=None, decoded=None, decision_adc=None)
    frame_bits = len(HEADER) + PAYLOAD_BITS
    if len(t) < 2 or t[-1] < frame_bits * bit_ms:
        return result
    # Evaluate possible origins on a fixed grid, at least 16 positions per bit.
    # Chunking bounds memory even for long captures/very short bit periods.
    grid = min(bit_ms / 16, 1.)
    limit = t[-1] - (frame_bits - .5) * bit_ms
    if limit / grid > 2_000_000:
        result['status'] = 'analysis_limit'
        return result
    expected = np.asarray([int(x) for x in HEADER])
    best = None
    best_origins = []
    for begin in range(0, int(limit/grid)+1, 4096):
        origins = np.arange(begin, min(begin+4096, int(limit/grid)+1)) * grid
        points = origins[:, None] + (np.arange(len(HEADER)) + .5) * bit_ms
        values = nearest(t, y, points, bit_ms)
        valid = np.isfinite(values).all(axis=1)
        if not valid.any():
            continue
        origins, values = origins[valid], values[valid]
        low, high = np.median(values[:, :8], axis=1), np.median(values[:, 8:16], axis=1)
        contrast = high-low
        threshold = (low+high)/2
        errors = ((values >= threshold[:, None]) != expected).sum(axis=1)
        valid = (contrast >= RULES['min_contrast_adc']) & (errors <= RULES['max_header_errors'])
        if not valid.any():
            continue
        origins, values, low, contrast, threshold = [x[valid] for x in (origins, values, low, contrast, threshold)]
        score = np.mean(((values-low[:, None])/contrast[:, None] - expected)**2, axis=1)
        minimum = float(np.min(score))
        indexes = np.flatnonzero(np.isclose(score, minimum, atol=1e-12, rtol=0))
        i = int(indexes[0])
        if best is None or minimum < best[0]-1e-12:
            best = (minimum, float(threshold[i]), float(contrast[i]))
            best_origins = [float(origins[k]) for k in indexes]
        elif abs(minimum-best[0]) <= 1e-12:
            best_origins.extend(float(origins[k]) for k in indexes)
    if best is None:
        return result
    # Center the earliest equally good contiguous phase window, rather than
    # biasing perfect square waves toward the first transition edge.
    end = 0
    while end+1 < len(best_origins) and best_origins[end+1]-best_origins[end] <= grid*1.01:
        end += 1
    origin = (best_origins[0]+best_origins[end])/2
    # Recompute training levels at the chosen phase; still no payload access.
    header_values = nearest(t, y, origin+(np.arange(len(HEADER))+.5)*bit_ms, bit_ms)
    low, high = float(np.median(header_values[:8])), float(np.median(header_values[8:16]))
    threshold, contrast = (low+high)/2, high-low
    values = nearest(t, y, origin + (len(HEADER)+np.arange(PAYLOAD_BITS)+.5)*bit_ms, bit_ms)
    bits = ''.join('?' if not math.isfinite(v) else str(int(v >= threshold)) for v in values)
    result.update(status='decoded', sync=True, origin_ms=origin, threshold_adc=threshold,
                  contrast_adc=contrast, decoded=bits,
                  decision_adc=[float(v) if math.isfinite(v) else None for v in values])
    return result


def score_payload(decoded, expected):
    # Ground truth enters only here, after sync and decisions are frozen.
    if decoded is None:
        return dict(bit_errors=0, decided_bits=0, erasures=PAYLOAD_BITS, success=False)
    pairs = [(a, b) for a, b in zip(decoded, expected) if a != '?']
    errors = sum(a != b for a, b in pairs)
    return dict(bit_errors=errors, decided_bits=len(pairs), erasures=PAYLOAD_BITS-len(pairs),
                success=len(pairs) == PAYLOAD_BITS and errors == 0)


def wilson(success, count):
    if not count:
        return None, None
    z = 1.959963984540054
    p = success/count
    center = (p+z*z/(2*count))/(1+z*z/count)
    radius = z*math.sqrt(p*(1-p)/count + z*z/(4*count*count))/(1+z*z/count)
    return max(0., center-radius), min(1., center+radius)


def write_csv(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def analyze_trial(folder):
    config, summary = read_json(folder/'config.json'), read_json(folder/'summary.json')
    experiment = config.get('experiment') or {}
    settings = config['settings']
    t, adc = read_samples(folder/'rx.csv')
    record_dt = np.diff(t)
    with (folder/'rx.csv').open(encoding='utf-8-sig', newline='') as f:
        sample_rows = list(csv.DictReader(f))
    # Consecutive ADC sequence numbers distinguish sampling timing from UART loss.
    dt = np.asarray([float(r['sample_dt_ms']) for r in sample_rows
                     if r['sample_dt_ms'] and r['sequence_gap'] == '0'])
    events = [json.loads(line) for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    terminal = next((e for e in reversed(events) if e.get('event') == 'laser_pattern'
                     and e.get('state') == 'completed' and e.get('request_id') == config['request_id']), None)
    trace = terminal.get('trace', []) if terminal else []
    sent = settings['bits'] * settings['repeats']
    tx_complete = bool(terminal and len(trace) == len(sent)
                       and all(e.get('index') == i and e.get('level') == int(sent[i])
                               for i, e in enumerate(trace)))
    gpio_times = [e.get('gpio_call_done_ms') for e in trace]
    scheduled = [e.get('scheduled_ms') for e in trace]
    timing_valid = bool(trace and all(isinstance(v, (int, float)) and math.isfinite(v)
                                   for v in gpio_times+scheduled))
    gpio_dt = np.diff(gpio_times) if timing_valid else np.array([])
    row = dict(trial=folder.name, mode=experiment.get('mode', 'manual'),
               bit_ms=settings['bit_ms'], bitrate_bps=1000/settings['bit_ms'],
               sample_ms=config['rx_adc_interval_ms'], rows=len(t),
               median_dt_ms=float(np.median(dt)) if len(dt) else None,
               p95_dt_ms=float(np.percentile(dt, 95)) if len(dt) else None,
               max_dt_ms=float(np.max(dt)) if len(dt) else None,
               max_record_dt_ms=float(np.max(record_dt)) if len(record_dt) else None,
               tx_median_interval_ms=float(np.median(gpio_dt)) if len(gpio_dt) else None,
               tx_max_lateness_ms=float(np.max(np.array(gpio_times)-scheduled)) if timing_valid else None,
               missing_records=summary['missing_records'], end_reason=summary['end_reason'],
               tx_complete=tx_complete, status='incomplete', sync=False, success=False,
               bit_errors=0, decided_bits=0, erasures=0, delta_pv_v=None)
    decoded = None
    if tx_complete and summary['end_reason'] == 'completed' and len(t):
        if row['mode'] != 'sweep':
            row['status'] = 'waveform_only'
        elif experiment.get('rules_version') != VERSION:
            row['status'] = 'unsupported_rules'
        else:
            decoded = decode(t, adc, settings['bit_ms'])
            row.update({k: v for k, v in decoded.items() if k != 'decision_adc'})
            row.update(score_payload(decoded['decoded'], experiment['payload']))
            if row['sync']:
                row['status'] = 'success' if row['success'] else ('erasure' if row['erasures'] else 'bit_error')
                values = decoded['decision_adc']
                groups = [[v for v, b in zip(values, experiment['payload']) if b == bit and v is not None] for bit in '01']
                if all(groups):
                    # Truth-labelled diagnostic statistic only, never fed back into the decoder.
                    row['delta_pv_v'] = float(np.mean(groups[1])-np.mean(groups[0])) * config['adc_reference_v'] * config['divider_gain']/1024
            if summary['missing_records'] or summary['clipped_rows'] or summary['invalid_lines']:
                row['status'] = 'acquisition_invalid'
                row['success'] = False
    return row, (t, adc, config, trace, decoded)


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        if row['mode'] == 'sweep':
            groups[(row['sample_ms'], row['bit_ms'])].append(row)
    results = []
    for (sample_ms, bit_ms), group in sorted(groups.items()):
        # Failed/aborted TX is an instrument issue, not an optical bit trial.
        eligible = [r for r in group if r['tx_complete'] and r['end_reason'] == 'completed']
        n, success = len(eligible), sum(r['success'] for r in eligible)
        decided = sum(r['decided_bits'] for r in eligible)
        errors = sum(r['bit_errors'] for r in eligible)
        lo, hi = wilson(success, n)
        deltas = [r['delta_pv_v'] for r in eligible if r['delta_pv_v'] is not None and r['status'] != 'acquisition_invalid']
        results.append(dict(sample_ms=sample_ms, bit_ms=bit_ms, bitrate_bps=1000/bit_ms,
                            attempted=len(group), completed_tx_blocks=n, incomplete=len(group)-n,
                            successes=success, block_success_pct=100*success/n if n else None,
                            success_ci95_low_pct=100*lo if n else None, success_ci95_high_pct=100*hi if n else None,
                            zero_failure_upper95=1-.05**(1/n) if n and n == success else None,
                            sync_failures=sum(not r['sync'] for r in eligible),
                            acquisition_invalid=sum(r['status'] == 'acquisition_invalid' for r in eligible),
                            erased_bits=sum(r['erasures'] for r in eligible), decided_bits=decided,
                            bit_errors=errors, conditional_ber=errors/decided if decided else None,
                            delta_pv_mean_v=float(np.mean(deltas)) if deltas else None,
                            delta_pv_sd_v=float(np.std(deltas, ddof=1)) if len(deltas)>1 else None,
                            delta_blocks=len(deltas),
                            actual_median_dt_ms=float(np.median([r['median_dt_ms'] for r in eligible if r['median_dt_ms'] is not None])) if eligible else None))
    return results


def make_plots(out, rows, aggregates, folders):
    from matplotlib.figure import Figure
    for field, ylabel, filename in [('block_success_pct', 'Block success (%)', 'block_success.png'),
                                    ('delta_pv_mean_v', 'PV difference (V)', 'voltage_difference.png'),
                                    ('conditional_ber', 'BER (synchronized, decided bits)', 'conditional_ber.png')]:
        if not aggregates:
            break
        fig = Figure(figsize=(7, 4), layout='constrained'); ax = fig.subplots()
        for interval in sorted({r['sample_ms'] for r in aggregates}):
            group = sorted([r for r in aggregates if r['sample_ms'] == interval and r[field] is not None], key=lambda r:r['bitrate_bps'])
            if not group:
                continue
            x, y = [r['bitrate_bps'] for r in group], [r[field] for r in group]
            errors = None
            if field == 'block_success_pct':
                errors = [[max(0, r[field]-r['success_ci95_low_pct']) for r in group],
                          [max(0, r['success_ci95_high_pct']-r[field]) for r in group]]
            elif field == 'delta_pv_mean_v':
                errors = [r['delta_pv_sd_v'] or 0 for r in group]
            ax.errorbar(x, y, yerr=errors, marker='o', capsize=3, label=f'{interval} ms')
        ax.set(xlabel='Requested bitrate (bit/s)', ylabel=ylabel)
        if ax.get_legend_handles_labels()[0]: ax.legend(title='Sampling target')
        if field == 'block_success_pct': ax.set_ylim(-3, 103)
        fig.savefig(out/filename, dpi=180)
    # One complete trial per condition; success/failure selection is reproducible (first in plan).
    seen = set()
    for row, folder in zip(rows, folders):
        key = (row['sample_ms'], row['bit_ms'])
        if key in seen: continue
        seen.add(key)
        t, adc = read_samples(folder/'rx.csv'); config = read_json(folder/'config.json')
        fig = Figure(figsize=(8, 5), layout='constrained'); axes = fig.subplots(2, 1)
        bits = config['settings']['bits'] * config['settings']['repeats']; tb = config['settings']['bit_ms']
        axes[0].step(np.arange(len(bits)+1)*tb/1000, [int(b) for b in bits]+[0], where='post', label='Tx requested')
        axes[0].set(xlabel='Tx pattern time (s)', ylabel='Bit'); axes[0].legend()
        axes[1].plot(t/1000, adc*config['adc_reference_v']*config['divider_gain']/1024,
                     '.', ms=2, label='PV samples')
        axes[1].set(xlabel='Rx capture time (s)', ylabel='PV voltage (V)'); axes[1].legend()
        fig.savefig(out/f'waveform_{row["sample_ms"]}ms_{row["bit_ms"]}ms.png', dpi=180)


def analyze_experiment(root):
    root = Path(root)
    manifest = read_json(root/'experiment.json')
    if manifest['rules'] != RULES:
        raise ValueError('실험 판정 버전이 현재 분석 코드와 다릅니다. 해당 버전 코드를 사용하세요.')
    folders = sorted(p for p in root.iterdir() if p.is_dir() and (p/'summary.json').exists())
    if not folders:
        raise ValueError('완료 또는 중단된 측정 파일이 없습니다.')
    out = root/('analysis_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+uuid.uuid4().hex[:6])
    out.mkdir()
    rows = [analyze_trial(p)[0] for p in folders]
    aggregates = aggregate(rows)
    write_csv(out/'trials.csv', rows); write_csv(out/'rates.csv', aggregates)
    write_json(out/'analysis.json', dict(rules=RULES, decoder_location='offline_host', bit_decoding_implemented=True, planned_trials=len(manifest['trials']),
        saved_trials=len(rows), conditions=aggregates,
        notes=['Test-frame decoder only; no servo commands or final protocol.',
               'BER is conditional on synchronization and decided samples. Sync failures and erasures are separate.',
               'Acquisition gaps count as block failures; incomplete TX is reported separately.',
               'Voltage difference uses known payload labels after decoding, conditional on acquired frames.',
               'Tx/Rx waveform axes have independent origins. Requested Tx is not measured optical output.',
               'Wilson 95% intervals describe repeated blocks, assuming independence; no automatic optimal rate selection.',
               'Sampling interval statistics use consecutive ADC sequence numbers; record gaps are reported separately.',
               '1ms Arduino timestamp resolution; nominal reference voltage; real hardware validation required.']))
    make_plots(out, rows, aggregates, folders)
    return out


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('folder')
    print(analyze_experiment(parser.parse_args().folder))
