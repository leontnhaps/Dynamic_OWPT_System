"""Compare an angle command with a successfully received optical speed-test frame."""
import csv
import json
from pathlib import Path

import numpy as np

from common.angle_demo import FRAME_LEN
from common.ook_experiment import HEADER, PI_DECODER, PI_RECEIVER, RULES, THRESHOLD_BITS
from common.ook_frame_decode import MIN_CONTRAST_V
from Tx.Controller.ook_pi_analysis import analyze_trial, decode, read_pi_csv


def _reference_candidates(stage, bit_ms, sample_ms):
    candidates = []
    for run in sorted(stage.glob('experiment_*'), reverse=True):
        try:
            plan = json.loads((run / 'experiment.json').read_text(encoding='utf-8-sig'))
            if (not isinstance(plan, dict) or plan.get('receiver') != PI_RECEIVER or plan.get('decoder_version') != PI_DECODER
                    or plan.get('rules') != RULES):
                continue
            indices = sorted(run.glob('analysis_*/trials.csv'), reverse=True)
            successful = None
            if indices:
                with indices[0].open(encoding='utf-8-sig', newline='') as f:
                    successful = {r['trial'] for r in csv.DictReader(f) if r['status'] == 'success'}
            for n, spec in reversed(list(enumerate(plan['trials'], 1))):
                folder = run / f'trial_{n:05d}'
                if (spec['mode'] == 'sweep' and spec['settings']['bit_ms'] == bit_ms
                        and (successful is None or folder.name in successful)
                        and (folder / 'rx.csv').is_file()):
                    candidates.append((folder, spec))
        except (OSError, ValueError, TypeError, KeyError, csv.Error):
            continue
    # Stable sort keeps the newest run and last trial first within each group.
    return sorted(candidates, key=lambda item: item[1]['sample_ms'] != sample_ms)


def _reference(stage, bit_ms, sample_ms):
    for folder, spec in _reference_candidates(stage, bit_ms, sample_ms):
        try:
            config = json.loads((folder / 'config.json').read_text(encoding='utf-8-sig'))
            with (folder / 'events.jsonl').open(encoding='utf-8') as f:
                physical = any(isinstance(e, dict) and e.get('event') == 'laser_pattern' and e.get('state') == 'completed'
                               and e.get('request_id') == config['request_id'] and e.get('simulated') is False
                               for e in map(json.loads, f))
            if physical:
                row, _, wave = analyze_trial(folder, spec)
                if row['success']:
                    return folder, row, wave
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return None


def _angle_header(times, volts, bit_ms):
    # Reuse the speed-test header fit. Accept an origin for plotting only if the
    # recorded samples actually reproduce every known header bit with contrast.
    if len(volts) < len(HEADER) or np.ptp(volts) < MIN_CONTRAST_V:
        return None
    fit = decode(times, volts, bit_ms)
    if not fit or fit['header_high_v'] - fit['header_low_v'] < MIN_CONTRAST_V:
        return None
    edges = np.searchsorted(times, fit['origin_ms'] + np.arange(len(HEADER) + 1) * bit_ms)
    if any(a == b or int(np.mean(volts[a:b]) >= fit['threshold_v']) != int(bit)
           for a, b, bit in zip(edges, edges[1:], HEADER)):
        return None
    return fit['origin_ms']


def compare_angle_waveform(stage_dir, attempt_dir, bit_ms, sample_ms=1):
    """Write a standalone PNG + its provenance; run on the PC outside the Tk thread."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    stage, attempt = Path(stage_dir), Path(attempt_dir)
    t, v, timing = read_pi_csv(attempt / 'rx.csv')
    origin = _angle_header(t, v, bit_ms)
    ref = _reference(stage, bit_ms, sample_ms)
    info = dict(schema_version=1, bit_ms=bit_ms, angle_sample_ms=sample_ms, angle_samples=len(t),
                angle_csv='rx.csv', angle_header_origin_ms=origin, angle_timing=timing,
                angle_intervals_over_bit=int(np.count_nonzero(np.diff(t) > bit_ms)),
                reference=None, time_axis='Each trace uses its own Rx monotonic sample times.')
    summary_file = attempt / 'summary.json'
    summary = json.loads(summary_file.read_text(encoding='utf-8-sig')) if summary_file.is_file() else {}
    for key in ('waveform_complete', 'angle_verified', 'end_reason', 'interval_confirmed', 'clipped_rows'):
        info[key] = summary.get(key)
    if ref:
        folder, row, (rt, rv) = ref
        source = folder.relative_to(stage)
        ref_origin = row['origin_ms']
        info['reference'] = dict(csv=(source / 'rx.csv').as_posix(), sample_ms=row['sample_ms'],
                                 header_origin_ms=ref_origin, verified_payload_bits=row['decided_bits'],
                                 intervals_over_bit=int(np.count_nonzero(np.diff(rt) > bit_ms)),
                                 samples=len(rt), max_record_dt_ms=row['max_record_dt_ms'])
    overlay = ref is not None and origin is not None
    fig = Figure(figsize=(10, 7.5 if overlay else 5.8), layout='constrained')
    canvas = FigureCanvasAgg(fig)
    axes = fig.subplots(1 + int(ref is not None) + int(overlay), 1, squeeze=False, sharey=True).ravel()
    status = {True: 'verified', False: 'unverified', None: 'unknown'}[info['angle_verified']]
    capture = {True: 'complete', False: 'partial', None: 'unknown'}[info['waveform_complete']]
    fig.suptitle(f'Angle OOK at {1000 / bit_ms:g} bit/s ({bit_ms:g} ms/bit)\n'
                 f'Command: {status}   |   Capture: {capture}   |   Rx sampling target: {sample_ms:g} ms')
    angle_color, reference_color = '#007495', '#a0481e'

    def trace(ax, times, volts, color, label):
        # Individual ADC reads: never draw a line across missing samples.
        ax.plot(times, volts, '.', markersize=2.7, alpha=.82, color=color, label=label)
        gaps = times[1:][np.diff(times) > bit_ms]
        if len(gaps):
            ax.vlines(gaps, 0, .06, transform=ax.get_xaxis_transform(), color='#ab3342', linewidth=.9)

    def frame_view(ax, times, volts, zero, color, label):
        trace(ax, times - zero, volts, color, label)
        ax.axvspan(0, len(HEADER) * bit_ms, color='#c8dce2', alpha=.28)
        ax.set_xlim(-2 * bit_ms, (FRAME_LEN + 2) * bit_ms)
        ax.set_xlabel('Time from fitted header start (ms)')

    current = axes[0]
    if origin is None:
        trace(current, t, v, angle_color, 'angle')
        current.set_xlabel('Time from first angle sample (ms)')
        current.set_title('Angle command  |  full recording, header unconfirmed', loc='left')
        if not len(t):
            current.text(.5, .5, 'No angle samples recorded', transform=current.transAxes, ha='center')
    else:
        frame_view(current, t, v, origin, angle_color, 'angle')
        current.set_title('Angle command  |  frame at fitted header', loc='left')
    gap = timing['max_record_dt_ms']
    gap_text = f'{gap:.2f} ms' if gap is not None else 'n/a'
    current.text(1, 1.02, f'{len(t)} reads | max interval {gap_text}', transform=current.transAxes,
                 ha='right', fontsize=9)
    if ref:
        frame_view(axes[1], rt, rv, ref_origin, reference_color, 'basic test')
        axes[1].set_title(f'Basic test  |  32/32 bits correct  |  {row["sample_ms"]:g} ms sampling', loc='left')
        axes[1].text(1, 1.02, folder.name, transform=axes[1].transAxes, ha='right', fontsize=9)
    if overlay:
        shared = axes[2]
        trace(shared, t - origin, v, angle_color, 'Angle command')
        trace(shared, rt - ref_origin, rv, reference_color, 'Basic test')
        shared.axvline(len(THRESHOLD_BITS) * bit_ms, linewidth=1, color='#627278', linestyle='--')
        shared.set(xlim=(0, len(HEADER) * bit_ms), xlabel='Time from each fitted header start (ms)')
        shared.set_title('Common 29-bit header  |  each signal aligned independently', loc='left')
        shared.legend(loc='best', markerscale=2)
    for ax in axes:
        ax.set_ylabel('Cell voltage (V)')
        ax.grid(alpha=.2)
    if ref is None:
        note = 'No successful basic test at this bitrate.'
        message = '같은 속도의 기본 시험 성공 기록 없음 · 이번 파형만 표시'
    elif origin is None:
        note = 'Angle header unconfirmed. The two time axes have different origins.'
        message = '이번 헤더 미확인 · 전체 파형과 같은 속도 성공 파형을 각각 표시'
    else:
        note = 'Shading marks the common header. The two 32-bit payloads can differ.'
        message = '같은 속도 기본 시험 성공 파형과 공통 헤더 비교'
    fig.supxlabel(note + '\nRed ticks mark sample intervals longer than one bit.', fontsize=10)
    target = attempt / 'comparison.png'
    canvas.print_png(target)
    (attempt / 'comparison.json').write_text(
        json.dumps(info, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return target, message
