"""Collect fixed-pose receiver runs and match each to its own completed PC transmission."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from uuid import uuid4

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.analysis.angle_reception import analyze


ROOT = Path(__file__).resolve().parents[2] / 'captures' / 'M6-4'
FIELDS = ('pair', 'angle_deg', 'pan_deg', 'bit_ms', 'period_s', 'planned', 'received',
          'missed', 'indeterminate', 'success_percent', 'marker_received', 'timing_basis',
          'duplicate_events', 'wrong_slot_events', 'unexpected_events', 'issue')


def index_tx(folder, prefix):
    index = defaultdict(list)
    for path in folder.glob('arc_tx_*.jsonl'):
        with path.open(encoding='utf-8-sig') as stream:
            first = stream.readline().strip()
        if not first:
            continue
        try:
            start = json.loads(first)
        except ValueError:
            continue  # Never accept a TX log lacking a readable run identifier.
        if (isinstance(start, dict) and start.get('event') == 'arc_start'
                and str(start.get('pair_id', '')).startswith(prefix)):
            index[start['pair_id']].append(path)
    return index


def collect(rx_root, tx_root, prefix):
    tx = index_tx(tx_root, prefix)
    rx = defaultdict(list)
    for path in rx_root.glob('angle_*/manifest.json'):
        manifest = json.loads(path.read_text(encoding='utf-8-sig'))
        pair = manifest.get('pair')
        if isinstance(pair, str) and pair.startswith(prefix):
            rx[pair].append((path.parent, manifest))
    summaries, trials = [], []
    for pair in sorted(rx.keys() | tx.keys()):
        runs, logs = rx.get(pair, []), tx.get(pair, [])
        if len(runs) != 1 or len(logs) != 1:
            summaries.append(dict(pair=pair, issue=f'expected one Rx and Tx run; found {len(runs)} and {len(logs)}'))
            continue
        folder, manifest = runs[0]
        try:
            outcomes, summary = analyze(folder, logs[0])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            summaries.append(dict(pair=pair, angle_deg=manifest.get('angle_deg'),
                                  bit_ms=manifest.get('bit_ms'), issue=str(exc)))
            continue
        summaries.append({**summary, 'issue': ''})
        trials.extend(dict(outcome, pair=pair) for outcome in outcomes)
    if not summaries:
        raise ValueError(f'No Rx or Tx runs matched {prefix!r}')
    return summaries, trials


def write_csv(path, fields, rows):
    with path.open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    try:
        groups = sorted({r['bit_ms'] for r in rows})
        for bit_ms in groups:
            group = sorted((r for r in rows if r['bit_ms'] == bit_ms), key=lambda r: r['angle_deg'])
            x = [r['angle_deg'] for r in group]
            lower = [100*r['received']/r['planned'] for r in group]
            upper = [100*(r['received'] + r['indeterminate'])/r['planned'] for r in group]
            ax.errorbar(x, lower, yerr=[[0]*len(x), [b-a for a, b in zip(lower, upper)]],
                        fmt='o', capsize=4, label=f'{1000/bit_ms:.2f} bit/s')
        ax.set(xlabel='Commanded incidence offset (degrees)', ylabel='Correct commands (%)',
               ylim=(-4, 105), title='Optical command reception at held angles')
        ax.grid(alpha=0.25)
        ax.legend()
        fig.subplots_adjust(bottom=.20)
        fig.text(.5, .035, 'Points: verified correct / planned.  Bars: possible range for indeterminate slots.',
                 ha='center', fontsize=8)
        fig.savefig(path, dpi=160, bbox_inches='tight')
    finally:
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', required=True, help='pair ID prefix for this series, e.g. a12-')
    parser.add_argument('--rx-root', type=Path, default=ROOT)
    parser.add_argument('--tx-root', type=Path, default=ROOT)
    parser.add_argument('--out-root', type=Path, default=ROOT)
    args = parser.parse_args(argv)
    if not args.prefix.strip():
        parser.error('A nonempty pair ID prefix is required.')
    try:
        summaries, trials = collect(args.rx_root, args.tx_root, args.prefix)
        dest = args.out_root / ('angle_series_' + datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid4().hex[:8])
        dest.mkdir(parents=True)
        write_csv(dest/'angles.csv', FIELDS, summaries)
        if trials:
            write_csv(dest/'commands.csv', list(trials[0]), trials)
        valid = [row for row in summaries if not row['issue']]
        if valid:
            try:
                plot_results(valid, dest/'reception.png')
            except ModuleNotFoundError as exc:
                if exc.name != 'matplotlib':
                    raise
                print('CSV saved. Install matplotlib to also generate the plot.', file=sys.stderr)
        print(f'{dest}\nAudited {len(valid)}/{len(summaries)} runs. See angles.csv for unmatched or incomplete runs.')
        return 0 if len(valid) == len(summaries) else 1
    except (OSError, KeyError, ValueError) as exc:
        parser.exit(1, f'Angle series analysis failed: {exc}\n')


if __name__ == '__main__':
    sys.exit(main())
