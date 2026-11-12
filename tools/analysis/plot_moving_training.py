"""Regenerate M5 episode/update graphs from saved CSVs without loading a model."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import threading
import uuid


# Matplotlib has shared font/rendering state even when no pyplot/Tk canvas is used.
_PLOT_LOCK = threading.Lock()


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else math.nan
    except (TypeError, ValueError, OverflowError):
        return math.nan


def _read_csv(path):
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def _series(rows, x_key, y_key):
    # Keep missing rows as NaNs so lines do not bridge missing episodes/updates.
    return ([_number(row.get(x_key)) for row in rows],
            [_number(row.get(y_key)) for row in rows])


def _draw(axis, rows, x_key, columns, ylabel, title, *, ratio=False):
    from matplotlib.ticker import MaxNLocator
    points = 0
    for key, label in columns:
        x, y = _series(rows, x_key, key)
        count = sum(math.isfinite(a) and math.isfinite(b) for a, b in zip(x, y))
        points += count
        if count:
            axis.plot(x, y, marker='o', markersize=3, linewidth=1.2, label=label)
    axis.set(title=title, xlabel='Episode' if x_key == 'episode' else 'Cumulative SAC update', ylabel=ylabel)
    # Matplotlib otherwise crops trailing episodes whose metrics are all missing.
    # Their IDs still belong on the axis even though no value can be plotted.
    ids = [value for row in rows if math.isfinite(value := _number(row.get(x_key)))]
    if ids:
        low, high = min(ids), max(ids)
        margin = .5 if low == high else max(.15, (high-low)*.02)
        axis.set_xlim(low-margin, high+margin)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
    axis.grid(True, alpha=.25)
    if ratio:
        axis.set_ylim(-.03, 1.03)
    if points:
        axis.legend(loc='best')
    else:
        axis.text(.5, .5, 'No measured values', transform=axis.transAxes,
                  ha='center', va='center', color='#666666')


def _save_figure(folder, name, panels):
    # Deliberately avoid pyplot, backend switching and any Tk object in workers.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=(10, 3.2*len(panels)), layout='constrained')
    FigureCanvasAgg(figure)
    temp = folder/f'.{name}.{uuid.uuid4().hex}.tmp'
    try:
        for i, panel in enumerate(panels, 1):
            axis = figure.add_subplot(len(panels), 1, i)
            _draw(axis, **panel)
        figure.savefig(temp, format='png', dpi=150)
        os.replace(temp, folder/name)
    finally:
        figure.clear()
        if temp.exists():
            temp.unlink()


def generate_plots(folder):
    """Save derived PNGs under ``folder/graphs``; graph errors are nonfatal.

    Call after episodes/updates have been flushed, outside the GUI thread. Missing
    metrics remain gaps. Offline initial_dataset rows do not become live-policy
    performance measurements. Original CSVs/checkpoints/images are read-only here.
    """
    report = dict(files=[], errors=[])
    with _PLOT_LOCK:
        try:
            folder = Path(folder)
            if not folder.is_dir():
                raise FileNotFoundError(f'Run folder does not exist: {folder}')
            episodes = _read_csv(folder/'episodes.csv')
            updates = _read_csv(folder/'updates.csv')
            graphs = folder/'graphs'
            graphs.mkdir(exist_ok=True)
        except Exception as exc:
            report['errors'].append(f'{type(exc).__name__}: {exc}')
            return report

        def episode_panel(columns, ylabel, title, **kwargs):
            return dict(rows=episodes, x_key='episode', columns=columns,
                        ylabel=ylabel, title=title, **kwargs)

        plots = {
            'episode_rewards.png': [
                episode_panel([('reward_sum', 'Valid reward sum')], 'Reward sum', 'Episode valid reward sum'),
                episode_panel([('reward_mean', 'Valid reward mean')], 'Reward mean', 'Episode valid reward mean')],
            'episode_error.png': [episode_panel([('rms_px', 'RMS'), ('p95_px', '95th percentile')],
                                                'Pixel error (px)', 'Episode measured tracking error')],
            'episode_ratios.png': [episode_panel([('hit_ratio', 'Hit ratio'), ('valid_ratio', 'Valid result ratio')],
                                                 'Ratio', 'Episode hit and valid result ratios', ratio=True)],
            'update_losses.png': [dict(rows=updates, x_key='update', columns=[(key, label)],
                                        ylabel=label, title=label+' by SAC update')
                                  for key, label in [('actor_loss', 'Actor loss'), ('critic_loss', 'Critic loss'),
                                                     ('alpha', 'Entropy coefficient alpha'), ('alpha_loss', 'Alpha loss')]],
        }
        for name, panels in plots.items():
            try:
                _save_figure(graphs, name, panels)
                report['files'].append(str(graphs/name))
            except Exception as exc:
                report['errors'].append(f'{name}: {type(exc).__name__}: {exc}')
        manifest = dict(schema='m5-3-moving-graphs-v1', generated_utc=datetime.now(timezone.utc).isoformat(),
                        sources={'episodes.csv': len(episodes), 'updates.csv': len(updates)},
                        measured_counts={key: sum(math.isfinite(x) and math.isfinite(y)
                            for x, y in zip(*_series(rows, x_key, key)))
                            for rows, x_key, keys in [(episodes, 'episode',
                                ('reward_sum', 'reward_mean', 'rms_px', 'p95_px', 'hit_ratio', 'valid_ratio')),
                                (updates, 'update', ('actor_loss', 'critic_loss', 'alpha', 'alpha_loss'))]
                            for key in keys},
                        note='Missing/nonfinite metrics remain gaps, not zero. Rewards summarize valid results only; '
                             'loss changes alone do not establish policy convergence or hardware tracking performance.',
                        files=[Path(name).name for name in report['files']], errors=list(report['errors']))
        temp = graphs/f'.manifest.{uuid.uuid4().hex}.tmp'
        try:
            temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
            os.replace(temp, graphs/'manifest.json')
        except Exception as exc:
            report['errors'].append(f'manifest.json: {type(exc).__name__}: {exc}')
        finally:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path, help='Saved M5 run folder containing episodes.csv / updates.csv')
    args = parser.parse_args(argv)
    report = generate_plots(args.folder)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
