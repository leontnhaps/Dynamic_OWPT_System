"""Compare paired Rx arc runs at each planned virtual angle (no hardware access)."""
import argparse
import csv
import json
import math
import sys
import uuid
from datetime import datetime
from pathlib import Path
from statistics import mean

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.arc_protocol import VERSION, stimulus_bits
from common.laser_pattern import pattern_settings

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "captures" / "M6-4"
PAIR_SETTINGS = (
    "pair", "profile", "steps", "period_s", "bit_ms", "sample_ms", "window_s",
    "radius_m", "initial_pan_deg", "tilt_deg", "simulated_motion",
    "min_contrast_v", "frame_tolerance_s", "pca9685_bus", "ina219_bus",
    "pca9685_address", "ina219_address",
)
SAMPLE_FIELDS = (
    "pan_deg", "tilt_deg", "frame_accepted", "window_source", "sample_count",
    "missing_samples", "mean_v", "status",
)


def load_run(folder, mode):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("mode") != mode:
        raise ValueError(f"{folder}: expected version 1 {mode} arc run")
    if manifest.get("status") != "completed":
        raise ValueError(f"{folder}: incomplete run ({manifest.get('status')!r})")
    settings = {key: manifest[key] for key in PAIR_SETTINGS}
    if type(settings["steps"]) is not int or not 1 <= settings["steps"] <= 180:
        raise ValueError(f"{folder}: invalid number of steps")

    rows = {}
    with (folder / "steps.csv").open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"step", "theta_deg", *SAMPLE_FIELDS}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{folder}: steps.csv missing required columns")
        for item in reader:
            step = int(item["step"])
            if (not 0 <= step <= settings["steps"] or step in rows
                    or float(item["theta_deg"]) != 180 - step):
                raise ValueError(f"{folder}: duplicate or invalid angle at step {step}")
            row = {key: item[key] for key in SAMPLE_FIELDS}
            row["sample_count"] = int(row["sample_count"])
            row["missing_samples"] = int(row["missing_samples"])
            row["frame_accepted"] = int(row["frame_accepted"])
            if (min(row["sample_count"], row["missing_samples"]) < 0 or
                    row["frame_accepted"] not in (0, 1) or
                    row["window_source"] not in ("scheduled", "detected")):
                raise ValueError(f"{folder}: invalid counts or window source at step {step}")
            value = row["mean_v"]
            row["mean_v"] = float(value) if value else None
            if row["mean_v"] is not None and (
                    not math.isfinite(row["mean_v"]) or not row["sample_count"]):
                raise ValueError(f"{folder}: inconsistent voltage at step {step}")
            rows[step] = row
    return settings, rows


def compare(baseline, corrected):
    a_settings, a_rows = load_run(baseline, "baseline")
    b_settings, b_rows = load_run(corrected, "corrected")
    mismatched = [name for name in PAIR_SETTINGS if a_settings[name] != b_settings[name]]
    if mismatched:
        raise ValueError(f"Pair settings differ: {', '.join(mismatched)}")

    angles = []
    for step in range(a_settings["steps"] + 1):
        row = {"step": step, "theta_deg": 180 - step}
        for side, rows in (("a", a_rows), ("b", b_rows)):
            recorded = rows.get(step, {"status": "missing_row"})
            for key in SAMPLE_FIELDS:
                row[f"{side}_{key}"] = recorded.get(key)
        a, b = row["a_mean_v"], row["b_mean_v"]
        paired = a is not None and b is not None
        row["delta_v"] = b - a if paired else None
        row["both_confirmed_complete"] = int(paired and all(
            row[f"{side}_frame_accepted"] == 1 and
            row[f"{side}_window_source"] == "detected" and
            row[f"{side}_status"] == "ok" and row[f"{side}_missing_samples"] == 0
            for side in ("a", "b")
        ))
        angles.append(row)

    summary = {"comparison_version": 1, "baseline": str(baseline.resolve()),
               "corrected": str(corrected.resolve()), "settings": a_settings,
               "planned_poses": len(angles)}
    for side, name in (("a", "baseline"), ("b", "corrected")):
        summary[f"{name}_coverage"] = {
            "accepted_frames": sum(r[f"{side}_frame_accepted"] == 1 for r in angles),
            "missed_frames": sum(r[f"{side}_frame_accepted"] == 0 for r in angles),
            "missing_rows": sum(r[f"{side}_status"] == "missing_row" for r in angles),
            "scheduled_windows": sum(r[f"{side}_window_source"] == "scheduled" for r in angles),
            "missing_voltage": sum(r[f"{side}_mean_v"] is None for r in angles),
            "incomplete_windows": sum(r[f"{side}_status"] != "ok" for r in angles),
        }
    paired = [r["delta_v"] for r in angles if r["delta_v"] is not None]
    confirmed = [r["delta_v"] for r in angles if r["both_confirmed_complete"]]
    summary.update(paired_voltage_poses=len(paired),
                   mean_paired_delta_v=mean(paired) if paired else None,
                   both_confirmed_complete_poses=len(confirmed),
                   mean_both_confirmed_complete_delta_v=mean(confirmed) if confirmed else None)
    return angles, summary


def find_pair(root, pair):
    runs = {"baseline": [], "corrected": []}
    for path in sorted(root.glob("arc_*/manifest.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        mode = data.get("mode")
        if mode in runs and data.get("pair") == pair and data.get("status") == "completed":
            runs[mode].append(path.parent)
    for mode, paths in runs.items():
        if len(paths) != 1:
            raise ValueError(f"{root}: found {len(paths)} completed {mode} runs for {pair!r}; "
                             "supply the two run directories explicitly")
    return runs["baseline"][0], runs["corrected"][0]


def audit_tx(path, settings, mode):
    """Audit the producer's chronological dispatch, GPIO trace and OFF acknowledgements."""
    run_id = None
    origin = None
    pending = None
    next_seq = 0
    finished = False
    maximum_lag = 0.0
    allowed_lag = 0.0
    with path.open(encoding="utf-8-sig") as stream:
        for lineno, line in enumerate(stream, 1):
            try:
                event = json.loads(line)
                kind = event["event"]
                if not isinstance(kind, str) or finished:
                    raise ValueError("unexpected event after end or invalid event name")
                if kind == "arc_start":
                    expected = {"steps": settings["steps"], "period_s": settings["period_s"],
                                "bit_ms": settings["bit_ms"], "pair_id": settings["pair"],
                                "mode": mode, "protocol_version": settings["profile"]}
                    if (run_id is not None or settings["profile"] != VERSION or
                            any(event.get(k) != v for k, v in expected.items())):
                        raise ValueError("start metadata does not match this Rx run")
                    run_id = event["run_id"]
                    if not isinstance(run_id, str) or not run_id:
                        raise ValueError("missing Tx run ID")
                    allowed_lag = float(event["max_dispatch_late_s"])
                    if not math.isfinite(allowed_lag) or allowed_lag < 0:
                        raise ValueError("invalid dispatch tolerance")
                elif kind == "arc_armed":
                    if not run_id or origin is not None:
                        raise ValueError("unexpected arc_armed")
                    origin = float(event["first_due_s"])
                    if not math.isfinite(origin):
                        raise ValueError("invalid start time")
                elif kind == "arc_dispatch":
                    if (origin is None or pending is not None or event["seq"] != next_seq
                            or next_seq > settings["steps"]):
                        raise ValueError("missing, duplicated, or out-of-order dispatch")
                    bits = stimulus_bits(next_seq, settings["bit_ms"])
                    expected = pattern_settings(dict(bits=bits, bit_ms=settings["bit_ms"], repeats=1))
                    request_id = f"{run_id}-frame-{next_seq}"
                    if event["request_id"] != request_id or event["settings"] != expected:
                        raise ValueError("unexpected request ID or stimulus")
                    due, late = float(event["due_s"]), float(event["late_s"])
                    if (not math.isclose(due, origin + next_seq * settings["period_s"],
                                         rel_tol=0, abs_tol=1e-5)
                            or not math.isclose(event["dispatch_s"] - due, late,
                                                rel_tol=0, abs_tol=1e-5)
                            or not 0 <= late <= allowed_lag):
                        raise ValueError("invalid dispatch timeline")
                    maximum_lag = max(maximum_lag, late)
                    pending = (request_id, expected)
                elif kind == "arc_response":
                    if pending is None or event["seq"] != next_seq:
                        raise ValueError("response without matching dispatch")
                    response = event["response"]
                    request_id, expected = pending
                    if response["event"] != "laser_pattern" or response["request_id"] != request_id:
                        raise ValueError("invalid Tx response")
                    if response["state"] == "started":
                        continue
                    if response["state"] != "completed" or response["settings"] != expected:
                        raise ValueError("Tx pattern failed or mismatched settings")
                    trace, bits = response["trace"], expected["bits"]
                    if not isinstance(trace, list) or len(trace) != len(bits) or any(
                            row["index"] != i or row["level"] != int(bits[i])
                            for i, row in enumerate(trace)):
                        raise ValueError("incomplete or inconsistent GPIO trace")
                    off = response["off_call_done_ms"]
                    if type(off) not in (float, int) or not math.isfinite(off) or off < 0:
                        raise ValueError("missing GPIO OFF call timestamp")
                    pending = None
                    next_seq += 1
                elif kind == "arc_end":
                    if (next_seq != settings["steps"] + 1 or pending is not None
                            or event["seq"] != settings["steps"] or event["reason"] != "completed"
                            or event["gpio_off_confirmed"] is not True):
                        raise ValueError("incomplete frame set or GPIO OFF unconfirmed")
                    finished = True
                else:
                    raise ValueError(f"unexpected Tx event {kind!r}")
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    if not finished:
        raise ValueError(f"{path}: missing completed arc_end")
    return {"file": str(path.resolve()), "run_id": run_id, "completed_frames": next_seq,
            "max_dispatch_lag_s": maximum_lag, "gpio_off_confirmed": True}


def plot(angles, path, audit_status):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theta = [r["theta_deg"] for r in angles]
    fig, (volts, delta, coverage) = plt.subplots(
        3, 1, figsize=(9, 7), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 2, 1.3]},
    )
    try:
        for side, label in (("a", "A  baseline"), ("b", "B  corrected")):
            volts.plot(theta, [r[f"{side}_mean_v"] for r in angles], label=label, lw=1.5)
        volts.legend()
        volts.set_ylabel("PV terminal voltage (V)")
        delta.axhline(0, color="0.65", lw=0.8)
        delta.plot(theta, [r["delta_v"] for r in angles], color="#6f52a2", lw=1.4)
        delta.set_ylabel("B minus A (V)")
        labels = []
        for side in ("a", "b"):
            for name, predicate in (
                    ("frame unconfirmed", lambda r: r[f"{side}_frame_accepted"] != 1),
                    ("fallback / incomplete", lambda r: r[f"{side}_window_source"] != "detected"
                     or r[f"{side}_status"] != "ok")):
                x = [r["theta_deg"] for r in angles if predicate(r)]
                coverage.scatter(x, [len(labels)] * len(x), s=14, marker="|", color="black")
                labels.append(f"{side.upper()} {name}")
        coverage.set_yticks(range(len(labels)))
        coverage.set_yticklabels(labels, fontsize=8)
        coverage.set_ylim(len(labels) - .5, -.5)
        coverage.set_xlabel("Virtual arc angle (degrees)")
        coverage.set_xlim(theta[0], theta[-1])
        evidence = "GPIO traces verified" if audit_status == "verified_gpio" else "Tx logs unverified"
        volts.set_title(f"Receiver arc comparison · {evidence}")
        for axes in (volts, delta, coverage):
            axes.grid(axis="x", alpha=0.22)
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, nargs="?", help="completed baseline run directory")
    parser.add_argument("corrected", type=Path, nargs="?", help="matching completed corrected run directory")
    parser.add_argument("--pair", help="find a unique completed baseline and corrected run by pair ID")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_ROOT, help="where --pair searches")
    parser.add_argument("--tx-a", type=Path, help="PC baseline arc_tx_*.jsonl")
    parser.add_argument("--tx-b", type=Path, help="PC corrected arc_tx_*.jsonl")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_ROOT,
                        help="parent of the new comparison directory")
    args = parser.parse_args(argv)
    if bool(args.tx_a) != bool(args.tx_b):
        parser.error("supply both --tx-a and --tx-b")
    try:
        if args.pair and not (args.baseline or args.corrected):
            baseline, corrected = find_pair(args.runs_root, args.pair)
        elif args.baseline and args.corrected and not args.pair:
            baseline, corrected = args.baseline, args.corrected
        else:
            parser.error("supply --pair or the two positional run directories")
        angles, summary = compare(baseline, corrected)
        audit = {"status": "unverified"}
        if args.tx_a:
            a = audit_tx(args.tx_a, summary["settings"], "A")
            b = audit_tx(args.tx_b, summary["settings"], "B")
            if a["run_id"] == b["run_id"]:
                raise ValueError("the two Tx run IDs must differ")
            audit = {"status": "verified_gpio", "baseline": a, "corrected": b}
        summary["tx_audit"] = audit
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = args.out_root / f"arc_pair_{stamp}_{uuid.uuid4().hex[:8]}"
        dest.mkdir(parents=True)
        with (dest / "angles.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(angles[0]))
            writer.writeheader()
            writer.writerows(angles)
        (dest / "comparison.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        try:
            plot(angles, dest / "voltage.png", audit["status"])
        except ModuleNotFoundError as exc:
            if exc.name != "matplotlib":
                raise
            print("Matplotlib is unavailable. CSV and JSON saved; PNG skipped.", file=sys.stderr)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Comparison failed: {exc}\n")
    print(f"{dest}\nPaired voltage: {summary['paired_voltage_poses']}/{len(angles)} poses"
          f"\nTransmitter GPIO records: {audit['status']}")


if __name__ == "__main__":
    main()
