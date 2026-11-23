import csv
import json
import tempfile
import unittest
from pathlib import Path

from common.arc_protocol import VERSION
from Tx.Controller.arc_sender import ArcSender
from tools.analysis.arc_voltage import SAMPLE_FIELDS, audit_tx, compare, find_pair


class ArcVoltageTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def write_run(self, mode, rows, **overrides):
        directory = self.root / f"arc_test_{mode}"
        directory.mkdir()
        manifest = dict(schema_version=1, mode=mode, status="completed", pair="pair01",
                        profile=VERSION, radius_m=5, steps=3, period_s=4, bit_ms=50,
                        sample_ms=5, window_s=[2.75, 3.25], initial_pan_deg=180,
                        tilt_deg=20, simulated_motion=True, min_contrast_v=.024,
                        frame_tolerance_s=.18, pca9685_bus=1, ina219_bus=3,
                        pca9685_address="0x40", ina219_address="0x40")
        manifest.update(overrides)
        (directory / "manifest.json").write_text(json.dumps(manifest))
        with (directory / "steps.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["step", "theta_deg", *SAMPLE_FIELDS])
            writer.writeheader()
            for step, voltage, accepted in rows:
                writer.writerow(dict(step=step, theta_deg=180-step, pan_deg=180-step, tilt_deg=20,
                                     frame_accepted=accepted, window_source="detected" if accepted else "scheduled",
                                     sample_count=50 if voltage is not None else 0, missing_samples=0,
                                     mean_v=voltage, status="ok" if voltage is not None else "no_samples"))
        return directory

    def test_missed_frames_remain_in_voltage_comparison_and_missing_rows_stay_visible(self):
        a = self.write_run("baseline", [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1)])
        b = self.write_run("corrected", [(0, 2, 1), (1, 5, 0), (3, None, 1)])
        rows, summary = compare(a, b)
        self.assertEqual([r["delta_v"] for r in rows], [1, 3, None, None])
        self.assertEqual(rows[2]["b_status"], "missing_row")
        self.assertEqual(summary["mean_paired_delta_v"], 2)
        self.assertEqual(summary["mean_both_confirmed_complete_delta_v"], 1)
        self.assertEqual(summary["paired_voltage_poses"], 2)
        self.assertEqual(summary["corrected_coverage"]["missed_frames"], 1)
        self.assertEqual(summary["corrected_coverage"]["missing_rows"], 1)

    def test_refuse_unpaired_runs(self):
        a = self.write_run("baseline", [])
        b = self.write_run("corrected", [], pair="pair02", period_s=5)
        with self.assertRaisesRegex(ValueError, "pair, period_s"):
            compare(a, b)

    def test_refuse_aborted_run(self):
        a = self.write_run("baseline", [], status="aborted")
        b = self.write_run("corrected", [])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            compare(a, b)

    def test_pair_discovery_and_ambiguity(self):
        a = self.write_run("baseline", [])
        b = self.write_run("corrected", [])
        self.assertEqual(find_pair(self.root, "pair01"), (a, b))
        duplicate = self.root / "arc_duplicate"
        duplicate.mkdir()
        (duplicate / "manifest.json").write_bytes((b / "manifest.json").read_bytes())
        with self.assertRaisesRegex(ValueError, "2 completed corrected"):
            find_pair(self.root, "pair01")

    def test_audit_real_sender_sequence_and_reject_invalid_evidence(self):
        sent, events = [], []
        now = 0.0

        def send(message):
            sent.append(message)
            return True

        sender = ArcSender(send, events.append, lambda _: None, 50, 4, 1, pair_id="pair01",
                           mode="A", clock=lambda: now)
        sender.start()
        sender.handle_event(dict(event="laser", request_id=sent[-1]["request_id"], laser_gpio_level=0))
        for seq in range(2):
            now = sender.epoch + 4 * seq
            sender.tick(True)
            command = sent[-1]
            sender.handle_event(dict(event="laser_pattern", request_id=command["request_id"],
                                     state="completed", off_call_done_ms=3000,
                                     settings={k: command[k] for k in ("bits", "bit_ms", "repeats")},
                                     trace=[dict(index=i, level=int(b)) for i,b in enumerate(command["bits"])]))
        sender.handle_event(dict(event="laser", request_id=sent[-1]["request_id"], laser_gpio_level=0))
        path = self.root / "tx.jsonl"
        settings = dict(steps=1, period_s=4, bit_ms=50, pair="pair01", profile=VERSION)

        def write(items):
            path.write_text("".join(json.dumps(item) + "\n" for item in items))

        write(events)
        self.assertEqual(audit_tx(path, settings, "A")["completed_frames"], 2)
        with self.assertRaisesRegex(ValueError, "metadata"):
            audit_tx(path, settings, "B")
        write(events[:-1])
        with self.assertRaisesRegex(ValueError, "missing completed"):
            audit_tx(path, settings, "A")
        write([*events[:-1], dict(events[-1], gpio_off_confirmed=False)])
        with self.assertRaisesRegex(ValueError, "GPIO OFF"):
            audit_tx(path, settings, "A")
        response = next(item["response"] for item in events if item["event"] == "arc_response")
        response["trace"].pop()
        write(events)
        with self.assertRaisesRegex(ValueError, "GPIO trace"):
            audit_tx(path, settings, "A")
