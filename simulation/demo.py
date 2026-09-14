"""평가 CSV를 영상 좌표계로 재생하는 세미나용 뷰어. 카메라 영상 합성이 아님."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", help="frames.csv/config.json이 있는 평가 결과 폴더")
    parser.add_argument("--controller", default="SAC", choices=["B0", "B1", "SAC"])
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--gif", help="선택: GUI 대신 GIF 저장 경로")
    args = parser.parse_args()
    root = Path(args.evaluation)
    cfg = json.loads((root / "config.json").read_text())
    with (root / "frames.csv").open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["controller"] == args.controller and int(r["seed"]) == args.seed]
    if not rows:
        parser.error("해당 controller/seed 데이터가 없습니다.")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set(xlim=(0, cfg["width"]), ylim=(cfg["height"], 0), xlabel="u (px)", ylabel="v (px)")
    ax.set_aspect("equal")
    ax.scatter([cfg["laser_u"]], [cfg["laser_v"]], marker="+", s=150, color="red", label="Laser reference")
    point, = ax.plot([], [], "bo", label="PV center (simulated)")
    trace, = ax.plot([], [], color="blue", alpha=.3)
    ax.legend(loc="lower left")

    def update(i):
        row = rows[i]
        point.set_data([float(row["u"])], [float(row["v"])])
        history = rows[max(0, i-30):i+1]
        trace.set_data([float(r["u"]) for r in history], [float(r["v"]) for r in history])
        ax.set_title(f"SIMULATION {args.controller} | {float(row['time_s']):.1f}s | error {float(row['error_px']):.1f}px")
        return point, trace

    animation = FuncAnimation(fig, update, frames=len(rows), interval=1000*cfg["dt"], blit=False)
    if args.gif:
        animation.save(args.gif, writer=PillowWriter(fps=1/cfg["dt"]))
    else:
        plt.show()


if __name__ == "__main__":
    main()
