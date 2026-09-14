"""동일 seed/궤적에서 B0, B1, SAC 평가 및 CSV/JSON/그림 저장."""
import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from .config import Config
from .control import proportional_action
from .env import TrackingEnv


def rollout(cfg, seed, controller, model=None, scenario=None):
    env = TrackingEnv(cfg)
    obs, _ = env.reset(seed=seed, options=dict(scenario=scenario) if scenario else None)
    rows = []
    for step in range(cfg.episode_steps):
        if controller == "B0":
            action = np.zeros(2)
        elif controller == "B1":
            action = proportional_action(obs, cfg, env.nominal_focal*np.pi/180)
        elif controller == "SAC" and model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            raise ValueError("Unknown controller or missing SAC model")
        obs, reward, _, _, info = env.step(action)
        eu, ev = info["true_error_px"]
        rows.append(dict(controller=controller, seed=seed, step=step+1,
                         time_s=info["time_s"], scenario=info["scenario"],
                         u=info["uv"][0], v=info["uv"][1], e_u=eu, e_v=ev,
                         error_px=float(np.hypot(eu, ev)), visible=int(info["visible"]),
                         pan_cmd_deg=info["command_deg"][0], tilt_cmd_deg=info["command_deg"][1],
                         pan_actual_sim_deg=info["actual_angles_deg"][0],
                         tilt_actual_sim_deg=info["actual_angles_deg"][1],
                         target_x_m=info["target_world_m"][0], target_y_m=info["target_world_m"][1],
                         reward=reward))
    env.close()
    errors = np.array([r["error_px"] for r in rows])
    commands = np.array([[r["pan_cmd_deg"], r["tilt_cmd_deg"]] for r in rows])
    changes = np.diff(commands, axis=0) / (2*cfg.angle_limit_deg)
    summary = dict(controller=controller, seed=seed, scenario=rows[0]["scenario"],
                   pointing_rms_px=float(np.sqrt(np.mean(errors**2))),
                   pointing_p95_px=float(np.percentile(errors, 95)),
                   command_variation=float(np.mean(np.sum(changes**2, axis=1))) if len(changes) else 0.,
                   max_command_step_deg=float(np.max(np.abs(np.diff(np.vstack([np.zeros(2), commands]), axis=0)))),
                   visible_fraction=float(np.mean([r["visible"] for r in rows])),
                   within_engineering_threshold_fraction=float(np.mean(errors <= cfg.engineering_threshold_px)),
                   episode_return=float(sum(r["reward"] for r in rows)))
    return rows, summary


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plot(path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    first_seed = rows[0]["seed"]
    for controller in dict.fromkeys(r["controller"] for r in rows):
        selected = [r for r in rows if r["controller"] == controller and r["seed"] == first_seed]
        for ax, field in zip(axes, ("error_px", "pan_cmd_deg", "tilt_cmd_deg")):
            ax.plot([r["time_s"] for r in selected], [r[field] for r in selected], label=controller)
    for ax, title in zip(axes, ("Pointing error (px)", "Pan command (deg)", "Tilt command (deg)")):
        ax.set_ylabel(title)
        ax.grid(alpha=.3)
        ax.legend()
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Coordinate simulation only | paired seed {first_seed}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="학습한 model.zip; 없으면 B0/B1만 실행")
    parser.add_argument("--config", help="모델 없이 평가할 때 설정")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--scenario", choices=["sine", "linear", "stationary"])
    parser.add_argument("--stress", action="store_true", help="미실측 가정: tau=.15s, 1 step 지연, 3px 노이즈")
    parser.add_argument("--out", default="simulation/runs/evaluation")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.model and args.config:
        parser.error("Model evaluation uses its saved config; use --stress for a declared stress test")
    cfg = Config.load(Path(args.model).parent / "config.json" if args.model else args.config)
    if args.stress:
        cfg = replace(cfg, actuator_tau_s=.15, command_delay_steps=1, measurement_noise_px=3.)
    model = None
    if args.model:
        import torch
        from stable_baselines3 import SAC
        torch.set_num_threads(1)
        model = SAC.load(args.model, device="cpu")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        parser.error("출력 폴더가 비어 있지 않습니다. 새 --out을 사용하세요.")
    out.mkdir(parents=True, exist_ok=True)
    rows, episodes = [], []
    for seed in range(args.seed_start, args.seed_start+args.episodes):
        for controller in (["B0", "B1", "SAC"] if model else ["B0", "B1"]):
            trajectory, metrics = rollout(cfg, seed, controller, model, args.scenario)
            rows.extend(trajectory)
            episodes.append(metrics)
    numeric = [k for k in episodes[0] if k not in ("controller", "seed", "scenario")]
    summary = {}
    for controller in dict.fromkeys(r["controller"] for r in episodes):
        selected = [r for r in episodes if r["controller"] == controller]
        summary[controller] = {k: dict(mean=float(np.mean([r[k] for r in selected])),
                                      std=float(np.std([r[k] for r in selected], ddof=1)) if len(selected)>1 else 0.)
                               for k in numeric}
    write_csv(out / "frames.csv", rows)
    write_csv(out / "episodes.csv", episodes)
    cfg.save(out / "config.json")
    result = dict(simulation_only=True, model=args.model, stress=args.stress, episodes=args.episodes,
                  seed_start=args.seed_start, scenario=args.scenario or "seeded mixture",
                  metrics_source="simulator ground truth for evaluation; not policy input",
                  summary=summary)
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    make_plot(out / "comparison.png", rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
