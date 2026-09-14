"""SAC 사전학습, 체크포인트 및 replay buffer 저장. 실제 장치 연결 없음."""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import platform
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from .calibration import report
from .config import Config
from .env import TrackingEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100_000, help="이번 실행에서 추가 학습할 step")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", help="Config JSON; 생략하면 기본값")
    parser.add_argument("--out", default="captures/M3/sac_straight")
    parser.add_argument("--resume", help="기존 model.zip 경로; 같은 폴더의 config/replay 필요")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        parser.error("출력 폴더가 비어 있지 않습니다. 새 --out을 사용하세요.")
    cfg = Config.load(args.config)
    if args.resume:
        source = Path(args.resume).parent
        saved_cfg = Config.load(source / "config.json")
        if args.config and asdict(saved_cfg) != asdict(cfg):
            parser.error("resume config must match the saved training config")
        cfg = saved_cfg
        if not (source / "replay_buffer.pkl").exists():
            parser.error("resume requires replay_buffer.pkl next to model.zip")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    env = Monitor(TrackingEnv(cfg), str(out / "monitor.csv"))
    env.reset(seed=args.seed)
    if args.resume:
        model = SAC.load(args.resume, env=env, device=args.device)
        model.load_replay_buffer(source / "replay_buffer.pkl")
        model.set_random_seed(args.seed)
    else:
        model = SAC("MlpPolicy", env, learning_rate=3e-4, buffer_size=100_000,
                    learning_starts=1_000, batch_size=128, tau=.005, gamma=.99,
                    train_freq=1, gradient_steps=1, ent_coef="auto",
                    policy_kwargs=dict(net_arch=[128, 128]), seed=args.seed,
                    device=args.device, verbose=1)
    cfg.save(out / "config.json")
    metadata = dict(algorithm="SAC", observation="M0 Tx-only 6D", action="absolute angle 2D; quantization and legacy slew from config",
                    seed=args.seed, requested_additional_steps=args.steps, resume=args.resume,
                    python=platform.python_version(),
                    versions={p: importlib.metadata.version(p) for p in
                              ("numpy", "torch", "gymnasium", "stable-baselines3")},
                    calibration=report(), hardware_validated=False)
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    callback = CheckpointCallback(save_freq=10_000, save_path=str(out / "checkpoints"), name_prefix="sac")
    try:
        model.learn(total_timesteps=args.steps, callback=callback, reset_num_timesteps=not bool(args.resume))
    finally:
        # Ctrl-C에서도 복구 가능한 최신 모델/버퍼를 보존한다.
        model.save(out / "model")
        model.save_replay_buffer(out / "replay_buffer.pkl")
        metadata["completed_total_steps"] = model.num_timesteps
        (out / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        env.close()
    print(f"Saved: {out / 'model.zip'}")


if __name__ == "__main__":
    main()

