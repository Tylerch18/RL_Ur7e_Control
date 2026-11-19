"""Train PPO on UR7e-Reach-v0 (Stable-Baselines3).

Works headless on any machine. For remote GPU training, SSH in, activate the
venv, and run this script — no display required.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train PPO on UR7e-Reach-v0")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logdir", type=Path, default=Path("logs/reach_ppo"))
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--progress", action="store_true", help="Show SB3 progress bar")
    args = parser.parse_args(argv)

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.callbacks import CheckpointCallback
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "RL extras not installed. Run:\n  uv sync --extra rl\n  # or: pip install '.[rl]'"
        ) from exc

    import ur7e_control  # noqa: F401  — registers Gymnasium envs

    args.logdir.mkdir(parents=True, exist_ok=True)
    env = make_vec_env("UR7e-Reach-v0", n_envs=args.n_envs, seed=args.seed)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(args.logdir / "tb"),
    )
    checkpoint = CheckpointCallback(
        save_freq=max(10_000 // args.n_envs, 1),
        save_path=str(args.logdir / "checkpoints"),
        name_prefix="ur7e_reach",
    )
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint,
        progress_bar=args.progress,
    )
    out = args.logdir / "ur7e_reach_final.zip"
    model.save(str(out))
    env.close()
    print(f"saved {out}")


if __name__ == "__main__":
    main()
