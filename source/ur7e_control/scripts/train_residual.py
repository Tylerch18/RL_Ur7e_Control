"""Train a residual tracker policy (SB3 PPO if available, else numpy ARS)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _eval_rms(env, policy_fn, n_episodes: int, seed: int) -> float:
    rms_list = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + 1000 + i)
        done = False
        errors = []
        while not done:
            action = policy_fn(obs)
            obs, _, terminated, truncated, info = env.step(action)
            errors.append(info.get("rms_so_far", 0.0))
            done = terminated or truncated
        rms_list.append(errors[-1] if errors else 0.0)
    return float(np.mean(rms_list))


def _baseline_policy(_obs: np.ndarray) -> np.ndarray:
    return np.zeros(6, dtype=np.float32)


def _train_ars(env_id: str, timesteps: int, seed: int, logdir: Path) -> Path:
    """Simple linear ARS residual (fallback when SB3 is unavailable)."""
    import gymnasium as gym

    import ur7e_control  # noqa: F401

    rng = np.random.default_rng(seed)
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    W = np.zeros((act_dim, obs_dim), dtype=np.float64)
    n_dir = 16
    nu = 0.08
    alpha = 0.02
    steps = 0
    episode = 0

    def policy(obs, weight=W):
        a = weight @ np.asarray(obs, dtype=np.float64)
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    while steps < timesteps:
        deltas = [rng.normal(0.0, 1.0, size=W.shape) for _ in range(n_dir)]
        rewards = []
        for delta in deltas:
            for sign in (+1.0, -1.0):
                Ww = W + sign * nu * delta
                obs, _ = env.reset(seed=seed + episode)
                episode += 1
                total = 0.0
                done = False
                while not done:
                    action = policy(obs, Ww)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    total += reward
                    steps += 1
                    done = terminated or truncated
                    if steps >= timesteps:
                        break
                rewards.append((sign, delta, total))
                if steps >= timesteps:
                    break
            if steps >= timesteps:
                break
        # Update from sorted returns.
        if len(rewards) >= 4:
            rewards.sort(key=lambda x: x[2], reverse=True)
            top = rewards[: max(2, len(rewards) // 4)]
            std = np.std([r for _, _, r in top]) + 1e-6
            grad = np.zeros_like(W)
            for sign, delta, ret in top:
                grad += sign * delta * ((ret - np.mean([r for _, _, r in top])) / std)
            W += alpha * grad / max(len(top), 1)

    env.close()
    out = logdir / "residual_ars.npz"
    logdir.mkdir(parents=True, exist_ok=True)
    np.savez(out, W=W)
    # Hold-out compare vs baseline.
    env = gym.make(env_id)
    base_rms = _eval_rms(env, _baseline_policy, n_episodes=5, seed=seed + 50_000)

    def trained(obs):
        return policy(obs, W)

    train_rms = _eval_rms(env, trained, n_episodes=5, seed=seed + 50_000)
    env.close()
    print(f"baseline RMS: {base_rms:.6f}")
    print(f"ARS residual RMS (same seeds): {train_rms:.6f}")
    print(f"saved {out}")
    return out


def _train_ppo(env_id: str, timesteps: int, seed: int, logdir: Path) -> Path:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    import ur7e_control  # noqa: F401

    logdir.mkdir(parents=True, exist_ok=True)
    env = make_vec_env(env_id, n_envs=4, seed=seed)
    model = PPO("MlpPolicy", env, verbose=1, seed=seed, tensorboard_log=str(logdir / "tb"))
    model.learn(total_timesteps=timesteps)
    out = logdir / "residual_ppo.zip"
    model.save(str(out))
    env.close()

    # Compare to zero residual on held-out seeds.
    import gymnasium as gym

    eval_env = gym.make(env_id)
    base_rms = _eval_rms(eval_env, _baseline_policy, n_episodes=5, seed=seed + 50_000)

    def trained(obs):
        a, _ = model.predict(obs, deterministic=True)
        return np.asarray(a, dtype=np.float32)

    train_rms = _eval_rms(eval_env, trained, n_episodes=5, seed=seed + 50_000)
    eval_env.close()
    print(f"baseline RMS: {base_rms:.6f}")
    print(f"PPO residual RMS (same seeds): {train_rms:.6f}")
    print(f"saved {out}")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train residual trajectory policy")
    p.add_argument("--timesteps", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--logdir", type=Path, default=Path("logs/residual"))
    p.add_argument("--algo", choices=("auto", "ppo", "ars"), default="auto")
    args = p.parse_args(argv)

    env_id = "UR7e-TrajResidual-v0"
    use_ppo = args.algo == "ppo"
    if args.algo == "auto":
        try:
            import stable_baselines3  # noqa: F401

            use_ppo = True
        except ImportError:
            use_ppo = False

    if use_ppo:
        _train_ppo(env_id, args.timesteps, args.seed, args.logdir)
    else:
        if args.algo == "auto":
            print("stable-baselines3 not available; using numpy ARS")
        else:
            print("using numpy ARS")
        _train_ars(env_id, args.timesteps, args.seed, args.logdir)


if __name__ == "__main__":
    main()
