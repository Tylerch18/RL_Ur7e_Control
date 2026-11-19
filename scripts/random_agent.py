#!/usr/bin/env python3
"""Roll out a random policy in UR7e-Reach-v0 (sanity check)."""

from __future__ import annotations

import argparse

import gymnasium as gym

import ur7e_control  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = gym.make("UR7e-Reach-v0")
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        total = 0.0
        for _ in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        print(
            f"episode={ep} return={total:.3f} "
            f"distance={info.get('distance', float('nan')):.4f} "
            f"success={info.get('success')}"
        )
    env.close()


if __name__ == "__main__":
    main()
