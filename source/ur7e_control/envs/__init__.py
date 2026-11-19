"""Gymnasium environments for UR7e."""

from gymnasium.envs.registration import register

from ur7e_control.envs.reach import Ur7eReachEnv

register(
    id="UR7e-Reach-v0",
    entry_point="ur7e_control.envs.reach:Ur7eReachEnv",
    max_episode_steps=200,
)

__all__ = ["Ur7eReachEnv"]
