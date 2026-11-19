"""Basic env smoke tests (no viewer)."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

import ur7e_control  # noqa: F401


def test_reset_and_step():
    env = gym.make("UR7e-Reach-v0")
    obs, info = env.reset(seed=0)
    assert obs.shape == (18,)
    assert "target" in info
    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info2 = env.step(action)
    assert obs2.shape == (18,)
    assert np.isfinite(reward)
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))
    assert "distance" in info2
    env.close()


def test_fixed_target_option():
    env = gym.make("UR7e-Reach-v0")
    target = np.array([0.4, 0.1, 0.3], dtype=np.float64)
    obs, info = env.reset(seed=1, options={"target": target})
    np.testing.assert_allclose(info["target"], target, atol=1e-6)
    np.testing.assert_allclose(obs[-3:], target.astype(np.float32), atol=1e-5)
    env.close()
