"""Basic env smoke tests (no viewer)."""

from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np

import ur7e_control  # noqa: F401
from ur7e_control.assets import ARM_DOF, load_ur7e_model


def test_model_has_hande():
    model = load_ur7e_model(with_target=True, with_gripper=True)
    names = [model.joint(i).name for i in range(model.njnt)]
    assert any("hande_left_finger" in n for n in names)
    assert any("hande_right_finger" in n for n in names)
    assert model.nu == 7
    assert model.nq == 8
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper/tcp") >= 0


def test_reset_and_step():
    env = gym.make("UR7e-Reach-v0")
    obs, info = env.reset(seed=0)
    assert obs.shape == (22,)
    assert env.action_space.shape == (7,)
    assert "target" in info
    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info2 = env.step(action)
    assert obs2.shape == (22,)
    assert np.isfinite(reward)
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))
    assert "distance" in info2
    assert "gripper_ctrl" in info2
    env.close()


def test_fixed_target_option():
    env = gym.make("UR7e-Reach-v0")
    target = np.array([0.4, 0.1, 0.3], dtype=np.float64)
    obs, info = env.reset(seed=1, options={"target": target})
    np.testing.assert_allclose(info["target"], target, atol=1e-6)
    np.testing.assert_allclose(obs[-3:], target.astype(np.float32), atol=1e-5)
    # Arm joints are the first 6 qpos entries.
    assert obs[:ARM_DOF].shape == (ARM_DOF,)
    env.close()
