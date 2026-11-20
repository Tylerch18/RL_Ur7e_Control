"""UR7e + Hand-E end-effector reach environment (MuJoCo + Gymnasium)."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from ur7e_control.assets import (
    ARM_DOF,
    ARM_HOME_QPOS,
    EE_SITE,
    GRIPPER_CTRL_OPEN,
    TARGET_BODY,
    apply_home,
    ee_xpos,
    load_ur7e_model,
)


class Ur7eReachEnv(gym.Env):
    """Reach a random 3D target with the UR7e + Hand-E TCP.

    Observation (22,):
      qpos (8), qvel (8), ee position (3), target position (3)

    Action (7,):
      arm joint position targets (6) + Hand-E command in [0, 255]
      (0 = open, 255 = closed)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        *,
        render_mode: str | None = None,
        frame_skip: int = 10,
        success_threshold: float = 0.05,
        target_low: tuple[float, float, float] = (0.30, -0.35, 0.15),
        target_high: tuple[float, float, float] = (0.65, 0.35, 0.55),
        ctrl_delta_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.success_threshold = success_threshold
        self.target_low = np.asarray(target_low, dtype=np.float64)
        self.target_high = np.asarray(target_high, dtype=np.float64)
        self.ctrl_delta_scale = float(ctrl_delta_scale)

        self.model = load_ur7e_model(with_target=True, with_gripper=True)
        self.data = mujoco.MjData(self.model)

        self._ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
        self._target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, TARGET_BODY)
        self._target_mocap_id = int(self.model.body_mocapid[self._target_body_id])
        if self._ee_site_id < 0 or self._target_mocap_id < 0:
            raise RuntimeError("UR7e+Hand-E model is missing ee site or mocap target")

        self.nu = int(self.model.nu)
        self.nq = int(self.model.nq)
        if self.nu != 7 or self.nq != 8:
            raise RuntimeError(f"expected nq=8 nu=7, got nq={self.nq} nu={self.nu}")

        ctrl_low = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        ctrl_high = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        if self.ctrl_delta_scale > 0:
            span = np.full(self.nu, self.ctrl_delta_scale, dtype=np.float32)
            # Keep gripper absolute [0, 255] even in delta-arm mode.
            span[ARM_DOF] = 255.0
            low = -span.copy()
            high = span.copy()
            low[ARM_DOF] = 0.0
            high[ARM_DOF] = 255.0
            self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            self.action_space = spaces.Box(low=ctrl_low, high=ctrl_high, dtype=np.float32)

        obs_dim = self.nq + self.nq + 3 + 3
        obs_high = np.full(obs_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        self._viewer: Any | None = None
        self._renderer: mujoco.Renderer | None = None
        self._ctrl_low = ctrl_low
        self._ctrl_high = ctrl_high

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos[: self.nq].astype(np.float32)
        qvel = self.data.qvel[: self.nq].astype(np.float32)
        ee = self.data.site_xpos[self._ee_site_id].astype(np.float32)
        target = self.data.mocap_pos[self._target_mocap_id].astype(np.float32)
        return np.concatenate([qpos, qvel, ee, target], dtype=np.float32)

    def _set_target(self, pos: np.ndarray) -> None:
        self.data.mocap_pos[self._target_mocap_id] = pos
        self.data.mocap_quat[self._target_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}

        apply_home(self.model, self.data, gripper_ctrl=GRIPPER_CTRL_OPEN)
        noise = self.np_random.uniform(low=-0.1, high=0.1, size=ARM_DOF)
        arm_q = np.clip(
            ARM_HOME_QPOS + noise,
            self._ctrl_low[:ARM_DOF],
            self._ctrl_high[:ARM_DOF],
        )
        self.data.qpos[:ARM_DOF] = arm_q
        self.data.ctrl[:ARM_DOF] = arm_q
        self.data.ctrl[ARM_DOF] = GRIPPER_CTRL_OPEN

        if "target" in options:
            target = np.asarray(options["target"], dtype=np.float64)
        else:
            target = self.np_random.uniform(low=self.target_low, high=self.target_high)
        self._set_target(target)

        mujoco.mj_forward(self.model, self.data)
        obs = self._get_obs()
        info = {"ee_pos": ee_xpos(self.model, self.data), "target": target.copy()}
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64).reshape(self.nu)
        if self.ctrl_delta_scale > 0:
            ctrl = self.data.ctrl.copy()
            ctrl[:ARM_DOF] = self.data.qpos[:ARM_DOF] + action[:ARM_DOF]
            ctrl[ARM_DOF] = action[ARM_DOF]
        else:
            ctrl = action
        self.data.ctrl[:] = np.clip(ctrl, self._ctrl_low, self._ctrl_high)

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        ee = self.data.site_xpos[self._ee_site_id]
        target = self.data.mocap_pos[self._target_mocap_id]
        dist = float(np.linalg.norm(ee - target))
        success = dist < self.success_threshold

        reward = -dist
        if success:
            reward += 1.0

        terminated = success
        truncated = False
        obs = self._get_obs()
        info = {
            "distance": dist,
            "success": success,
            "ee_pos": ee.copy(),
            "target": target.copy(),
            "gripper_ctrl": float(self.data.ctrl[ARM_DOF]),
        }

        if self.render_mode == "human":
            self.render()
        return obs, float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()

        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        return None

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
