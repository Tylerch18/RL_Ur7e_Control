"""Residual RL env: 50 Hz policy on top of 500 Hz scripted tracker."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ur7e_control.backends.sim_ur7e import SimUR7e, sample_plant_params
from ur7e_control.control.limits import CONTROL_DT, HOME_Q
from ur7e_control.control.tracker import JointTracker
from ur7e_control.trajectory.min_jerk import MinJerkTrajectory, random_waypoints_near_home
from ur7e_control.trajectory.validate import validate_trajectory

POLICY_HZ = 50
INNER_STEPS = int(round((1.0 / POLICY_HZ) / CONTROL_DT))  # 10
RESIDUAL_SCALE = 0.1  # rad/s
ERROR_TERMINATE = 0.1


class Ur7eTrajResidualEnv(gym.Env):
    """Learn a residual joint-velocity correction on top of the scripted tracker."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        offset_scale: float = 0.25,
        kp: float = 8.0,
        domain_randomize: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.offset_scale = float(offset_scale)
        self.kp = float(kp)
        self.domain_randomize = bool(domain_randomize)
        self.np_random = np.random.default_rng(seed)

        # Obs: e(6), qd(6), qd_ref(6), qdd_ref(6), prev_a(6), phase(1) = 31
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(31,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        self.robot = SimUR7e(rng=self.np_random)
        self.tracker = JointTracker(kp=self.kp)
        self.traj: MinJerkTrajectory | None = None
        self.t = 0.0
        self.prev_action = np.zeros(6, dtype=np.float64)
        self._episode_errors: list[float] = []

    def _obs(self, q, qd, q_ref, qd_ref, qdd_ref) -> np.ndarray:
        e = q_ref - q
        phase = 0.0 if self.traj is None or self.traj.duration <= 0 else self.t / self.traj.duration
        # Scale roughly to O(1).
        return np.concatenate(
            [
                e / 0.1,
                qd / 1.0,
                qd_ref / 1.0,
                qdd_ref / 5.0,
                self.prev_action,
                [phase],
            ]
        ).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        options = options or {}

        if self.domain_randomize:
            self.robot.params = sample_plant_params(self.np_random)
            self.robot._delay_steps = max(0, int(round(self.robot.params.delay_s / self.robot.dt)))

        waypoints = options.get("waypoints")
        if waypoints is None:
            waypoints = random_waypoints_near_home(
                HOME_Q, n_waypoints=4, offset_scale=self.offset_scale, rng=self.np_random
            )
        self.traj = MinJerkTrajectory(np.asarray(waypoints, dtype=np.float64))
        validate_trajectory(self.traj)

        q0 = self.traj.waypoints[0]
        self.robot.reset(q0)
        self.tracker.reset()
        self.t = 0.0
        self.prev_action = np.zeros(6, dtype=np.float64)
        self._episode_errors = []

        q, qd = self.robot.read()
        q_ref, qd_ref, qdd_ref = self.traj.sample(0.0)
        obs = self._obs(q, qd, q_ref, qd_ref, qdd_ref)
        return obs, {"duration": self.traj.duration}

    def step(self, action: np.ndarray):
        assert self.traj is not None
        action = np.asarray(action, dtype=np.float64).reshape(6)
        action = np.clip(action, -1.0, 1.0)
        residual = action * RESIDUAL_SCALE

        total_err = 0.0
        terminated = False
        for _ in range(INNER_STEPS):
            q, qd = self.robot.read()
            q_ref, qd_ref, qdd_ref = self.traj.sample(self.t)
            e = q_ref - q
            if np.any(np.abs(e) > ERROR_TERMINATE):
                terminated = True
            qd_cmd = self.tracker.compute(q, q_ref, qd_ref, qdd_ref, residual_qd=residual)
            self.robot.command_velocity(qd_cmd)
            self.robot.wait_period()
            total_err += float(np.linalg.norm(e))
            self.t += CONTROL_DT
            if self.t >= self.traj.duration:
                break
            if terminated:
                break

        q, qd = self.robot.read()
        q_ref, qd_ref, qdd_ref = self.traj.sample(min(self.t, self.traj.duration))
        e_now = q_ref - q
        self._episode_errors.append(float(np.linalg.norm(e_now)))

        reward = (
            -float(np.linalg.norm(e_now))
            - 0.01 * float(np.linalg.norm(action))
            - 0.01 * float(np.linalg.norm(action - self.prev_action))
        )
        self.prev_action = action.copy()

        truncated = self.t >= self.traj.duration
        obs = self._obs(q, qd, q_ref, qd_ref, qdd_ref)
        info = {
            "rms_so_far": float(np.sqrt(np.mean(np.square(self._episode_errors)))),
            "t": self.t,
        }
        return obs, reward, terminated, truncated, info
