"""Scripted joint-space tracker shared by sim and real."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ur7e_control.control.limits import A_MAX, CONTROL_DT, V_MAX
from ur7e_control.trajectory.types import Trajectory

Array = NDArray[np.floating]


class JointTracker:
    """``qd_cmd = qd_ref + lead * qdd_ref + Kp * (q_ref - q)`` with clip + rate limit."""

    def __init__(
        self,
        *,
        kp: float | Array = 8.0,
        lead: float = 0.0,
        v_max: Array | None = None,
        a_max: Array | None = None,
        dt: float = CONTROL_DT,
    ) -> None:
        self.kp = np.asarray(kp, dtype=np.float64)
        if self.kp.ndim == 0:
            self.kp = np.full(6, float(self.kp))
        self.lead = float(lead)
        self.v_max = np.asarray(v_max if v_max is not None else V_MAX, dtype=np.float64)
        self.a_max = np.asarray(a_max if a_max is not None else A_MAX, dtype=np.float64)
        self.dt = float(dt)
        self._qd_prev = np.zeros(6, dtype=np.float64)

    def reset(self, qd_init: Array | None = None) -> None:
        self._qd_prev = np.zeros(6) if qd_init is None else np.asarray(qd_init, dtype=np.float64).copy()

    def compute(
        self,
        q: Array,
        q_ref: Array,
        qd_ref: Array,
        qdd_ref: Array,
        *,
        residual_qd: Array | None = None,
    ) -> Array:
        q = np.asarray(q, dtype=np.float64)
        q_ref = np.asarray(q_ref, dtype=np.float64)
        qd_ref = np.asarray(qd_ref, dtype=np.float64)
        qdd_ref = np.asarray(qdd_ref, dtype=np.float64)

        qd_cmd = qd_ref + self.lead * qdd_ref + self.kp * (q_ref - q)
        if residual_qd is not None:
            qd_cmd = qd_cmd + np.asarray(residual_qd, dtype=np.float64)

        # Velocity clip.
        qd_cmd = np.clip(qd_cmd, -self.v_max, self.v_max)
        # Acceleration / rate limit.
        dq = qd_cmd - self._qd_prev
        max_step = self.a_max * self.dt
        dq = np.clip(dq, -max_step, max_step)
        qd_cmd = self._qd_prev + dq
        self._qd_prev = qd_cmd.copy()
        return qd_cmd

    def step_refs(self, traj: Trajectory, t: float) -> tuple[Array, Array, Array]:
        return traj.sample(t)
