"""Minimum-jerk (quintic) joint trajectories through waypoints."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ur7e_control.control.limits import A_MAX, V_MAX
from ur7e_control.trajectory.validate import validate_waypoints

Array = NDArray[np.floating]


def _segment_duration(delta: Array, v_max: Array, a_max: Array) -> float:
    """Duration so a rest-to-rest quintic respects per-joint v/a limits."""
    # For s(τ)=10τ³-15τ⁴+6τ⁵ on [0,1], max |ṡ| = 1.875/T * |Δ|, max |s̈| = 5.7735/T² * |Δ|
    # (approx peak factors for unit quintic). Use conservative factors 1.875 and 5.8.
    abs_d = np.abs(np.asarray(delta, dtype=np.float64))
    t_v = 1.875 * abs_d / np.maximum(v_max, 1e-9)
    t_a = np.sqrt(5.8 * abs_d / np.maximum(a_max, 1e-9))
    return float(max(np.max(t_v), np.max(t_a), 1e-3))


def _quintic_scalars(tau: float) -> tuple[float, float, float]:
    """Position / velocity / accel shape for rest-to-rest quintic (τ in [0,1])."""
    tau = float(np.clip(tau, 0.0, 1.0))
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    sd = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
    sdd = 60 * tau - 180 * tau**2 + 120 * tau**3
    return s, sd, sdd


class MinJerkTrajectory:
    """Piecewise rest-to-rest quintic through joint waypoints.

    Segment durations are auto-timed from ``v_max`` / ``a_max`` unless ``durations``
    is provided. Angles are never wrapped.
    """

    def __init__(
        self,
        waypoints: Array,
        *,
        v_max: Array | None = None,
        a_max: Array | None = None,
        durations: Array | None = None,
    ) -> None:
        self.waypoints = np.asarray(waypoints, dtype=np.float64)
        validate_waypoints(self.waypoints)
        if self.waypoints.shape[0] < 2:
            raise ValueError("need at least 2 waypoints")

        self.v_max = np.asarray(v_max if v_max is not None else V_MAX, dtype=np.float64)
        self.a_max = np.asarray(a_max if a_max is not None else A_MAX, dtype=np.float64)

        n_seg = self.waypoints.shape[0] - 1
        if durations is None:
            durs = []
            for i in range(n_seg):
                delta = self.waypoints[i + 1] - self.waypoints[i]
                durs.append(_segment_duration(delta, self.v_max, self.a_max))
            self.segment_durations = np.asarray(durs, dtype=np.float64)
        else:
            self.segment_durations = np.asarray(durations, dtype=np.float64)
            if self.segment_durations.shape != (n_seg,):
                raise ValueError(f"durations must have shape ({n_seg},)")

        self._starts = np.concatenate([[0.0], np.cumsum(self.segment_durations)])
        self._duration = float(self._starts[-1])

    @property
    def duration(self) -> float:
        return self._duration

    def sample(self, t: float) -> tuple[Array, Array, Array]:
        t = float(np.clip(t, 0.0, self._duration))
        # Last sample sits on the final waypoint.
        if t >= self._duration:
            q = self.waypoints[-1].copy()
            return q, np.zeros(6), np.zeros(6)

        idx = int(np.searchsorted(self._starts, t, side="right") - 1)
        idx = int(np.clip(idx, 0, len(self.segment_durations) - 1))
        t0 = self._starts[idx]
        T = self.segment_durations[idx]
        tau = 0.0 if T <= 0 else (t - t0) / T
        s, sd, sdd = _quintic_scalars(tau)
        q0 = self.waypoints[idx]
        q1 = self.waypoints[idx + 1]
        delta = q1 - q0
        q = q0 + s * delta
        qd = (sd / T) * delta
        qdd = (sdd / (T * T)) * delta
        return q, qd, qdd


def random_waypoints_near_home(
    home: Array,
    *,
    n_waypoints: int = 4,
    offset_scale: float = 0.25,
    rng: np.random.Generator | None = None,
) -> Array:
    """Sample waypoints as home + small offsets (for residual env / demos)."""
    rng = rng or np.random.default_rng()
    home = np.asarray(home, dtype=np.float64)
    wps = [home.copy()]
    for _ in range(n_waypoints - 2):
        offset = rng.uniform(-offset_scale, offset_scale, size=6)
        wps.append(home + offset)
    wps.append(home.copy())
    return np.asarray(wps, dtype=np.float64)
