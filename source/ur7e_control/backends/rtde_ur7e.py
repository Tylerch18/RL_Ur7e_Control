"""Real UR7e backend via ``ur_rtde`` with dry-run default and hard safety gates."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ur7e_control.control.interface import RobotInterface
from ur7e_control.control.limits import CONTROL_DT, HOME_Q, SAFE_A_MAX, SAFE_V_MAX

Array = NDArray[np.floating]


class RealRobotSafetyError(RuntimeError):
    """Raised when a live-run safety check fails."""


class RTDEUR7e(RobotInterface):
    """RTDE joint-velocity interface.

    Defaults to **dry-run** (commands logged, no motion). Live motion requires
    ``live=True`` which must only be set when the operator passes ``--ip`` and
    ``--i-am-at-the-robot``.
    """

    def __init__(
        self,
        *,
        ip: str | None = None,
        live: bool = False,
        i_am_at_the_robot: bool = False,
        v_max: Array | None = None,
        a_max: Array | None = None,
        start_tol: float = 0.05,
        dt: float = CONTROL_DT,
        q0_dry: Array | None = None,
    ) -> None:
        self.ip = ip
        self.live = bool(live)
        self.i_am_at_the_robot = bool(i_am_at_the_robot)
        self.v_max = np.asarray(v_max if v_max is not None else SAFE_V_MAX, dtype=np.float64)
        self.a_max = np.asarray(a_max if a_max is not None else SAFE_A_MAX, dtype=np.float64)
        self.start_tol = float(start_tol)
        self.dt = float(dt)
        self._period_start: float | None = None
        self._rtde_c: Any | None = None
        self._rtde_r: Any | None = None
        self._stopped = False
        self._last_cmd = np.zeros(6, dtype=np.float64)
        # Dry-run plant (integrator) so scripted loops work without hardware.
        self._q = np.asarray(q0_dry if q0_dry is not None else HOME_Q, dtype=np.float64).copy()
        self._qd = np.zeros(6, dtype=np.float64)

        if self.live:
            self._connect_live()

    def _connect_live(self) -> None:
        if not self.ip:
            raise RealRobotSafetyError("live mode requires --ip")
        if not self.i_am_at_the_robot:
            raise RealRobotSafetyError(
                "live mode requires --i-am-at-the-robot (refuse to move without explicit flag)"
            )
        try:
            from rtde_control import RTDEControlInterface  # type: ignore
            from rtde_receive import RTDEReceiveInterface  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RealRobotSafetyError(
                "ur_rtde is not installed. pip install 'ur7e-control[real]'"
            ) from exc

        self._rtde_r = RTDEReceiveInterface(self.ip)
        self._rtde_c = RTDEControlInterface(self.ip)

    def move_to_start(self, q_start: Array, *, speed: float = 0.3, accel: float = 0.5) -> None:
        """MoveJ to trajectory start (live) or set dry-run state; refuse if far."""
        q_start = np.asarray(q_start, dtype=np.float64)
        q, _ = self.read()
        err = np.max(np.abs(q - q_start))
        if err > self.start_tol and self.live:
            # On live, moveJ then re-check.
            assert self._rtde_c is not None
            self._rtde_c.moveJ(q_start.tolist(), speed, accel)
            q, _ = self.read()
            err = np.max(np.abs(q - q_start))
            if err > self.start_tol:
                raise RealRobotSafetyError(
                    f"|q_actual - q_start|={err:.4f} exceeds tol={self.start_tol}"
                )
        elif not self.live:
            if err > self.start_tol:
                # Snap dry-run state to start (no real motion).
                self._q = q_start.copy()
                self._qd[:] = 0.0

    def read(self) -> tuple[Array, Array]:
        if self.live:
            assert self._rtde_r is not None
            q = np.asarray(self._rtde_r.getActualQ(), dtype=np.float64)
            qd = np.asarray(self._rtde_r.getActualQd(), dtype=np.float64)
            return q, qd
        return self._q.copy(), self._qd.copy()

    def command_velocity(self, qd_cmd: Array) -> None:
        if self._stopped:
            return
        qd_cmd = np.clip(np.asarray(qd_cmd, dtype=np.float64), -self.v_max, self.v_max)
        # Rate-limit using a_max.
        dq = qd_cmd - self._last_cmd
        max_step = self.a_max * self.dt
        dq = np.clip(dq, -max_step, max_step)
        qd_cmd = self._last_cmd + dq
        self._last_cmd = qd_cmd.copy()

        if self.live:
            assert self._rtde_c is not None
            # speedJ(qd, acceleration, time) — time≈dt for streaming.
            a = float(np.min(self.a_max))
            self._rtde_c.speedJ(qd_cmd.tolist(), a, self.dt)
        else:
            self._qd = qd_cmd.copy()
            self._q = self._q + self._qd * self.dt

    def wait_period(self) -> None:
        if self.live:
            assert self._rtde_c is not None
            # Prefer RTDE period helpers when available.
            if hasattr(self._rtde_c, "initPeriod") and self._period_start is None:
                self._rtde_c.initPeriod()
            if hasattr(self._rtde_c, "waitPeriod"):
                self._rtde_c.waitPeriod()
                return
        now = time.perf_counter()
        if self._period_start is None:
            self._period_start = now
            return
        elapsed = now - self._period_start
        sleep = self.dt - elapsed
        if sleep > 0:
            time.sleep(sleep)
        elif self.live and elapsed > 1.5 * self.dt:
            self.stop()
            raise RealRobotSafetyError(f"missed RTDE period: elapsed={elapsed:.4f}s")
        self._period_start = time.perf_counter()

    def stop(self) -> None:
        self._stopped = True
        self._last_cmd[:] = 0.0
        self._qd[:] = 0.0
        if self.live and self._rtde_c is not None:
            try:
                self._rtde_c.speedStop()
            finally:
                try:
                    self._rtde_c.stopScript()
                except Exception:
                    pass

    def close(self) -> None:
        self.stop()
        self._rtde_c = None
        self._rtde_r = None
