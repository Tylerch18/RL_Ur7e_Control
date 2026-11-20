"""Simulated UR7e backend: speedJ-like plant (not torque / mesh dynamics)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ur7e_control.control.interface import RobotInterface
from ur7e_control.control.limits import A_MAX, CONTROL_DT, HOME_Q, V_MAX

Array = NDArray[np.floating]


@dataclass
class SimPlantParams:
    """Domain-randomizable speedJ plant parameters."""

    delay_s: float = 0.006
    tau_s: float = 0.025
    gain: Array | None = None
    a_max: Array | None = None
    meas_noise_std: float = 1e-4

    def __post_init__(self) -> None:
        if self.gain is None:
            self.gain = np.ones(6, dtype=np.float64)
        else:
            self.gain = np.asarray(self.gain, dtype=np.float64)
        if self.a_max is None:
            self.a_max = A_MAX.copy()
        else:
            self.a_max = np.asarray(self.a_max, dtype=np.float64)


def sample_plant_params(rng: np.random.Generator | None = None) -> SimPlantParams:
    """Draw plant params from the default domain-randomization ranges."""
    rng = rng or np.random.default_rng()
    return SimPlantParams(
        delay_s=float(rng.uniform(0.0, 0.012)),
        tau_s=float(rng.uniform(0.010, 0.040)),
        gain=rng.uniform(0.9, 1.1, size=6),
        a_max=A_MAX.copy(),
        meas_noise_std=1e-4,
    )


class SimUR7e(RobotInterface):
    """Integrate a delayed first-order velocity plant at 500 Hz.

    Optionally mirrors ``q`` into MuJoCo for visualization (dynamics of the mesh
    model are not used for tracking).
    """

    def __init__(
        self,
        *,
        q0: Array | None = None,
        params: SimPlantParams | None = None,
        dt: float = CONTROL_DT,
        visualize: bool = False,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.dt = float(dt)
        self.params = params or SimPlantParams()
        self.rng = rng or np.random.default_rng()
        self.q = np.asarray(q0 if q0 is not None else HOME_Q, dtype=np.float64).copy()
        self.qd = np.zeros(6, dtype=np.float64)
        self._qd_filt = np.zeros(6, dtype=np.float64)
        self._cmd_buffer: list[Array] = []
        self._delay_steps = max(0, int(round(self.params.delay_s / self.dt)))
        self._stopped = False
        self._visualize = visualize
        self._model = None
        self._data = None
        if visualize:
            self._init_viewer_model()

    def _init_viewer_model(self) -> None:
        # Lazy import so headless tracking does not require Menagerie load.
        from ur7e_control.assets import apply_home, load_ur7e_model

        self._model = load_ur7e_model(with_target=False, with_gripper=True)
        self._data = __import__("mujoco").MjData(self._model)
        apply_home(self._model, self._data)
        self._data.qpos[:6] = self.q
        __import__("mujoco").mj_forward(self._model, self._data)

    def randomize(self, rng: np.random.Generator | None = None) -> None:
        self.params = sample_plant_params(rng or self.rng)
        self._delay_steps = max(0, int(round(self.params.delay_s / self.dt)))
        self._cmd_buffer.clear()

    def reset(self, q0: Array | None = None) -> None:
        self.q = np.asarray(q0 if q0 is not None else HOME_Q, dtype=np.float64).copy()
        self.qd[:] = 0.0
        self._qd_filt[:] = 0.0
        self._cmd_buffer.clear()
        self._stopped = False
        if self._data is not None and self._model is not None:
            self._data.qpos[:6] = self.q
            self._data.qvel[:6] = 0.0
            __import__("mujoco").mj_forward(self._model, self._data)

    def read(self) -> tuple[Array, Array]:
        noise = self.rng.normal(0.0, self.params.meas_noise_std, size=6)
        return self.q + noise, self.qd + noise

    def command_velocity(self, qd_cmd: Array) -> None:
        if self._stopped:
            return
        cmd = np.clip(np.asarray(qd_cmd, dtype=np.float64), -V_MAX, V_MAX)
        self._cmd_buffer.append(cmd.copy())
        # Apply delayed command.
        if len(self._cmd_buffer) <= self._delay_steps:
            delayed = np.zeros(6)
        else:
            delayed = self._cmd_buffer[-(self._delay_steps + 1)]
            # Drop old samples to bound memory.
            if len(self._cmd_buffer) > self._delay_steps + 50:
                self._cmd_buffer = self._cmd_buffer[-(self._delay_steps + 50) :]

        # First-order lag toward gain * delayed command.
        target = self.params.gain * delayed
        alpha = self.dt / max(self.params.tau_s, 1e-6)
        alpha = min(alpha, 1.0)
        qd_des = self._qd_filt + alpha * (target - self._qd_filt)
        # Accel clamp.
        dq = qd_des - self.qd
        max_step = self.params.a_max * self.dt
        dq = np.clip(dq, -max_step, max_step)
        self.qd = self.qd + dq
        self._qd_filt = self.qd.copy()
        self.q = self.q + self.qd * self.dt

        if self._data is not None and self._model is not None:
            self._data.qpos[:6] = self.q
            self._data.ctrl[:6] = self.q
            __import__("mujoco").mj_forward(self._model, self._data)

    def wait_period(self) -> None:
        # Discrete plant already advanced one ``dt`` in ``command_velocity``.
        return

    def stop(self) -> None:
        self._stopped = True
        self.qd[:] = 0.0
        self._qd_filt[:] = 0.0
        self._cmd_buffer.clear()
