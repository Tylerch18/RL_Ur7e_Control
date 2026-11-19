#!/usr/bin/env python3
"""Interactive UR7e demo: hold a red target and track it with joint PD ctrl.

On macOS run with mjpython:
  mjpython scripts/demo_viewer.py
"""

from __future__ import annotations

import time

import mujoco
import mujoco.viewer
import numpy as np

from ur7e_control.assets import HOME_QPOS, load_ur7e_model


def main() -> None:
    model = load_ur7e_model(with_target=True)
    data = mujoco.MjData(model)

    data.qpos[:6] = HOME_QPOS
    data.ctrl[:] = HOME_QPOS
    mocap_id = int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")])
    data.mocap_pos[mocap_id] = np.array([0.45, 0.15, 0.35])
    mujoco.mj_forward(model, data)

    print("UR7e MuJoCo demo — close the window to quit.")
    print("The red sphere is the reach target; the arm tracks moving joint setpoints.")

    t0 = time.time()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t = time.time() - t0
            # Slow joint setpoints around home so the arm visibly moves.
            data.ctrl[:] = HOME_QPOS + 0.35 * np.array(
                [
                    np.sin(0.6 * t),
                    0.4 * np.sin(0.45 * t + 0.5),
                    0.5 * np.sin(0.5 * t + 1.0),
                    0.4 * np.sin(0.7 * t + 1.5),
                    0.5 * np.sin(0.8 * t),
                    np.sin(0.9 * t + 0.3),
                ]
            )
            # Orbit the target slowly in front of the arm.
            data.mocap_pos[mocap_id] = np.array(
                [
                    0.45 + 0.12 * np.cos(0.35 * t),
                    0.15 * np.sin(0.35 * t),
                    0.32 + 0.08 * np.sin(0.5 * t),
                ]
            )

            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            leftover = model.opt.timestep - (time.time() - step_start)
            if leftover > 0:
                time.sleep(leftover)


if __name__ == "__main__":
    main()
