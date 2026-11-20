#!/usr/bin/env python3
"""Interactive UR7e + Hand-E demo.

On macOS run with mjpython:
  mjpython scripts/demo_viewer.py
"""

from __future__ import annotations

import time

import mujoco
import mujoco.viewer
import numpy as np

from ur7e_control.assets import (
    ARM_DOF,
    ARM_HOME_QPOS,
    GRIPPER_CTRL_CLOSED,
    GRIPPER_CTRL_OPEN,
    apply_home,
    load_ur7e_model,
)


def main() -> None:
    model = load_ur7e_model(with_target=True, with_gripper=True)
    data = mujoco.MjData(model)
    apply_home(model, data, gripper_ctrl=GRIPPER_CTRL_OPEN)

    mocap_id = int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")])
    data.mocap_pos[mocap_id] = np.array([0.45, 0.15, 0.35])
    mujoco.mj_forward(model, data)

    print("UR7e + Hand-E demo — close the window to quit.")
    print("Arm moves around home; Hand-E opens/closes; red sphere is the reach target.")

    t0 = time.time()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t = time.time() - t0
            data.ctrl[:ARM_DOF] = ARM_HOME_QPOS + 0.35 * np.array(
                [
                    np.sin(0.6 * t),
                    0.4 * np.sin(0.45 * t + 0.5),
                    0.5 * np.sin(0.5 * t + 1.0),
                    0.4 * np.sin(0.7 * t + 1.5),
                    0.5 * np.sin(0.8 * t),
                    np.sin(0.9 * t + 0.3),
                ]
            )
            # Open/close the Hand-E (0=open, 255=closed).
            phase = 0.5 * (1.0 + np.sin(0.7 * t))
            data.ctrl[ARM_DOF] = GRIPPER_CTRL_OPEN + phase * (GRIPPER_CTRL_CLOSED - GRIPPER_CTRL_OPEN)

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
