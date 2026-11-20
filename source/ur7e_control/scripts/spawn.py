"""Smoke-test: load UR7e + Hand-E and hold the home pose (optional viewer)."""

from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np

from ur7e_control.assets import (
    ARM_DOF,
    GRIPPER_CTRL_CLOSED,
    GRIPPER_CTRL_OPEN,
    apply_home,
    ee_xpos,
    load_ur7e_model,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Spawn UR7e + Hand-E in MuJoCo")
    parser.add_argument("--seconds", type=float, default=5.0, help="How long to run")
    parser.add_argument("--headless", action="store_true", help="No interactive viewer")
    parser.add_argument("--close-gripper", action="store_true", help="Start with Hand-E closed")
    parser.add_argument("--target", nargs=3, type=float, metavar=("X", "Y", "Z"), default=None)
    args = parser.parse_args(argv)

    model = load_ur7e_model(with_target=True, with_gripper=True)
    data = mujoco.MjData(model)
    grip = GRIPPER_CTRL_CLOSED if args.close_gripper else GRIPPER_CTRL_OPEN
    apply_home(model, data, gripper_ctrl=grip)

    if args.target is not None:
        mocap_id = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")]
        data.mocap_pos[mocap_id] = np.asarray(args.target, dtype=np.float64)
        mujoco.mj_forward(model, data)

    print(f"joints: {[model.joint(i).name for i in range(model.njnt)]}")
    print(f"actuators: {[model.actuator(i).name for i in range(model.nu)]}")
    print(f"ee_pos (Hand-E TCP): {ee_xpos(model, data)}")

    if args.headless:
        # Briefly exercise open→close so fingers move under headless CI.
        end = time.time() + args.seconds
        while time.time() < end:
            t = time.time()
            data.ctrl[ARM_DOF] = GRIPPER_CTRL_OPEN + 0.5 * (1.0 + np.sin(2.0 * t)) * (
                GRIPPER_CTRL_CLOSED - GRIPPER_CTRL_OPEN
            )
            mujoco.mj_step(model, data)
        print(f"ee_pos_final: {ee_xpos(model, data)}")
        print(f"finger_qpos: {data.qpos[ARM_DOF:ARM_DOF + 2]}")
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        end = time.time() + args.seconds
        while viewer.is_running() and time.time() < end:
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()
