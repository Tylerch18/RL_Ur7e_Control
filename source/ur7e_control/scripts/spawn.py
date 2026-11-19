"""Smoke-test: load UR7e and hold the home pose (optional viewer)."""

from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np

from ur7e_control.assets import HOME_QPOS, ee_xpos, load_ur7e_model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Spawn UR7e in MuJoCo")
    parser.add_argument("--seconds", type=float, default=5.0, help="How long to run")
    parser.add_argument("--headless", action="store_true", help="No interactive viewer")
    parser.add_argument("--target", nargs=3, type=float, metavar=("X", "Y", "Z"), default=None)
    args = parser.parse_args(argv)

    model = load_ur7e_model(with_target=True)
    data = mujoco.MjData(model)
    data.qpos[:6] = HOME_QPOS
    data.ctrl[:] = HOME_QPOS
    if args.target is not None:
        mocap_id = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")]
        data.mocap_pos[mocap_id] = np.asarray(args.target, dtype=np.float64)
    mujoco.mj_forward(model, data)

    print(f"joints: {[model.joint(i).name for i in range(6)]}")
    print(f"ee_pos: {ee_xpos(model, data)}")

    if args.headless:
        end = time.time() + args.seconds
        while time.time() < end:
            mujoco.mj_step(model, data)
        print(f"ee_pos_final: {ee_xpos(model, data)}")
        return

    # Interactive viewer (on macOS prefer: mjpython -m ur7e_control.scripts.spawn)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        end = time.time() + args.seconds
        while viewer.is_running() and time.time() < end:
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            # Real-time pacing
            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()
