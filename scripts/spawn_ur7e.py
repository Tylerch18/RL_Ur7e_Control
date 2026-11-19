#!/usr/bin/env python3
"""Smoke-test: spawn a UR7e (ur5e USD) in Isaac Lab / Isaac Sim.

Run on a Linux + NVIDIA GPU host after setup:
  source scripts/activate_env.sh
  python scripts/spawn_ur7e.py
  python scripts/spawn_ur7e.py --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Spawn UR7e in Isaac Lab.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports that require the Kit app must come after AppLauncher.
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from ur7e_control.robots import UR7E_CFG  # noqa: E402


@configclass
class Ur7eSceneCfg(InteractiveSceneCfg):
    ground = sim_utils.GroundPlaneCfg()
    dome_light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    robot = UR7E_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main() -> None:
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0))
    scene = InteractiveScene(Ur7eSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5))
    sim.reset()

    robot: Articulation = scene["robot"]
    print(f"[UR7e] spawned. joint names: {robot.data.joint_names}")

    while simulation_app.is_running():
        # Hold the default pose (zero joint-velocity targets).
        robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_vel))
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
