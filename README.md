# RL UR7e Control

Isaac Lab (on Isaac Sim) environment for Universal Robots **UR7e** simulation and RL control.

## Why Isaac Lab (not Isaac Gym)

| Stack | Status | Use for this project |
| --- | --- | --- |
| **Isaac Lab + Isaac Sim** | Current | **Yes** — RL, assets, PhysX |
| Isaac Gym / IsaacGymEnvs | Deprecated | No |

## Platform requirement

Isaac Sim needs **Linux x86_64 + NVIDIA GPU** (or Windows 11 + NVIDIA). It does **not** run natively on macOS / Apple Silicon.

On a Mac you can develop this repo; create and run the sim env on a lab GPU machine or cloud GPU (e.g. [Isaac Launchable / NVIDIA Brev](https://github.com/isaac-sim/IsaacLab)).

## Create the env (Linux GPU host)

```bash
cd RL_Ur7e_Control
bash scripts/setup_linux_env.sh
source scripts/activate_env.sh
```

This creates `env_isaaclab/` (Python 3.12), installs Isaac Sim 6 + Isaac Lab, and editable-installs this project.

## Smoke test

```bash
source scripts/activate_env.sh
python scripts/spawn_ur7e.py --headless
```

## UR7e asset note

Isaac Lab has no built-in UR7e USD. Official UR7e kinematics match the **UR5e** reach class (850 mm); we spawn Isaac Sim’s `UniversalRobots/ur5e/ur5e.usd` via `ur7e_control.robots.UR7E_CFG`. Payload differs (7.5 kg vs 5 kg) — fine for most control/RL work; swap in a URDF from [Universal_Robots_ROS2_Description](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description) (`ur_type:=ur7e`) if you need an exact UR7e model later.
