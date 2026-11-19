# RL UR7e Control

Cross-platform **MuJoCo + Gymnasium** simulation and RL for the Universal Robots **UR7e**.

Runs on **macOS (Apple Silicon / Intel), Windows, and Linux**. Train locally or over SSH on a remote GPU machine — no Isaac Sim required.

## Why MuJoCo (not Isaac)

| | MuJoCo (this repo) | Isaac Lab / Sim |
| --- | --- | --- |
| Mac / Win / Linux laptops | Yes | No Mac |
| Light install | Yes | Heavy + NVIDIA |
| Remote headless train | SSH + this env | Possible, heavier |
| Photoreal vision / RTX | Limited | Strong |

## Setup

```bash
bash scripts/setup_env.sh
source scripts/activate_env.sh
```

Creates `.venv` (Python 3.12), installs MuJoCo + Gymnasium + SB3, and prefetches the Menagerie arm meshes.

## Quick checks

```bash
# Headless spawn
python -m ur7e_control.scripts.spawn --headless

# Interactive viewer (on macOS prefer mjpython for the native viewer)
python -m ur7e_control.scripts.spawn
# mjpython -m ur7e_control.scripts.spawn

# Random-policy sanity check
python scripts/random_agent.py
```

## Train

Local or remote (headless):

```bash
source scripts/activate_env.sh
python -m ur7e_control.scripts.train_reach --timesteps 200000 --n-envs 8
```

Remote pattern:

```bash
ssh user@gpu-box
cd RL_Ur7e_Control && source scripts/activate_env.sh
python -m ur7e_control.scripts.train_reach --timesteps 1_000_000 --n-envs 16
```

## Environment

- **Gym id:** `UR7e-Reach-v0`
- **Action:** 6 absolute joint position targets (rad)
- **Obs:** joint pos/vel (12) + EE xyz + target xyz (18)
- **Reward:** `-||ee − target||` (+1 on success within 5 cm)

## UR7e asset note

There is no dedicated Menagerie UR7e MJCF. Official UR7e kinematics match the **UR5e** reach class (850 mm), so we load `universal_robots_ur5e` via `mujoco-menagerie` and label it UR7e. Payload differs (7.5 kg vs 5 kg).
