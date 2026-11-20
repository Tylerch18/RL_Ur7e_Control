# RL UR7e Control

Cross-platform **MuJoCo + Gymnasium** simulation and RL for the Universal Robots **UR7e** with a **Robotiq Hand-E** gripper, plus a **joint-space trajectory tracker** designed for sim-to-real.

Runs on **macOS (Apple Silicon / Intel), Windows, and Linux**.

## Finished

Completed from the joint-space trajectory / sim-to-real plan:

- [x] Shared trajectory API (`MinJerkTrajectory`, validation, Cartesian stub)
- [x] `RobotInterface` + `JointTracker` at 500 Hz (clip + rate-limit)
- [x] `SimUR7e` speedJ-like plant (delay, lag, gain, noise, domain randomization)
- [x] `RTDEUR7e` real backend (dry-run default; live gated by `--ip` + `--i-am-at-the-robot`)
- [x] `run_scripted.py` — sim/real runner, NPZ logs, plots, RMS metrics
- [x] `sysid.py` — fit delay/tau from logs
- [x] `UR7e-TrajResidual-v0` + `train_residual.py` (PPO / ARS) + `play_residual.py`
- [x] Pytest coverage + real-robot safety procedure (below)
- [x] MuJoCo UR7e + Hand-E assets, cross-platform `.venv` setup

**Not done yet (by design):** Cartesian IK trajectories, learned policy on the real arm, fitting sim DR from real hardware logs.

## Setup

```bash
bash scripts/setup_env.sh
source scripts/activate_env.sh
```

Optional extras:

```bash
uv pip install -e ".[rl,dev]"     # PPO + matplotlib
uv pip install -e ".[real]"       # ur_rtde for a real UR
```

## Scripted trajectory tracking (sim)

```bash
source scripts/activate_env.sh
python -m ur7e_control.scripts.run_scripted --backend sim --seed 0
python -m ur7e_control.scripts.sysid logs/scripted/scripted_sim_seed0.npz
```

## Residual RL (sim)

```bash
python -m ur7e_control.scripts.train_residual --timesteps 20000
```

Gym id: `UR7e-TrajResidual-v0` — residual joint-velocity on top of the 500 Hz scripted tracker.

### What trajectory is that?

Not a fixed Cartesian path. Each episode samples **4 joint waypoints** near home
(`home → random → random → home`), connected by **min-jerk quintics**. The policy
only adds a small velocity residual on top of the scripted tracker (it does **not** choose waypoints).

Visualize a rollout (plots + optional MuJoCo viewer):

```bash
python -m ur7e_control.scripts.play_residual --model logs/residual/residual_ppo.zip --seed 0
python -m ur7e_control.scripts.play_residual --baseline --seed 0   # scripted only
mjpython -m ur7e_control.scripts.play_residual --model logs/residual/residual_ppo.zip --viewer
```

Compare the desired trajectory with PPO's executed motion in the viewer:

```powershell
.\.venv\Scripts\python.exe -m ur7e_control.scripts.play_residual --viewer --loop --speed 0.25 --seed 0
```

The orange dashed path is the reference; the blue trail is the scripted tracker
plus PPO's residual correction. Moving markers show the desired and actual
Hand-E tip at the same instant, with the TCP distance error displayed in mm.
PPO runs at 50 Hz while the tracker runs at 500 Hz. `--speed` changes playback
timing only. Close the viewer to finish.

A headless run saves `logs/play/play_residual_seed0.png` (3D and top-view TCP
paths with tracking errors), `play_residual_seed0_joints.png` (six joint angle
comparisons), and the NPZ samples. These comparisons use the MuJoCo Hand-E TCP
in world coordinates, matching the viewer. The reference is generated from
joint waypoints, rather than recorded demonstrations.

## Reach env (legacy demo)

- `UR7e-Reach-v0` — EE reach with Hand-E command channel

## Architecture (trajectory)

- `MinJerkTrajectory` — joint waypoints, quintic segments, auto-timed from vel/acc limits
- `JointTracker` — `qd_cmd = qd_ref + lead*qdd_ref + Kp*(q_ref - q)` (clip + rate-limit)
- `SimUR7e` — speedJ-like plant (delay, lag, gain, noise, domain randomization)
- `RTDEUR7e` — real backend via `ur_rtde` (**dry-run by default**)

Control rate: **500 Hz** (`dt = 0.002`).

## Real-robot safety procedure

**Do not run learned policies on hardware until the scripted baseline is clean.**

1. Verify the exact trajectory in sim with the **same** velocity/acceleration caps you will use on the arm:
   ```bash
   python -m ur7e_control.scripts.run_scripted --backend sim --offset-scale 0.15
   ```
2. On the real network, start with **dry-run** (default — no motion):
   ```bash
   python -m ur7e_control.scripts.run_scripted --backend real --ip 192.168.x.x
   ```
3. Live motion requires **both** `--ip` and `--i-am-at-the-robot` (and `--live`):
   ```bash
   python -m ur7e_control.scripts.run_scripted \
     --backend real --ip 192.168.x.x --i-am-at-the-robot --live \
     --offset-scale 0.1
   ```
4. First runs use safe caps (~0.4 rad/s, ~1 rad/s²) and small offsets from home.
5. The backend `moveJ`s to the trajectory start and **refuses** to stream if `|q_actual − q_start|` is large.
6. Abort (`speedStop`) on tracking error, missed RTDE period, or any exception; always stop in `finally`.
7. Log full-rate NPZ (`t, q_ref, qd_ref, q, qd, qd_cmd`) and run `sysid.py` before trusting sim DR ranges.

## Assets

- **Arm:** Menagerie `universal_robots_ur5e` (same nominal kinematics as UR7e; do **not** trust mesh inertias for dynamics).
- **Gripper:** `assets/robotiq_hande/` (BSD-3-Clause).

## Do not commit

`logs/`, `.venv/`, `MUJOCO_LOG.TXT`, checkpoints — see `.gitignore`. **Do** commit vendored Hand-E assets.
