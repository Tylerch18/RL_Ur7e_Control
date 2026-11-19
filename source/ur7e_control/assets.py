"""Load the UR7e MuJoCo model.

Isaac / Menagerie do not ship a dedicated UR7e MJCF. The UR7e shares the same
nominal kinematics and mesh family as the UR5e (850 mm reach). We load DeepMind
Menagerie's ``universal_robots_ur5e`` and treat it as UR7e for this project.
Payload differs (7.5 kg vs 5 kg); that is fine for most control / RL work.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import mujoco_menagerie as mm
import numpy as np

MENAGERIE_ROBOT = "universal_robots_ur5e"
EE_SITE = "attachment_site"
TARGET_BODY = "target"
TARGET_MOCAP = "target"
HOME_QPOS = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0], dtype=np.float64)


def robot_path() -> Path:
    """Return the cached Menagerie model directory for the UR5e/UR7e stand-in."""
    return Path(mm.get(MENAGERIE_ROBOT).path())


def load_ur7e_spec(*, with_target: bool = True) -> mujoco.MjSpec:
    """Build an editable MjSpec for the UR7e scene (floor + arm [+ reach target])."""
    robot = mm.get(MENAGERIE_ROBOT)
    # Prefer the packaged scene (floor / lights); fall back to the bare arm.
    entry = robot.default_scene or robot.default_model or "scene"
    spec = robot.spec(entry)
    spec.modelname = "ur7e"

    if with_target:
        # Mocap target the env can move each reset / step without welding joints.
        target = spec.worldbody.add_body(name=TARGET_BODY, mocap=True, pos=[0.45, 0.0, 0.35])
        target.add_geom(
            name="target_geom",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.025, 0, 0],
            rgba=[0.9, 0.2, 0.2, 0.85],
            contype=0,
            conaffinity=0,
        )
    return spec


def load_ur7e_model(*, with_target: bool = True) -> mujoco.MjModel:
    """Compile and return an ``MjModel`` for the UR7e scene."""
    return load_ur7e_spec(with_target=with_target).compile()


def ee_xpos(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """End-effector position from the flange attachment site."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    if sid < 0:
        raise RuntimeError(f"site '{EE_SITE}' not found in model")
    return data.site_xpos[sid].copy()
