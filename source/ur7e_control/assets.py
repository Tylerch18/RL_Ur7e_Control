"""Load the UR7e + Robotiq Hand-E MuJoCo model.

The arm uses DeepMind Menagerie's ``universal_robots_ur5e`` (same nominal
kinematics as UR7e). The Hand-E MJCF/meshes are vendored under
``assets/robotiq_hande`` (from the Menagerie Hand-E draft model).
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import mujoco_menagerie as mm
import numpy as np

MENAGERIE_ROBOT = "universal_robots_ur5e"
GRIPPER_PREFIX = "gripper/"
EE_SITE = f"{GRIPPER_PREFIX}tcp"
TARGET_BODY = "target"
ARM_DOF = 6
GRIPPER_CTRL_OPEN = 0.0
GRIPPER_CTRL_CLOSED = 255.0

# Arm home (rad). Gripper finger joints start fully open (0 m).
ARM_HOME_QPOS = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0], dtype=np.float64)
# Back-compat alias used by older scripts.
HOME_QPOS = ARM_HOME_QPOS

_REPO_ROOT = Path(__file__).resolve().parents[2]
HANDE_XML = _REPO_ROOT / "assets" / "robotiq_hande" / "hande.xml"


def robot_path() -> Path:
    """Return the cached Menagerie model directory for the UR5e/UR7e stand-in."""
    return Path(mm.get(MENAGERIE_ROBOT).path())


def hande_path() -> Path:
    """Path to the vendored Hand-E MJCF."""
    if not HANDE_XML.is_file():
        raise FileNotFoundError(f"Hand-E MJCF missing: {HANDE_XML}")
    return HANDE_XML


def load_ur7e_spec(*, with_target: bool = True, with_gripper: bool = True) -> mujoco.MjSpec:
    """Build an editable MjSpec for the UR7e scene (floor + arm [+ Hand-E] [+ target])."""
    robot = mm.get(MENAGERIE_ROBOT)
    entry = robot.default_scene or robot.default_model or "scene"
    spec = robot.spec(entry)
    spec.modelname = "ur7e_hande" if with_gripper else "ur7e"

    if with_gripper:
        # Prefer Hand-E contact settings on the composed model.
        spec.option.impratio = 10
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        gripper = mujoco.MjSpec.from_file(str(hande_path()))
        # Align child options with the parent so attach() does not warn.
        gripper.option.timestep = spec.option.timestep
        gripper.option.tolerance = spec.option.tolerance
        gripper.option.impratio = spec.option.impratio
        gripper.option.cone = spec.option.cone
        spec.attach(gripper, prefix=GRIPPER_PREFIX, site=spec.site("attachment_site"))
        # Menagerie home keyframe is for the bare arm; drop the stale empty key.
        for key in list(spec.keys):
            if key.name == "home":
                spec.delete(key)

    if with_target:
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


def load_ur7e_model(*, with_target: bool = True, with_gripper: bool = True) -> mujoco.MjModel:
    """Compile and return an ``MjModel`` for the UR7e scene."""
    return load_ur7e_spec(with_target=with_target, with_gripper=with_gripper).compile()


def apply_home(model: mujoco.MjModel, data: mujoco.MjData, *, gripper_ctrl: float = GRIPPER_CTRL_OPEN) -> None:
    """Set arm home pose and gripper command (0=open, 255=closed)."""
    data.qpos[:ARM_DOF] = ARM_HOME_QPOS
    if model.nq > ARM_DOF:
        data.qpos[ARM_DOF:] = 0.0
    data.qvel[:] = 0.0
    data.ctrl[:ARM_DOF] = ARM_HOME_QPOS
    if model.nu > ARM_DOF:
        data.ctrl[ARM_DOF] = float(np.clip(gripper_ctrl, GRIPPER_CTRL_OPEN, GRIPPER_CTRL_CLOSED))
    mujoco.mj_forward(model, data)


def ee_xpos(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """TCP position at the Hand-E tip site (falls back to flange site)."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    if sid < 0:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    if sid < 0:
        raise RuntimeError(f"site '{EE_SITE}' not found in model")
    return data.site_xpos[sid].copy()
