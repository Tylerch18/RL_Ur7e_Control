"""UR7e articulation config for Isaac Lab.

Isaac Lab does not ship a dedicated UR7e USD. The UR7e shares the same
nominal kinematics and mesh family as the UR5e (850 mm reach class); the
main datasheet difference is payload (7.5 kg vs 5 kg). We therefore spawn
Isaac Sim's ur5e USD and label the config as UR7e for this project.

Official URDF/xacro (if you later need a true UR7e import):
  https://github.com/UniversalRobots/Universal_Robots_ROS2_Description
  (ur_type:=ur7e)
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Prefer PhysX schema helpers when available (Isaac Lab 3.x); fall back for 2.x.
try:
    from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

    _rigid_props = PhysxRigidBodyCfg(disable_gravity=True, max_depenetration_velocity=5.0)
    _articulation_props = PhysxArticulationCfg(
        enabled_self_collisions=False,
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=1,
    )
except ImportError:  # pragma: no cover - older Isaac Lab
    _rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=True,
        max_depenetration_velocity=5.0,
    )
    _articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=1,
    )

UR7E_USD = f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur5e/ur5e.usd"

UR7E_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=UR7E_USD,
        rigid_props=_rigid_props,
        articulation_props=_articulation_props,
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.5707963267948966,
            "elbow_joint": 1.5707963267948966,
            "wrist_1_joint": -1.5707963267948966,
            "wrist_2_joint": -1.5707963267948966,
            "wrist_3_joint": 0.0,
        },
        pos=(0.0, 0.0, 0.0),
    ),
    actuators={
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*"],
            stiffness=800.0,
            damping=40.0,
            friction=0.0,
            armature=0.0,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["elbow_joint"],
            stiffness=400.0,
            damping=40.0,
            friction=0.0,
            armature=0.0,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_.*"],
            stiffness=200.0,
            damping=8.0,
            friction=0.0,
            armature=0.0,
        ),
    },
)
"""UR7e arm (Isaac Sim ur5e USD stand-in; same nominal kinematics)."""
