"""UR7e joint / velocity / acceleration limits (conservative for sim-to-real)."""

from __future__ import annotations

import numpy as np

N_JOINTS = 6
CONTROL_HZ = 500
CONTROL_DT = 1.0 / CONTROL_HZ

# Official: ±360° and 180°/s. Elbow is tighter in practice (self-collision).
# Keep a conservative joint box around typical reach poses.
JOINT_LOW = np.array([-2 * np.pi, -2 * np.pi, -np.pi, -2 * np.pi, -2 * np.pi, -2 * np.pi], dtype=np.float64)
JOINT_HIGH = np.array([2 * np.pi, 2 * np.pi, np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi], dtype=np.float64)

# Datasheet max: π rad/s. First real runs use much lower caps.
V_MAX = np.full(N_JOINTS, np.pi, dtype=np.float64)
A_MAX = np.full(N_JOINTS, 5.0, dtype=np.float64)

# Safe first-contact hardware caps (rad/s, rad/s²).
SAFE_V_MAX = np.full(N_JOINTS, 0.4, dtype=np.float64)
SAFE_A_MAX = np.full(N_JOINTS, 1.0, dtype=np.float64)

# Nominal home (same as Menagerie UR5e / UR7e stand-in).
HOME_Q = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0], dtype=np.float64)

# Published UR5e/UR7e DH (meters / rad) for FK sanity checks.
DH_A = np.array([0.0, -0.425, -0.3922, 0.0, 0.0, 0.0], dtype=np.float64)
DH_D = np.array([0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996], dtype=np.float64)
DH_ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0], dtype=np.float64)
