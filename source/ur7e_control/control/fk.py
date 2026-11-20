"""Forward kinematics from published UR5e/UR7e DH parameters."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ur7e_control.control.limits import DH_A, DH_ALPHA, DH_D

Array = NDArray[np.floating]


def _dh(a: float, alpha: float, d: float, theta: float) -> Array:
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def fk_flange(q: Array) -> Array:
    """Return 4x4 transform of the tool flange in base frame."""
    q = np.asarray(q, dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    for i in range(6):
        T = T @ _dh(DH_A[i], DH_ALPHA[i], DH_D[i], float(q[i]))
    return T


def fk_position(q: Array) -> Array:
    """Cartesian position of the flange origin."""
    return fk_flange(q)[:3, 3].copy()
