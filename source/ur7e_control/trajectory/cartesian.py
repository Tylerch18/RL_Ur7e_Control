"""Cartesian trajectory hook (IK deferred — joint-space first)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.floating]
IKFn = Callable[[Array], Array]  # pose (e.g. 7,) -> q (6,)


class CartesianTrajectory:
    """Placeholder for pose-path + injected IK.

    Not implemented in this pass. Joint-space tracking is the active path.
    """

    def __init__(
        self,
        poses: Array,
        ik: IKFn | None = None,
        **_: Any,
    ) -> None:
        self.poses = np.asarray(poses, dtype=np.float64)
        self.ik = ik

    @property
    def duration(self) -> float:
        raise NotImplementedError(
            "CartesianTrajectory is a hook only; use MinJerkTrajectory for joint-space tracking"
        )

    def sample(self, t: float) -> tuple[Array, Array, Array]:
        raise NotImplementedError(
            "CartesianTrajectory requires an IK implementation; not built in this pass"
        )
