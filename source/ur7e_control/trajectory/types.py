"""Trajectory protocol for joint-space (and future Cartesian) tracking."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.floating]


@runtime_checkable
class Trajectory(Protocol):
    """Time-parameterized joint trajectory."""

    @property
    def duration(self) -> float:
        """Total duration in seconds."""

    def sample(self, t: float) -> tuple[Array, Array, Array]:
        """Return ``(q, qd, qdd)`` at time ``t`` (clamped to ``[0, duration]``)."""
