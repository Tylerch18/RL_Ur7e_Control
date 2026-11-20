"""Robot interface shared by sim and real backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.floating]


class RobotInterface(ABC):
    """Minimal control interface for joint-velocity tracking at a fixed period."""

    @abstractmethod
    def read(self) -> tuple[Array, Array]:
        """Return measured ``(q, qd)``."""

    @abstractmethod
    def command_velocity(self, qd_cmd: Array) -> None:
        """Send a joint-velocity command (rad/s), analogous to ``speedJ``."""

    @abstractmethod
    def wait_period(self) -> None:
        """Block until the next control period (500 Hz)."""

    @abstractmethod
    def stop(self) -> None:
        """Safely stop motion."""
