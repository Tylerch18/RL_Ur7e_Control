"""Full-rate trajectory logging to NPZ."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.floating]


class TrajectoryLogger:
    """Buffer ``t, q_ref, qd_ref, q, qd, qd_cmd`` and write NPZ."""

    def __init__(self) -> None:
        self.t: list[float] = []
        self.q_ref: list[Array] = []
        self.qd_ref: list[Array] = []
        self.q: list[Array] = []
        self.qd: list[Array] = []
        self.qd_cmd: list[Array] = []

    def record(
        self,
        t: float,
        q_ref: Array,
        qd_ref: Array,
        q: Array,
        qd: Array,
        qd_cmd: Array,
    ) -> None:
        self.t.append(float(t))
        self.q_ref.append(np.asarray(q_ref, dtype=np.float64).copy())
        self.qd_ref.append(np.asarray(qd_ref, dtype=np.float64).copy())
        self.q.append(np.asarray(q, dtype=np.float64).copy())
        self.qd.append(np.asarray(qd, dtype=np.float64).copy())
        self.qd_cmd.append(np.asarray(qd_cmd, dtype=np.float64).copy())

    def as_arrays(self) -> dict[str, Array]:
        return {
            "t": np.asarray(self.t, dtype=np.float64),
            "q_ref": np.asarray(self.q_ref, dtype=np.float64),
            "qd_ref": np.asarray(self.qd_ref, dtype=np.float64),
            "q": np.asarray(self.q, dtype=np.float64),
            "qd": np.asarray(self.qd, dtype=np.float64),
            "qd_cmd": np.asarray(self.qd_cmd, dtype=np.float64),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.as_arrays())
        return path

    def rms_tracking_error(self) -> float:
        data = self.as_arrays()
        err = data["q"] - data["q_ref"]
        return float(np.sqrt(np.mean(err**2)))
