"""Fit speedJ-like delay and first-order tau from commanded-vs-actual qd logs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _fit_delay_steps(qd_cmd: np.ndarray, qd: np.ndarray, max_delay: int = 20) -> int:
    """Choose integer delay maximizing mean cross-correlation across joints."""
    best_d, best_score = 0, -np.inf
    for d in range(max_delay + 1):
        if d >= len(qd_cmd) - 2:
            break
        a = qd_cmd[:-d or None]
        b = qd[d:]
        n = min(len(a), len(b))
        a = a[:n] - a[:n].mean(axis=0)
        b = b[:n] - b[:n].mean(axis=0)
        denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
        score = float(np.sum(a * b) / denom)
        if score > best_score:
            best_score, best_d = score, d
    return best_d


def _fit_tau(qd_cmd_aligned: np.ndarray, qd: np.ndarray, dt: float) -> float:
    """Fit scalar tau for qd_dot ≈ (gain*u - qd)/tau with gain≈1 via least squares."""
    # Discrete: qd[k+1] - qd[k] = (dt/tau) * (u[k] - qd[k])
    u = qd_cmd_aligned[:-1]
    y = qd[:-1]
    dy = qd[1:] - qd[:-1]
    # dy = alpha * (u - y), alpha = dt/tau
    x = (u - y).reshape(-1)
    b = dy.reshape(-1)
    denom = float(np.dot(x, x)) + 1e-12
    alpha = float(np.dot(x, b) / denom)
    alpha = float(np.clip(alpha, 1e-4, 0.99))
    tau = dt / alpha
    return float(np.clip(tau, 0.005, 0.1))


def fit_log(path: Path, *, dt: float = 0.002) -> dict[str, float]:
    data = np.load(path)
    qd_cmd = np.asarray(data["qd_cmd"], dtype=np.float64)
    qd = np.asarray(data["qd"], dtype=np.float64)
    delay_steps = _fit_delay_steps(qd_cmd, qd)
    if delay_steps > 0:
        u = qd_cmd[:-delay_steps]
        y = qd[delay_steps:]
    else:
        u, y = qd_cmd, qd
    n = min(len(u), len(y))
    tau = _fit_tau(u[:n], y[:n], dt)
    return {
        "delay_s": delay_steps * dt,
        "tau_s": tau,
        "delay_steps": float(delay_steps),
        "suggested_delay_lo": 0.0,
        "suggested_delay_hi": max(0.012, delay_steps * dt * 1.5),
        "suggested_tau_lo": max(0.008, tau * 0.6),
        "suggested_tau_hi": min(0.06, tau * 1.5),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Sysid delay/tau from NPZ logs")
    p.add_argument("log", type=Path, help="NPZ from run_scripted")
    p.add_argument("--dt", type=float, default=0.002)
    args = p.parse_args(argv)

    result = fit_log(args.log, dt=args.dt)
    print(f"log: {args.log}")
    print(f"fitted delay: {result['delay_s']*1000:.2f} ms ({int(result['delay_steps'])} steps)")
    print(f"fitted tau:   {result['tau_s']*1000:.2f} ms")
    print(
        "suggested sim DR ranges:\n"
        f"  delay [{result['suggested_delay_lo']*1000:.1f}, {result['suggested_delay_hi']*1000:.1f}] ms\n"
        f"  tau   [{result['suggested_tau_lo']*1000:.1f}, {result['suggested_tau_hi']*1000:.1f}] ms"
    )


if __name__ == "__main__":
    main()
