"""Tests for trajectory API, FK, scripted baseline, residual env."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import ur7e_control  # noqa: F401
from ur7e_control.backends.sim_ur7e import SimPlantParams, SimUR7e
from ur7e_control.control.fk import fk_position
from ur7e_control.control.limits import HOME_Q
from ur7e_control.control.runner import run_scripted_tracking
from ur7e_control.control.tracker import JointTracker
from ur7e_control.trajectory.cartesian import CartesianTrajectory
from ur7e_control.trajectory.min_jerk import MinJerkTrajectory
from ur7e_control.trajectory.validate import TrajectoryValidationError, validate_trajectory


def test_fk_home_reach_class():
    # At home, flange should be roughly in the 0.5–0.9 m reach ball.
    p = fk_position(HOME_Q)
    assert p.shape == (3,)
    r = float(np.linalg.norm(p))
    assert 0.3 < r < 0.95


def test_min_jerk_boundary_conditions():
    wps = np.vstack([HOME_Q, HOME_Q + np.array([0.2, -0.1, 0.15, 0.1, -0.1, 0.05]), HOME_Q])
    traj = MinJerkTrajectory(wps)
    q0, qd0, qdd0 = traj.sample(0.0)
    qT, qdT, qddT = traj.sample(traj.duration)
    np.testing.assert_allclose(q0, wps[0], atol=1e-9)
    np.testing.assert_allclose(qT, wps[-1], atol=1e-9)
    np.testing.assert_allclose(qd0, 0.0, atol=1e-8)
    np.testing.assert_allclose(qdT, 0.0, atol=1e-8)
    validate_trajectory(traj)


def test_validate_rejects_out_of_box():
    bad = np.vstack([HOME_Q, HOME_Q + np.array([0.0, 0.0, 4.0, 0.0, 0.0, 0.0])])
    with pytest.raises(TrajectoryValidationError):
        MinJerkTrajectory(bad)


def test_cartesian_stub_raises():
    traj = CartesianTrajectory(np.zeros((2, 7)))
    with pytest.raises(NotImplementedError):
        _ = traj.duration
    with pytest.raises(NotImplementedError):
        traj.sample(0.0)


def test_scripted_baseline_converges_in_sim():
    wps = np.vstack([HOME_Q, HOME_Q + 0.15, HOME_Q])
    traj = MinJerkTrajectory(wps)
    robot = SimUR7e(
        q0=wps[0],
        params=SimPlantParams(delay_s=0.004, tau_s=0.02, gain=np.ones(6), meas_noise_std=0.0),
    )
    tracker = JointTracker(kp=12.0)
    logger = run_scripted_tracking(robot, traj, tracker, error_abort=0.2)
    rms = logger.rms_tracking_error()
    assert rms < 0.05, f"RMS {rms} too high"


def test_residual_env_api_and_zero_action():
    env = gym.make("UR7e-TrajResidual-v0")
    obs, info = env.reset(seed=0)
    assert obs.shape == (31,)
    assert env.action_space.shape == (6,)
    # Zero residual should run without crashing and stay near baseline.
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))
        assert np.isfinite(reward)
        if terminated or truncated:
            break
    env.close()
