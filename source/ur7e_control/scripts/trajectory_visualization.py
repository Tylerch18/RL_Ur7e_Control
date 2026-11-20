"""Compare reference and executed trajectories in the MuJoCo TCP frame."""

from pathlib import Path

import mujoco
import numpy as np

from ur7e_control.assets import EE_SITE, apply_home, load_ur7e_model

REFERENCE_COLOR = np.array([0.95, 0.49, 0.08, 1.0])
ACTUAL_COLOR = np.array([0.08, 0.55, 0.95, 1.0])


def tcp_positions(joints, model=None, *, data=None):
    """Map joint samples to Hand-E TCP positions in the viewer's world frame."""
    model = model if model is not None else load_ur7e_model(with_target=False)
    data = data if data is not None else mujoco.MjData(model)
    apply_home(model, data)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    if site < 0:
        raise ValueError(f"missing TCP site: {EE_SITE}")
    points = []
    for q in joints:
        data.qpos[:6] = q
        mujoco.mj_kinematics(model, data)
        points.append(data.site_xpos[site].copy())
    return np.asarray(points)


class TrajectoryOverlay:
    """Draw the desired path, executed trail, and synchronized TCP markers."""

    def __init__(self, model, traj, actual_label="PPO + tracker"):
        self.model = model
        self.data = mujoco.MjData(model)
        self.duration = traj.duration
        self.actual_label = actual_label
        samples = [traj.sample(t)[0] for t in np.linspace(0, traj.duration, 241)]
        self.reference = tcp_positions(samples, model)
        self.waypoints = tcp_positions(traj.waypoints, model)
        self.reset()

    def reset(self):
        self.trail = []
        self.ticks = 0

    @staticmethod
    def _sphere(scene, point, radius, color):
        if scene.ngeom >= scene.maxgeom:
            return
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
            np.full(3, radius), point, np.eye(3).ravel(), color,
        )
        scene.ngeom += 1

    @staticmethod
    def _line(scene, start, end, radius, color):
        if scene.ngeom >= scene.maxgeom or np.linalg.norm(end - start) < 1e-9:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3), np.zeros(3), np.eye(3).ravel(), color,
        )
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, start, end)
        scene.ngeom += 1

    def update(self, viewer, q_ref, q, t):
        reference, actual = tcp_positions([q_ref, q], self.model, data=self.data)
        if self.ticks % 4 == 0:
            self.trail.append(actual)
        self.ticks += 1
        # Bound rendering work even for unusually long trajectories.
        if len(self.trail) > 600:
            self.trail = self.trail[::2]
        with viewer.lock():
            scene = viewer.user_scn
            scene.ngeom = 0
            for i in range(0, len(self.reference) - 1, 2):
                self._line(scene, self.reference[i], self.reference[i + 1], 0.0015, REFERENCE_COLOR)
            for start, end in zip(self.trail[:-1], self.trail[1:]):
                self._line(scene, start, end, 0.0015, ACTUAL_COLOR)
            if self.trail:
                self._line(scene, self.trail[-1], actual, 0.0015, ACTUAL_COLOR)
            for point in self.waypoints:
                self._sphere(scene, point, 0.004, REFERENCE_COLOR)
            self._sphere(scene, reference, 0.007, REFERENCE_COLOR)
            self._sphere(scene, actual, 0.005, ACTUAL_COLOR)
        if hasattr(viewer, "set_texts"):
            error_mm = np.linalg.norm(actual - reference) * 1000
            viewer.set_texts((
                None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                "Reference\n" + self.actual_label + "\nTCP error\nTime",
                f"Orange dashed\nBlue trail\n{error_mm:.2f} mm\n{t:.2f} / {self.duration:.2f} s",
            ))


def plot_comparison(log, waypoints, out: Path, title: str):
    """Save 3D/XY TCP comparisons, errors, and six joint overlays."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    data = log.as_arrays()
    if len(data["t"]) == 0:
        raise ValueError("trajectory log is empty")
    model = load_ur7e_model(with_target=False)
    reference = tcp_positions(data["q_ref"], model)
    actual = tcp_positions(data["q"], model)
    waypoint_tcp = tcp_positions(waypoints, model)
    error_mm = (actual - reference) * 1000
    distance_mm = np.linalg.norm(error_mm, axis=1)
    rms_mm = float(np.sqrt(np.mean(distance_mm ** 2)))
    ref_color, actual_color = REFERENCE_COLOR[:3], ACTUAL_COLOR[:3]
    fig = plt.figure(figsize=(12, 9), layout="constrained")
    fig.set_constrained_layout_pads(h_pad=0.12, w_pad=0.08, hspace=0.10, wspace=0.08)
    fig.suptitle(f"{title}\nTCP RMS {rms_mm:.2f} mm | peak {distance_mm.max():.2f} mm")

    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.plot(*reference.T, "--", color=ref_color, label="Reference (desired)", linewidth=2)
    ax.plot(*actual.T, color=actual_color, label="Executed", linewidth=1.6)
    ax.scatter(*waypoint_tcp.T, color=ref_color, s=22)
    ax.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)", title="Hand-E tip: 3D trajectory")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(3))
    span = np.ptp(np.vstack([reference, actual]), axis=0)
    ax.set_box_aspect(np.maximum(span, 0.02))
    ax.legend(fontsize=9, loc="upper left")

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(reference[:, 0], reference[:, 1], "--", color=ref_color, label="Reference", linewidth=2)
    ax.plot(actual[:, 0], actual[:, 1], color=actual_color, label="Executed")
    ax.scatter(waypoint_tcp[:, 0], waypoint_tcp[:, 1], color=ref_color, s=25)
    ax.set(xlabel="World X (m)", ylabel="World Y (m)", title="Hand-E tip: top view")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(2, 2, 3)
    ax.plot(data["t"], distance_mm, color=actual_color)
    ax.set(xlabel="Simulation time (s)", ylabel="TCP distance error (mm)", title="Gap at the same instant")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(2, 2, 4)
    for i, axis in enumerate("XYZ"):
        ax.plot(data["t"], error_mm[:, i], label=axis)
    ax.axhline(0, color="0.6", linewidth=0.8)
    ax.set(xlabel="Simulation time (s)", ylabel="Executed - reference (mm)", title="TCP error by world axis")
    ax.legend()
    ax.grid(alpha=0.25)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True, layout="constrained")
    fig.suptitle(f"{title} | joint angles")
    for i, ax in enumerate(axes.flat):
        ax.plot(data["t"], np.rad2deg(data["q_ref"][:, i]), "--", color=ref_color, label="Reference", linewidth=2)
        ax.plot(data["t"], np.rad2deg(data["q"][:, i]), color=actual_color, label="Executed")
        ax.set(title=f"Joint {i + 1}", ylabel="Angle (deg)")
        ax.grid(alpha=0.25)
    axes[0, 0].legend()
    for ax in axes[-1]:
        ax.set_xlabel("Simulation time (s)")
    fig.savefig(out.with_stem(out.stem + "_joints"), dpi=150)
    plt.close(fig)
    np.savez(
        out.with_suffix(".npz"), **data, waypoints=waypoints,
        tcp_ref=reference, tcp=actual, tcp_waypoints=waypoint_tcp,
    )
    return out
