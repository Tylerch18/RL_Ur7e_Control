# Robotiq Hand-E (MJCF)

A 50 mm parallel gripper with force-limited position control and an example
controller that approximates its holding brake. Tested with **MuJoCo 3.14.0**.
The mesh geometry comes from the [Hand-E URDF](https://github.com/cambel/robotiq/blob/noetic-devel/robotiq_description/urdf/robotiq_hand_e.urdf).

## Manual XML-only pickup (minimal Python)

```bash
uv run python robotiq_hande/manual.py
```

The launcher only loads `scene_manual.xml` and opens MuJoCo's
[managed viewer](https://mujoco.readthedocs.io/en/stable/python.html#managed-viewer),
which handles simulation stepping and the GUI. There is no Python gripper
controller, automatic grasp detection, brake switching or motion sequence.
`scene_manual.xml` attaches the existing `hande.xml`, including its actuator,
without copying or changing its gains or force limits.

The open gripper starts aligned around the upper part of a free **0.75 kg box**.
Expand **Control** in the viewer's right panel. If paused, press **Space**.
Use the actuator sliders in this order, waiting about two seconds between steps:

1. Set **`fingers_actuator` to 255** to close and squeeze the box.
2. Set **`lift` to 0.15** to raise the gripper 150 mm.
3. Optionally set **`transfer` to 0.15** to move sideways 150 mm.
4. Set **`lift` back to 0** to lower the box onto the table.
5. Set **`fingers_actuator` to 0** to open and release it.
6. Raise **`lift`** again to confirm the box stays on the table.

`lift` and `transfer` are position actuators with commands in metres, from 0 to
0.2. Their targets are smoothed by XML actuator filters. The gripper's command
remains 0–255. Keep the closing command applied while lifting: this example tests
the XML position servo alone, without the stronger Python holding mode.
The fingers grip the box on the flat tip surfaces, above the angled pad sections.
The box has no attachment to the gripper and can fall if the grip is insufficient.

The full close/lift/transfer/lower/release sequence was checked headlessly with
direct actuator commands and produced no solver warnings.

## Run the pick-and-place example

From the repository root:

```bash
uv sync
uv run python robotiq_hande/run.py --check
uv run python robotiq_hande/run.py --viewer --check
```

The example mounts the gripper on a dynamic two-axis gantry with finite-force
position servos. It picks a **30 × 40 × 80 mm, 0.75 kg billet** from a table,
lifts it **180 mm**, transfers it **200 mm**, applies an additional **20 N
downward load**, then places and releases it. The object has a free joint and
is supported only by contacts. There is no tether, object weld or global fluid
damping. The gantry approximates a robot wrist; it does not model an entire arm.

| Time (s) | Action |
| --- | --- |
| 0–0.5 | Settle on table |
| 0.5–1.5 | Approach |
| 1.5–2.5 | Close at 25 mm/s per finger, up to 60 N per finger |
| 2.5–3.5 | Lift |
| 3.5–4.5 | Transfer |
| 4.5–6 | Hold; ramp the extra downward load on and off |
| 6–7 | Lower |
| 7–7.8 | Release and open |
| 7.8–9 | Retreat and settle |

The brake engages after both fingers establish contact above 85% of the force
setting and their speeds remain below 2 mm/s for 80 ms. The commanded trajectory
continues on a failed grasp so that failures remain visible in the plots and
checks; this example is a simulation benchmark, not a hardware controller.

The script writes these files to `robotiq_hande/results/` (or `--output PATH`):

- `performance.png` and `.pdf`: lift, transfer, contact forces, aperture, relative
  slip, external load and table support.
- `pick_place.csv`: measurements at every simulation step, with units in headers.
- `metrics.json`: parameters, measured results and pass/fail criteria.
- `scenario.png`: six snapshots, when `--render` is selected.

For headless snapshots on Linux with EGL:

```bash
MUJOCO_GL=egl uv run python robotiq_hande/run.py --render --check
```

Change the payload, closing force or external load:

```bash
uv run python robotiq_hande/run.py --mass 1.5 --force 60 --load 20 --check
```

`--check` returns a nonzero exit code if the object is not grasped before lifting,
lifts less than 170 mm, drifts more than 1 mm relative to the latched grasp,
changes aperture by more than 0.2 mm while holding, loses finger contact, or is
not placed within 2 mm and released. It also checks finger synchronization,
finite state, solver warnings and absence of table support during the interval
2.75–6.75 s. Drift includes both contact deformation and slip; it is measured
through the entire lift/transfer/lower sequence, 2.5–7 s.

## Verify resistance to being pushed open

```bash
uv run python robotiq_hande/verify.py
```

This produces `backdrive.png`, `.pdf`, three CSV traces and `verification.json`.
The comparison applies **20 N outward to each finger** with no object, using
identical geometry and a 0.5 ms timestep:

| Actuation | Expected steady movement per finger |
| --- | --- |
| Previous position law, Kp = 5,000 N/m | 8 mm |
| Corrected position law, Kp = 40,000 N/m | 1 mm |
| Engaged brake, Kp = 1,000,000 N/m | 0.04 mm |

The first trace isolates the old actuator law; it is not a replay of the old
tethered scene. Total aperture change is twice the per-finger displacement.
The checks also cover full-stroke collision clearance, fixed-block chatter,
finite brake force and yielding under overload, a 1.5 kg payload, half the
timestep, and an excessive object load that must fail the grasp.

## Model and controller

`hande.xml` remains a standalone component; `scene.xml` is a simple inspection
scene. The actuator interface remains `ctrl = 0` for open and `255` for closed.
**The XML alone supplies position control. Use `HandEController` to engage the
holding brake.**

The tendon length is `(q_left + q_right) / 2`. Each finger receives half the
actuator force, so the XML's ±370 N actuator cap corresponds to ±185 N per
finger. Position error generates force; the force cap alone does not create
holding force. The example reduces the closing limit to 60 N per finger.

`controller.py` exposes:

```python
from robotiq_hande.controller import HandEController

controller = HandEController(model, data, force=60, brake_force=185)
controller.move(opening=0.0, speed=0.025)  # metres and metres/second per finger
# Before every mj_step:
controller.update()
# After a settled grasp has been detected:
controller.hold()
# To release the brake and reopen:
controller.move(opening=0.05)
```

Each controller needs its own `MjModel`: mode changes update actuator gains and
force limits. Do not share one model across independently controlled `MjData`
instances. Call `update()` before every physics step. The grasp-detection example
is in `run.py`.

### Holding brake approximation

Robotiq documents a [brake that engages at the end of a move](https://assets.robotiq.com/website-assets/support_documents/document/online/Hand-E_TM_InstructionManual_HTML5_20190306.zip/Hand-E_TM_InstructionManual_HTML5/Content/4.%20Control.htm).
Here it is represented by a stiff, damped position restraint with a finite force
cap, acting through the same tendon. Engagement captures the current position
and transfers the closing preload; opening releases the restraint. The brake
can deflect and yields if its capacity is exceeded. Its default 185 N per-finger
holding cap is **an assumed simulation parameter, not a published brake rating**.
It has elastic recovery and does not simulate brake wear, backlash, firmware
re-grasp logic or power-loss behavior.

The [manufacturer product sheet](https://robotiq.com/hubfs/Hand-E_Adaptive_Gripper_ProductSheet_EN_2024-11_WEB.pdf)
lists up to 185 N closing force per finger and a 1 kg gripper mass. These inform
the force limit and mass budget. Contact friction, brake stiffness, housing
inertia and pad compliance are approximations requiring hardware measurements
for physical calibration. Passing this benchmark does not establish a hardware
payload rating.

### Other corrections

- Explicit 0.5 ms timestep, Newton solver and reduced actuator damping avoid the
  contact chatter reproduced with the old settings.
- A stiffer finger equality keeps motion symmetric; it does not provide braking.
- Corrected mirrored finger COM locations. Finger inertias retain the source
  approximation; the housing uses an explicit approximate cylinder-like inertia.
- Visual geoms have zero mass and each gripper body has an explicit inertia,
  avoiding duplicate mass from overlapping visual/collision meshes.
- Pads use `friction="0.8 0.003 0.0001"`, with `condim=4`. Torsional friction has
  a 3 mm length scale; rolling friction is inactive at this contact dimension.
- Pad priority 2 overrides the example object's priority 0, so the intended
  contact softness is used without blending with the object's defaults.
- Finger collisions use mounting-heel and pad boxes. The supplied whole-finger
  mesh hulls bridge recesses and intersect during closing, so they are not used.
  The simplified boxes approximate the fingers rather than reproducing fine CAD.

## License

This model is released under the [BSD-2-Clause License](LICENSE).
