# Real robot ROS 2 workspace

Two packages, with one node each:

```text
real/
  models/                         Trained PPO checkpoint
  src/
    ur7e_state/                   Read and order the six arm joints
      ur7e_state/joint_state_node.py
    ur7e_policy/                  Receive state; future PPO and command logic
      ur7e_policy/policy_node.py
      config/real.yaml
      launch/real.launch.py
```

The official UR ROS 2 driver supplies `/joint_states` and executes commands.
This workspace does not replace that driver.

```text
UR driver -> /joint_states -> joint_state_node -> /ur7e/arm_state
                                                     |
                                                policy_node
                                                     |
                              /forward_velocity_controller/commands
                                                     |
                                                 UR driver
```

## What works now

- `joint_state_node` selects the configured six joints by name and republishes
  their positions and velocities in a consistent order.
- `policy_node` receives that state and declares the command publisher.
- The launch file starts both nodes using `config/real.yaml`.

**This is a skeleton. PPO inference, reference generation, tracking, and command
publishing are not implemented here yet. It sends no motion commands.**
The existing PPO outputs joint-velocity corrections, not absolute joint angles.

## Build and run on the lab computer

Use a terminal with ROS 2 already sourced and `colcon` installed:

```bash
cd real
colcon build --symlink-install
source install/setup.bash
ros2 launch ur7e_policy real.launch.py
```

Start the lab's UR driver separately. To inspect the extracted joint state:

```bash
ros2 topic echo /ur7e/arm_state
```

Change the joint names in `src/ur7e_policy/config/real.yaml` if the lab driver
uses a prefix. Both input positions and velocities must be present.

## Next step

Complete `policy_node.py`: use the reference trajectory and measured joints to
build the same 31 observations, run PPO at 50 Hz, and combine its correction with
the existing 500 Hz tracker. Validate timing, limits, and stopping on stale data
before enabling command publishing.

Use `models/residual_ppo.zip` for the checkpoint. The launch file accepts its
absolute path:

```bash
ros2 launch ur7e_policy real.launch.py model_path:="$(pwd)/models/residual_ppo.zip"
```

The path is stored by the skeleton node; the model is not loaded yet. Robot IP,
External Control, and controller activation remain part of the UR driver setup.
