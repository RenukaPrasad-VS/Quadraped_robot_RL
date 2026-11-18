# go2_gazebo_rl: the trained Go2 policy in Gazebo (sim2sim)

ROS 2 Humble + Gazebo Fortress package that runs the policy trained in Isaac Lab (`Go2-Stairs-Blind-v0`) on the same stairs + side-slope course. No training happens here. Gazebo serves as an independent physics engine for checking that the policy holds up outside the simulator it was trained in.

```
Gazebo (Unitree go2_description URDF, IMU on trunk) ──ros_gz_bridge──► /go2/imu, /go2/joint_states
                                                                              │
                                    policy_node (50 Hz): 45 inputs built from deploy.yaml
                                    → MLP (raw model_*.pt via NumPy, or policy.onnx)
                                    → target = standing pose + 0.25·action
                                                                              │
Gazebo JointPositionController (kp=25, ±23.5 Nm) + joint damping (kd=0.5) ◄── /go2/joint/<j>/cmd_pos
```

The actuation reproduces Unitree's Go2 motor model used in training: τ = kp(q*−q) − kd·q̇, clipped. Gains are read from the training run's `deploy.yaml`.

## Build

```bash
source /opt/ros/humble/setup.bash
cd ~/go2_stairs_rl/ros2_ws && colcon build --symlink-install
source install/setup.bash
```

## Run

```bash
ros2 launch go2_gazebo_rl sim.launch.py                       # newest blind checkpoint, GUI, 12 cm stairs, 8° slopes
ros2 launch go2_gazebo_rl sim.launch.py step_height:=0.15 slope_deg:=10 speed:=0.6
ros2 launch go2_gazebo_rl sim.launch.py policy:=~/go2_stairs_rl/results/go2_stairs_blind_model_1200/model_1200.pt   # the included trained policy
ros2 launch go2_gazebo_rl sim.launch.py policy_enabled:=false # only stand in the training pose
ros2 launch go2_gazebo_rl sim.launch.py gui:=false            # headless
```

Sequence: the robot spawns at the course start, blends into the standing pose (2 s), settles (1 s), then the policy walks it forward and holds heading 0. You can override the command by publishing a `geometry_msgs/Twist` on `/cmd_vel`.

`course_monitor` prints progress and IMU roll/pitch every 2 s. It finishes when the robot passes the second side slope, tilts past 1 rad (counted as a fall), or reaches `timeout`. The JSON report goes to `~/go2_stairs_rl/logs/gazebo_eval/`, with the same metrics as the Isaac Lab `scripts/course.sh`, so the two simulators can be compared directly.

Launch arguments: `policy`, `deploy_cfg`, `step_height`, `slope_deg`, `speed`, `timeout`, `gui`, `policy_enabled`, `go2_description`.

## Live training in Gazebo

`train.launch.py` keeps training the policy inside Gazebo with PPO while you watch. It starts from an Isaac Lab checkpoint (default: the newest blind one), because a policy that already walks needs only tens of thousands of Gazebo steps to adapt. Training from scratch needs tens of millions, which at Gazebo's ~50 steps/s for one robot would take days.

```bash
ros2 launch go2_gazebo_rl train.launch.py                     # world:=stairs (default), GUI
ros2 launch go2_gazebo_rl train.launch.py world:=terrain      # varied terrain world
ros2 launch go2_gazebo_rl train.launch.py world:=terrain lane:=3   # obstacles lane only
ros2 launch go2_gazebo_rl train.launch.py init_policy:=~/go2_stairs_rl/results/go2_stairs_blind_model_1200/model_1200.pt
```

Two worlds, each six lanes side by side (3 m wide, 6 m apart, with a catch floor 1 m below for trenches and edges):

| `world:=stairs` (original) | Lane selection |
|---|---|
| The evaluation course at six levels: stairs 5 / 7 / 9 / 11 / 13 / 15 cm with side slopes 5° to 10° | Curriculum: finishing a lane promotes the robot to the next level; covering less than half of it demotes the robot |

| `world:=terrain` lane | What it contains (harder along the lane) |
|---|---|
| 0 stairs | stair sets 6, 10, 14 cm |
| 1 slopes | up/down ramps 6° and 11°, each followed by a side slope of the same angle |
| 2 blocks | random stepping-block fields 0–4, 0–8, 0–12 cm |
| 3 obstacles | hurdles 5 cm, debris field (12 boxes, 5–10 cm), hurdles 9 cm, trenches 15 and 20 cm |
| 4 surfaces | ice (μ 0.2), normal, wet (0.4), rubber (1.2), ice, then a wet 6° side slope |
| 5 exam | the evaluation course (12 cm stairs, ±8° side slopes) |

In the terrain world, lanes are sampled adaptively: the trainer keeps a running completion score per lane, and picks lanes with weight 1.2 − score. Lanes it does badly on come up about twice as often as mastered ones. `lane:=N` fixes one lane in either world.

- **Episode:** up to 40 s, ending on a fall (tilt > 1 rad or off the lane), the goal, or the time limit. Between episodes the robot first comes to rest (the policy runs with a zero command, or a fallen robot lies still for 1.5 s). It is then teleported to its lane start and blended back into the standing pose.
- **PPO:** the world pauses for each update (every `rollout_steps` steps, about 40 s of walking). The actor starts from Isaac's weights and normalizer; the critic is new and gets 3 warm-up updates before the actor changes. KL early-stopping keeps each update close to the previous policy. Rewards are Isaac's main terms (velocity tracking, tilt, vertical/roll rates, action rate, torque, joint velocity) plus forward course progress, +10 on the goal and −2 on a fall.
- **Output:** `~/go2_stairs_rl/logs/gazebo_train/<time>/` holds `model_<it>.pt` (rsl-rl layout, so `sim.launch.py policy:=...` can evaluate it), `progress.jsonl` and TensorBoard logs (`tensorboard --logdir ~/go2_stairs_rl/logs/gazebo_train`).
- The terminal prints one line per episode (lane, outcome, distance) and one `UPDATE` line per PPO iteration.

## Status

- `sim.launch.py` with the included `model_1200.pt`: finishes the default course (12 cm stairs, ±8° side slopes) in 37 s, max roll 9.7°, max pitch 20.1° (`results/gazebo_eval/`).
- `train.launch.py` (Gazebo fine-tuning): implemented and smoke-tested only (a few PPO iterations; the critic warm-up ran, the actor was not yet updated).

## Notes

- The project is assumed to be at `~/go2_stairs_rl`; set `GO2_STAIRS_RL_ROOT` otherwise. `third_party/unitree_ros` comes from `scripts/setup_third_party.sh`.

- Only the **blind** policy runs here. The height-scan policy would need a simulated depth sensor plus an elevation map.
- Each launch uses its own Gazebo transport partition, so other or leftover Gazebo servers can't interfere. On shutdown it also stops its Gazebo server explicitly, because the `ign gazebo` wrapper doesn't always do that.
- Physics: DART at 1 ms. Training used PhysX at 5 ms, and the robot model comes from a URDF here versus a USD in training. Expect some performance gap, which is what this package measures.
