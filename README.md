# Go2 stair climbing with reinforcement learning

Unitree Go2 in Isaac Sim / Isaac Lab, trained with PPO to walk over stairs (up and down), 5–10° side slopes and rough ground, and to recover its balance when it starts to tip. It uses the simulated trunk IMU for body angles and angular rates.

The robot and its low-level control come from Unitree's official RL repository, [`unitree_rl_lab`](https://github.com/unitreerobotics/unitree_rl_lab), used unmodified. Specifically:

- the Go2 motor model (`UnitreeActuatorCfg_Go2HV`: PD with kp=25, kd=0.5 and Go2 torque–speed limits);
- standing pose, joint order and their velocity-task MDP;
- the deploy-config exporter, so policies can go to Unitree's MuJoCo sim2sim and the real Go2 through `third_party/unitree_rl_lab/deploy/robots/go2`.

This project adds the stair/slope task, the IMU observations, a fixed test course and the scripts.

## Results

Policy: `Go2-Stairs-Blind-v0` (IMU + joints only, no terrain sensing), checkpoint `results/go2_stairs_blind_model_1200/model_1200.pt`, i.e. 1,200 PPO iterations (~1.2 h on an RTX 5070 Ti laptop GPU, 2048 parallel robots). Training reached curriculum level ~4.75 of 9 (around 11 cm risers in the training mix) with 94 % of episodes ending by time-out rather than a fall.

Evaluation course (see below): 6 stairs up, landing, stairs down, then a +θ and a −θ side slope; 17.8 m at 0.5 m/s, heading held at 0.

| Simulator | Course | Robots finished | Max roll / pitch (finishers) | RMS roll / pitch |
|---|---|---|---|---|
| Isaac Lab (PhysX, training sim) | 12 cm stairs, ±8° | **7 / 8** (one fell on the first stairs) | 6–11° / 19–26° | ~3° / ~7° |
| Isaac Lab, with random pushes every 5–10 s | 12 cm stairs, ±8° | **8 / 8** | 7–10° / 17–25° | ~3° / ~7° |
| Isaac Lab, harder than the training level reached | 15 cm stairs, ±10° | **7 / 8** (one stuck at the foot of the stairs until time-out) | 9–15° / 24–35° | ~4° / ~9° |

For comparison, the 400-iteration checkpoint walks on flat ground but stalls at the first step. Raw reports: `results/isaac_course_eval/*.json`.

Pitch of ~20° is expected: the body follows the stair incline. Roll stays under ~11° on the 8° side slopes, i.e. the robot keeps itself level across the slope.

## How the control stack fits together

```
 IMU (ang. rate, gravity dir.) ─┐
 joint pos/vel, last action ────┤
 velocity command ──────────────┼─► PPO policy (MLP 512-256-128) ─► 12 joint offsets ─► Go2 PD motors ─► physics
 height scan (non-blind only) ──┘        50 Hz                         ×0.25 rad          200 Hz
```

The policy replaces a hand-written gait: trotting, foot lift on stair edges and the fast corrective steps when the body starts to roll or pitch all emerge from training. Those corrections are the "rapid movements before falling" you see on real RL-controlled robots. Random shoves every 5–10 s during training force the policy to learn them.

Two policy variants:

| Task | Policy sees | Use |
|---|---|---|
| `Go2-Stairs-v0` | IMU, joints, command, **terrain height scan** | best stair performance in sim (needs a height map on hardware) |
| `Go2-Stairs-Blind-v0` | IMU, joints, command | deployable with Unitree's stock C++ deploy code (same observation names); **this is the trained one** |

In both variants the critic (used only in training) also sees true velocity and the height scan.

## Training terrain

A 10×20 grid of 8 m tiles. Rows are difficulty levels; robots that cover enough distance are promoted to a harder row, and robots that fall are demoted.

- stairs up and stairs down: 30 cm treads, 4–18 cm risers depending on level
- pyramid slopes 2–15°, approached from every direction (side slope = roll, up/down slope = pitch)
- random boxes 2.5–10 cm, uniform rough ground
- friction 0.3–1.2, payload −1…+3 kg, random pushes

## Evaluation course

`scripts/course.py` builds one fixed course and drives robots along it at a constant forward speed, holding heading 0:

```
start flat 3 m | 6 stairs up | 2 m landing | stairs down | 2 m flat | 4 m side slope (+θ) | 1 m flat | 4 m side slope (−θ) | flat
```

Defaults: 12 cm risers and θ = 8°, both adjustable. For each robot it reports whether it finished, which segment it fell on, max/RMS roll and pitch from the IMU, and peak tilt rate. Results are saved to `logs/course_eval/*.json`.

## Setup

Tested on Ubuntu 22.04, RTX 5070 Ti laptop (12 GB).

- Isaac Sim 5.1 standalone. The scripts look in `~/issacsim/isaac-sim-standalone-5.1.0-linux-x86_64`; set `ISAACSIM_DIR` to use another location.
- Isaac Lab 2.3.x installed into that Python, with `rsl-rl-lib` 5.0.1.
- Internet access the first time: the Go2 model is streamed from the Isaac Lab asset server.

Unitree's repositories are not included here. Fetch them at the exact commits used (and install `unitree_rl_lab` into the Isaac Sim Python):

```bash
./scripts/setup_third_party.sh            # add --no-install to only clone
```

This clones `unitree_rl_lab` @ `4960b84` into `third_party/`.

## Usage

All commands are run from the project root. The `.sh` wrappers find Isaac Sim (`ISAACSIM_DIR`) and use its Python, so no environment needs to be activated.

```bash
cd ~/go2_stairs_rl
export ISAACSIM_DIR=~/issacsim/isaac-sim-standalone-5.1.0-linux-x86_64   # only if Isaac Sim is elsewhere
```

### 1. One-time setup

```bash
./scripts/setup_third_party.sh               # clone unitree_rl_lab and pip-install it into Isaac Sim's Python
```

### 2. Run the included trained policy on the evaluation course

```bash
# Watch in the Isaac Sim window
./scripts/course.sh --task Go2-Stairs-Blind-v0 --checkpoint results/go2_stairs_blind_model_1200/model_1200.pt --speed 0.5

# Headless, harder course with random pushes (report saved to logs/course_eval/*.json)
./scripts/course.sh --headless --task Go2-Stairs-Blind-v0 --checkpoint results/go2_stairs_blind_model_1200/model_1200.pt \
    --step_height 0.15 --slope_deg 10 --push

# Watch a single robot instead of 8
./scripts/course.sh --task Go2-Stairs-Blind-v0 --checkpoint results/go2_stairs_blind_model_1200/model_1200.pt --speed 0.5 --num_envs 1

# Untrained baseline (robot just stands)
./scripts/course.sh --headless --policy zero
```

The course runs 8 robots side by side by default (`--num_envs 8`). They all use the same trained policy; running several at once is what gives the success rate (e.g. 7 / 8 finished) rather than a single, possibly lucky, run. Use `--num_envs 1` to watch just one.

### 3. Watch the policy on the training terrains

Also exports `policy.pt` / `policy.onnx` next to the checkpoint.

```bash
./scripts/play.sh --task Go2-Stairs-Blind-v0 --checkpoint results/go2_stairs_blind_model_1200/model_1200.pt
./scripts/play.sh --task Go2-Stairs-Blind-v0 --checkpoint results/go2_stairs_blind_model_1200/model_1200.pt --num_envs 1   # one robot (default 16)
```

### 4. Train

Checkpoints are written to `logs/rsl_rl/<experiment>/<timestamp>/model_*.pt`.

```bash
./scripts/train.sh --headless --task Go2-Stairs-Blind-v0      # blind policy: 2048 robots, 6000 iterations
./scripts/train.sh --headless                                 # height-scan variant (Go2-Stairs-v0)
./scripts/train.sh --headless --task Go2-Stairs-Blind-v0 --num_envs 1024   # less GPU memory

# Continue training from the included checkpoint
./scripts/train.sh --headless --task Go2-Stairs-Blind-v0 --resume results/go2_stairs_blind_model_1200/model_1200.pt

# Training curves: reward, terrain level, episode length
"$ISAACSIM_DIR/python.sh" -m tensorboard.main --logdir logs/rsl_rl
```

Without `--checkpoint`, `play.sh` and `course.sh` use the newest checkpoint under `logs/rsl_rl/`. If GPU memory runs out, lower `--num_envs` (1024 is fine on 12 GB at a modest speed cost). Add `--help` to any script to list all its options.

### 5. Optional: live training in the Isaac Sim window (`go2_isaac_live`)

Independent of the steps above (uses Isaac Lab's own Go2 tasks). See `go2_isaac_live/README.md` for details.

```bash
cd go2_isaac_live
./run.sh train --task flat                      # 256 robots on flat ground, watch them learn to walk
./run.sh train --task stairs --num_envs 512     # stair curriculum + pushes
./run.sh train --task stairs --num_envs 2048 --headless   # Ctrl+C saves, then opens the checkpoint
./run.sh play --task stairs                     # watch the newest checkpoint
./run.sh tensorboard                            # curves in go2_isaac_live/logs/
```

## Repository layout

- `go2_stairs/robot.py`: Unitree's Go2 config, with the USD path pointed at the Isaac Lab asset server
- `go2_stairs/env_cfg.py`: the task: IMU observations, height scan, commands, rewards, pushes, course overrides
- `go2_stairs/terrains.py`: training terrain mix and the evaluation-course mesh
- `go2_stairs/mdp.py`: IMU observation terms (everything else comes from `unitree_rl_lab` / Isaac Lab)
- `go2_stairs/agents/ppo_cfg.py`: PPO hyper-parameters
- `scripts/train.py`, `play.py`, `course.py`: entry points (wrapped by the `.sh` files); `setup_third_party.sh`
- `results/`: the trained checkpoint with its `params/` (including `deploy.yaml`), and the evaluation reports
- `go2_isaac_live/`: optional tool to watch and steer PPO training live in the Isaac Sim window (pause, save, follow a robot). Independent of the rest; not used for the results above.

## Limitations and next steps

- **Training is partial.** The blind run was stopped at iteration 1,398 of the planned 6,000 (curriculum level ~4.75 of 9). Continuing it (`--resume`) should push the stair height it handles towards the 18 cm in the curriculum.
- **The height-scan policy (`Go2-Stairs-v0`) is implemented but was not trained.** On hardware it would also need an elevation map.
- **Not tested on a real Go2.** The blind policy's observations and `deploy.yaml` match Unitree's deploy pipeline, so the next step would be `unitree_mujoco` sim2sim and then hardware.
