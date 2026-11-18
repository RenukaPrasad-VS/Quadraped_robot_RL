# go2_isaac_live (optional): watch and steer Go2 training live in Isaac Sim

This package spawns many Unitree Go2 robots in one Isaac Sim scene and trains them with PPO while you watch. It runs on Isaac Sim + Isaac Lab + rsl-rl only. It has **no Gazebo or ROS dependencies** and doesn't depend on the sibling `go2_stairs` package or `unitree_rl_lab`.

The environments use Isaac Lab's own Go2 locomotion tasks (`isaaclab_tasks` velocity tracking, Go2 model and DC motors from `isaaclab_assets`). These reliably learn to walk from scratch.

## Quick start

```bash
cd ~/go2_stairs_rl/go2_isaac_live

./run.sh train --task flat                      # 256 robots on flat ground; walking in ~10-20 min
./run.sh train --task stairs --num_envs 512     # stair curriculum (4-18 cm, up and down) + pushes
./run.sh train --task rough                     # Isaac Lab rough terrain (slopes, stairs, boxes, noise)

./run.sh play --task stairs                     # watch the newest checkpoint (32 robots)
./run.sh play --task flat --vx 1.0 --wz 0.3     # same command for all robots
./run.sh tensorboard                            # curves in logs/
```

**Train without visuals, look afterwards:**

```bash
./run.sh train --task stairs --num_envs 2048 --headless
# ... press Ctrl+C whenever you want to stop: it saves model_<it>.pt, exports policy.pt/.onnx,
# then opens that checkpoint in Isaac Sim (32 robots). Add --no-play to skip opening it.
# Pressing Ctrl+C twice aborts without saving.
```

Useful training options: `--num_envs N` (robots to spawn), `--max_iterations N`, `--resume logs/<exp>/<run>/model_X.pt`, `--start_paused`, `--turbo` (start with the viewport hidden), `--headless` (no window or panel; fastest).

## The Live Training panel

It docks on the right side of the Isaac Sim window:

- live stats: PPO iteration, steps/s, mean episode return and length, mean terrain level (curriculum), action noise, losses, a return-per-iteration plot;
- **Pause / Resume** (`P`): physics and learning stop, the viewport stays interactive;
- **Save now** (`K`): writes `model_<it>.pt` after the current update (autosave runs every 50/100 iterations anyway);
- **camera**: follow next/previous robot (`N`/`B`), jump to robot #i, overview of the whole terrain (`O`);
- **Turbo** (`T`): hides the viewport so training runs at full speed. Press again to watch;
- **Real-time** (`R`): slows the sim to wall-clock speed so the gait looks natural;
- **Stop training & save**: ends cleanly, saves, and exports `exported/policy.pt` + `policy.onnx`.

Keyboard shortcuts work when the viewport has focus (click it once).

## Tasks

| `--task` | Terrain | Default robots | Iterations |
|---|---|---|---|
| `flat` | plane | 256 | 1000 |
| `rough` | Isaac Lab `ROUGH_TERRAINS_CFG` with curriculum | 512 | 3000 |
| `stairs` | 60 % stairs up/down (4-18 cm risers, 30 cm treads), slopes, boxes, rough; curriculum; pushes every 10-15 s | 512 | 5000 |

Policy: MLP 512-256-128 (flat 128-128-128), 50 Hz, joint position targets (scale 0.25) around the standing pose. Inputs are base velocities, gravity direction, command, joint pos/vel and last action, plus a height scan on rough/stairs.

## How many robots?

More robots means more data per second, but a heavier viewport. On this RTX 5070 Ti laptop (12 GB):

- 256-512: smooth to watch;
- 1024-2048: faster learning. Use Turbo, and turn it off now and then to look;
- 4096 headless: fastest, the standard Isaac Lab setting.

If GPU memory runs out, lower `--num_envs`.

## Output

`logs/<go2_live_flat|go2_live_rough|go2_live_stairs>/<timestamp>/`: `model_*.pt` (standard rsl-rl checkpoints), `params/env.yaml`, `params/agent.yaml`, TensorBoard events, `exported/policy.{pt,onnx}` at the end.

## Files

- `go2_isaac_live/env_cfg.py`: the three environments, stair terrain, follow camera, playback config
- `go2_isaac_live/ppo_cfg.py`: PPO settings
- `go2_isaac_live/live_runner.py`: rsl-rl's PPO loop with pause/stop/save hooks and live statistics
- `go2_isaac_live/live_panel.py`: the omni.ui panel and keyboard shortcuts
- `scripts/train_live.py`, `scripts/play.py`, `run.sh`

## Status

Optional tool, separate from the main `go2_stairs` task. It is not used for the results in the top-level README. Smoke-tested on this machine: headless training (flat), GUI training with the panel (stairs, 64 robots), checkpoint save, TorchScript/ONNX export, GUI playback. The stairs runs were stopped after about 200 iterations to demo the panel, so no trained policy from this tool is included.
