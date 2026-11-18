"""Watch a trained policy (no learning). Defaults to the newest checkpoint of the task.

    ./run.sh play --task flat
    ./run.sh play --task stairs --num_envs 32 --checkpoint logs/go2_live_stairs/<run>/model_3000.pt
    ./run.sh play --task flat --vx 1.0 --wz 0.5     # fixed velocity command for every robot
"""

import argparse
import glob
import os
import sys

from isaaclab.app import AppLauncher

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PKG_ROOT)
from go2_isaac_live import TASKS  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="flat", choices=sorted(TASKS))
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--vx", type=float, default=None, help="Fixed forward speed command (m/s).")
parser.add_argument("--vy", type=float, default=0.0)
parser.add_argument("--wz", type=float, default=0.0)
parser.add_argument("--steps", type=int, default=0, help="Stop after N policy steps (0 = until the window closes).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import importlib.metadata as metadata  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

from go2_isaac_live.env_cfg import make_play_cfg  # noqa: E402
from go2_isaac_live.live_runner import LiveRunner  # noqa: E402


def newest_checkpoint(experiment: str) -> str:
    for run in sorted(glob.glob(os.path.join(PKG_ROOT, "logs", experiment, "*")), reverse=True):
        models = glob.glob(os.path.join(run, "model_*.pt"))
        if models:
            return max(models, key=lambda p: int(os.path.basename(p)[6:-3]))
    raise FileNotFoundError(f"no checkpoints in logs/{experiment}; train first (./run.sh train --task ...)")


def main():
    task_id = TASKS[args.task]
    env_cfg = make_play_cfg(load_cfg_from_registry(task_id, "env_cfg_entry_point"))
    agent_cfg = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")
    env_cfg.sim.device = agent_cfg.device = args.device
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env_cfg.scene.num_envs = args.num_envs
    if args.vx is not None:
        r = env_cfg.commands.base_velocity.ranges
        r.lin_vel_x, r.lin_vel_y, r.ang_vel_z = (args.vx,) * 2, (args.vy,) * 2, (args.wz,) * 2
        env_cfg.commands.base_velocity.heading_command = False
        env_cfg.commands.base_velocity.rel_standing_envs = 0.0

    ckpt = os.path.abspath(args.checkpoint) if args.checkpoint else newest_checkpoint(agent_cfg.experiment_name)
    print(f"[go2_isaac_live] playing {ckpt}")
    env = RslRlVecEnvWrapper(gym.make(task_id, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = LiveRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    panel = None
    if not args.headless:
        from go2_isaac_live.live_panel import LivePanel

        panel = LivePanel(env, f"{task_id} (playback)", os.path.dirname(ckpt))
        panel.labels["status"].text = f"Status: playing {os.path.basename(ckpt)}"

    obs = env.get_observations()
    step = 0
    while simulation_app.is_running() and (args.steps == 0 or step < args.steps):
        if panel is not None:
            panel.on_sim_step(runner)
            if panel.stop_requested:
                break
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
        step += 1
    if panel is not None:
        panel.close()
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        raise
    finally:
        simulation_app.close()
