"""Watch a trained policy on the training terrains and export it (TorchScript + ONNX)."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.dirname(__file__))
from imu_monitor import ImuMonitor, add_imu_args, follow_imu_robot  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Go2-Stairs-v0", choices=["Go2-Stairs-v0", "Go2-Stairs-Blind-v0"])
parser.add_argument("--checkpoint", default=None, help="Defaults to the newest checkpoint of the task.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=0, help="Stop after N policy steps (0 = run until closed).")
add_imu_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import go2_stairs  # noqa: E402, F401

from common import load_cfgs, resolve_checkpoint  # noqa: E402


def main():
    env_cfg, agent_cfg = load_cfgs(args.task, args.device)
    env_cfg.scene.num_envs = args.num_envs
    follow_imu_robot(args, env_cfg)
    env_cfg.scene.terrain.max_init_terrain_level = None  # spread over all difficulty rows
    gen = env_cfg.scene.terrain.terrain_generator
    gen.num_rows, gen.num_cols, gen.curriculum = 5, 5, False
    env_cfg.curriculum.terrain_levels = None
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.push_robot = None

    checkpoint = resolve_checkpoint(args.checkpoint, agent_cfg.experiment_name)
    print(f"[go2_stairs] loading {checkpoint}")
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    export_dir = os.path.join(os.path.dirname(checkpoint), "exported")
    try:
        runner.export_policy_to_jit(path=export_dir, filename="policy.pt")  # rsl-rl >= 4 (includes normalizer)
        runner.export_policy_to_onnx(path=export_dir, filename="policy.onnx")
        print(f"[go2_stairs] exported policy to {export_dir}")
    except Exception as exc:  # export is optional; don't block playback
        print(f"[go2_stairs] policy export skipped: {exc}")

    imu_monitor = ImuMonitor(args, env.unwrapped)
    obs = env.get_observations()
    step = 0
    while simulation_app.is_running() and (args.steps == 0 or step < args.steps):
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
        step += 1
        imu_monitor.update(step)
    imu_monitor.close()
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        simulation_app.close()
