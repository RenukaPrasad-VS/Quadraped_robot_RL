"""Train a Go2 stair-climbing policy with PPO (rsl-rl)."""

import argparse
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Go2-Stairs-v0", choices=["Go2-Stairs-v0", "Go2-Stairs-Blind-v0"])
parser.add_argument("--num_envs", type=int, default=None, help="Parallel robots (default 2048).")
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume", default=None, help="Checkpoint (.pt) to continue from.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import go2_stairs  # noqa: E402, F401
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from common import LOG_ROOT, load_cfgs  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True


def main():
    env_cfg, agent_cfg = load_cfgs(args.task, args.device)
    if args.num_envs:
        env_cfg.scene.num_envs = args.num_envs
    if args.max_iterations:
        agent_cfg.max_iterations = args.max_iterations
    env_cfg.seed = agent_cfg.seed = args.seed

    log_dir = os.path.join(LOG_ROOT, agent_cfg.experiment_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    if args.resume:
        runner.load(args.resume)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    export_deploy_cfg(env.unwrapped, log_dir)  # params/deploy.yaml for unitree_rl_lab/deploy
    print(f"[go2_stairs] {args.task}: {env_cfg.scene.num_envs} robots, logs -> {log_dir}")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
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
