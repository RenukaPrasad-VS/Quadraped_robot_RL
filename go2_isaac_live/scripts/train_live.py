"""Spawn N Go2 robots in Isaac Sim and train them with PPO while you watch.

    ./run.sh train --task flat --num_envs 256
    ./run.sh train --task stairs --num_envs 512 --max_iterations 5000
    ./run.sh train --task rough --resume logs/go2_live_rough/<run>/model_500.pt
    ./run.sh train --task flat --headless          # no window, no panel (fastest); Ctrl+C = stop + save,
                                                   # then run.sh opens the result in Isaac Sim (--no-play to skip)
"""

import argparse
import os
import signal
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PKG_ROOT)
from go2_isaac_live import TASKS  # noqa: E402  (only registers gym ids, no Isaac imports)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="flat", choices=sorted(TASKS))
parser.add_argument("--num_envs", type=int, default=None, help="Robots to spawn (default: flat 256, rough/stairs 512).")
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume", default=None, help="Checkpoint model_*.pt to continue from.")
parser.add_argument("--start_paused", action="store_true", help="Open paused; press P / Resume to start.")
parser.add_argument("--lite", action="store_true",
                    help="Low-memory mode: 256 robots (unless --num_envs), smaller terrain grid (6 levels x 8 columns).")
parser.add_argument("--turbo", action="store_true", help="Start with the viewport hidden (toggle with T).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import importlib.metadata as metadata  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

from go2_isaac_live.live_runner import LiveRunner  # noqa: E402

LOG_ROOT = os.path.join(PKG_ROOT, "logs")
torch.backends.cuda.matmul.allow_tf32 = True


class HeadlessControl:
    """Stands in for the GUI panel in --headless runs: Ctrl+C stops cleanly (save + export)."""

    stop_requested = False

    def __init__(self):
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

    def _on_signal(self, *_):
        if self.stop_requested:
            print("\n[go2_isaac_live] second Ctrl+C: aborting without saving", flush=True)
            os._exit(130)
        self.stop_requested = True
        print("\n[go2_isaac_live] Ctrl+C: stopping, saving checkpoint and exporting...", flush=True)

    def on_sim_step(self, runner) -> float:
        return 0.0

    def consume_save_request(self) -> bool:
        return False

    def on_iteration(self, **_):
        pass

    def close(self):
        pass


def main():
    task_id = TASKS[args.task]
    env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")
    env_cfg.sim.device = agent_cfg.device = args.device
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    if args.lite:
        env_cfg.scene.num_envs = 256
        gen = env_cfg.scene.terrain.terrain_generator
        if gen is not None:
            gen.num_rows, gen.num_cols, gen.border_width = 6, 8, 10.0
            if env_cfg.scene.terrain.max_init_terrain_level is not None:
                env_cfg.scene.terrain.max_init_terrain_level = min(env_cfg.scene.terrain.max_init_terrain_level, 2)
    if args.num_envs:
        env_cfg.scene.num_envs = args.num_envs
    if args.max_iterations:
        agent_cfg.max_iterations = args.max_iterations
    env_cfg.seed = agent_cfg.seed = args.seed

    log_dir = os.path.join(LOG_ROOT, agent_cfg.experiment_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    env = RslRlVecEnvWrapper(gym.make(task_id, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = LiveRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    if args.resume:
        runner.load(os.path.abspath(args.resume))
        print(f"[go2_isaac_live] resumed from {args.resume} (iteration {runner.current_learning_iteration})")
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    panel = None
    if args.headless:
        panel = HeadlessControl()
        print("[go2_isaac_live] headless: press Ctrl+C to stop, save and export (twice to abort without saving)")
    else:
        from go2_isaac_live.live_panel import LivePanel

        panel = LivePanel(env, task_id, log_dir)
        panel.attach_runner(runner)
        if args.start_paused:
            panel.toggle_pause()
        if args.turbo:
            panel.toggle_turbo()

    print(f"[go2_isaac_live] {task_id}: {env.num_envs} robots, {agent_cfg.max_iterations} iterations, logs -> {log_dir}")
    last = runner.learn_live(agent_cfg.max_iterations, panel=panel)
    print(f"[go2_isaac_live] final checkpoint: {last}")
    if last:  # run.sh opens this checkpoint in Isaac Sim after a headless run
        with open(os.path.join(LOG_ROOT, ".last_checkpoint"), "w") as f:
            f.write(last + "\n")

    if last:
        export_dir = os.path.join(log_dir, "exported")
        try:
            runner.export_policy_to_jit(export_dir, "policy.pt")
            runner.export_policy_to_onnx(export_dir, "policy.onnx")
            print(f"[go2_isaac_live] exported TorchScript/ONNX policy to {export_dir}")
        except Exception as exc:
            print(f"[go2_isaac_live] export skipped: {exc}")
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
