"""Helpers shared by the scripts (imported after the simulator has started)."""

import glob
import importlib.metadata as metadata
import os

from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import load_cfg_from_registry

LOG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "rsl_rl"))


def load_cfgs(task: str, device: str):
    env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    env_cfg.sim.device = device
    agent_cfg.device = device
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    return env_cfg, agent_cfg


def resolve_checkpoint(path: str | None, experiment: str) -> str:
    """Return ``path`` or, if omitted, the newest model_*.pt of the newest run."""
    if path:
        return os.path.abspath(path)
    runs = sorted(glob.glob(os.path.join(LOG_ROOT, experiment, "*")))
    for run in reversed(runs):
        models = glob.glob(os.path.join(run, "model_*.pt"))
        if models:
            return max(models, key=lambda p: int(os.path.basename(p)[6:-3]))
    raise FileNotFoundError(f"No checkpoints under {os.path.join(LOG_ROOT, experiment)}; train first.")
