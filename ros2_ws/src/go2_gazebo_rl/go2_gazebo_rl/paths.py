"""Locate the go2_stairs_rl project, Unitree's Go2 description and trained policies."""

import glob
import os

PROJECT_ROOT = os.path.expanduser(os.environ.get("GO2_STAIRS_RL_ROOT", "~/go2_stairs_rl"))
GO2_DESCRIPTION = os.path.join(PROJECT_ROOT, "third_party", "unitree_ros", "robots", "go2_description")


def newest_checkpoint(experiment: str = "go2_stairs_blind") -> str:
    """Newest model_*.pt of the newest run that has one ('' if none)."""
    for run in sorted(glob.glob(os.path.join(PROJECT_ROOT, "logs", "rsl_rl", experiment, "*")), reverse=True):
        models = glob.glob(os.path.join(run, "model_*.pt"))
        if models:
            return max(models, key=lambda p: int(os.path.basename(p)[6:-3]))
    return ""


def deploy_cfg_for(policy_path: str) -> str:
    """params/deploy.yaml of the training run a checkpoint/export belongs to."""
    d = os.path.dirname(os.path.abspath(policy_path))
    for run_dir in (d, os.path.dirname(d)):  # model_*.pt or exported/policy.onnx
        cand = os.path.join(run_dir, "params", "deploy.yaml")
        if os.path.isfile(cand):
            return cand
    return ""
