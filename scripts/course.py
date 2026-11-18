"""Run Go2 robots over the fixed stairs + side-slope course and score them.

Course (walked along +x): flat | stairs up | landing | stairs down | flat |
side slope tilted one way | flat | side slope tilted the other way | flat.

Per robot it reports: reached the end or not, where it fell, and roll/pitch
statistics measured by the simulated IMU. Results also go to
logs/course_eval/<timestamp>.json.
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Go2-Stairs-v0", choices=["Go2-Stairs-v0", "Go2-Stairs-Blind-v0"])
parser.add_argument("--checkpoint", default=None, help="Defaults to the newest checkpoint of the task.")
parser.add_argument("--policy", default="trained", choices=["trained", "zero"],
                    help="'zero' holds the standing pose (baseline / smoke test, no checkpoint needed).")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--step_height", type=float, default=0.12, help="Stair riser height in metres.")
parser.add_argument("--slope_deg", type=float, default=8.0, help="Side-slope angle in degrees (5-10 requested).")
parser.add_argument("--speed", type=float, default=0.6, help="Commanded forward speed (m/s).")
parser.add_argument("--push", action="store_true", help="Also apply random shoves every 5-10 s.")
parser.add_argument("--max_seconds", type=float, default=60.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import go2_stairs  # noqa: E402, F401
from go2_stairs.env_cfg import apply_course  # noqa: E402
from go2_stairs.terrains import StairSlopeCourseCfg, course_length  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from common import load_cfgs, resolve_checkpoint  # noqa: E402


def segments(cfg: StairSlopeCourseCfg):
    """(name, end_x) pairs measured from the spawn point."""
    s0 = cfg.start_length / 2.0
    out, x = [], s0
    for name, length in [
        ("start flat", cfg.start_length / 2.0),
        ("stairs up", cfg.num_steps * cfg.step_depth),
        ("landing", cfg.landing_length),
        ("stairs down", (cfg.num_steps - 1) * cfg.step_depth),
        ("flat", 2.0),
        ("side slope A", cfg.slope_length),
        ("flat", 1.0),
        ("side slope B", cfg.slope_length),
    ]:
        x += length
        out.append((name, x - s0))
    return out


def segment_at(progress: float, segs) -> str:
    for name, end in segs:
        if progress < end:
            return name
    return "finish"


def main():
    env_cfg, agent_cfg = load_cfgs(args.task, args.device)
    env_cfg.scene.num_envs = args.num_envs
    apply_course(env_cfg, args.step_height, args.slope_deg, args.speed, args.push)
    env_cfg.episode_length_s = args.max_seconds

    course_cfg = env_cfg.scene.terrain.terrain_generator.sub_terrains["course"]
    goal = course_length(course_cfg)
    segs = segments(course_cfg)

    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    base = env.unwrapped
    if args.policy == "trained":
        checkpoint = resolve_checkpoint(args.checkpoint, agent_cfg.experiment_name)
        print(f"[course] loading {checkpoint}")
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(checkpoint)
        policy = runner.get_inference_policy(device=base.device)
    else:
        checkpoint = None
        zeros = torch.zeros(base.num_envs, base.action_manager.total_action_dim, device=base.device)
        policy = lambda obs: zeros  # noqa: E731

    print(f"[course] steps {args.step_height * 100:.0f} cm x {course_cfg.num_steps}, side slopes ±{args.slope_deg:.1f}°,"
          f" speed {args.speed} m/s, goal {goal:.1f} m, push={args.push}, policy={args.policy}")

    robot, imu = base.scene["robot"], base.scene.sensors["imu"]
    origins = base.scene.env_origins
    n = base.num_envs
    active = torch.ones(n, dtype=torch.bool, device=base.device)
    status = ["running"] * n
    progress = torch.zeros(n, device=base.device)
    max_roll = torch.zeros(n, device=base.device)
    max_pitch = torch.zeros(n, device=base.device)
    sq_roll = torch.zeros(n, device=base.device)
    sq_pitch = torch.zeros(n, device=base.device)
    max_rate = torch.zeros(n, device=base.device)
    samples = torch.zeros(n, device=base.device)

    obs = env.get_observations()
    max_steps = int(args.max_seconds / base.step_dt)
    for step in range(max_steps):
        if not simulation_app.is_running():
            break
        with torch.inference_mode():
            obs, _, dones, extras = env.step(policy(obs))

        g = imu.data.projected_gravity_b
        roll = torch.atan2(g[:, 1], -g[:, 2]).abs()
        pitch = torch.asin(torch.clamp(-g[:, 0], -1.0, 1.0)).abs()
        rate = torch.linalg.norm(imu.data.ang_vel_b[:, :2], dim=1)
        live = active & ~dones.bool()
        progress = torch.where(live, robot.data.root_pos_w[:, 0] - origins[:, 0], progress)
        max_roll = torch.where(live, torch.maximum(max_roll, roll), max_roll)
        max_pitch = torch.where(live, torch.maximum(max_pitch, pitch), max_pitch)
        max_rate = torch.where(live, torch.maximum(max_rate, rate), max_rate)
        sq_roll += live * roll**2
        sq_pitch += live * pitch**2
        samples += live

        time_outs = extras.get("time_outs", torch.zeros_like(dones)).bool()
        for i in torch.nonzero(active).flatten().tolist():
            if progress[i] >= goal:
                status[i], active[i] = "finished", False
            elif dones[i]:
                status[i] = "timeout" if time_outs[i] else f"fell on {segment_at(progress[i].item(), segs)}"
                active[i] = False
        if step % int(2.0 / base.step_dt) == 0:
            a = active
            print(f"[course] t={step * base.step_dt:5.1f}s active={int(a.sum())}/{n} "
                  f"mean progress={progress.mean():5.2f} m  max roll={math.degrees(max_roll.max()):4.1f}° "
                  f"max pitch={math.degrees(max_pitch.max()):4.1f}°")
        if not active.any():
            break

    for i in torch.nonzero(active).flatten().tolist():
        status[i] = f"stopped on {segment_at(progress[i].item(), segs)} (time limit)"

    rows = []
    for i in range(n):
        k = max(samples[i].item(), 1.0)
        rows.append({
            "robot": i,
            "status": status[i],
            "progress_m": round(progress[i].item(), 2),
            "max_roll_deg": round(math.degrees(max_roll[i].item()), 1),
            "max_pitch_deg": round(math.degrees(max_pitch[i].item()), 1),
            "rms_roll_deg": round(math.degrees(math.sqrt(sq_roll[i].item() / k)), 1),
            "rms_pitch_deg": round(math.degrees(math.sqrt(sq_pitch[i].item() / k)), 1),
            "max_tilt_rate_dps": round(math.degrees(max_rate[i].item()), 0),
        })
    finished = sum(r["status"] == "finished" for r in rows)
    print("\n robot | status                         | progress | max roll | max pitch | rms roll | rms pitch")
    for r in rows:
        print(f" {r['robot']:5d} | {r['status'][:30]:30s} | {r['progress_m']:6.2f} m | {r['max_roll_deg']:6.1f}° |"
              f" {r['max_pitch_deg']:7.1f}° | {r['rms_roll_deg']:6.1f}° | {r['rms_pitch_deg']:7.1f}°")
    print(f"\n[course] finished {finished}/{n}  (goal {goal:.1f} m)")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "logs", "course_eval")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.abspath(os.path.join(out_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".json"))
    with open(out, "w") as f:
        json.dump({"args": vars(args), "checkpoint": checkpoint, "goal_m": goal,
                   "finished": finished, "robots": rows}, f, indent=2, default=str)
    print(f"[course] results -> {out}")
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
