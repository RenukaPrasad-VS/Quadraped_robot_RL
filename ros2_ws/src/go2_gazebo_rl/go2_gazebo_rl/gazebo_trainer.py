"""Live PPO fine-tuning of the Go2 stair policy inside Gazebo.

The actor starts from an Isaac Lab checkpoint (it already walks). Training
continues here on Gazebo's physics with a stair-height curriculum: the world
has side-by-side lanes with increasing stair height and side-slope angle.
A robot that finishes its lane is promoted to the next one; a robot that
covers less than half of it is demoted. This mirrors Isaac Lab's terrain
curriculum.

Loop (one robot, Gazebo at its real-time factor, policy at 50 Hz sim time):
  reset  -> teleport to the lane start (set_pose), blend legs to the standing pose
  run    -> obs -> sample action -> joint targets; reward from the next state
  done   -> fall (tilt > 1 rad / off the lane), goal reached, or time limit
  update -> every `rollout_steps` steps the world is paused for a PPO update

Checkpoints use the rsl-rl key layout, so `sim.launch.py policy:=...` and the
policy node can evaluate them directly.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
from datetime import datetime

import numpy as np
import rclpy
import torch
import torch.nn as nn
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64

from .terrain_lanes import WORLDS, Lane
from .paths import PROJECT_ROOT, deploy_cfg_for, newest_checkpoint
from .policy import DeployCfg
from .policy_node import quat_rotate_inverse, wrap, yaw_of

LANE_SPACING = 6.0
SPAWN_Z = 0.48


def lane_y(i: int) -> float:
    return i * LANE_SPACING


# ---------------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------------
def mlp(sizes: list[int]) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-2):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))

    def update(self, x: torch.Tensor):
        b_mean, b_var, n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta, tot = b_mean - self.mean, self.count + n
        self.mean += delta * n / tot
        self.var = (self.var * self.count + b_var * n + delta**2 * self.count * n / tot) / tot
        self.count = tot

    def forward(self, x):
        return (x - self.mean) / (self.var.sqrt() + self.eps)


class Actor(nn.Module):
    """Same layout as the rsl-rl actor; its normalizer stays frozen at the Isaac statistics."""

    def __init__(self, sd: dict, init_std: float):
        super().__init__()
        idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith(".weight")})
        sizes = [sd[f"mlp.{idx[0]}.weight"].shape[1]] + [sd[f"mlp.{i}.weight"].shape[0] for i in idx]
        self.mlp = mlp(sizes)
        self.mlp.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("mlp.")})
        self.register_buffer("obs_mean", sd["obs_normalizer._mean"][0].float())
        self.register_buffer("obs_std", sd["obs_normalizer._std"][0].float())
        std = sd["distribution.std_param"].float().clamp(0.05, init_std)
        self.log_std = nn.Parameter(std.log())
        self._template = sd

    def mean(self, obs):
        return self.mlp((obs - self.obs_mean) / (self.obs_std + 1e-2))

    def dist(self, obs):
        return torch.distributions.Normal(self.mean(obs), self.log_std.exp())

    def rsl_state_dict(self) -> dict:
        sd = {k: v.clone() for k, v in self._template.items()}
        for k, v in self.mlp.state_dict().items():
            sd["mlp." + k] = v.detach().clone()
        sd["distribution.std_param"] = self.log_std.detach().exp().clone()
        return sd


# ---------------------------------------------------------------------------------
# Trainer node
# ---------------------------------------------------------------------------------
class GazeboTrainer(Node):
    def __init__(self):
        super().__init__("go2_gazebo_trainer")
        P = self.declare_parameter
        P("init_policy", "")
        P("cmd_vx", 0.6)
        P("episode_s", 40.0)
        P("rollout_steps", 2000)
        P("world", "stairs")  # "stairs" (stair-height curriculum) or "terrain" (varied terrain types)
        P("lane", -1)  # -1 = curriculum / adaptive sampling over all lanes, else train on that lane only
        P("actor_lr", 1.0e-4)
        P("critic_lr", 1.0e-3)
        P("init_std", 0.3)
        P("value_warmup", 3)
        P("max_iterations", 10000)
        P("model_name", "go2")
        p = lambda n: self.get_parameter(n).value  # noqa: E731

        init = p("init_policy") or newest_checkpoint()
        if not init:
            raise RuntimeError("No Isaac Lab checkpoint to start from (train Go2-Stairs-Blind-v0 first).")
        self.cfg = DeployCfg(deploy_cfg_for(init))
        ck = torch.load(init, map_location="cpu", weights_only=False)
        self.actor = Actor(ck["actor_state_dict"], p("init_std"))
        self.obs_dim = self.cfg.obs_dim
        self.critic_dim = self.obs_dim + 4  # + body linear velocity (3) + height above lane start
        self.critic = mlp([self.critic_dim, 256, 128, 1])
        self.critic_norm = RunningNorm(self.critic_dim)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=p("actor_lr"))
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=p("critic_lr"))

        self.world = p("world")
        build, self.curriculum = WORLDS[self.world]
        self.lanes: list[Lane] = build()
        self.fixed_lane = int(p("lane"))
        # Exponential moving average of the fraction of each lane completed; lanes the robot is
        # worse at are sampled more often (weight = 1.2 - completion).
        self.lane_score = np.zeros(len(self.lanes))
        self.lane_episodes = np.zeros(len(self.lanes), dtype=int)
        self.rng = np.random.default_rng(0)
        self.level = self.fixed_lane if self.fixed_lane >= 0 else 0
        self.model = p("model_name")
        self.cmd = np.array([p("cmd_vx"), 0.0, 0.0], dtype=np.float32)
        self.dt = self.cfg.step_dt
        self.n_act = len(self.cfg.joint_names)

        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = os.path.join(PROJECT_ROOT, "logs", "gazebo_train", stamp)
        os.makedirs(os.path.join(self.log_dir, "params"), exist_ok=True)
        shutil.copy(deploy_cfg_for(init), os.path.join(self.log_dir, "params", "deploy.yaml"))
        with open(os.path.join(self.log_dir, "params", "trainer.json"), "w") as f:
            json.dump({"init_policy": init, "world": self.world, "curriculum": self.curriculum, "lanes": [l.name for l in self.lanes], "lane": self.fixed_lane,
                       **{n: p(n) for n in ("cmd_vx", "episode_s", "rollout_steps", "actor_lr", "critic_lr",
                                            "init_std", "value_warmup")}}, f, indent=2)
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.tb = SummaryWriter(self.log_dir)
        except Exception:  # tensorboard optional
            self.tb = None

        # ROS I/O
        self.pubs = [self.create_publisher(Float64, f"/{self.model}/joint/{j}/cmd_pos", 10)
                     for j in self.cfg.joint_names]
        self.create_subscription(Imu, f"/{self.model}/imu", self.on_imu, qos_profile_sensor_data)
        self.create_subscription(JointState, f"/{self.model}/joint_states", self.on_joints, qos_profile_sensor_data)
        self.create_subscription(Odometry, f"/model/{self.model}/odometry", self.on_odom, qos_profile_sensor_data)

        self.q = np.zeros(self.n_act, np.float32)
        self.dq = np.zeros(self.n_act, np.float32)
        self.quat = None
        self.gyro = np.zeros(3, np.float32)
        self.pos = None
        self.prev_pos = None
        self.have_joints = False

        self.buf = {k: [] for k in ("obs", "cobs", "act", "logp", "val", "rew", "done", "timeout")}
        self.iteration = 0
        self.total_steps = 0
        self.ep_stats = []
        self.phase = "WAIT"
        self.phase_t0 = 0.0
        self.pending = None  # (obs, cobs, act, logp, val) waiting for its reward
        self.get_logger().info(f"init policy: {init}")
        self.get_logger().info(f"logs/checkpoints: {self.log_dir}")
        self.get_logger().info(f"world: {self.world} ({self.curriculum} lane selection)")
        for i, l in enumerate(self.lanes):
            self.get_logger().info(f"lane {i} {l.name}: " + ", ".join(n for n, _, _ in l.segs[1:-1]))
        self.create_timer(self.dt, self.tick)

    # --- sensors -----------------------------------------------------------------
    def on_imu(self, m: Imu):
        o, w = m.orientation, m.angular_velocity
        self.quat = np.array([o.x, o.y, o.z, o.w])
        self.gyro = np.array([w.x, w.y, w.z], np.float32)

    def on_joints(self, m: JointState):
        idx = {n: i for i, n in enumerate(m.name)}
        try:
            ids = [idx[j] for j in self.cfg.joint_names]
        except KeyError:
            return
        self.q = np.array([m.position[i] for i in ids], np.float32)
        if len(m.velocity) == len(m.name):
            self.dq = np.array([m.velocity[i] for i in ids], np.float32)
        self.have_joints = True

    def on_odom(self, m: Odometry):
        p = m.pose.pose.position
        self.pos = np.array([p.x, p.y, p.z])

    # --- Gazebo services (CLI; Fortress has no Python transport bindings) ----------
    def gz_service(self, service: str, reqtype: str, req: str):
        cmd = ["ign", "service", "-s", f"/world/course/{service}", "--reqtype", reqtype,
               "--reptype", "ignition.msgs.Boolean", "--timeout", "3000", "--req", req]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if "true" not in r.stdout:
            self.get_logger().warn(f"gz service {service} failed: {r.stdout.strip()} {r.stderr.strip()}")

    def pause(self, paused: bool):
        self.gz_service("control", "ignition.msgs.WorldControl", f"pause: {'true' if paused else 'false'}")

    def teleport(self, level: int):
        """Place the robot upright at the lane start. set_pose keeps velocities, so callers bring the
        robot to rest first (see the SETTLE_DOWN phase); Fortress world reset proved unreliable."""
        x, y = self.lanes[level].spawn_x, lane_y(level)
        self.gz_service("set_pose", "ignition.msgs.Pose",
                        f'name: "{self.model}", position: {{x: {x}, y: {y}, z: {SPAWN_Z}}}, orientation: {{w: 1}}')

    # --- helpers -----------------------------------------------------------------
    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def publish(self, target):
        self.target = np.asarray(target, np.float32)
        for pub, v in zip(self.pubs, self.target):
            pub.publish(Float64(data=float(v)))

    def gravity_b(self):
        return quat_rotate_inverse(self.quat, np.array([0.0, 0.0, -1.0]))

    def observation(self):
        yaw = yaw_of(self.quat)
        self.cmd[2] = float(np.clip(0.5 * wrap(0.0 - yaw), -1.0, 1.0))  # hold heading along the lane
        values = {
            "base_ang_vel": self.gyro,
            "projected_gravity": self.gravity_b().astype(np.float32),
            "velocity_commands": self.cmd,
            "joint_pos_rel": self.q - self.cfg.default_q,
            "joint_vel_rel": self.dq,
            "last_action": self.last_action,
        }
        parts = []
        for name, scale, clip in self.cfg.obs_terms:
            v = values[name] * scale
            parts.append(np.clip(v, *clip) if clip else v)
        return np.concatenate(parts).astype(np.float32)

    def body_velocity(self):
        """Finite-difference world velocity -> body frame (odometry twist frames vary by version)."""
        if self.prev_pos is None:
            return np.zeros(3), np.zeros(3)
        v_w = (self.pos - self.prev_pos) / self.dt
        return v_w, quat_rotate_inverse(self.quat, v_w)

    # --- episode control -----------------------------------------------------------
    def start_reset(self):
        self.teleport(self.level)
        self.phase, self.phase_t0 = "STAND", self.now()
        self.q_start = None  # captured on the first STAND tick, once post-reset joint states arrive
        self.pending = None

    def begin_episode(self):
        c = self.lanes[self.level]
        self.ep = {"level": self.level, "t0": self.now(), "ret": 0.0, "len": 0,
                   "x0": c.spawn_x, "y0": lane_y(self.level), "z0": self.pos[2], "max_x": c.spawn_x}
        self.last_action = np.zeros(self.n_act, np.float32)
        self.prev_pos = self.pos.copy()
        self.phase = "RUN"

    def end_episode(self, status: str):
        c = self.lanes[self.ep["level"]]
        progress = self.ep["max_x"] - c.spawn_x
        goal = c.goal_x - c.spawn_x
        lane = self.ep["level"]
        frac = 1.0 if status == "goal" else float(np.clip(progress / goal, 0.0, 1.0))
        a = 0.3 if self.lane_episodes[lane] else 1.0
        self.lane_score[lane] = (1 - a) * self.lane_score[lane] + a * frac
        self.lane_episodes[lane] += 1
        if self.fixed_lane >= 0:
            self.level = self.fixed_lane
        elif self.curriculum == "levels":
            # stairs world: lanes are difficulty levels (Isaac Lab-style promote / demote)
            if status == "goal":
                self.level = min(lane + 1, len(self.lanes) - 1)
            elif frac < 0.5:
                self.level = max(lane - 1, 0)
        else:
            w = 1.2 - self.lane_score
            self.level = int(self.rng.choice(len(self.lanes), p=w / w.sum()))
        self.ep_stats.append({"status": status, "ret": self.ep["ret"], "len": self.ep["len"],
                              "progress": progress, "level": self.ep["level"],
                              "segment": c.segment_at(self.ep["max_x"])})
        self.get_logger().info(
            f"episode: lane {self.ep['level']} {c.name:9s} {status:7s} "
            f"progress {progress:5.2f}/{goal:.1f} m (reached {c.segment_at(self.ep['max_x'])}), "
            f"return {self.ep['ret']:6.2f} -> next: lane {self.level} {self.lanes[self.level].name}"
        )
        # Bring the robot to rest before teleporting: policy with a zero command after a time-out/goal,
        # holding the current pose after a fall.
        self.settle_mode = "hold" if status == "fell" else "stop"
        self.hold_q = self.q.copy()
        self.phase, self.phase_t0 = "SETTLE_DOWN", self.now()

    # --- main loop -------------------------------------------------------------------
    def tick(self):
        if self.phase == "WAIT":
            if self.have_joints and self.quat is not None and self.pos is not None:
                self.get_logger().info("sensors up, starting first episode")
                self.start_reset()
            return
        if self.phase == "SETTLE_DOWN":
            if self.settle_mode == "hold":
                self.publish(self.hold_q)
            else:
                cmd_saved = self.cmd.copy()
                self.cmd[:] = 0.0
                obs = self.observation()
                self.cmd[:] = cmd_saved
                self.cmd[2] = 0.0
                with torch.no_grad():
                    act = self.actor.mean(torch.from_numpy(obs)[None])[0].numpy()
                self.last_action = act.astype(np.float32)
                self.publish(self.cfg.action_offset + self.cfg.action_scale * self.last_action)
            if self.now() - self.phase_t0 >= 1.5:
                self.start_reset()
            return
        if self.phase == "STAND":
            if self.now() < self.phase_t0:  # sim clock rewound by the world reset
                self.phase_t0 = self.now()
            if self.q_start is None:
                self.q_start = self.q.copy()
            a = min((self.now() - self.phase_t0) / 2.0, 1.0)
            a = 0.5 - 0.5 * math.cos(math.pi * a)
            self.publish((1 - a) * self.q_start + a * self.cfg.default_q)
            if self.now() - self.phase_t0 >= 2.5:
                self.begin_episode()
            return

        # RUN: finish the previous transition with the reward of the state we just reached
        v_w, v_b = self.body_velocity()
        g = self.gravity_b()
        c = self.lanes[self.ep["level"]]
        x = self.pos[0]
        self.ep["max_x"] = max(self.ep["max_x"], x)
        tilt = math.acos(max(-1.0, min(1.0, -g[2])))
        fell = tilt > 1.0 or self.pos[2] < -0.3 or abs(self.pos[1] - self.ep["y0"]) > c.width / 2 + 0.5
        goal = x >= c.goal_x
        timeout = self.now() - self.ep["t0"] >= self.get_parameter("episode_s").value

        if self.pending is not None:
            r = self.reward(v_w, v_b, g, fell, goal)
            self.store(r, done=fell or goal or timeout, timeout=timeout and not (fell or goal))
            self.ep["ret"] += r
            self.ep["len"] += 1
        self.prev_pos = self.pos.copy()

        if fell or goal or timeout:
            if len(self.buf["rew"]) >= self.get_parameter("rollout_steps").value:
                self.update()
            self.end_episode("fell" if fell else "goal" if goal else "timeout")
            return
        if len(self.buf["rew"]) >= self.get_parameter("rollout_steps").value:
            self.update()  # mid-episode: bootstrap from the value of the current state

        obs = self.observation()
        cobs = np.concatenate([obs, v_b.astype(np.float32), [self.pos[2] - self.ep["z0"]]]).astype(np.float32)
        with torch.no_grad():
            o = torch.from_numpy(obs)[None]
            dist = self.actor.dist(o)
            act = dist.sample()
            logp = dist.log_prob(act).sum(-1)
            val = self.critic(self.critic_norm(torch.from_numpy(cobs)[None]))[0, 0]
        act = act[0].numpy()
        self.pending = (obs, cobs, act, float(logp), float(val))
        self.prev_action = self.last_action
        self.last_action = act.astype(np.float32)
        self.publish(self.cfg.action_offset + self.cfg.action_scale * self.last_action)

    def reward(self, v_w, v_b, g, fell, goal) -> float:
        """Isaac Lab-style terms (weight x term x dt) + course progress."""
        yaw = yaw_of(self.quat)
        cy, sy = math.cos(yaw), math.sin(yaw)
        v_yaw = np.array([cy * v_w[0] + sy * v_w[1], -sy * v_w[0] + cy * v_w[1]])
        tau = np.clip(self.cfg.kp * (self.target - self.q) - self.cfg.kd * self.dq, -23.5, 23.5)
        terms = {
            "track_lin_vel": 1.5 * math.exp(-float(np.sum((self.cmd[:2] - v_yaw) ** 2)) / 0.25),
            "track_ang_vel": 0.75 * math.exp(-(self.cmd[2] - self.gyro[2]) ** 2 / 0.25),
            "lin_vel_z": -0.5 * float(v_b[2] ** 2),
            "ang_vel_xy": -0.05 * float(self.gyro[0] ** 2 + self.gyro[1] ** 2),
            "flat_orientation": -1.0 * float(g[0] ** 2 + g[1] ** 2),
            "action_rate": -0.05 * float(np.sum((self.last_action - self.prev_action) ** 2)),
            "torques": -2e-4 * float(np.sum(tau**2)),
            "joint_vel": -1e-3 * float(np.sum(self.dq**2)),
            "progress": 2.0 * float(np.clip(v_w[0], -1.0, 1.0)),
        }
        r = sum(terms.values()) * self.dt
        if fell:
            r -= 2.0
        if goal:
            r += 10.0
        return r

    def store(self, r, done, timeout):
        obs, cobs, act, logp, val = self.pending
        for k, v in zip(("obs", "cobs", "act", "logp", "val", "rew", "done", "timeout"),
                        (obs, cobs, act, logp, val, r, done, timeout)):
            self.buf[k].append(v)
        self.total_steps += 1
        self.pending = None

    # --- PPO update ------------------------------------------------------------------
    def update(self):
        self.pause(True)
        t_start = time.time()
        B = {k: torch.as_tensor(np.array(v), dtype=torch.float32) for k, v in self.buf.items()}
        n = B["rew"].shape[0]
        # Bootstrap value for the last state if the rollout was cut mid-episode.
        with torch.no_grad():
            self.critic_norm.update(B["cobs"])
            values = self.critic(self.critic_norm(B["cobs"]))[:, 0]
            last_val = 0.0
            if not bool(B["done"][-1]):
                last_val = float(values[-1])
        gamma, lam = 0.99, 0.95
        adv = torch.zeros(n)
        gae = 0.0
        for t in reversed(range(n)):
            next_val = last_val if t == n - 1 else float(values[t + 1])
            nonterminal = 1.0 - float(B["done"][t])
            # time-outs are not true terminations: bootstrap with the current value estimate
            r = float(B["rew"][t]) + (gamma * float(values[t]) if B["timeout"][t] else 0.0)
            delta = r + gamma * next_val * nonterminal - float(values[t])
            gae = delta + gamma * lam * nonterminal * gae
            adv[t] = gae
        returns = adv + values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        train_actor = self.iteration >= self.get_parameter("value_warmup").value
        stats = {"value_loss": 0.0, "kl": 0.0, "clip_frac": 0.0}
        n_mb, epochs, updates = 4, 5, 0
        for _ in range(epochs):
            perm = torch.randperm(n)
            for mb in perm.chunk(n_mb):
                v = self.critic(self.critic_norm(B["cobs"][mb]))[:, 0]
                v_loss = ((v - returns[mb]) ** 2).mean()
                self.opt_critic.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.opt_critic.step()
                stats["value_loss"] += float(v_loss)
                if train_actor:
                    dist = self.actor.dist(B["obs"][mb])
                    logp = dist.log_prob(B["act"][mb]).sum(-1)
                    ratio = (logp - B["logp"][mb]).exp()
                    s1, s2 = ratio * adv[mb], ratio.clamp(0.8, 1.2) * adv[mb]
                    a_loss = -torch.min(s1, s2).mean()
                    self.opt_actor.zero_grad()
                    a_loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                    self.opt_actor.step()
                    with torch.no_grad():
                        stats["kl"] += float((B["logp"][mb] - logp).mean())
                        stats["clip_frac"] += float(((ratio - 1).abs() > 0.2).float().mean())
                updates += 1
            if train_actor and stats["kl"] / updates > 0.02:
                break  # keep the fine-tuned policy close to the Isaac one
        for k in stats:
            stats[k] /= max(updates, 1)

        self.iteration += 1
        eps = self.ep_stats
        summary = {
            "iteration": self.iteration, "steps": self.total_steps, "episodes": len(eps),
            "mean_return": float(np.mean([e["ret"] for e in eps])) if eps else float("nan"),
            "mean_progress_m": float(np.mean([e["progress"] for e in eps])) if eps else float("nan"),
            "goal_rate": float(np.mean([e["status"] == "goal" for e in eps])) if eps else float("nan"),
            "fall_rate": float(np.mean([e["status"] == "fell" for e in eps])) if eps else float("nan"),
            "lane_completion": {l.name: round(float(v), 2) for l, v in zip(self.lanes, self.lane_score)},
            "action_std": float(self.actor.log_std.exp().mean()),
            "actor_trained": train_actor, **stats, "update_s": round(time.time() - t_start, 1),
        }
        self.get_logger().info("UPDATE " + json.dumps({k: round(v, 3) if isinstance(v, float) else v
                                                         for k, v in summary.items()}))
        if self.tb:
            for k, v in summary.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    self.tb.add_scalar(f"gazebo/{k}", v, self.iteration)
            for name, v in summary["lane_completion"].items():
                self.tb.add_scalar(f"lane_completion/{name}", v, self.iteration)
        with open(os.path.join(self.log_dir, "progress.jsonl"), "a") as f:
            f.write(json.dumps(summary) + "\n")
        torch.save({"actor_state_dict": self.actor.rsl_state_dict(),
                    "critic_state_dict": {"mlp": self.critic.state_dict(), "norm": self.critic_norm.state_dict()},
                    "iter": self.iteration, "lane_completion": self.lane_score.tolist()},
                   os.path.join(self.log_dir, f"model_{self.iteration}.pt"))
        self.buf = {k: [] for k in self.buf}
        self.ep_stats = []
        if self.iteration >= self.get_parameter("max_iterations").value:
            self.get_logger().info("max_iterations reached")
            raise SystemExit
        self.pause(False)


def main():
    rclpy.init()
    node = GazeboTrainer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
