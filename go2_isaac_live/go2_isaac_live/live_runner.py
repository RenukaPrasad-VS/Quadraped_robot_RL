"""rsl-rl PPO runner, opened up so training can be watched and steered while it runs.

It is rsl-rl's ``OnPolicyRunner.learn`` loop split into steps, with hooks:

* after every simulation step: pause (the viewport keeps rendering), stop, camera follow;
* after every PPO update: live statistics for the panel, on-demand checkpoint saving.

Checkpoints are ordinary rsl-rl ``model_<it>.pt`` files.
"""

from __future__ import annotations

import collections
import os
import time

import torch
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import check_nan


class LiveStats:
    """Episode returns/lengths finished during training, and per-iteration history for plotting."""

    def __init__(self, num_envs: int, device: str, window: int = 200, history: int = 300):
        self.ep_return = torch.zeros(num_envs, device=device)
        self.ep_len = torch.zeros(num_envs, device=device)
        self.returns: collections.deque[float] = collections.deque(maxlen=window)
        self.lengths: collections.deque[float] = collections.deque(maxlen=window)
        self.return_history: collections.deque[float] = collections.deque(maxlen=history)
        self.terrain_level = float("nan")
        self.extra: dict[str, float] = {}

    def step(self, rewards: torch.Tensor, dones: torch.Tensor, extras: dict):
        self.ep_return += rewards
        self.ep_len += 1
        done_ids = dones.nonzero(as_tuple=False).flatten()
        if done_ids.numel():
            self.returns.extend(self.ep_return[done_ids].tolist())
            self.lengths.extend(self.ep_len[done_ids].tolist())
            self.ep_return[done_ids] = 0.0
            self.ep_len[done_ids] = 0.0
        log = extras.get("log") or extras.get("episode") or {}
        for key, value in log.items():
            if key.startswith(("Curriculum/", "Metrics/", "Episode_Termination/")):
                v = float(value.mean()) if isinstance(value, torch.Tensor) else float(value)
                self.extra[key] = v
                if key == "Curriculum/terrain_levels":
                    self.terrain_level = v

    @property
    def mean_return(self) -> float:
        return sum(self.returns) / len(self.returns) if self.returns else float("nan")

    @property
    def mean_length(self) -> float:
        return sum(self.lengths) / len(self.lengths) if self.lengths else float("nan")


class LiveRunner(OnPolicyRunner):
    def learn_live(self, num_learning_iterations: int, panel=None, init_at_random_ep_len: bool = True) -> str | None:
        """Train like ``OnPolicyRunner.learn`` while serving ``panel`` (may be None for headless runs).

        Returns the path of the last checkpoint written.
        """
        env = self.env
        if init_at_random_ep_len:
            env.episode_length_buf = torch.randint_like(env.episode_length_buf, high=int(env.max_episode_length))
        obs = env.get_observations().to(self.device)
        self.alg.train_mode()
        self.logger.init_logging_writer()
        stats = LiveStats(env.num_envs, self.device)
        steps_per_it = self.cfg["num_steps_per_env"]
        use_rnd = bool(self.cfg["algorithm"].get("rnd_cfg"))
        last_ckpt = None

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        it = start_it
        while it < total_it:
            t0 = time.time()
            paused_s = 0.0
            with torch.inference_mode():
                for _ in range(steps_per_it):
                    if panel is not None:
                        paused_s += panel.on_sim_step(self)
                        if panel.stop_requested:
                            break
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = env.step(actions.to(env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    self.logger.process_env_step(rewards, dones, extras, self.alg.intrinsic_rewards if use_rnd else None)
                    stats.step(rewards, dones, extras)
                else:
                    collect_time = time.time() - t0 - paused_s
                    t1 = time.time()
                    self.alg.compute_returns(obs)
            if panel is not None and panel.stop_requested:
                break  # rollout cut short: drop it, keep the last complete update

            loss_dict = self.alg.update()
            learn_time = time.time() - t1
            self.current_learning_iteration = it
            self.logger.log(
                it=it, start_it=start_it, total_it=total_it, collect_time=collect_time, learn_time=learn_time,
                loss_dict=loss_dict, learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std, rnd_weight=None,
            )
            stats.return_history.append(stats.mean_return)

            save_now = it % self.cfg["save_interval"] == 0 or (panel is not None and panel.consume_save_request())
            if save_now and self.logger.log_dir:
                last_ckpt = os.path.join(self.logger.log_dir, f"model_{it}.pt")
                self.save(last_ckpt)
            if panel is not None:
                fps = steps_per_it * env.num_envs / max(collect_time + learn_time, 1e-6)
                panel.on_iteration(
                    it=it, total_it=total_it, fps=fps, stats=stats, loss_dict=loss_dict,
                    action_std=float(self.alg.get_policy().output_std.mean()), last_ckpt=last_ckpt,
                    collect_time=collect_time, learn_time=learn_time,
                )
            it += 1

        if self.logger.log_dir:
            if it > start_it:  # at least one PPO update happened; otherwise there is nothing new to save
                last_ckpt = os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt")
                self.save(last_ckpt)
            self.logger.stop_logging_writer()
        return last_ckpt
