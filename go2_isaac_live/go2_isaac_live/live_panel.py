"""'Go2 Live Training' panel inside the Isaac Sim window, plus keyboard shortcuts.

Keys (click the viewport first so it has keyboard focus):
    P  pause / resume training (the viewport keeps running while paused)
    K  save a checkpoint after the current PPO update
    N  follow next robot        B  follow previous robot
    O  overview camera (whole terrain)
    T  turbo: hide the viewport so training runs at full speed (press again to watch)
    R  real-time: slow the simulation to wall-clock speed so the gait looks natural

Do NOT use Isaac Sim's own timeline Stop button: it destroys the physics simulation and
training cannot continue. If it happens anyway, the panel saves a checkpoint and exits
cleanly so you can resume with --resume.
"""

from __future__ import annotations

import math
import os
import sys
import time

import carb
import omni.appwindow
import omni.timeline
import omni.ui as ui

KEY_HELP = "P pause | K save | N/B next/prev robot | O overview | T turbo | R real-time"


def _fmt(v: float, nd: int = 2) -> str:
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


class LivePanel:
    def __init__(self, env, task_name: str, log_dir: str):
        """``env`` is the rsl-rl wrapped env; ``env.unwrapped`` is the Isaac Lab ManagerBasedRLEnv."""
        self.env = env
        self.base = env.unwrapped
        self.sim = self.base.sim
        self.cam = getattr(self.base, "viewport_camera_controller", None)
        self.num_envs = self.base.num_envs

        self.paused = False
        self.stop_requested = False
        self.realtime = False
        self.turbo = False
        self._save_requested = False
        self.follow_idx = 0
        self._last_step_wall = time.time()
        self.runner = None
        self.log_dir = log_dir

        self._build_window(task_name, log_dir)
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._key_sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_key)
        self._timeline_sub = (omni.timeline.get_timeline_interface().get_timeline_event_stream()
                              .create_subscription_to_pop_by_type(int(omni.timeline.TimelineEventType.STOP),
                                                                  self._on_timeline_stop))
        self.follow(0)

    def attach_runner(self, runner):
        """Lets the panel save a checkpoint if the timeline is stopped."""
        self.runner = runner

    def _on_timeline_stop(self, _event):
        """Isaac Sim's timeline Stop destroys the PhysX views; Isaac Lab would then hang forever.

        Save what we have and exit so training can be resumed with --resume.
        """
        msg = "[go2_isaac_live] Isaac Sim timeline was STOPPED (toolbar stop button). Physics cannot continue."
        print(msg, flush=True)
        try:
            if self.runner is not None and self.runner.logger.log_dir:
                it = self.runner.current_learning_iteration
                path = os.path.join(self.runner.logger.log_dir, f"model_{it}.pt")
                self.runner.save(path)
                if self.runner.logger.writer is not None:
                    self.runner.logger.writer.flush()
                print(f"[go2_isaac_live] saved {path}\n[go2_isaac_live] resume with: ./run.sh train ... --resume {path}",
                      flush=True)
        except Exception as exc:  # never hang here
            print(f"[go2_isaac_live] could not save checkpoint: {exc}", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    # ------------------------------------------------------------------ UI
    def _build_window(self, task_name: str, log_dir: str):
        self.window = ui.Window("Go2 Live Training", width=380, height=620,
                                dock_preference=ui.DockPreference.RIGHT_TOP)
        self.labels: dict[str, ui.Label] = {}
        rows = ["status", "iteration", "robots", "speed", "return", "ep_len", "terrain", "std", "loss", "camera", "ckpt"]
        with self.window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=4, height=0):
                    ui.Label(f"Task: {task_name}", style={"font_size": 18})
                    ui.Label(f"Logs: {log_dir}", word_wrap=True, style={"color": 0xFF9A9A9A})
                    ui.Spacer(height=4)
                    for r in rows:
                        self.labels[r] = ui.Label("", word_wrap=True)
                    ui.Spacer(height=4)
                    ui.Label("Mean episode return per PPO iteration")
                    self.plot = ui.Plot(ui.Type.LINE, 0.0, 1.0, 0.0, 0.0, height=120,
                                        style={"color": 0xFF00C8FF, "background_color": 0xFF202020})
                    ui.Spacer(height=6)
                    with ui.HStack(height=28, spacing=4):
                        self.btn_pause = ui.Button("Pause (P)", clicked_fn=self.toggle_pause)
                        ui.Button("Save now (K)", clicked_fn=self.request_save)
                    with ui.HStack(height=28, spacing=4):
                        ui.Button("< Prev (B)", clicked_fn=lambda: self.follow(self.follow_idx - 1))
                        ui.Button("Next robot (N) >", clicked_fn=lambda: self.follow(self.follow_idx + 1))
                        ui.Button("Overview (O)", clicked_fn=self.overview)
                    with ui.HStack(height=28, spacing=4):
                        ui.Label("Follow robot #", width=100)
                        self.idx_model = ui.SimpleIntModel(0)
                        ui.IntField(model=self.idx_model, width=70)
                        ui.Button("Go", width=50, clicked_fn=lambda: self.follow(self.idx_model.get_value_as_int()))
                    with ui.HStack(height=28, spacing=4):
                        self.btn_turbo = ui.Button("Turbo: off (T)", clicked_fn=self.toggle_turbo)
                        self.btn_rt = ui.Button("Real-time: off (R)", clicked_fn=self.toggle_realtime)
                    ui.Spacer(height=6)
                    ui.Button("Stop training & save", height=30, clicked_fn=self.request_stop,
                              style={"background_color": 0xFF3030A0})
                    ui.Spacer(height=4)
                    ui.Label(KEY_HELP, word_wrap=True, style={"color": 0xFF9A9A9A})
        self.labels["status"].text = "Status: starting..."
        self.labels["robots"].text = f"Robots: {self.num_envs}"

    def _on_key(self, event, *args, **kwargs):
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True
        key = event.input.name
        actions = {
            "P": self.toggle_pause,
            "K": self.request_save,
            "N": lambda: self.follow(self.follow_idx + 1),
            "B": lambda: self.follow(self.follow_idx - 1),
            "O": self.overview,
            "T": self.toggle_turbo,
            "R": self.toggle_realtime,
        }
        if key in actions:
            actions[key]()
        return True

    # ------------------------------------------------------------- actions
    def toggle_pause(self):
        self.paused = not self.paused
        self.btn_pause.text = "Resume (P)" if self.paused else "Pause (P)"
        self.labels["status"].text = "Status: PAUSED" if self.paused else "Status: training"

    def request_save(self):
        self._save_requested = True
        self.labels["ckpt"].text = "Checkpoint: will save after this PPO update..."

    def consume_save_request(self) -> bool:
        requested, self._save_requested = self._save_requested, False
        return requested

    def request_stop(self):
        self.stop_requested = True
        self.paused = False
        self.labels["status"].text = "Status: stopping, saving final checkpoint..."

    def follow(self, idx: int):
        self.follow_idx = idx % self.num_envs
        self.idx_model.set_value(self.follow_idx)
        if self.cam is not None:
            self.cam.set_view_env_index(self.follow_idx)
            self.cam.update_view_to_asset_root("robot")
        self.labels["camera"].text = f"Camera: following robot #{self.follow_idx}"

    def overview(self):
        if self.cam is None:
            return
        self.cam.update_view_to_world()
        terrain = self.base.scene.terrain
        origins = getattr(terrain, "terrain_origins", None)
        if origins is not None:
            flat = origins.reshape(-1, 3)
            center, span = flat.mean(0).tolist(), float((flat.max(0).values - flat.min(0).values).max())
        else:
            env_o = self.base.scene.env_origins
            center, span = env_o.mean(0).tolist(), float((env_o.max(0).values - env_o.min(0).values).max())
        d = max(span, 10.0)
        self.cam.update_view_location(eye=(center[0] - 0.8 * d, center[1] - 0.8 * d, 0.7 * d), lookat=tuple(center))
        self.labels["camera"].text = "Camera: overview"

    def toggle_turbo(self):
        self.turbo = not self.turbo
        mode = self.sim.RenderMode.NO_RENDERING if self.turbo else self.sim.RenderMode.FULL_RENDERING
        self.sim.set_render_mode(mode)
        self.btn_turbo.text = f"Turbo: {'ON' if self.turbo else 'off'} (T)"

    def toggle_realtime(self):
        self.realtime = not self.realtime
        self.btn_rt.text = f"Real-time: {'ON' if self.realtime else 'off'} (R)"

    # ---------------------------------------------------------- runner hooks
    def on_sim_step(self, runner) -> float:
        """Called before every env step. Blocks while paused. Returns the seconds spent paused/throttled."""
        waited = 0.0
        if self.realtime:
            dt = self.base.step_dt - (time.time() - self._last_step_wall)
            if dt > 0:
                time.sleep(dt)
                waited += dt
        if self.paused:
            t0 = time.time()
            while self.paused and not self.stop_requested:
                self.sim.render()  # keep viewport and UI alive; physics does not advance
                time.sleep(0.01)
            waited += time.time() - t0
        self._last_step_wall = time.time()
        return waited

    def on_iteration(self, *, it, total_it, fps, stats, loss_dict, action_std, last_ckpt, collect_time, learn_time):
        L = self.labels
        if not self.paused and not self.stop_requested:
            L["status"].text = "Status: training"
        L["iteration"].text = f"Iteration: {it} / {total_it}   ({collect_time:.2f}s sim + {learn_time:.2f}s PPO)"
        L["speed"].text = f"Speed: {fps:,.0f} steps/s"
        L["return"].text = f"Mean episode return: {_fmt(stats.mean_return)}"
        L["ep_len"].text = f"Mean episode length: {_fmt(stats.mean_length, 0)} steps (of {self.base.max_episode_length})"
        L["terrain"].text = f"Mean terrain level: {_fmt(stats.terrain_level)}"
        L["std"].text = f"Action noise std: {_fmt(action_std, 3)}"
        loss_txt = ", ".join(f"{k}={_fmt(float(v), 3)}" for k, v in list(loss_dict.items())[:3])
        L["loss"].text = f"Losses: {loss_txt}"
        if last_ckpt:
            L["ckpt"].text = f"Last checkpoint: {last_ckpt}"
        hist = [v for v in stats.return_history if not math.isnan(v)]
        if len(hist) >= 2:
            try:
                lo, hi = min(hist), max(hist)
                self.plot.scale_min, self.plot.scale_max = lo, hi if hi > lo else lo + 1.0
                self.plot.set_data(*hist)
            except Exception:  # plotting is cosmetic
                pass

    def close(self):
        self._timeline_sub = None  # normal shutdown stops the timeline too; don't treat that as an accident
        # stop the follow-camera callback, otherwise it fires after the scene is destroyed on env.close()
        if self.cam is not None:
            self.cam.update_view_to_world()
        if self._key_sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._key_sub)
            self._key_sub = None
        if self.window is not None:
            self.window.visible = False
            self.window = None
