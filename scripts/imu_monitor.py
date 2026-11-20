"""Live IMU monitoring for one robot, like ``ros2 topic echo``.

* ``--echo_imu``: print the IMU of robot ``--imu_robot`` in the terminal at ``--imu_hz``,
  laid out like a ``sensor_msgs/msg/Imu`` message, plus roll/pitch/yaw in degrees.
* ``--ros2``: also publish it as real ROS 2 topics (uses the ROS 2 Humble libraries bundled
  with Isaac Sim; the ``.sh`` wrappers set up the environment when ``--ros2`` is passed):

    /go2/imu       sensor_msgs/msg/Imu                 orientation, angular rate, linear acceleration
    /go2/imu/rpy   geometry_msgs/msg/Vector3Stamped    roll, pitch, yaw in degrees (x, y, z)

  then in another terminal: ``ros2 topic echo /go2/imu`` or ``ros2 run rqt_plot rqt_plot /go2/imu/rpy/vector``.
* ``--show_imu``: show it inside the Isaac Sim window (not with ``--headless``): the camera follows the
  robot, arrows are drawn on it and a "Go2 IMU" panel shows live values and scrolling plots.

    green arrow  body up-axis (shows how far the trunk is tilted)
    red arrow    angular velocity (rotation axis; longer = faster)
    blue arrow   linear acceleration with gravity removed (shoves, impacts, stepping)
"""

import math


def add_imu_args(parser):
    g = parser.add_argument_group("IMU monitoring")
    g.add_argument("--echo_imu", action="store_true", help="Print the IMU of one robot in the terminal.")
    g.add_argument("--ros2", action="store_true", help="Publish the IMU as ROS 2 topics /go2/imu and /go2/imu/rpy.")
    g.add_argument("--show_imu", action="store_true", help="Show the IMU in the Isaac Sim window (arrows + plot panel).")
    g.add_argument("--imu_robot", type=int, default=0, help="Index of the robot to monitor.")
    g.add_argument("--imu_hz", type=float, default=10.0, help="Echo / publish rate in Hz (max = policy rate, 50 Hz).")


def quat_to_rpy(w, x, y, z):
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def follow_imu_robot(args, env_cfg):
    """With --show_imu, point the viewport camera at the monitored robot and keep following it."""
    if getattr(args, "show_imu", False) and not args.headless:
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = args.imu_robot
        env_cfg.viewer.eye = (2.0, 2.0, 1.2)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.0)


class ImuMonitor:
    """Call ``update()`` once per env step; it prints / publishes at the requested rate."""

    def __init__(self, args, base_env):
        if args.show_imu and args.headless:
            print("[imu] --show_imu needs the Isaac Sim window; ignored with --headless")
            args.show_imu = False
        self.enabled = args.echo_imu or args.ros2 or args.show_imu
        if not self.enabled:
            return
        self.echo = args.echo_imu
        self.robot = base_env.scene["robot"]
        self.imu = base_env.scene.sensors["imu"]
        self.step_dt = base_env.step_dt
        if not 0 <= args.imu_robot < base_env.num_envs:
            raise ValueError(f"--imu_robot {args.imu_robot} out of range (0..{base_env.num_envs - 1})")
        self.i = args.imu_robot
        self.every = max(1, round(1.0 / (args.imu_hz * self.step_dt)))
        self.count = 0
        self.node = None
        self.view = ImuView(self.i, base_env.device) if args.show_imu else None
        if args.ros2:
            self._start_ros2()

    def _start_ros2(self):
        try:
            import rclpy
            from geometry_msgs.msg import Vector3Stamped
            from sensor_msgs.msg import Imu
        except ImportError as exc:
            raise RuntimeError(
                "rclpy not found. Launch through the .sh wrappers (they set up Isaac Sim's bundled ROS 2 "
                f"libraries when --ros2 is passed). Original error: {exc}"
            ) from exc
        rclpy.init()
        self.rclpy, self.Imu, self.Vector3Stamped = rclpy, Imu, Vector3Stamped
        self.node = rclpy.create_node("go2_imu")
        self.pub_imu = self.node.create_publisher(Imu, "/go2/imu", 10)
        self.pub_rpy = self.node.create_publisher(Vector3Stamped, "/go2/imu/rpy", 10)
        print("[imu] publishing ROS 2 topics /go2/imu (sensor_msgs/Imu) and /go2/imu/rpy (degrees)")

    def update(self, step: int):
        if not self.enabled:
            return
        self.count += 1
        if self.count % self.every:
            return
        d = self.imu.data
        q = d.quat_w[self.i].tolist()  # (w, x, y, z)
        av = d.ang_vel_b[self.i].tolist()
        la = d.lin_acc_b[self.i].tolist()  # includes gravity, like a real accelerometer
        rpy = [math.degrees(a) for a in quat_to_rpy(*q)]
        t = step * self.step_dt
        if self.echo:
            self._print(t, q, av, la, rpy)
        if self.node is not None:
            self._publish(t, q, av, la, rpy)
        if self.view is not None:
            self.view.update(self.robot.data.root_pos_w[self.i], d.quat_w[self.i], d.ang_vel_b[self.i],
                             d.lin_acc_b[self.i], av, la, rpy)

    def _print(self, t, q, av, la, rpy):
        sec = int(t)
        print(
            "header:\n"
            f"  stamp:\n    sec: {sec}\n    nanosec: {int(round((t - sec) * 1e9))}\n"
            f"  frame_id: robot_{self.i}/base\n"
            "orientation:\n"
            f"  x: {q[1]:.6f}\n  y: {q[2]:.6f}\n  z: {q[3]:.6f}\n  w: {q[0]:.6f}\n"
            "angular_velocity:   # rad/s, body frame\n"
            f"  x: {av[0]:.6f}\n  y: {av[1]:.6f}\n  z: {av[2]:.6f}\n"
            "linear_acceleration:   # m/s^2, body frame, includes gravity\n"
            f"  x: {la[0]:.6f}\n  y: {la[1]:.6f}\n  z: {la[2]:.6f}\n"
            f"rpy_deg:\n  roll: {rpy[0]:.2f}\n  pitch: {rpy[1]:.2f}\n  yaw: {rpy[2]:.2f}\n"
            "---",
            flush=True,
        )

    def _publish(self, t, q, av, la, rpy):
        stamp = self.node.get_clock().now().to_msg()
        frame = f"robot_{self.i}/base"
        m = self.Imu()
        m.header.stamp, m.header.frame_id = stamp, frame
        m.orientation.w, m.orientation.x, m.orientation.y, m.orientation.z = q
        m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = av
        m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = la
        self.pub_imu.publish(m)
        r = self.Vector3Stamped()
        r.header.stamp, r.header.frame_id = stamp, frame
        r.vector.x, r.vector.y, r.vector.z = rpy
        self.pub_rpy.publish(r)

    def close(self):
        if getattr(self, "view", None) is not None:
            self.view.close()
            self.view = None
        if getattr(self, "node", None) is not None:
            self.node.destroy_node()
            self.rclpy.shutdown()
            self.node = None


class ImuView:
    """Arrows on the robot + an omni.ui panel with live values and scrolling plots."""

    HISTORY = 200  # samples shown in the plots

    def __init__(self, robot_index, device):
        import omni.ui as ui
        import torch
        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG

        self.torch, self.device = torch, device
        self.arrows = {
            "up": VisualizationMarkers(GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/imu/up")),
            "gyro": VisualizationMarkers(RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/imu/gyro")),
            "acc": VisualizationMarkers(BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/imu/acc")),
        }
        self.gravity = torch.tensor([0.0, 0.0, 9.81], device=device)
        self.hist = {k: [0.0] * self.HISTORY for k in ("roll", "pitch", "gyro")}

        self.window = ui.Window(f"Go2 IMU (robot {robot_index})", width=420, height=520)
        with self.window.frame:
            with ui.VStack(spacing=4):
                self.lbl_rpy = ui.Label("", height=20)
                self.lbl_gyro = ui.Label("", height=20)
                self.lbl_acc = ui.Label("", height=20)
                ui.Label("arrows: green = body up-axis, red = angular velocity, blue = acceleration (no gravity)",
                         height=20, word_wrap=True, style={"color": 0xFF999999})
                self.plots = {}
                for key, title, lo, hi, color in [
                    ("roll", "roll [deg]  (-30 .. 30)", -30.0, 30.0, 0xFF4040FF),
                    ("pitch", "pitch [deg]  (-40 .. 40)", -40.0, 40.0, 0xFF40FF40),
                    ("gyro", "roll/pitch rate |w_xy| [rad/s]  (0 .. 5)", 0.0, 5.0, 0xFFFF8040),
                ]:
                    ui.Label(title, height=18)
                    self.plots[key] = ui.Plot(ui.Type.LINE, lo, hi, *self.hist[key], height=100,
                                              style={"color": color, "background_color": 0xFF202020})

    @staticmethod
    def _arrow_quat(torch, v):
        """Quaternion (w, x, y, z) turning the +x arrow onto world vector v."""
        yaw = torch.atan2(v[1], v[0])
        pitch = -torch.atan2(v[2], torch.hypot(v[0], v[1]))
        cy, sy, cp, sp = torch.cos(yaw / 2), torch.sin(yaw / 2), torch.cos(pitch / 2), torch.sin(pitch / 2)
        return torch.stack([cy * cp, -sy * sp, cy * sp, sy * cp])

    def _draw(self, name, origin, vec, length):
        torch = self.torch
        q = self._arrow_quat(torch, vec)
        self.arrows[name].visualize(
            translations=origin.unsqueeze(0),
            orientations=q.unsqueeze(0),
            scales=torch.tensor([[max(length, 1e-3), 0.1, 0.1]], device=self.device),
        )

    def update(self, base_pos, quat, ang_vel_b, lin_acc_b, av, la, rpy):
        from isaaclab.utils.math import quat_apply

        torch = self.torch
        origin = base_pos + torch.tensor([0.0, 0.0, 0.35], device=self.device)
        q = quat.unsqueeze(0)
        up_w = quat_apply(q, torch.tensor([[0.0, 0.0, 1.0]], device=self.device))[0]
        gyro_w = quat_apply(q, ang_vel_b.unsqueeze(0))[0]
        acc_w = quat_apply(q, lin_acc_b.unsqueeze(0))[0] - self.gravity
        self._draw("up", origin, up_w, 0.5)
        self._draw("gyro", origin, gyro_w, min(0.15 * float(torch.linalg.norm(gyro_w)), 1.0))
        self._draw("acc", origin, acc_w, min(0.05 * float(torch.linalg.norm(acc_w)), 1.0))

        self.lbl_rpy.text = f"roll {rpy[0]:6.1f}   pitch {rpy[1]:6.1f}   yaw {rpy[2]:7.1f}   deg"
        self.lbl_gyro.text = f"ang vel  x {av[0]:6.2f}  y {av[1]:6.2f}  z {av[2]:6.2f}   rad/s"
        self.lbl_acc.text = f"lin acc  x {la[0]:6.2f}  y {la[1]:6.2f}  z {la[2]:6.2f}   m/s^2"
        for key, val in (("roll", rpy[0]), ("pitch", rpy[1]), ("gyro", math.hypot(av[0], av[1]))):
            h = self.hist[key]
            h.pop(0)
            h.append(val)
            self.plots[key].set_data(*h)

    def close(self):
        self.window.visible = False
