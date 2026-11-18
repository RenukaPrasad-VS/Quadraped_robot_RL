"""Run the Isaac Lab-trained Go2 policy in Gazebo.

Inputs (ROS): /go2/imu (sensor_msgs/Imu), /go2/joint_states (sensor_msgs/JointState),
optional /cmd_vel (geometry_msgs/Twist).
Outputs: /go2/joint/<joint>/cmd_pos (std_msgs/Float64) for the 12 Gazebo PD controllers,
/go2/policy_active (std_msgs/Bool).

Phases: WAIT for sensors -> STAND (blend to the training standing pose) ->
SETTLE -> POLICY at the training control rate (step_dt in deploy.yaml, 50 Hz).
Observations are assembled from deploy.yaml exactly as in training:
term order, scales and clipping.
"""

from __future__ import annotations

import math

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Bool, Float64

from .paths import deploy_cfg_for, newest_checkpoint
from .policy import DeployCfg, Policy


def quat_rotate_inverse(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate world vector v into the body frame given body orientation q (x, y, z, w)."""
    x, y, z, w = q_xyzw
    qv = np.array([x, y, z])
    return v * (2.0 * w * w - 1.0) - np.cross(qv, v) * w * 2.0 + qv * np.dot(qv, v) * 2.0


def yaw_of(q_xyzw: np.ndarray) -> float:
    x, y, z, w = q_xyzw
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class Go2PolicyNode(Node):
    def __init__(self):
        super().__init__("go2_policy")
        self.declare_parameter("policy_path", "")
        self.declare_parameter("deploy_cfg", "")
        self.declare_parameter("cmd_vx", 0.5)
        self.declare_parameter("cmd_vy", 0.0)
        self.declare_parameter("hold_heading", True)
        self.declare_parameter("heading_stiffness", 0.5)  # training value
        self.declare_parameter("stand_time", 2.0)
        self.declare_parameter("settle_time", 1.0)
        self.declare_parameter("model_name", "go2")
        self.declare_parameter("stand_only", False)  # hold the standing pose, never run the policy

        path = self.get_parameter("policy_path").value or newest_checkpoint()
        if not path:
            raise RuntimeError("No policy found: train Go2-Stairs-Blind-v0 first or pass policy_path:=...")
        cfg_path = self.get_parameter("deploy_cfg").value or deploy_cfg_for(path)
        if not cfg_path:
            raise RuntimeError(f"No params/deploy.yaml next to {path}; pass deploy_cfg:=...")
        self.cfg = DeployCfg(cfg_path)
        self.policy = Policy(path)
        if self.policy.obs_dim != self.cfg.obs_dim:
            raise RuntimeError(
                f"Policy expects {self.policy.obs_dim} inputs but deploy.yaml describes {self.cfg.obs_dim}. "
                "Only the blind policy (Go2-Stairs-Blind-v0) runs here; the height-scan one needs a terrain map."
            )
        self.get_logger().info(f"policy: {path}")
        self.get_logger().info(f"deploy cfg: {cfg_path}")
        self.get_logger().info(f"obs terms: {[t[0] for t in self.cfg.obs_terms]} ({self.cfg.obs_dim} values)")

        model = self.get_parameter("model_name").value
        self.names = self.cfg.joint_names  # policy (Isaac Lab) joint order
        self.pubs = [self.create_publisher(Float64, f"/{model}/joint/{j}/cmd_pos", 10) for j in self.names]
        self.active_pub = self.create_publisher(Bool, f"/{model}/policy_active", 10)
        self.create_subscription(Imu, f"/{model}/imu", self.on_imu, qos_profile_sensor_data)
        self.create_subscription(JointState, f"/{model}/joint_states", self.on_joints, qos_profile_sensor_data)
        self.create_subscription(Twist, "/cmd_vel", self.on_cmd_vel, 10)

        n = len(self.names)
        self.q = np.zeros(n, dtype=np.float32)
        self.dq = np.zeros(n, dtype=np.float32)
        self.quat = None
        self.gyro = np.zeros(3, dtype=np.float32)
        self.have_joints = False
        self.last_action = np.zeros(n, dtype=np.float32)
        self.cmd = np.array([self.get_parameter("cmd_vx").value, self.get_parameter("cmd_vy").value, 0.0],
                            dtype=np.float32)
        self.hold_heading = self.get_parameter("hold_heading").value
        self.heading_k = self.get_parameter("heading_stiffness").value
        self.heading_target = None
        self.phase, self.phase_t0, self.q_start = "WAIT", None, None
        self.target = None
        self.timer = self.create_timer(self.cfg.step_dt, self.step)
        self.create_timer(2.0, self.status)

    def status(self):
        if self.quat is None or self.target is None:
            return
        g = quat_rotate_inverse(self.quat, np.array([0.0, 0.0, -1.0]))
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -g[2]))))
        err = float(np.abs(self.target - self.q).mean())
        self.get_logger().info(
            f"[{self.phase}] tilt={tilt:4.1f}° gravity_b=({g[0]:+.2f},{g[1]:+.2f},{g[2]:+.2f}) "
            f"|q*-q|={err:.3f} rad cmd=({self.cmd[0]:+.2f},{self.cmd[1]:+.2f},{self.cmd[2]:+.2f})"
        )

    # --- callbacks ------------------------------------------------------------------
    def on_imu(self, msg: Imu):
        o, w = msg.orientation, msg.angular_velocity
        self.quat = np.array([o.x, o.y, o.z, o.w], dtype=np.float64)
        self.gyro = np.array([w.x, w.y, w.z], dtype=np.float32)

    def on_joints(self, msg: JointState):
        index = {name: i for i, name in enumerate(msg.name)}
        try:
            ids = [index[j] for j in self.names]
        except KeyError:
            return
        self.q = np.array([msg.position[i] for i in ids], dtype=np.float32)
        if len(msg.velocity) == len(msg.name):
            self.dq = np.array([msg.velocity[i] for i in ids], dtype=np.float32)
        self.have_joints = True

    def on_cmd_vel(self, msg: Twist):
        self.cmd[0], self.cmd[1] = msg.linear.x, msg.linear.y
        if abs(msg.angular.z) > 0.05 or not self.hold_heading:
            self.cmd[2] = msg.angular.z
            self.heading_target = None  # re-latch heading when turning stops

    # --- observation ---------------------------------------------------------------
    def observation(self) -> np.ndarray:
        if self.hold_heading:
            yaw = yaw_of(self.quat)
            if self.heading_target is None:
                self.heading_target = yaw
            self.cmd[2] = float(np.clip(self.heading_k * wrap(self.heading_target - yaw), -1.0, 1.0))
        values = {
            "base_ang_vel": self.gyro,
            "projected_gravity": quat_rotate_inverse(self.quat, np.array([0.0, 0.0, -1.0])).astype(np.float32),
            "velocity_commands": self.cmd,
            "joint_pos_rel": self.q - self.cfg.default_q,
            "joint_vel_rel": self.dq,
            "last_action": self.last_action,
        }
        parts = []
        for name, scale, clip in self.cfg.obs_terms:
            if name not in values:
                raise RuntimeError(f"observation term '{name}' is not available in Gazebo")
            v = values[name] * scale
            parts.append(np.clip(v, *clip) if clip else v)
        return np.concatenate(parts).astype(np.float32)

    # --- control loop --------------------------------------------------------------
    def publish(self, target: np.ndarray):
        self.target = np.asarray(target, dtype=np.float32)
        for pub, value in zip(self.pubs, target):
            pub.publish(Float64(data=float(value)))

    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def step(self):
        t = self.now()
        if self.phase == "WAIT":
            if self.have_joints and self.quat is not None:
                self.phase, self.phase_t0, self.q_start = "STAND", t, self.q.copy()
                self.get_logger().info("sensors up -> standing")
            return
        if self.phase == "STAND":
            a = min((t - self.phase_t0) / self.get_parameter("stand_time").value, 1.0)
            a = 0.5 - 0.5 * math.cos(math.pi * a)
            self.publish((1 - a) * self.q_start + a * self.cfg.default_q)
            if a >= 1.0:
                self.phase, self.phase_t0 = "SETTLE", t
            return
        if self.phase == "SETTLE":
            self.publish(self.cfg.default_q)
            if t - self.phase_t0 >= self.get_parameter("settle_time").value and not self.get_parameter("stand_only").value:
                self.phase = "POLICY"
                self.get_logger().info(f"policy running: cmd vx={self.cmd[0]:.2f} m/s, hold_heading={self.hold_heading}")
            return

        action = self.policy(self.observation())
        if self.cfg.action_clip is not None:
            action = np.clip(action, self.cfg.action_clip[:, 0], self.cfg.action_clip[:, 1])
        self.last_action = action.astype(np.float32)
        self.publish(self.cfg.action_offset + self.cfg.action_scale * self.last_action)
        self.active_pub.publish(Bool(data=True))


def main():
    rclpy.init()
    node = Go2PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
