"""Live PPO fine-tuning of the Go2 stair policy inside Gazebo (starts from an Isaac Lab checkpoint).

ros2 launch go2_gazebo_rl train.launch.py [init_policy:=model_X.pt] [gui:=true] [rtf:=1.0]
                                          [lane:=-1] [rollout_steps:=2000] [cmd_vx:=0.6]

world:=stairs  the original world: the course at six stair heights (5-15 cm), promote/demote curriculum.
world:=terrain six different terrains (stairs, slopes, blocks, obstacles, surfaces, exam), adaptive sampling.
lane:=N trains on one lane only.
"""

import os
import subprocess
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription, OpaqueFunction,
                            RegisterEventHandler, SetEnvironmentVariable, TimerAction)
from launch.events import Shutdown
from launch.event_handlers import OnProcessExit, OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from go2_gazebo_rl.gazebo_trainer import LANE_SPACING, SPAWN_Z, lane_y
from go2_gazebo_rl.terrain_lanes import WORLDS, lanes_world_sdf
from go2_gazebo_rl.paths import GO2_DESCRIPTION, deploy_cfg_for, newest_checkpoint
from go2_gazebo_rl.policy import SDK_JOINTS, DeployCfg
from go2_gazebo_rl.robot_description import go2_urdf

MODEL, WORLD = "go2", "course"


def setup(context):
    arg = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    init = os.path.expanduser(arg("init_policy")) or newest_checkpoint()
    if not init:
        raise RuntimeError("No Isaac Lab checkpoint found; train Go2-Stairs-Blind-v0 first or pass init_policy:=")
    cfg = DeployCfg(deploy_cfg_for(init))
    lane = int(arg("lane"))
    world = arg("world")
    if world not in WORLDS:
        raise RuntimeError(f"world:={world} unknown; choose one of {sorted(WORLDS)}")
    lanes = WORLDS[world][0]()
    first = max(lane, 0)

    world_file = os.path.join(tempfile.gettempdir(), f"go2_train_{world}_{PARTITION}.sdf")
    with open(world_file, "w") as f:
        f.write(lanes_world_sdf(lanes, spacing=LANE_SPACING, real_time_factor=float(arg("rtf"))))
    urdf = go2_urdf(GO2_DESCRIPTION, kp=float(cfg.kp[0]), kd=float(cfg.kd[0]))

    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")),
        launch_arguments={"gz_args": f"-r {'' if arg('gui').lower() == 'true' else '-s '}{world_file}"}.items(),
    )
    rsp = Node(package="robot_state_publisher", executable="robot_state_publisher",
               parameters=[{"robot_description": urdf, "use_sim_time": True}], output="log")
    spawn = Node(package="ros_gz_sim", executable="create", output="screen",
                 arguments=["-world", WORLD, "-name", MODEL, "-topic", "robot_description",
                            "-x", str(lanes[first].spawn_x), "-y", str(lane_y(first)), "-z", str(SPAWN_Z)])
    bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge", output="log",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            f"/{MODEL}/imu@sensor_msgs/msg/Imu[gz.msgs.IMU",
            f"/model/{MODEL}/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry",
            f"/world/{WORLD}/model/{MODEL}/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model",
            *[f"/{MODEL}/joint/{j}/cmd_pos@std_msgs/msg/Float64]gz.msgs.Double" for j in SDK_JOINTS],
        ],
        remappings=[(f"/world/{WORLD}/model/{MODEL}/joint_state", f"/{MODEL}/joint_states")],
    )
    trainer = Node(
        package="go2_gazebo_rl", executable="gazebo_trainer", output="screen",
        parameters=[{"use_sim_time": True, "init_policy": init, "world": world, "lane": lane,
                     "rollout_steps": int(arg("rollout_steps")), "cmd_vx": float(arg("cmd_vx")),
                     "max_iterations": int(arg("max_iterations")), "episode_s": float(arg("episode_s"))}],
    )
    stop_with_trainer = RegisterEventHandler(OnProcessExit(target_action=trainer,
                                                           on_exit=[EmitEvent(event=Shutdown(reason="trainer exited"))]))
    # The `ign gazebo` wrapper does not always forward shutdown to the server: stop it by its unique world file.
    stop_gz = RegisterEventHandler(OnShutdown(on_shutdown=[OpaqueFunction(
        function=lambda _ctx: subprocess.run(["pkill", "-INT", "-f", os.path.basename(world_file)]) and [])]))
    return [stop_gz, gz, rsp, bridge, TimerAction(period=3.0, actions=[spawn]),
            RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[trainer])), stop_with_trainer]


# Unique Gazebo transport partition per launch: leftover/other Gazebo servers can never cross-talk.
PARTITION = f"go2_stairs_rl_{os.getpid()}"


def generate_launch_description():
    return LaunchDescription([
        SetEnvironmentVariable("IGN_PARTITION", PARTITION),
        SetEnvironmentVariable("GZ_PARTITION", PARTITION),
        DeclareLaunchArgument("init_policy", default_value="", description="Isaac Lab model_*.pt (default: newest blind)"),
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("rtf", default_value="1.0", description="Gazebo real-time factor target"),
        DeclareLaunchArgument("world", default_value="stairs",
                              description="stairs = original stair-height curriculum world; terrain = varied terrain world"),
        DeclareLaunchArgument("lane", default_value="-1",
                              description="-1 = all lanes; else one lane (terrain world: 0 stairs, 1 slopes, 2 blocks, "
                                          "3 obstacles, 4 surfaces, 5 exam; stairs world: 0 = 5 cm ... 5 = 15 cm)"),
        DeclareLaunchArgument("rollout_steps", default_value="2000", description="policy steps per PPO update"),
        DeclareLaunchArgument("cmd_vx", default_value="0.6"),
        DeclareLaunchArgument("max_iterations", default_value="10000"),
        DeclareLaunchArgument("episode_s", default_value="40.0", description="episode time limit (s)"),
        OpaqueFunction(function=setup),
    ])
