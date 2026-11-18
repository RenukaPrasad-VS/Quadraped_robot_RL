"""Gazebo SDF version of the Isaac Lab evaluation course (same geometry).

Walked along +x, centred on y = 0, 3 m wide, with a flat border at z = 0
around it (as in Isaac Lab's terrain generator):

    flat | stairs up | landing | stairs down | flat | side slope (+θ) | flat | side slope (−θ) | flat
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass
class Course:
    step_height: float = 0.12
    step_depth: float = 0.30
    num_steps: int = 6
    landing_length: float = 2.0
    start_length: float = 3.0
    slope_deg: float = 8.0
    slope_length: float = 4.0
    width: float = 3.0
    end_length: float = 3.0

    @property
    def spawn_x(self) -> float:
        return self.start_length / 2.0

    def segments(self) -> list[tuple[str, float, float]]:
        """(name, x_start, x_end) in world coordinates."""
        n, d = self.num_steps, self.step_depth
        spec = [
            ("start flat", self.start_length),
            ("stairs up", n * d),
            ("landing", self.landing_length),
            ("stairs down", (n - 1) * d),
            ("flat", 2.0),
            ("side slope A", self.slope_length),
            ("flat", 1.0),
            ("side slope B", self.slope_length),
            ("end flat", self.end_length),
        ]
        out, x = [], 0.0
        for name, length in spec:
            out.append((name, x, x + length))
            x += length
        return out

    @property
    def goal_x(self) -> float:
        """End of the second side slope (same goal as the Isaac Lab course script)."""
        return self.segments()[-1][1]

    @property
    def length(self) -> float:
        return self.segments()[-1][2]

    def segment_at(self, x: float) -> str:
        for name, x0, x1 in self.segments():
            if x < x1:
                return name
        return "finish"


_BOTTOM = -1.0
_COLORS = {"stairs": "0.75 0.55 0.35 1", "slope": "0.45 0.6 0.75 1", "flat": "0.6 0.6 0.6 1", "border": "0.35 0.4 0.35 1"}


def _box(name: str, center, size, rpy=(0.0, 0.0, 0.0), kind: str = "flat", mu: float = 1.0) -> str:
    pose = f"{center[0]:.4f} {center[1]:.4f} {center[2]:.4f} {rpy[0]:.5f} {rpy[1]:.5f} {rpy[2]:.5f}"
    sz = f"{size[0]:.4f} {size[1]:.4f} {size[2]:.4f}"
    color = _COLORS[kind]
    return f"""
      <link name="{name}">
        <pose>{pose}</pose>
        <collision name="c"><geometry><box><size>{sz}</size></box></geometry>
          <surface><friction><ode><mu>{mu}</mu><mu2>{mu}</mu2></ode></friction></surface>
        </collision>
        <visual name="v"><geometry><box><size>{sz}</size></box></geometry>
          <material><ambient>{color}</ambient><diffuse>{color}</diffuse></material>
        </visual>
      </link>"""


def _block(name, x0, x1, y0, y1, top, kind="flat") -> str:
    return _box(name, ((x0 + x1) / 2, (y0 + y1) / 2, (top + _BOTTOM) / 2), (x1 - x0, y1 - y0, top - _BOTTOM), kind=kind)


def course_links(c: Course, y0: float = 0.0, prefix: str = "", border: float = 5.0) -> str:
    """Links for one course lane centred on y = y0, with a flat border of width ``border`` at z = 0."""
    w2 = c.width / 2
    n, h, d = c.num_steps, c.step_height, c.step_depth
    segs = {name: (x0, x1) for name, x0, x1 in c.segments() if name != "flat"}
    P = prefix
    links = []

    def blk(name, x0, x1, ya, yb, top, kind="flat"):
        links.append(_block(P + name, x0, x1, y0 + ya, y0 + yb, top, kind))

    x0, x1 = segs["start flat"]
    blk("start_flat", x0, x1, -w2, w2, 0.0)
    x = segs["stairs up"][0]
    for i in range(n):
        blk(f"step_up_{i}", x, x + d, -w2, w2, (i + 1) * h, "stairs")
        x += d
    x0, x1 = segs["landing"]
    blk("landing", x0, x1, -w2, w2, n * h, "stairs")
    x = segs["stairs down"][0]
    for i in range(n - 1):
        blk(f"step_down_{i}", x, x + d, -w2, w2, (n - 1 - i) * h, "stairs")
        x += d
    flats = [(x0, x1) for name, x0, x1 in c.segments() if name == "flat"]
    for i, (x0, x1) in enumerate(flats):
        blk(f"flat_{i}", x0, x1, -w2, w2, 0.0)
    x0, x1 = segs["end flat"]
    blk("end_flat", x0, x1, -w2, w2, 0.0)

    # Side slopes: a 1 m thick box rolled by ±θ whose top surface is level with the
    # ground on the centreline (z = y·tanθ), matching the Isaac Lab prism.
    half_t = 0.5
    for label, sign in (("A", 1.0), ("B", -1.0)):
        x0, x1 = segs[f"side slope {label}"]
        th = math.radians(sign * c.slope_deg)
        center = ((x0 + x1) / 2, y0 + half_t * math.sin(th), -half_t * math.cos(th))
        links.append(_box(P + f"side_slope_{label}", center, (x1 - x0, c.width / math.cos(th), 2 * half_t),
                          (th, 0, 0), "slope"))

    # Flat border at z = 0 around the course, like Isaac Lab's terrain border.
    b = border
    blk("border_left", -b, c.length + b, w2, w2 + b, 0.0, "border")
    blk("border_right", -b, c.length + b, -w2 - b, -w2, 0.0, "border")
    blk("border_back", -b, 0.0, -w2, w2, 0.0, "border")
    blk("border_front", c.length, c.length + b, -w2, w2, 0.0, "border")
    return "".join(links)


def lane_y(i: int, spacing: float) -> float:
    return i * spacing


def world_sdf(courses, physics_dt: float = 0.001, lane_spacing: float = 6.0, real_time_factor: float = 1.0) -> str:
    """One course (lane 0 at y = 0) or several side-by-side lanes at y = i * lane_spacing."""
    if isinstance(courses, Course):
        links = course_links(courses)
    else:
        border = (lane_spacing - max(c.width for c in courses)) / 2.0
        links = "".join(course_links(c, lane_y(i, lane_spacing), f"lane{i}_", border) for i, c in enumerate(courses))
    return f"""<?xml version="1.0"?>
<sdf version="1.9">
  <world name="course">
    <physics name="default" type="dart">
      <max_step_size>{physics_dt}</max_step_size>
      <real_time_factor>{real_time_factor}</real_time_factor>
    </physics>
    <gravity>0 0 -9.81</gravity>
    <plugin filename="ignition-gazebo-physics-system" name="ignition::gazebo::systems::Physics"/>
    <plugin filename="ignition-gazebo-user-commands-system" name="ignition::gazebo::systems::UserCommands"/>
    <plugin filename="ignition-gazebo-scene-broadcaster-system" name="ignition::gazebo::systems::SceneBroadcaster"/>
    <plugin filename="ignition-gazebo-imu-system" name="ignition::gazebo::systems::Imu"/>
    <scene><ambient>0.6 0.6 0.6 1</ambient><background>0.75 0.82 0.9 1</background><shadows>true</shadows></scene>
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse><specular>0.2 0.2 0.2 1</specular>
      <direction>-0.4 0.3 -0.85</direction>
    </light>
    <model name="course">
      <static>true</static>{links}
    </model>
  </world>
</sdf>
"""


def main():
    p = argparse.ArgumentParser(description="Write the Go2 stairs/side-slope course as a Gazebo world.")
    p.add_argument("output")
    p.add_argument("--step_height", type=float, default=0.12)
    p.add_argument("--slope_deg", type=float, default=8.0)
    a = p.parse_args()
    with open(a.output, "w") as f:
        f.write(world_sdf(Course(step_height=a.step_height, slope_deg=a.slope_deg)))
    print(a.output)


if __name__ == "__main__":
    main()
