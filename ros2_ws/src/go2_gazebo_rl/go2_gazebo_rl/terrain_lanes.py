"""Training worlds for live training in Gazebo. Two worlds are available:

world "stairs"  (stair_curriculum_lanes): six copies of the evaluation course with stair height
                5, 7, 9, 11, 13, 15 cm and side slopes 5-10°; lanes are curriculum levels.
world "terrain" (training_lanes): six different terrain types, each harder along its length:

  0 stairs     stair sets of 6, 10, 14 cm
  1 slopes     up/down ramps (6°, 11°), each followed by a side slope of the same angle
  2 blocks     random stepping-block fields, 0-4, 0-8, 0-12 cm
  3 obstacles  hurdles 5 cm, debris field, hurdles 9 cm, trenches 15 and 20 cm
  4 surfaces   ice (mu 0.2), normal, wet (0.4), rubber (1.2), ice, then a slippery 6° side slope
  5 exam       the evaluation course: 6 x 12 cm stairs up/down + 8° side slopes both ways

Lanes are 3 m wide, centred on y = i * spacing, with a flat z = 0 border between them,
and a catch floor at z = -1 under the whole world (trenches, falling off an edge).
Geometry is deterministic (fixed random seed), so every run sees the same world.
"""

from __future__ import annotations

import math

import numpy as np

_BOTTOM = -1.0
_HALF_T = 0.5  # half thickness of tilted ramp / slope boxes

COLORS = {
    "flat": "0.6 0.6 0.6 1", "border": "0.35 0.4 0.35 1", "stairs": "0.75 0.55 0.35 1",
    "slope": "0.45 0.6 0.75 1", "blocks": "0.62 0.5 0.4 1", "obstacle": "0.8 0.3 0.25 1",
    "ice": "0.75 0.9 1.0 1", "wet": "0.3 0.35 0.45 1", "rubber": "0.12 0.12 0.12 1",
}
MU = {"ice": 0.2, "wet": 0.4, "rubber": 1.2}


def _box_sdf(name, center, size, rpy=(0.0, 0.0, 0.0), kind="flat", mu=None) -> str:
    mu = MU.get(kind, 1.0) if mu is None else mu
    pose = " ".join(f"{v:.4f}" for v in (*center, *rpy))
    sz = " ".join(f"{v:.4f}" for v in size)
    c = COLORS[kind]
    return f"""
      <link name="{name}">
        <pose>{pose}</pose>
        <collision name="c"><geometry><box><size>{sz}</size></box></geometry>
          <surface><friction><ode><mu>{mu}</mu><mu2>{mu}</mu2></ode></friction></surface>
        </collision>
        <visual name="v"><geometry><box><size>{sz}</size></box></geometry>
          <material><ambient>{c}</ambient><diffuse>{c}</diffuse></material>
        </visual>
      </link>"""


class Lane:
    """Primitives in lane-local coordinates (x along the lane from 0, y across, centred on 0)."""

    def __init__(self, name: str, width: float = 3.0, start_length: float = 3.0):
        self.name, self.width = name, width
        self.prims = []  # (center, size, rpy, kind, mu)
        self.segs = []  # (name, x0, x1)
        self.x = 0.0
        self.flat(start_length, "start flat")
        self.spawn_x = start_length / 2.0

    # --- primitives ---------------------------------------------------------------
    def block(self, x0, x1, top, kind="flat", ya=None, yb=None, mu=None):
        ya = -self.width / 2 if ya is None else ya
        yb = self.width / 2 if yb is None else yb
        self.prims.append((((x0 + x1) / 2, (ya + yb) / 2, (top + _BOTTOM) / 2),
                           (x1 - x0, yb - ya, top - _BOTTOM), (0.0, 0.0, 0.0), kind, mu))

    def box_on_floor(self, cx, cy, sx, sy, h, yaw=0.0, kind="obstacle"):
        self.prims.append(((cx, cy, h / 2), (sx, sy, h), (0.0, 0.0, yaw), kind, None))

    def ramp(self, x0, z0, x1, z1, kind="slope", mu=None):
        """Box tilted about y whose top surface runs from (x0, z0) to (x1, z1)."""
        th = math.atan2(z1 - z0, x1 - x0)
        xm, zm = (x0 + x1) / 2, (z0 + z1) / 2
        center = (xm + _HALF_T * math.sin(th), 0.0, zm - _HALF_T * math.cos(th))
        self.prims.append((center, (math.hypot(x1 - x0, z1 - z0), self.width, 2 * _HALF_T), (0.0, -th, 0.0), kind, mu))

    def side_slope_prim(self, x0, x1, deg, kind="slope", mu=None):
        """Box tilted about x: level with the floor on the centreline, z = y * tan(deg)."""
        th = math.radians(deg)
        center = ((x0 + x1) / 2, _HALF_T * math.sin(th), -_HALF_T * math.cos(th))
        self.prims.append((center, (x1 - x0, self.width / math.cos(th), 2 * _HALF_T), (th, 0.0, 0.0), kind, mu))

    # --- segments -----------------------------------------------------------------
    def seg(self, name, length, build):
        x0 = self.x
        build(x0)
        self.x += length
        self.segs.append((name, x0, self.x))

    def flat(self, length, name="flat", kind="flat", mu=None):
        self.seg(name, length, lambda x0: self.block(x0, x0 + length, 0.0, kind, mu=mu))

    def stairs(self, h, n_up=4, n_down=3, d=0.30, landing=1.5):
        def build(x0):
            x = x0
            for i in range(n_up):
                self.block(x, x + d, (i + 1) * h, "stairs")
                x += d
            self.block(x, x + landing, n_up * h, "stairs")
            x += landing
            for i in range(n_down):
                self.block(x, x + d, (n_down - i) * h * n_up / (n_down + 1), "stairs")
                x += d
        self.seg(f"stairs {h * 100:.0f} cm", n_up * d + landing + n_down * d, build)

    def ramp_pair(self, deg, run=2.0, top=0.8):
        h = run * math.tan(math.radians(deg))

        def build(x0):
            self.ramp(x0, 0.0, x0 + run, h)
            self.block(x0 + run, x0 + run + top, h, "slope")
            self.ramp(x0 + run + top, h, x0 + 2 * run + top, 0.0)
        self.seg(f"ramp {deg:.0f}°", 2 * run + top, build)

    def side_slope(self, deg, length, kind="slope", mu=None, name=None):
        self.seg(name or f"side slope {deg:+.0f}°", length,
                 lambda x0: self.side_slope_prim(x0, x0 + length, deg, kind, mu))

    def block_field(self, hmax, length=4.0, cell=0.5, seed=0):
        rng = np.random.RandomState(seed)

        def build(x0):
            ny = int(round(self.width / cell))
            for ix in range(int(round(length / cell))):
                for iy in range(ny):
                    ya = -self.width / 2 + iy * cell
                    self.block(x0 + ix * cell, x0 + (ix + 1) * cell, float(rng.uniform(0.0, hmax)), "blocks",
                               ya=ya, yb=ya + cell)
        self.seg(f"blocks 0-{hmax * 100:.0f} cm", length, build)

    def hurdles(self, h, count=3, spacing=1.2, depth=0.08):
        length = count * spacing

        def build(x0):
            self.block(x0, x0 + length, 0.0)
            for i in range(count):
                self.box_on_floor(x0 + (i + 0.5) * spacing, 0.0, depth, self.width, h)
        self.seg(f"hurdles {h * 100:.0f} cm", length, build)

    def debris(self, length=4.0, count=12, seed=1):
        rng = np.random.RandomState(seed)

        def build(x0):
            self.block(x0, x0 + length, 0.0)
            for _ in range(count):
                s = rng.uniform(0.2, 0.35)
                self.box_on_floor(x0 + rng.uniform(0.3, length - 0.3), rng.uniform(-1.2, 1.2), s, s,
                                  rng.uniform(0.05, 0.10), yaw=rng.uniform(0, math.pi))
        self.seg("debris", length, build)

    def trenches(self, widths=(0.15, 0.20), gap_floor=1.0):
        length = gap_floor * (len(widths) + 1) + sum(widths)

        def build(x0):
            x = x0
            for w in widths:
                self.block(x, x + gap_floor, 0.0)
                x += gap_floor + w  # leave the trench open (catch floor 1 m below)
            self.block(x, x + gap_floor, 0.0)
        self.seg("trenches " + "/".join(f"{w * 100:.0f}" for w in widths) + " cm", length, build)

    def finish(self, end_length=3.0):
        self.goal_x = self.x
        self.flat(end_length, "end flat")
        self.length = self.x

    # --- queries used by the trainer ---------------------------------------------------
    def segment_at(self, x: float) -> str:
        for name, x0, x1 in self.segs:
            if x < x1:
                return name
        return "finish"

    def links(self, y0: float, prefix: str, border: float) -> str:
        out = [_box_sdf(f"{prefix}p{i}", (c[0], c[1] + y0, c[2]), s, r, k, mu)
               for i, (c, s, r, k, mu) in enumerate(self.prims)]
        w2, b, L = self.width / 2, border, self.length
        for name, (x0, x1, ya, yb) in {"border_l": (-b, L + b, w2, w2 + b), "border_r": (-b, L + b, -w2 - b, -w2),
                                       "border_back": (-b, 0.0, -w2, w2), "border_front": (L, L + b, -w2, w2)}.items():
            out.append(_box_sdf(f"{prefix}{name}", ((x0 + x1) / 2, y0 + (ya + yb) / 2, (0.0 + _BOTTOM) / 2),
                                (x1 - x0, yb - ya, -_BOTTOM), kind="border"))
        return "".join(out)


def training_lanes() -> list[Lane]:
    lanes = []

    s = Lane("stairs")
    for h in (0.06, 0.10, 0.14):
        s.stairs(h)
        s.flat(1.0)
    s.finish()
    lanes.append(s)

    r = Lane("slopes")
    for deg, side in ((6.0, 1.0), (11.0, -1.0)):
        r.ramp_pair(deg)
        r.flat(0.4)
        r.side_slope(side * deg, 2.5)
        r.flat(0.6)
    r.finish()
    lanes.append(r)

    b = Lane("blocks")
    for i, hmax in enumerate((0.04, 0.08, 0.12)):
        b.block_field(hmax, seed=10 + i)
        b.flat(1.0)
    b.finish()
    lanes.append(b)

    o = Lane("obstacles")
    o.hurdles(0.05)
    o.flat(1.0)
    o.debris()
    o.flat(1.0)
    o.hurdles(0.09)
    o.flat(1.0)
    o.trenches()
    o.finish()
    lanes.append(o)

    f = Lane("surfaces")
    for kind in ("ice", "flat", "wet", "rubber", "ice"):
        f.flat(2.0, name=f"{kind} patch" if kind != "flat" else "normal patch", kind=kind)
    f.flat(0.5)
    f.side_slope(6.0, 3.0, kind="wet", name="wet side slope +6°")
    f.finish()
    lanes.append(f)

    e = Lane("exam")
    e.stairs(0.12, n_up=6, n_down=5, landing=2.0)
    e.flat(2.0)
    e.side_slope(8.0, 4.0)
    e.flat(1.0)
    e.side_slope(-8.0, 4.0)
    e.finish()
    lanes.append(e)
    return lanes


STAIR_LEVELS = [(0.05, 5.0), (0.07, 6.0), (0.09, 7.0), (0.11, 8.0), (0.13, 9.0), (0.15, 10.0)]


def stair_curriculum_lanes() -> list[Lane]:
    """The original training world: the evaluation course at six difficulty levels."""
    lanes = []
    for h, deg in STAIR_LEVELS:
        e = Lane(f"stairs {h * 100:.0f}cm/{deg:.0f}°")
        e.stairs(h, n_up=6, n_down=5, landing=2.0)
        e.flat(2.0)
        e.side_slope(deg, 4.0)
        e.flat(1.0)
        e.side_slope(-deg, 4.0)
        e.finish()
        lanes.append(e)
    return lanes


# name -> (lane builder, curriculum mode)
WORLDS = {"stairs": (stair_curriculum_lanes, "levels"), "terrain": (training_lanes, "adaptive")}


def lanes_world_sdf(lanes: list[Lane], spacing: float = 6.0, physics_dt: float = 0.001,
                    real_time_factor: float = 1.0) -> str:
    border = (spacing - max(l.width for l in lanes)) / 2.0
    links = "".join(l.links(i * spacing, f"lane{i}_", border) for i, l in enumerate(lanes))
    L = max(l.length for l in lanes)
    span = (len(lanes) - 1) * spacing
    links += _box_sdf("catch_floor", (L / 2, span / 2, _BOTTOM - 0.5), (L + 40, span + 40, 1.0), kind="border")
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


if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "terrain"
    lanes = WORLDS[name][0]()
    for i, l in enumerate(lanes):
        print(f"lane {i} {l.name:10s} length {l.length:5.1f} m goal {l.goal_x - l.spawn_x:5.1f} m from spawn, "
              f"{len(l.prims)} parts: " + ", ".join(n for n, _, _ in l.segs))
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w") as fh:
            fh.write(lanes_world_sdf(lanes))
