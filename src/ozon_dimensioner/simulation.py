"""Idealized line scanning by ray/mesh-section intersection.

The model includes first-surface occlusion in the laser plane, discrete scan
positions and finite angular samples. Camera/laser triangulation, reflectance,
refraction, belt mechanics and proprietary sensor optics require bench data.
"""

from dataclasses import dataclass
from itertools import product
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


@dataclass
class Mesh:
    vertices: np.ndarray
    triangles: np.ndarray
    name: str
    reference_dimensions: np.ndarray | None = None

    def placed(self, angles=(0.0, 0.0, 0.0), x=0.0):
        r = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
        v = self.vertices @ r.T
        v[:, 0] += x
        v[:, 2] -= v[:, 2].min()
        return Mesh(v, self.triangles, self.name, self.reference_dimensions)


def cuboid(dimensions=(200.0, 100.0, 10.0)):
    d = np.asarray(dimensions, float)
    v = np.array(list(product([-0.5, 0.5], repeat=3))) * d
    return Mesh(v, ConvexHull(v).simplices, "cuboid", np.sort(d)[::-1])


def wedge():
    v = np.array(
        [
            [x, y, z]
            for y in [-60.0, 60.0]
            for x, z in [(0.0, 0.0), (180.0, 0.0), (0.0, 90.0)]
        ]
    )
    v -= v.mean(axis=0)
    return Mesh(v, ConvexHull(v).simplices, "wedge")


def cylinder(radius=40.0, height=130.0, sides=48):
    a = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    v = np.array(
        [[radius * np.cos(t), radius * np.sin(t), z] for z in [0.0, height] for t in a]
    )
    return Mesh(
        v,
        ConvexHull(v).simplices,
        "cylinder",
        np.array([height, 2 * radius, 2 * radius]),
    )


def l_shape():
    # Concave x-z outline, extruded along y. Caps tessellated into two rectangles.
    outline = np.array(
        [
            [0.0, 0.0],
            [160.0, 0.0],
            [160.0, 40.0],
            [50.0, 40.0],
            [50.0, 120.0],
            [0.0, 120.0],
        ]
    )
    v = np.array([[x, y, z] for y in [-50.0, 50.0] for x, z in outline])
    f = []
    for j in range(6):
        k = (j + 1) % 6
        f.extend([[j, k, k + 6], [j, k + 6, j + 6]])
    for offset in [0, 6]:
        f.extend(
            [
                [offset + i for i in t]
                for t in [(0, 1, 3), (1, 2, 3), (0, 3, 5), (3, 4, 5)]
            ]
        )
    v[:, 0] -= 80
    return Mesh(v, np.array(f), "l_shape")


@dataclass(frozen=True)
class Sensor:
    name: str
    x: float
    z: float
    target_x: float = 0.0
    target_z: float = 200.0
    plane_y: float = 0.0
    samples: int = 1920


SENSORS = (
    Sensor("top", 0.0, 1500.0, plane_y=0.0),
    Sensor("left", -1100.0, 700.0, plane_y=-100.0),
    Sensor("right", 1100.0, 700.0, plane_y=100.0),
)


def cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def section_segments(mesh, y):
    segments = []
    for tri in mesh.vertices[mesh.triangles]:
        hits = []
        for j in range(3):
            a, b = tri[j], tri[(j + 1) % 3]
            if (a[1] <= y < b[1]) or (b[1] <= y < a[1]):
                t = (y - a[1]) / (b[1] - a[1])
                hits.append((a + t * (b - a))[[0, 2]])
        if len(hits) == 2 and np.linalg.norm(hits[0] - hits[1]) > 1e-8:
            segments.append(hits)
    return np.asarray(segments)


def scan_section(segments, sensor):
    if len(segments) == 0:
        return np.empty((0, 2))
    origin = np.array([sensor.x, sensor.z])
    d = np.array([sensor.target_x, sensor.target_z]) - origin
    center = np.arctan2(d[1], d[0])
    # Conservative cone inside the published 390..2000 mm trapezoid.
    half = np.arctan(1000.0 / 1875.0)
    angles = center + np.linspace(-half, half, sensor.samples)
    endpoints = segments.reshape(-1, 2) - origin
    rel = (np.arctan2(endpoints[:, 1], endpoints[:, 0]) - center + np.pi) % (
        2 * np.pi
    ) - np.pi
    angles = angles[(angles - center >= rel.min()) & (angles - center <= rel.max())]
    rays = np.column_stack((np.cos(angles), np.sin(angles)))
    a = segments[:, 0] - origin
    e = segments[:, 1] - segments[:, 0]
    denominator = cross2(rays[:, None, :], e[None, :, :])
    valid = np.abs(denominator) > 1e-12
    safe = np.where(valid, denominator, 1.0)
    t = cross2(a, e)[None, :] / safe
    u = cross2(a[None, :, :], rays[:, None, :]) / safe
    t = np.where(valid & (t > 0) & (u >= 0) & (u <= 1), t, np.inf)
    nearest = t.min(axis=1)
    ok = np.isfinite(nearest)
    pts = origin + rays[ok] * nearest[ok, None]
    forward = d / np.linalg.norm(d)
    depth = (pts - origin) @ forward
    return pts[(depth >= 350.0) & (depth <= 1875.0)]


def simulate(
    mesh,
    seed=42,
    step_mm=1.25,
    noise_mm=0.15,
    dropout=0.0,
    sensors=SENSORS,
    encoder_scale=1.0,
    missing_profile_fraction=0.0,
):
    if (
        step_mm <= 0
        or noise_mm < 0
        or not 0 <= dropout <= 1
        or not 0 <= missing_profile_fraction <= 1
    ):
        raise ValueError("Invalid scan parameters")
    rng = np.random.default_rng(seed)
    low, high = mesh.vertices[:, 1].min(), mesh.vertices[:, 1].max()
    positions = np.arange(low + step_mm / 2, high, step_mm)
    points, ids, profiles = [], [], []
    expected = len(positions) * len(sensors)
    received = 0
    for profile_id, y in enumerate(positions):
        section = section_segments(mesh, y)
        for sensor_id, sensor in enumerate(sensors):
            if rng.random() < missing_profile_fraction:
                continue
            received += 1
            xz = scan_section(section, sensor)
            if len(xz) == 0:
                continue
            xz = xz[rng.random(len(xz)) >= dropout]
            xz += rng.normal(0.0, noise_mm, xz.shape)
            # Local scan plane: object coordinate = plane_y - belt displacement.
            displacement = sensor.plane_y - y
            measured_y = sensor.plane_y - displacement * encoder_scale
            pts = np.column_stack((xz[:, 0], np.full(len(xz), measured_y), xz[:, 1]))
            points.append(pts)
            ids.extend([sensor_id] * len(pts))
            profiles.extend([profile_id] * len(pts))
    cloud = np.vstack(points) if points else np.empty((0, 3))
    return {
        "points": cloud,
        "sensor_ids": np.array(ids),
        "profile_ids": np.array(profiles),
        "expected_profiles": expected,
        "received_profiles": received,
        "step_mm": step_mm,
        "noise_mm": noise_mm,
        "seed": seed,
        "simulation_model": "idealized_first_surface_sections",
    }


def coverage_calculation():
    """Check all corners of x=[-300,300], z=[0,sqrt(400²+300²+300²)]."""
    height = np.linalg.norm([400.0, 300.0, 300.0])
    corners = np.array(list(product([-300.0, 300.0], [0.0, height])))
    rows = []
    for sensor in SENSORS:
        p = np.array([sensor.x, sensor.z])
        forward = np.array([sensor.target_x, sensor.target_z]) - p
        forward /= np.linalg.norm(forward)
        side = np.array([-forward[1], forward[0]])
        depth = (corners - p) @ forward
        lateral = (corners - p) @ side
        # Linear interpolation of published full-field endpoints: engineering envelope.
        width = 390.0 + (depth - 350.0) * (2000.0 - 390.0) / 1525.0
        margin = width / 2 - np.abs(lateral)
        rows.append(
            {
                "sensor": sensor.name,
                "min_depth_mm": float(depth.min()),
                "max_depth_mm": float(depth.max()),
                "min_half_field_margin_mm": float(margin.min()),
                "depth_span_mm": float(np.ptp(depth)),
                "inside_nominal_envelope": bool(
                    np.all(margin > 0) and depth.min() >= 350 and depth.max() <= 1875
                ),
            }
        )
    return rows
