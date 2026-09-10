"""Enclosing boxes: bounded orientation search and constrained refinement.

Every returned box encloses the complete input. Numerical optimization yields
an upper bound; global optimality for arbitrary geometry remains uncertified.
"""

from dataclasses import dataclass
from itertools import product
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class Box:
    center: np.ndarray
    axes: np.ndarray  # columns are local axes expressed in world coordinates
    extents: np.ndarray  # descending; paired with axes
    method: str

    @property
    def volume(self):
        return float(np.prod(self.extents))

    def corners(self):
        return (
            np.array(list(product([-0.5, 0.5], repeat=3))) * self.extents @ self.axes.T
            + self.center
        )

    def contains(self, points, atol=1e-7):
        return np.all(
            np.abs((points - self.center) @ self.axes) <= self.extents / 2 + atol,
            axis=1,
        )

    def as_dict(self):
        return {
            "center_mm": self.center.tolist(),
            "axes": self.axes.tolist(),
            "dimensions_mm": self.extents.tolist(),
            "volume_mm3": self.volume,
            "method": self.method,
            "global_optimum_certified": False,
        }


def validate_points(points):
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 4:
        raise ValueError("At least four 3D points are required")
    if not np.isfinite(p).all():
        raise ValueError("Coordinates must be finite")
    if np.linalg.matrix_rank(p - p.mean(axis=0), tol=1e-8) < 3:
        raise ValueError("A three-dimensional point cloud is required")
    return p


def box_in_frame(points, axes, method):
    projected = points @ axes
    lo, hi = projected.min(axis=0), projected.max(axis=0)
    extents = hi - lo
    center = ((hi + lo) / 2) @ axes.T
    order = np.argsort(-extents, kind="stable")
    axes = axes[:, order].copy()
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    return Box(center, axes, extents[order], method)


def aabb(points):
    return box_in_frame(validate_points(points), np.eye(3), "aabb")


def pca_box(points):
    p = validate_points(points)
    _, axes = np.linalg.eigh(np.cov(p.T))
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    return box_in_frame(p, axes, "pca")


MAX_HULL_FRAMES = 256
RANDOM_FRAMES = Rotation.random(64, random_state=8128).as_matrix()


def minimum_box(points):
    """Search the same fixed configuration in production and validation.

    Hull orientations are capped independently of hull size. SLSQP optimizes
    rotation and six supporting planes with explicit enclosure constraints.
    Coordinates are normalized for consistent optimization tolerances.
    """
    p = validate_points(points)
    origin = p.mean(axis=0)
    centered = p - origin
    hull = ConvexHull(centered)
    scale = np.linalg.norm(np.ptp(centered, axis=0))
    vertices = centered[hull.vertices] / scale
    candidates = [aabb(vertices), pca_box(vertices)]
    triangles = centered[hull.simplices] / scale
    # Stratify by face area: flat large faces and curved small faces both enter.
    areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    order = np.argsort(areas, kind="stable")
    count = min(len(order), MAX_HULL_FRAMES // 3)
    triangles = triangles[order[np.linspace(0, len(order) - 1, count, dtype=int)]]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    frames = []
    for j in range(3):
        u = triangles[:, (j + 1) % 3] - triangles[:, j]
        u /= np.linalg.norm(u, axis=1)[:, None]
        frames.extend(np.stack((u, np.cross(normals, u), normals), axis=2))
    frames.extend(RANDOM_FRAMES)
    for axes in frames:
        candidates.append(box_in_frame(vertices, axes, "hull_slsqp"))
    min_volume = min(b.volume for b in candidates)

    def preference(b):
        # Equal-volume boxes can have different side lengths (e.g. a wedge).
        # Canonical tie: smallest longest side, then intermediate side.
        return (round(b.volume / min_volume, 10), *np.round(b.extents, 7))

    candidates.sort(key=preference)
    best = candidates[0]
    starts = []
    local_starts = 12 if len(vertices) <= 64 else 6
    dispersed_starts = 8 if len(vertices) <= 64 else 2
    for candidate in candidates:
        # Signed/permuted equivalent frames share the same absolute axis products.
        if all(
            np.max(np.abs(candidate.axes.T @ s.axes), axis=0).min() < 0.985
            for s in starts
        ):
            starts.append(candidate)
        if len(starts) >= local_starts:
            break
    # Additional dispersed starts escape basins absent from the best hull frames.
    starts.extend(
        box_in_frame(vertices, axes, "hull_slsqp")
        for axes in RANDOM_FRAMES[:dispersed_starts]
    )
    for start in starts:
        q = vertices @ start.axes
        x0 = np.r_[np.zeros(3), q.min(axis=0), q.max(axis=0)]
        active = np.unique(np.r_[q.argmin(axis=0), q.argmax(axis=0)])

        def objective(x):
            return np.prod(x[6:] - x[3:6])

        def constraints(x):
            axes = Rotation.from_rotvec(x[:3]).as_matrix() @ start.axes
            projected = vertices[active] @ axes
            return np.r_[(projected - x[3:6]).ravel(), (x[6:] - projected).ravel()]

        # Constraint exchange keeps the numerical subproblem small. Each round
        # checks every hull vertex and adds the most violated supporting points.
        for _ in range(10):
            result = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=[(-np.pi, np.pi)] * 3 + [(None, None)] * 6,
                constraints={"type": "ineq", "fun": constraints},
                options={"ftol": 1e-11, "maxiter": 100},
            )
            axes = Rotation.from_rotvec(result.x[:3]).as_matrix() @ start.axes
            q = vertices @ axes
            support = np.unique(np.r_[q.argmin(axis=0), q.argmax(axis=0)])
            if np.all(np.isin(support, active)):
                break
            active = np.union1d(active, support)
            x0 = np.r_[result.x[:3], q.min(axis=0), q.max(axis=0)]
        axes = Rotation.from_rotvec(result.x[:3]).as_matrix() @ start.axes
        # Re-evaluate extrema even if the optimizer reports an iteration limit.
        candidate = box_in_frame(vertices, axes, "hull_slsqp")
        if preference(candidate) < preference(best):
            best = candidate
    # A final full-hull solve removes residual active-set error from the winner.
    axes = best.axes.copy()
    projected = vertices @ axes

    def full_constraints(x):
        q = vertices @ (Rotation.from_rotvec(x[:3]).as_matrix() @ axes)
        return np.r_[(q - x[3:6]).ravel(), (x[6:] - q).ravel()]

    result = minimize(
        lambda x: np.prod(x[6:] - x[3:6]),
        np.r_[np.zeros(3), projected.min(axis=0), projected.max(axis=0)],
        method="SLSQP",
        constraints={"type": "ineq", "fun": full_constraints},
        options={"ftol": 1e-11, "maxiter": 40},
    )
    candidate = box_in_frame(
        vertices, Rotation.from_rotvec(result.x[:3]).as_matrix() @ axes, "hull_slsqp"
    )
    if preference(candidate) < preference(best):
        best = candidate
    # Recompute on the complete original cloud and return world coordinates.
    return box_in_frame(p, best.axes, "hull_slsqp")


def tolerance(dimensions):
    dimensions = np.asarray(dimensions, dtype=float)
    if not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
        raise ValueError("Reference dimensions must be positive and finite")
    return np.maximum(0.05 * dimensions, 5.0)


def dimension_errors(predicted, reference):
    predicted, reference = np.asarray(predicted, float), np.asarray(reference, float)
    if predicted.shape != (3,) or reference.shape != (3,):
        raise ValueError("Expected three dimensions")
    if not np.isfinite(predicted).all() or np.any(predicted <= 0):
        raise ValueError("Predicted dimensions must be positive and finite")
    # Warehouse canonical convention: longest, intermediate, shortest edge.
    p, r = np.sort(predicted)[::-1], np.sort(reference)[::-1]
    error = np.abs(p - r)
    normalized = error / tolerance(r)
    return {
        "absolute_mm": error.tolist(),
        "normalized": normalized.tolist(),
        "passed": bool(np.all(normalized <= 1.0)),
    }
