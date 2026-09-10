"""Batch measurement and explicit qualification gates for recorded clouds."""

from dataclasses import dataclass
from datetime import datetime, timezone
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from . import __version__
from .geometry import minimum_box


@dataclass(frozen=True)
class AcquisitionQuality:
    expected_profiles: int
    received_profiles: int
    calibration_valid: bool = False
    surface_qualified: bool = False
    motion_valid: bool = False


def cloud_components(points, cell_mm=4.0):
    """26-neighbour occupied-cell connectivity, used only as a quality gate.

    Original extrema are retained. Disconnected reflections and disconnected
    genuine protrusions both need a repeat scan; neither is silently discarded.
    """
    cells = np.unique(np.floor(points / cell_mm), axis=0)
    if len(cells) < 2:
        return len(cells)
    pairs = cKDTree(cells).query_pairs(1.0, p=np.inf, output_type="ndarray")
    graph = coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
        shape=(len(cells), len(cells)),
    )
    return int(connected_components(graph, directed=False, return_labels=False))


def measure(
    points,
    measurement_id,
    quality,
    sensor_ids=None,
    *,
    calibration_id="unspecified",
    measured_at=None,
    item_id=None,
):
    p = np.asarray(points, float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("Expected an Nx3 cloud in millimetres")
    if not measurement_id:
        raise ValueError("measurement_id is required")
    if any(
        type(flag) is not bool
        for flag in (
            quality.calibration_valid,
            quality.surface_qualified,
            quality.motion_valid,
        )
    ):
        raise ValueError("Quality attestations must be explicit booleans")
    reasons = []
    if (
        any(
            isinstance(count, (bool, np.bool_))
            or not isinstance(count, (int, np.integer))
            for count in (quality.expected_profiles, quality.received_profiles)
        )
        or quality.expected_profiles <= 0
        or not 0 <= quality.received_profiles <= quality.expected_profiles
    ):
        raise ValueError("Invalid profile counters")
    if quality.received_profiles / quality.expected_profiles < 0.99:
        reasons.append("profile_loss")
    if not quality.calibration_valid:
        reasons.append("calibration_expired")
    if (
        not isinstance(calibration_id, str)
        or not calibration_id.strip()
        or calibration_id == "unspecified"
    ):
        reasons.append("calibration_missing")
    if not quality.motion_valid:
        reasons.append("motion_invalid")
    if not quality.surface_qualified:
        reasons.append("surface_unqualified")
    finite = np.isfinite(p).all(axis=1)
    if len(p) and np.mean(~finite) > 0.01:
        reasons.append("invalid_coordinates")
    # Production belt subtraction uses calibrated uncertainty; 1 mm is the demo setting.
    keep = finite & (p[:, 2] > 1.0)
    cleaned = p[keep]
    if len(cleaned) < 20:
        reasons.append("insufficient_points")
    components = cloud_components(cleaned) if len(cleaned) else 0
    if components > 1:
        reasons.append("disconnected_cloud")
    if sensor_ids is None:
        reasons.append("sensor_ids_missing")
    else:
        ids = np.asarray(sensor_ids)
        if ids.shape != (len(p),):
            raise ValueError("sensor_ids must match point count")
        if (
            not np.issubdtype(ids.dtype, np.integer)
            or not np.isin(ids, [0, 1, 2]).all()
        ):
            reasons.append("sensor_ids_invalid")
        else:
            unique, counts = np.unique(ids[keep], return_counts=True)
            supported = unique[counts >= 10]
            if len(supported) < 2:
                reasons.append("insufficient_views")
            else:
                # Independently transformed scan planes should agree at both ends
                # of the item. Occlusion can also trigger review, by design.
                intervals = np.array(
                    [
                        [
                            cleaned[ids[keep] == i, 1].min(),
                            cleaned[ids[keep] == i, 1].max(),
                        ]
                        for i in supported
                    ]
                )
                if np.ptp(intervals, axis=0).max() > 3.0:
                    reasons.append("views_inconsistent")
    payload = {
        "schema_version": "1.0",
        "measurement_id": measurement_id,
        "units": "mm",
        "status": "review" if reasons else "ok",
        "reasons": reasons,
        "algorithm_version": __version__,
        "calibration_id": calibration_id,
        "measured_at": measured_at or datetime.now(timezone.utc).isoformat(),
        "item_id": item_id,
        "dimensions_mm": None,
        "candidate_box": None,
        "points_used": len(cleaned),
        "quality_checks": {
            "connected_components": components,
            "connectivity_cell_mm": 4.0,
            "view_end_tolerance_mm": 3.0,
        },
    }
    if len(cleaned) >= 20:
        try:
            box = minimum_box(cleaned)
            payload["candidate_box"] = box.as_dict()
            if np.any(box.extents > np.array([400.0, 300.0, 300.0]) * 1.05) or np.any(
                box.extents < 5.0
            ):
                reasons.append("dimensions_out_of_range")
                payload["status"] = "review"
            if not reasons:
                payload["dimensions_mm"] = dict(
                    zip(["length", "width", "height"], box.extents.tolist())
                )
        except ValueError:
            reasons.append("degenerate_cloud")
            payload["status"] = "review"
    return payload
