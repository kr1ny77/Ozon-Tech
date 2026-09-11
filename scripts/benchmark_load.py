"""Repeated tests of both point count and convex-hull complexity."""

import json
from pathlib import Path
import platform
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from ozon_dimensioner.geometry import minimum_box, dimension_errors
from ozon_dimensioner.simulation import cuboid
from ozon_dimensioner.pipeline import measure, AcquisitionQuality


def run():
    rng = np.random.default_rng(17)
    rows = []
    for kind, count in [
        ("box_interior", 100_000),
        ("box_interior", 1_000_000),
        ("box_interior", 3_700_000),
        ("ellipsoid_surface", 1000),
        ("ellipsoid_surface", 3000),
        ("ellipsoid_surface", 10000),
        ("ellipsoid_with_interior", 3_700_000),
    ]:
        if kind == "box_interior":
            corners = cuboid((400, 300, 300)).placed((0, 0, 27)).vertices
            p = rng.uniform(-0.5, 0.5, (count - 8, 3)) * [400, 300, 300]
            p = p @ Rotation.from_euler("z", 27, degrees=True).as_matrix().T + [
                0,
                0,
                150,
            ]
            p = np.vstack((p, corners))
        else:
            shell_count = 10000 if kind == "ellipsoid_with_interior" else count
            p = np.random.default_rng(444).normal(size=(shell_count, 3))
            p = p / np.linalg.norm(p, axis=1)[:, None] * [200, 150, 150]
            if kind == "ellipsoid_with_interior":
                interior = rng.uniform(-0.45, 0.45, (count - shell_count, 3)) * [
                    200,
                    150,
                    150,
                ]
                p = np.vstack((p, interior))
            p = p @ Rotation.from_euler(
                "xyz", [19, 37, 61], degrees=True
            ).as_matrix().T + [0, 0, 300]
        hull_vertices = len(ConvexHull(p).vertices)
        minimum_box(p)
        times = []
        passed = True
        for _ in range(5):
            start = time.perf_counter()
            box = minimum_box(p)
            times.append((time.perf_counter() - start) * 1000)
            passed &= bool(
                box.contains(p).all()
                and dimension_errors(box.extents, [400, 300, 300])["passed"]
            )
        row = {
            "shape": kind,
            "points": count,
            "hull_vertices": hull_vertices,
            "repeats": len(times),
            "obb_ms_runs": times,
            "obb_ms": float(np.median(times)),
            "max_ms": max(times),
            "passed": passed,
        }
        if kind == "ellipsoid_with_interior" or (
            kind == "ellipsoid_surface" and count == 10000
        ):
            ids = np.arange(count, dtype=np.int32) % 3
            pipeline_times = []
            for repeat in range(3):
                start = time.perf_counter()
                payload = measure(
                    p,
                    f"load-{kind}-{repeat}",
                    AcquisitionQuality(100, 100, True, True, True),
                    ids,
                    calibration_id="synthetic-load",
                )
                pipeline_times.append((time.perf_counter() - start) * 1000)
                assert payload["candidate_box"] is not None
            row.update(
                {
                    "pipeline_ms_runs": pipeline_times,
                    "pipeline_median_ms": float(np.median(pipeline_times)),
                    "pipeline_max_ms": max(pipeline_times),
                    "pipeline_status": payload["status"],
                    "pipeline_reasons": payload["reasons"],
                }
            )
        rows.append(row)
        print(row, flush=True)
    result = {
        "purpose": "OBB kernel and full measure function; combined point-count and hull-complexity tests",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "timing_scope": "Five kernel observations per shape and three full measure observations for complex cases; target IPC and production p99 remain unmeasured",
        "runs": rows,
    }
    Path("results/load_benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    assert all(row["passed"] for row in rows)


if __name__ == "__main__":
    run()
