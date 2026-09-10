"""Independent numerical witnesses for the production OBB configuration."""

import json
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.transform import Rotation
from ozon_dimensioner.geometry import minimum_box, box_in_frame
from ozon_dimensioner.simulation import wedge, l_shape


def run():
    cases = [
        (mesh.name, mesh.placed((21, 37, 19)).vertices) for mesh in (wedge(), l_shape())
    ]
    for seed in [10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
        rng = np.random.default_rng(seed)
        points = rng.normal(size=(10 if seed == 10 else 24, 3)) * [80, 60, 40]
        cases.append((f"random_hull_seed_{seed}", points))
    for seed, count in [(444, 1000), (445, 1000), (444, 3000), (445, 3000)]:
        rng = np.random.default_rng(seed)
        points = rng.normal(size=(count, 3))
        points = points / np.linalg.norm(points, axis=1)[:, None] * [200, 150, 150]
        points = (
            points
            @ Rotation.from_euler("xyz", [19, 37, 61], degrees=True).as_matrix().T
        )
        cases.append((f"ellipsoid_{count}_seed_{seed}", points))
    rows = []
    for name, points in cases:
        p = points - points.mean(axis=0)
        box = minimum_box(p)
        witnesses = []

        def objective(angles):
            matrices = Rotation.from_euler("xyz", np.atleast_2d(angles.T)).as_matrix()
            return np.prod(np.ptp(np.matmul(p, matrices), axis=1), axis=1)

        for seed in (110, 210, 310):
            result = differential_evolution(
                objective,
                [(-np.pi, np.pi)] * 3,
                seed=seed,
                maxiter=600,
                popsize=20,
                tol=1e-9,
                polish=False,
                vectorized=True,
                updating="deferred",
            )
            witness = box_in_frame(
                p, Rotation.from_euler("xyz", result.x).as_matrix(), "independent_de"
            )
            assert witness.contains(p).all()
            witnesses.append(witness.as_dict())
        ratio = box.volume / min(w["volume_mm3"] for w in witnesses)
        row = {
            "shape": name,
            "points_mm": p.tolist(),
            "prototype_box": box.as_dict(),
            "independent_boxes": witnesses,
            "ratio_to_best_de": ratio,
            "all_points_enclosed": bool(box.contains(p).all()),
            "within_numerical_comparison_limit": bool(ratio <= 1.005),
        }
        rows.append(row)
        print(name, f"ratio={ratio:.8f}", flush=True)
    result = {
        "configuration": "minimum_box defaults, identical to pipeline",
        "cases": len(rows),
        "volume_comparison_limit": 1.005,
        "max_ratio_to_best_de": max(r["ratio_to_best_de"] for r in rows),
        "passed": all(
            r["within_numerical_comparison_limit"] and r["all_points_enclosed"]
            for r in rows
        ),
        "scope": "Numerical volume cross-check; per-edge tolerance and global optimality require separate evidence",
        "runs": rows,
    }
    Path("results/geometry_crosscheck.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    assert result["passed"], "Production OBB exceeded independent comparison limit"


if __name__ == "__main__":
    run()
