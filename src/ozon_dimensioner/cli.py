"""Reproducible synthetic demonstration and saved-cloud measurement."""

import argparse
from datetime import datetime, timedelta, timezone
import csv
import json
from pathlib import Path
import platform
import time
import numpy as np
from .geometry import aabb, pca_box, dimension_errors
from .pipeline import measure, AcquisitionQuality
from .simulation import cuboid, wedge, l_shape, cylinder, simulate, coverage_calculation
from .outbox import Outbox, http_sender


def save_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def demo(output, seed=42):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cases = []
    for d in [
        (10, 10, 10),
        (20, 15, 10),
        (100, 100, 100),
        (200, 100, 10),
        (400, 300, 300),
        (95, 60, 25),
    ]:
        for angle in [0, 27, 63]:
            cases.append((cuboid(d).placed((0, 0, angle)), "nominal", {}))
    for _ in range(12):
        dims = rng.uniform([15, 10, 10], [250, 180, 150])
        cases.append((cuboid(dims).placed(rng.uniform(-60, 60, 3)), "nominal", {}))
    cases.extend(
        [
            (cuboid((80, 50, 30)).placed((0, 0, 31), x=x), "edge", {})
            for x in [-250, 250]
        ]
    )
    cases.extend(
        [
            (wedge().placed((15, 10, 27)), "irregular", {}),
            (l_shape().placed((0, 0, 27)), "irregular", {}),
            (cylinder().placed((0, 0, 19)), "curved", {}),
        ]
    )
    base = cuboid((200, 100, 40)).placed((0, 0, 27))
    cases.extend(
        [
            (base, "dropout", {"dropout": 0.35}),
            (base, "profile_loss", {"missing_profile_fraction": 0.08}),
            (base, "unknown_surface", {"dropout": 0.85}),
            (base, "encoder_fault", {"encoder_scale": 1.12}),
            (base, "undetected_encoder_fault", {"encoder_scale": 1.12}),
            (base, "expired_calibration", {}),
        ]
    )
    cases.extend(
        (base, group, {})
        for group in [
            "isolated_reflection",
            "reflection_cluster",
            "missing_sensor_ids",
            "single_view",
            "invalid_sensor_ids",
            "missing_attestations",
        ]
    )
    rows, payloads = [], []
    for i, (mesh, group, params) in enumerate(cases):
        start = time.perf_counter()
        data = simulate(mesh, seed=seed + i, **params)
        ids = data["sensor_ids"]
        if group in ("isolated_reflection", "reflection_cluster"):
            count = 1 if group == "isolated_reflection" else 30
            extra = np.random.default_rng(seed + i).normal(0, 0.1, (count, 3)) + [
                0,
                0,
                500,
            ]
            data["points"] = np.vstack((data["points"], extra))
            ids = np.r_[ids, np.zeros(count, dtype=int)]
        elif group == "missing_sensor_ids":
            ids = None
        elif group == "single_view":
            mask = ids == 0
            data["points"], ids = data["points"][mask], ids[mask]
        elif group == "invalid_sensor_ids":
            ids = ids + 100
        scan_ms = (time.perf_counter() - start) * 1000
        quality = AcquisitionQuality(
            data["expected_profiles"],
            data["received_profiles"],
            calibration_valid=group != "expired_calibration",
            surface_qualified=group != "unknown_surface",
            motion_valid=group != "encoder_fault",
        )
        if group == "missing_attestations":
            quality = AcquisitionQuality(
                data["expected_profiles"], data["received_profiles"]
            )
        start = time.perf_counter()
        payload = measure(
            data["points"],
            f"demo-{i:03d}",
            quality,
            ids,
            calibration_id="synthetic-v1",
            measured_at=(
                datetime(2026, 9, 9, tzinfo=timezone.utc) + timedelta(seconds=3 * i)
            ).isoformat(),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        payload["data_origin"] = "synthetic"
        payload["qualification_basis"] = "test_scenario_metadata"
        payload["case_group"] = group
        payloads.append(payload)
        p = data["points"][data["points"][:, 2] > 1.0]
        row = {
            "case": i,
            "shape": mesh.name,
            "group": group,
            "status": payload["status"],
            "points": len(data["points"]),
            "scan_simulation_ms": scan_ms,
            "processing_ms": elapsed_ms,
            "passed": None,
            "max_error_mm": None,
            "max_normalized_error": None,
            "aabb_volume_ratio": None,
            "pca_volume_ratio": None,
        }
        if payload["candidate_box"]:
            b = payload["candidate_box"]
            row["aabb_volume_ratio"] = aabb(p).volume / b["volume_mm3"]
            row["pca_volume_ratio"] = pca_box(p).volume / b["volume_mm3"]
            if mesh.reference_dimensions is not None:
                err = dimension_errors(b["dimensions_mm"], mesh.reference_dimensions)
                row.update(
                    passed=err["passed"],
                    max_error_mm=max(err["absolute_mm"]),
                    max_normalized_error=max(err["normalized"]),
                )
                payload["reference_dimensions_mm"] = mesh.reference_dimensions.tolist()
                payload["errors"] = err
        rows.append(row)
        if i in [3, 13, 32, 33, 37, 39]:
            np.savez_compressed(
                out / f"cloud_{i:03d}.npz",
                points=data["points"],
                sensor_ids=data["sensor_ids"],
                vertices=mesh.vertices,
                triangles=mesh.triangles,
            )
        print(
            f"{i + 1:02d}/{len(cases)} {group:25s} {payload['status']:6s} {elapsed_ms:8.1f} ms",
            flush=True,
        )
    evaluated = [
        r
        for r in rows
        if r["group"] in ("nominal", "edge", "curved", "dropout")
        and r["passed"] is not None
    ]
    accepted = [r for r in rows if r["status"] == "ok" and r["passed"] is not None]
    nominal_accepted = [r for r in evaluated if r["status"] == "ok"]
    ms = [r["processing_ms"] for r in rows]
    summary = {
        "seed": seed,
        "total_cases": len(rows),
        "reference_evaluable_cases": len(evaluated),
        "reference_pass_count": sum(r["passed"] for r in evaluated),
        "reference_pass_rate": np.mean([r["passed"] for r in evaluated]).item(),
        "nominal_accepted_count": len(nominal_accepted),
        "nominal_review_count": len(evaluated) - len(nominal_accepted),
        "nominal_acceptance_rate": len(nominal_accepted) / len(evaluated),
        "review_count": sum(r["status"] == "review" for r in rows),
        "accepted_reference_cases": len(accepted),
        "accepted_reference_failure_count": sum(not r["passed"] for r in accepted),
        "error_max_mm": max(r["max_error_mm"] for r in evaluated),
        "error_p95_mm": float(
            np.percentile([r["max_error_mm"] for r in evaluated], 95)
        ),
        "processing_p50_ms": float(np.percentile(ms, 50)),
        "processing_p95_ms": float(np.percentile(ms, 95)),
        "processing_max_ms": max(ms),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "coverage": coverage_calculation(),
        "limitations": [
            "Synthetic idealized geometry; no physical hardware measurements",
            "Qualification flags supplied by scenario; no material classifier",
            "Hull/SLSQP OBB is approximate; arbitrary global optimum uncertified",
            "Camera occlusion, refraction, radiometry and conveyor mechanics omitted",
        ],
    }
    save_json(out / "summary.json", summary)
    save_json(out / "measurements.json", payloads)
    with (out / "benchmark.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("demo")
    d.add_argument("--output", default="results")
    d.add_argument("--seed", type=int, default=42)
    m = sub.add_parser("measure")
    m.add_argument("input")
    m.add_argument("--output", required=True)
    m.add_argument("--measurement-id", required=True)
    m.add_argument("--calibration-id", required=True)
    m.add_argument("--expected-profiles", type=int, required=True)
    m.add_argument("--received-profiles", type=int, required=True)
    m.add_argument(
        "--qualified",
        action="store_true",
        help="Operator attests prior surface qualification",
    )
    m.add_argument(
        "--calibration-valid",
        action="store_true",
        help="Attest a passed calibration check",
    )
    m.add_argument(
        "--motion-valid", action="store_true", help="Attest an independent motion check"
    )
    s = sub.add_parser("send")
    s.add_argument("input")
    s.add_argument("--database", default="outbox.sqlite")
    s.add_argument("--url", required=True)
    args = parser.parse_args()
    if args.command == "demo":
        demo(args.output, args.seed)
    elif args.command == "measure":
        with np.load(args.input, allow_pickle=False) as data:
            quality = AcquisitionQuality(
                args.expected_profiles,
                args.received_profiles,
                calibration_valid=args.calibration_valid,
                surface_qualified=args.qualified,
                motion_valid=args.motion_valid,
            )
            payload = measure(
                data["points"],
                args.measurement_id,
                quality,
                data.get("sensor_ids"),
                calibration_id=args.calibration_id,
            )
        save_json(args.output, payload)
    else:
        outbox = Outbox(args.database)
        try:
            payloads = json.loads(Path(args.input).read_text())
            if isinstance(payloads, dict):
                payloads = [payloads]
            for payload in payloads:
                outbox.enqueue(payload)
            delivered = outbox.flush(http_sender(args.url))
            print(json.dumps({"delivered": delivered, "pending": outbox.pending()}))
        finally:
            outbox.close()


if __name__ == "__main__":
    main()
