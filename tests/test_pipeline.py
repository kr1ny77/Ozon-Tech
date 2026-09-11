import numpy as np
import pytest
from functools import partial
from ozon_dimensioner.pipeline import (
    measure as measure_raw,
    AcquisitionQuality,
    cloud_components,
)
from ozon_dimensioner.simulation import (
    cuboid,
    simulate,
    coverage_calculation,
    section_segments,
    scan_section,
    SENSORS,
)
from ozon_dimensioner.geometry import dimension_errors


measure = partial(measure_raw, calibration_id="test-v1")


def quality(**kwargs):
    flags = dict(calibration_valid=True, surface_qualified=True, motion_valid=True)
    flags.update(kwargs)
    return AcquisitionQuality(100, 100, **flags)


@pytest.fixture
def cloud():
    return simulate(cuboid((80, 50, 30)).placed((0, 0, 27)), noise_mm=0.0)


def test_profile_reconstruction_and_measurement(cloud):
    p = measure(cloud["points"], "test", quality(), cloud["sensor_ids"])
    assert p["status"] == "ok"
    assert dimension_errors(list(p["dimensions_mm"].values()), [80, 50, 30])["passed"]
    assert p["dimension_convention"] == "obb_edges_descending"
    assert p["obb_dimensions_sorted_mm"] == list(p["dimensions_mm"].values())
    assert p["height_z_mm"] == pytest.approx(30.0)
    np.testing.assert_allclose(
        np.array(p["obb_axes"]).T @ np.array(p["obb_axes"]), np.eye(3), atol=1e-8
    )


def test_vertical_height_is_distinct_from_shortest_obb_edge():
    data = simulate(cuboid((80, 50, 30)).placed((45, 0, 0)), noise_mm=0)
    p = measure(data["points"], "tilted", quality(), data["sensor_ids"])
    assert p["status"] == "ok"
    assert p["height_z_mm"] > p["dimensions_mm"]["height"] + 20


@pytest.mark.parametrize("offset", [0, 1_000_000_000])
def test_grid_connectivity_preserves_diagonals_and_gaps(offset):
    points = (
        np.array(
            [[-0.1, -0.1, -0.1], [0.1, 0.1, 0.1], [4.1, 4.1, 4.1], [16.1, 16.1, 16.1]]
        )
        + offset
    )
    assert cloud_components(points) == 2
    assert cloud_components(np.array([[0, 0, 0], [4, 4, 4], [8_000_000, 0, 0]])) == 2


@pytest.mark.parametrize(
    "q,reason",
    [
        (AcquisitionQuality(100, 90, surface_qualified=True), "profile_loss"),
        (AcquisitionQuality(100, 100), "surface_unqualified"),
        (quality(calibration_valid=False), "calibration_expired"),
        (quality(motion_valid=False), "motion_invalid"),
    ],
)
def test_review_retains_candidate_withholds_warehouse_dimensions(cloud, q, reason):
    p = measure(cloud["points"], "test", q, cloud["sensor_ids"])
    assert p["status"] == "review" and reason in p["reasons"]
    assert p["dimensions_mm"] is None
    assert p["candidate_box"] is not None


def test_missing_and_degenerate_points():
    p = measure(np.empty((0, 3)), "empty", quality())
    assert p["status"] == "review"
    p = measure(np.tile([1, 1, 10], (30, 1)), "flat", quality())
    assert "degenerate_cloud" in p["reasons"]


def test_sensor_coverage():
    assert all(r["inside_nominal_envelope"] for r in coverage_calculation())
    assert all(r["depth_span_mm"] < 1000 for r in coverage_calculation())


def test_scanner_first_surface():
    mesh = cuboid((80, 50, 30)).placed()
    points = scan_section(section_segments(mesh, 0.0), SENSORS[0])
    np.testing.assert_allclose(points[:, 1], 30.0, atol=1e-8)


def test_deterministic_scan():
    mesh = cuboid((20, 15, 10)).placed()
    a = simulate(mesh, seed=7)
    b = simulate(mesh, seed=7)
    np.testing.assert_array_equal(a["points"], b["points"])


@pytest.mark.parametrize(
    "expected,received",
    [
        (0, 0),
        (np.inf, np.inf),
        (np.nan, 0),
        (100.5, 100),
        (100, 99.5),
        (True, True),
        (100, 101),
    ],
)
def test_invalid_profile_counters(cloud, expected, received):
    with pytest.raises(ValueError, match="profile counters"):
        measure(cloud["points"], "test", AcquisitionQuality(expected, received))


def test_single_view_flag(cloud):
    p = measure(
        cloud["points"], "test", quality(), np.zeros(len(cloud["points"]), dtype=int)
    )
    assert "insufficient_views" in p["reasons"]


@pytest.mark.parametrize("count", [1, 30])
def test_detached_reflection_withholds_dimensions(cloud, count):
    extra = np.random.default_rng(18).normal(0, 0.1, (count, 3)) + [0, 0, 500]
    points = np.vstack((cloud["points"], extra))
    ids = np.r_[cloud["sensor_ids"], np.zeros(count, dtype=int)]
    p = measure(points, "reflection", quality(), ids)
    assert p["status"] == "review"
    assert "disconnected_cloud" in p["reasons"]
    assert p["dimensions_mm"] is None
    # Keep the anomaly in the diagnostic box rather than erase a possible feature.
    assert p["candidate_box"]["dimensions_mm"][0] > 450


def test_connected_thin_feature_is_preserved(cloud):
    feature = np.column_stack((np.zeros(24), np.zeros(24), np.arange(29, 101, 3)))
    points = np.vstack((cloud["points"], feature))
    p = measure(
        points,
        "feature",
        quality(),
        np.r_[cloud["sensor_ids"], np.zeros(24, dtype=int)],
    )
    assert p["status"] == "ok"
    b = p["candidate_box"]
    projection = (feature - np.array(b["center_mm"])) @ np.array(b["axes"])
    assert np.all(np.abs(projection) <= np.array(b["dimensions_mm"]) / 2 + 1e-6)


def test_missing_sensor_ids_withholds_dimensions(cloud):
    top = cloud["sensor_ids"] == 0
    p = measure(cloud["points"][top], "missing", quality())
    assert p["status"] == "review" and p["dimensions_mm"] is None
    assert "sensor_ids_missing" in p["reasons"]


@pytest.mark.parametrize(
    "ids_kind", ["unknown", "fractional", "nan", "mixed_object", "token_second_view"]
)
def test_sensor_provenance_validation(cloud, ids_kind):
    ids = cloud["sensor_ids"].copy()
    if ids_kind == "unknown":
        ids += 100
    elif ids_kind == "fractional":
        ids = ids.astype(float) + 0.5
    elif ids_kind == "nan":
        ids = ids.astype(float)
        ids[ids == 0] = np.nan
    elif ids_kind == "mixed_object":
        ids = ids.astype(object)
        ids[ids == 0] = None
    else:
        ids[:] = 0
        ids[0] = 1
    p = measure(cloud["points"], "provenance", quality(), ids)
    assert p["status"] == "review" and p["dimensions_mm"] is None
    assert p["candidate_box"] is not None
    if ids_kind != "token_second_view":
        assert "sensor_ids_invalid" in p["reasons"]


def test_hidden_encoder_scale_fault_detected():
    data = simulate(cuboid((200, 100, 40)).placed((0, 0, 27)), encoder_scale=1.12)
    p = measure(data["points"], "encoder", quality(), data["sensor_ids"])
    assert p["status"] == "review"
    assert "views_inconsistent" in p["reasons"]


def test_qualification_defaults_and_calibration_identity(cloud):
    p = measure_raw(
        cloud["points"], "defaults", AcquisitionQuality(100, 100), cloud["sensor_ids"]
    )
    assert p["status"] == "review"
    assert set(
        [
            "calibration_expired",
            "calibration_missing",
            "motion_invalid",
            "surface_unqualified",
        ]
    ) <= set(p["reasons"])


def test_textual_attestations_rejected(cloud):
    with pytest.raises(ValueError, match="booleans"):
        measure(
            cloud["points"], "bad", quality(motion_valid="false"), cloud["sensor_ids"]
        )
