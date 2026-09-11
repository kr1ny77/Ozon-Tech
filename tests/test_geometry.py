import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from ozon_dimensioner.geometry import (
    minimum_box,
    aabb,
    pca_box,
    dimension_errors,
    tolerance,
)
from ozon_dimensioner.simulation import cuboid, wedge


@pytest.mark.parametrize(
    "dimensions", [(10, 10, 10), (200, 100, 10), (400, 300, 300), (99, 100, 101)]
)
@pytest.mark.parametrize("angles", [(0, 0, 0), (23, 41, 67)])
def test_rotated_boxes_analytic(dimensions, angles):
    mesh = cuboid(dimensions).placed(angles)
    b = minimum_box(mesh.vertices)
    np.testing.assert_allclose(b.extents, np.sort(dimensions)[::-1], atol=1e-5)
    assert b.contains(mesh.vertices).all()
    assert np.linalg.det(b.axes) == pytest.approx(1.0)


def test_tetrahedron_analytic_minimum():
    p = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]], float)
    r = Rotation.from_euler("xyz", [23, 17, 31], degrees=True).as_matrix()
    b = minimum_box(p @ r.T)
    assert b.volume == pytest.approx(1.0, rel=2e-4)
    assert b.contains(p @ r.T).all()


def test_noisy_cloud_contains_all_points_and_beats_baselines():
    rng = np.random.default_rng(3)
    p = rng.uniform(-1, 1, (150, 3)) * [100, 50, 20]
    p = p @ Rotation.from_euler("xyz", [20, 30, 60], degrees=True).as_matrix().T
    b = minimum_box(p)
    assert b.contains(p).all()
    assert b.volume <= min(aabb(p).volume, pca_box(p).volume) * (1 + 1e-10)


def test_translation_and_scale():
    p = wedge().placed((20, 30, 40)).vertices
    b = minimum_box(p)
    q = minimum_box(p * 3 + [10000, -50, 73])
    np.testing.assert_allclose(q.extents, b.extents * 3, rtol=1e-5)


@pytest.mark.parametrize(
    "p",
    [
        [],
        np.zeros((5, 2)),
        np.zeros((5, 3)),
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [np.nan, 0, 1]],
    ],
)
def test_invalid_cloud(p):
    with pytest.raises(ValueError):
        minimum_box(p)


def test_tolerance_boundary_and_all_axes():
    np.testing.assert_allclose(tolerance([10, 100, 200]), [5, 5, 10])
    assert dimension_errors([210, 105, 15], [200, 100, 10])["passed"]
    assert not dimension_errors([210.001, 105, 15], [200, 100, 10])["passed"]
    assert dimension_errors([10, 100, 200], [200, 100, 10])["passed"]


@pytest.mark.parametrize("dimensions", [[0, 1, 2], [-1, 2, 3], [np.nan, 1, 2]])
def test_bad_reference(dimensions):
    with pytest.raises(ValueError):
        tolerance(dimensions)


def test_irregular_hull_regression_against_independent_feasible_box():
    p = np.random.default_rng(10).normal(size=(10, 3)) * [80, 60, 40]
    b = minimum_box(p)
    # Independent differential-evolution witness, preserved by validate_geometry.py.
    assert b.volume <= 2033001.404 * (1 + 1e-6)
    assert b.contains(p).all()
    np.testing.assert_allclose(b.axes.T @ b.axes, np.eye(3), atol=1e-9)


@pytest.mark.parametrize("count", [1000, 3000, 10000])
def test_curved_surface_with_many_hull_vertices(count):
    p = np.random.default_rng(444).normal(size=(count, 3))
    p = p / np.linalg.norm(p, axis=1)[:, None] * [200, 150, 150]
    p = p @ Rotation.from_euler("xyz", [19, 37, 61], degrees=True).as_matrix().T
    b = minimum_box(p)
    assert b.contains(p).all()
    assert dimension_errors(b.extents, [400, 300, 300])["passed"]
    assert b.volume <= 400 * 300 * 300


def test_wedge_equal_minima_use_canonical_edges():
    p = wedge().placed((21, 37, 19)).vertices
    b = minimum_box(p)
    np.testing.assert_allclose(b.extents, [180, 120, 90], atol=1e-5)
    # The hypotenuse-aligned box has the same volume and different edge lengths.
    hypotenuse = np.hypot(180, 90)
    alternative = [hypotenuse, 120, 180 * 90 / hypotenuse]
    assert np.prod(alternative) == pytest.approx(b.volume)
    assert not dimension_errors(b.extents, alternative)["passed"]
