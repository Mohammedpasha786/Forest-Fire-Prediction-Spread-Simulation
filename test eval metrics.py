import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.eval_metrics import (
    jaccard_similarity, boundary_displacement_error_m,
    area_growth_rate_error_pct, evaluate_spread_fidelity,
)


def test_jaccard_identical_masks():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    assert jaccard_similarity(mask, mask) == 1.0


def test_jaccard_disjoint_masks():
    a = np.zeros((20, 20), dtype=bool)
    a[0:5, 0:5] = True
    b = np.zeros((20, 20), dtype=bool)
    b[15:20, 15:20] = True
    assert jaccard_similarity(a, b) == 0.0


def test_jaccard_partial_overlap():
    a = np.zeros((10, 10), dtype=bool)
    a[0:6, 0:6] = True
    b = np.zeros((10, 10), dtype=bool)
    b[3:9, 3:9] = True
    score = jaccard_similarity(a, b)
    assert 0.0 < score < 1.0


def test_boundary_displacement_zero_for_identical_masks():
    mask = np.zeros((30, 30), dtype=bool)
    mask[10:20, 10:20] = True
    err = boundary_displacement_error_m(mask, mask, cell_size_m=30)
    assert err == 0.0


def test_boundary_displacement_increases_with_offset():
    base = np.zeros((50, 50), dtype=bool)
    base[20:30, 20:30] = True

    near_offset = np.zeros((50, 50), dtype=bool)
    near_offset[21:31, 20:30] = True

    far_offset = np.zeros((50, 50), dtype=bool)
    far_offset[30:40, 20:30] = True

    err_near = boundary_displacement_error_m(base, near_offset, cell_size_m=30)
    err_far = boundary_displacement_error_m(base, far_offset, cell_size_m=30)
    assert err_far > err_near


def test_area_growth_rate_error_zero_for_identical_series():
    series = [0, 5, 12, 20, 35]
    err = area_growth_rate_error_pct(series, series)
    assert err == 0.0


def test_area_growth_rate_error_positive_for_different_series():
    sim = [0, 5, 12, 20, 35]
    ref = [0, 4, 10, 22, 30]
    err = area_growth_rate_error_pct(sim, ref)
    assert err > 0.0


def test_evaluate_spread_fidelity_combines_metrics():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    result = evaluate_spread_fidelity(mask, mask, sim_area_series=[0, 1, 2], ref_area_series=[0, 1, 2])
    assert "jaccard_similarity" in result
    assert "boundary_displacement_error_m" in result
    assert "area_growth_rate_error_pct" in result
    assert result["jaccard_similarity"] == 1.0
