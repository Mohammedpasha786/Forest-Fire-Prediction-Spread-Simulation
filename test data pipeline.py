import os
import sys
import shutil
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_prep.synthetic_data_generator import (
    _fractal_terrain, _compute_slope_aspect, _generate_lulc,
    _fuel_load_from_lulc_ndvi, generate_all,
)
from data_prep.feature_stack import build_feature_stack


@pytest.fixture
def tmp_dirs():
    synth_dir = tempfile.mkdtemp()
    proc_dir = tempfile.mkdtemp()
    yield synth_dir, proc_dir
    shutil.rmtree(synth_dir, ignore_errors=True)
    shutil.rmtree(proc_dir, ignore_errors=True)


def test_fractal_terrain_shape_and_range():
    rng = np.random.default_rng(0)
    dem = _fractal_terrain(64, 64, rng)
    assert dem.shape == (64, 64)
    assert dem.min() >= 0  # elevation normalized to start at 0
    assert np.isfinite(dem).all()


def test_slope_aspect_derivation():
    rng = np.random.default_rng(0)
    dem = _fractal_terrain(64, 64, rng)
    slope, aspect = _compute_slope_aspect(dem, cell_size_m=30)
    assert slope.shape == dem.shape
    assert aspect.shape == dem.shape
    assert slope.min() >= 0
    assert (aspect >= 0).all() and (aspect <= 360).all()


def test_flat_terrain_has_zero_slope():
    dem = np.full((20, 20), 500.0, dtype=np.float32)
    slope, _ = _compute_slope_aspect(dem, cell_size_m=30)
    assert np.allclose(slope, 0.0, atol=1e-5)


def test_lulc_classes_in_valid_range():
    rng = np.random.default_rng(0)
    dem = _fractal_terrain(50, 50, rng)
    slope, _ = _compute_slope_aspect(dem, 30)
    lulc = _generate_lulc(dem, slope, rng, 50, 50)
    assert lulc.min() >= 0 and lulc.max() <= 7
    assert lulc.dtype == np.uint8


def test_fuel_load_bounds():
    lulc = np.array([[0, 1, 2, 7]], dtype=np.uint8)
    ndvi = np.array([[0.0, 0.0, 0.0, 0.9]], dtype=np.float32)
    fuel = _fuel_load_from_lulc_ndvi(lulc, ndvi)
    assert fuel.min() >= 0 and fuel.max() <= 1
    assert fuel[0, 0] == 0.0  # water -> zero fuel
    assert fuel[0, 3] > fuel[0, 0]  # dense forest > water


def test_generate_all_creates_expected_files(tmp_dirs):
    synth_dir, _ = tmp_dirs
    config = {
        "region": {"resolution_m": 30, "crs": "EPSG:32644"},
        "synthetic_data": {
            "grid_height": 48, "grid_width": 48, "n_historical_days": 10,
            "fire_season_doy_range": [60, 165], "random_seed": 7,
        },
    }
    generate_all(config, synth_dir)
    expected_files = [
        "dem.tif", "slope.tif", "aspect.tif", "lulc.tif", "ndvi.tif",
        "fuel_load.tif", "temperature.tif", "rh.tif", "wind_speed.tif",
        "wind_dir.tif", "rainfall.tif", "dist_to_road.tif", "fire_history.tif",
        "fire_history_timeseries.npy", "metadata.json",
    ]
    for fname in expected_files:
        assert os.path.exists(os.path.join(synth_dir, fname)), f"Missing {fname}"


def test_feature_stack_shape_and_label_consistency(tmp_dirs):
    synth_dir, proc_dir = tmp_dirs
    config = {
        "region": {"resolution_m": 30, "crs": "EPSG:32644"},
        "synthetic_data": {
            "grid_height": 48, "grid_width": 48, "n_historical_days": 10,
            "fire_season_doy_range": [60, 165], "random_seed": 7,
        },
    }
    generate_all(config, synth_dir)
    stack, label, manifest = build_feature_stack(synth_dir, proc_dir)

    assert stack.shape[1:] == (48, 48)
    assert stack.shape[0] == manifest["n_channels"]
    assert label.shape == (48, 48)
    assert set(np.unique(label)).issubset({0, 1, 2, 3})
    assert os.path.exists(os.path.join(proc_dir, "feature_stack.npy"))
    assert os.path.exists(os.path.join(proc_dir, "label.npy"))
    assert os.path.exists(os.path.join(proc_dir, "feature_manifest.json"))


def test_feature_stack_is_finite(tmp_dirs):
    synth_dir, proc_dir = tmp_dirs
    config = {
        "region": {"resolution_m": 30, "crs": "EPSG:32644"},
        "synthetic_data": {
            "grid_height": 32, "grid_width": 32, "n_historical_days": 5,
            "fire_season_doy_range": [60, 165], "random_seed": 1,
        },
    }
    generate_all(config, synth_dir)
    stack, _, _ = build_feature_stack(synth_dir, proc_dir)
    assert np.isfinite(stack).all(), "Feature stack must not contain NaN/Inf"
