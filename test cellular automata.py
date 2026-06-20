import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulation.cellular_automata import FireSpreadCA, moisture_from_weather, UNBURNED, BURNING, BURNED, NONBURNABLE


@pytest.fixture
def simple_grid():
    h, w = 40, 40
    fuel = np.full((h, w), 0.7, dtype=np.float32)
    slope = np.zeros((h, w), dtype=np.float32)
    aspect = np.zeros((h, w), dtype=np.float32)
    wind_speed = np.full((h, w), 15.0, dtype=np.float32)
    wind_dir = np.full((h, w), 270.0, dtype=np.float32)  # blowing from west
    moisture = np.full((h, w), 0.1, dtype=np.float32)
    lulc = np.full((h, w), 7, dtype=np.uint8)  # all dense forest (burnable)
    return dict(fuel=fuel, slope=slope, aspect=aspect, wind_speed=wind_speed,
                wind_dir=wind_dir, moisture=moisture, lulc=lulc, h=h, w=w)


def test_ignition_sets_burning_state(simple_grid):
    g = simple_grid
    ca = FireSpreadCA(g["fuel"], g["slope"], g["aspect"], g["wind_speed"], g["wind_dir"],
                       g["moisture"], g["lulc"], cell_size_m=30,
                       config={"base_spread_probability": 0.2})
    ca.ignite([(20, 20)])
    assert ca.state[20, 20] == BURNING
    assert ca.burn_timer[20, 20] > 0


def test_nonburnable_cells_never_ignite(simple_grid):
    g = simple_grid
    lulc = g["lulc"].copy()
    lulc[10:30, 10:30] = 0  # water body in the middle
    ca = FireSpreadCA(g["fuel"], g["slope"], g["aspect"], g["wind_speed"], g["wind_dir"],
                       g["moisture"], lulc, cell_size_m=30,
                       config={"base_spread_probability": 0.9, "spotting_enabled": False})
    assert (ca.state[10:30, 10:30] == NONBURNABLE).all()

    ca.ignite([(20, 20)])  # attempting to ignite inside water -> should be ignored
    assert ca.state[20, 20] == NONBURNABLE

    rng = np.random.default_rng(0)
    for _ in range(20):
        ca.step(rng)
    assert (ca.state[10:30, 10:30] == NONBURNABLE).all()


def test_fire_spreads_over_time(simple_grid):
    g = simple_grid
    ca = FireSpreadCA(g["fuel"], g["slope"], g["aspect"], g["wind_speed"], g["wind_dir"],
                       g["moisture"], g["lulc"], cell_size_m=30,
                       config={"base_spread_probability": 0.3, "wind_effect_factor": 0.3,
                               "slope_effect_factor": 0.0, "moisture_suppression_factor": 0.3,
                               "spotting_enabled": False},
                       burn_duration_steps=8)
    ca.ignite([(20, 20)])
    rng = np.random.default_rng(1)
    areas = [ca.burned_area_ha()]
    for _ in range(12):
        ca.step(rng)
        areas.append(ca.burned_area_ha())

    assert areas[-1] > areas[0], "Fire should grow from a single ignition point given sufficient spread probability"
    assert np.all(np.diff(areas) >= 0), "Burned+burning area must be monotonically non-decreasing"


def test_burn_duration_scales_with_fuel(simple_grid):
    g = simple_grid
    fuel = g["fuel"].copy()
    fuel[:, :20] = 0.1   # low fuel half
    fuel[:, 20:] = 1.0   # high fuel half
    ca = FireSpreadCA(fuel, g["slope"], g["aspect"], g["wind_speed"], g["wind_dir"],
                       g["moisture"], g["lulc"], cell_size_m=30, burn_duration_steps=10)
    low_fuel_duration = ca._initial_burn_duration(5, 5)
    high_fuel_duration = ca._initial_burn_duration(5, 25)
    assert high_fuel_duration > low_fuel_duration


def test_wind_alignment_biases_spread_direction(simple_grid):
    """With strong westerly wind (blowing toward the east), the eastward
    bearing (90 deg) should get a spread-probability boost relative to the
    westward bearing (270 deg)."""
    g = simple_grid
    ca = FireSpreadCA(g["fuel"], g["slope"], g["aspect"], g["wind_speed"], g["wind_dir"],
                       g["moisture"], g["lulc"], cell_size_m=30,
                       config={"wind_effect_factor": 0.5})
    east_mult = ca._wind_alignment_factor(90)
    west_mult = ca._wind_alignment_factor(270)
    assert np.all(east_mult > west_mult)


def test_moisture_from_weather_bounds():
    rh = np.array([[0, 50, 100]], dtype=np.float32)
    rainfall = np.array([[0, 10, 50]], dtype=np.float32)
    m = moisture_from_weather(rh, rainfall)
    assert m.min() >= 0.0 and m.max() <= 1.0
    assert m[0, 2] > m[0, 0], "Higher RH/rainfall should yield higher moisture suppression"


def test_run_terminates(simple_grid):
    g = simple_grid
    ca = FireSpreadCA(g["fuel"], g["slope"], g["aspect"], g["wind_speed"], g["wind_dir"],
                       g["moisture"], g["lulc"], cell_size_m=30,
                       config={"base_spread_probability": 0.0, "spotting_enabled": False},
                       burn_duration_steps=2)
    ca.ignite([(20, 20)])
    history = ca.run(n_steps=50, seed=0)
    # With zero spread probability, fire should burn out and the run loop should break early
    assert len(history) <= 51
    assert not (ca.state == BURNING).any()
