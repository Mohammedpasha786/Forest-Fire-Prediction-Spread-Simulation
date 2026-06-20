Generates a complete, self-consistent synthetic dataset that mimics the real
inputs described in the problem statement, so the full pipeline (feature
stacking -> U-NET training -> CA spread simulation) runs end-to-end without
any external downloads or API keys.

Outputs (all GeoTIFF, 30 m grid, EPSG:32644) under data/synthetic/:
    lulc.tif              - 8-class land use / land cover
    dem.tif                - elevation (m)
    slope.tif              - slope (degrees), derived from DEM
    aspect.tif             - aspect (degrees 0-360), derived from DEM
    ndvi.tif               - vegetation index proxy (fuel availability)
    temperature.tif        - air temperature (deg C), day-of-interest
    rh.tif                 - relative humidity (%)
    wind_speed.tif         - wind speed (km/h)
    wind_dir.tif           - wind direction (degrees, meteorological convention)
    rainfall.tif           - antecedent 7-day rainfall (mm)
    dist_to_road.tif       - distance to nearest road (m), human stressor proxy
    fire_history.tif       - binary VIIRS-style historical fire occurrence (label)
    fire_history_timeseries.npy - (n_days, H, W) daily fire occurrence for LSTM context

Real-data note: scripts/data_prep/download_real_data.py contains the
equivalent fetchers for Bhuvan, Bhoonidhi, IMD/ERA-5, and FIRMS/VIIRS. Swap
USE_SYNTHETIC = False in config.yaml once API credentials are available; the
rest of the pipeline (feature_stack.py onward) is agnostic to data origin
because it only reads from data/processed/.

import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter, distance_transform_edt
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "config.yaml")


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _save_raster(path, array, transform, crs, dtype="float32", nodata=None):
    array = array.astype(dtype)
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="lzw",
    ) as dst:
        dst.write(array, 1)


def _fractal_terrain(h, w, rng, roughness=0.55, base_scale=1800.0):
    """Diamond-square-like fractal terrain via successive Gaussian octaves.
    Produces a Himalayan-foothills-like elevation surface (valleys + ridgelines)."""
    elevation = np.zeros((h, w), dtype=np.float64)
    n_octaves = 6
    amplitude = base_scale
    for octave in range(n_octaves):
        sigma = max(1.0, (h / (2 ** (octave + 1))))
        noise = rng.normal(0, 1, size=(h, w))
        smoothed = gaussian_filter(noise, sigma=sigma)
        elevation += amplitude * smoothed / (np.std(smoothed) + 1e-6)
        amplitude *= roughness
    # Add a north-south ridge bias (Uttarakhand: Himalaya in north, Terai in south)
    row_grad = np.linspace(1.0, 0.15, h).reshape(-1, 1)
    elevation = elevation * row_grad + np.linspace(2200, 250, h).reshape(-1, 1)
    elevation -= elevation.min()
    return elevation.astype(np.float32)


def _compute_slope_aspect(dem, cell_size_m):
    """Horn's method for slope/aspect from a DEM array."""
    gy, gx = np.gradient(dem, cell_size_m)
    slope_rad = np.arctan(np.sqrt(gx ** 2 + gy ** 2))
    slope_deg = np.degrees(slope_rad)

    aspect_rad = np.arctan2(gy, -gx)
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = np.where(aspect_deg < 0, 90.0 - aspect_deg, 90.0 - aspect_deg)
    aspect_deg = np.mod(aspect_deg, 360.0)
    return slope_deg.astype(np.float32), aspect_deg.astype(np.float32)


def _generate_lulc(dem, slope, rng, h, w):
    """Elevation/slope-aware LULC classes (0-7):
    0 water, 1 settlement, 2 barren, 3 agriculture, 4 grassland,
    5 scrub, 6 forest_open, 7 forest_dense
    """
    lulc = np.full((h, w), 6, dtype=np.uint8)  # default open forest
    elev_norm = (dem - dem.min()) / (dem.max() - dem.min() + 1e-6)

    lulc[(elev_norm < 0.08) & (slope < 3)] = 0          # river valleys -> water
    lulc[(elev_norm < 0.25) & (slope < 5)] = 3          # low flat -> agriculture
    lulc[(elev_norm < 0.20) & (slope < 4)][rng.random(lulc[(elev_norm < 0.20) & (slope < 4)].shape) < 0.15] = 1

    dense_forest_mask = (elev_norm > 0.18) & (elev_norm < 0.65) & (slope > 8)
    lulc[dense_forest_mask] = 7

    grassland_mask = elev_norm > 0.80
    lulc[grassland_mask] = 4

    barren_mask = elev_norm > 0.92
    lulc[barren_mask] = 2

    # Sprinkle scrub patches at forest fringes (common fire-prone interface)
    scrub_noise = gaussian_filter(rng.random((h, w)), sigma=6)
    fringe = (lulc == 6) & (scrub_noise > 0.62)
    lulc[fringe] = 5

    # Scatter small settlements along low-slope river corridors
    settlement_noise = gaussian_filter(rng.random((h, w)), sigma=4)
    settlement_mask = (elev_norm < 0.30) & (slope < 6) & (settlement_noise > 0.85)
    lulc[settlement_mask] = 1

    return lulc


def _fuel_load_from_lulc_ndvi(lulc, ndvi):
    """Relative fuel availability per cell in [0,1], used by both U-NET feature
    stack and the CA spread model. Dense forest + high NDVI = highest fuel."""
    base_fuel = {0: 0.0, 1: 0.0, 2: 0.05, 3: 0.15, 4: 0.45, 5: 0.55, 6: 0.70, 7: 0.95}
    fuel = np.zeros(lulc.shape, dtype=np.float32)
    for cls, val in base_fuel.items():
        fuel[lulc == cls] = val
    fuel = np.clip(fuel * (0.5 + 0.5 * ndvi), 0, 1)
    return fuel.astype(np.float32)


def _generate_weather_fields(h, w, rng, doy):
    """Spatially smooth weather rasters consistent with a pre-monsoon fire-season day."""
    elev_factor = gaussian_filter(rng.normal(0, 1, (h, w)), sigma=25)

    base_temp = 34.0 - 6.0 * (doy < 100)  # slightly cooler early season
    temperature = base_temp + 4.0 * elev_factor + rng.normal(0, 0.5, (h, w))
    temperature = gaussian_filter(temperature, sigma=8).astype(np.float32)

    rh = 28.0 - 6.0 * elev_factor + rng.normal(0, 2, (h, w))
    rh = np.clip(gaussian_filter(rh, sigma=8), 8, 70).astype(np.float32)

    wind_speed_base = 14.0 + 5.0 * gaussian_filter(rng.normal(0, 1, (h, w)), sigma=30)
    wind_speed = np.clip(wind_speed_base, 2, 45).astype(np.float32)

    prevailing_dir = 290.0  # NW-erly pre-monsoon wind typical of foothill corridors
    wind_dir = (prevailing_dir + 25 * gaussian_filter(rng.normal(0, 1, (h, w)), sigma=40)) % 360
    wind_dir = wind_dir.astype(np.float32)

    rainfall = np.clip(
        gaussian_filter(rng.exponential(scale=1.2, size=(h, w)), sigma=10) - 1.0, 0, None
    ).astype(np.float32)

    return temperature, rh, wind_speed, wind_dir, rainfall


def _generate_fire_history(lulc, fuel, temperature, rh, wind_speed, slope, rng, h, w, n_days, fire_season_range, seed):
    """Synthetic VIIRS-style daily binary fire occurrence over n_days, used as
    (a) the U-NET training label for day 'today' and (b) LSTM temporal context."""
    timeseries = np.zeros((n_days, h, w), dtype=np.uint8)
    danger_index = (
        0.35 * fuel
        + 0.25 * (temperature - temperature.min()) / (temperature.max() - temperature.min() + 1e-6)
        + 0.20 * (1 - (rh - rh.min()) / (rh.max() - rh.min() + 1e-6))
        + 0.15 * (wind_speed - wind_speed.min()) / (wind_speed.max() - wind_speed.min() + 1e-6)
        + 0.05 * (slope - slope.min()) / (slope.max() - slope.min() + 1e-6)
    )
    danger_index = np.clip(danger_index, 0, 1)
    no_burn_mask = np.isin(lulc, [0, 1, 2])  # water, settlement, barren never ignite

    for day in range(n_days):
        in_season = fire_season_range[0] <= (day % 365) <= fire_season_range[1]
        season_mult = 1.0 if in_season else 0.05
        ignition_prob = danger_index * 0.004 * season_mult
        random_field = rng.random((h, w))
        ignitions = (random_field < ignition_prob) & (~no_burn_mask)

        if ignitions.sum() > 0:
            spread_mask = gaussian_filter(ignitions.astype(np.float32), sigma=2.5) > 0.08
            spread_mask &= danger_index > 0.3
            spread_mask &= ~no_burn_mask
            timeseries[day] = (ignitions | spread_mask).astype(np.uint8)

    return timeseries


def generate_all(config: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(config["synthetic_data"]["random_seed"])

    h = config["synthetic_data"]["grid_height"]
    w = config["synthetic_data"]["grid_width"]
    res = config["region"]["resolution_m"]
    crs = config["region"]["crs"]
    n_days = config["synthetic_data"]["n_historical_days"]
    season_range = config["synthetic_data"]["fire_season_doy_range"]

    origin_x, origin_y = 300000.0, 3450000.0  # arbitrary UTM44N anchor inside Uttarakhand extent
    transform = from_origin(origin_x, origin_y, res, res)

    print(f"[1/9] Generating fractal DEM ({h}x{w} @ {res}m)...")
    dem = _fractal_terrain(h, w, rng)
    _save_raster(os.path.join(out_dir, "dem.tif"), dem, transform, crs)

    print("[2/9] Deriving slope & aspect (Horn's method)...")
    slope, aspect = _compute_slope_aspect(dem, res)
    _save_raster(os.path.join(out_dir, "slope.tif"), slope, transform, crs)
    _save_raster(os.path.join(out_dir, "aspect.tif"), aspect, transform, crs)

    print("[3/9] Generating LULC (8-class)...")
    lulc = _generate_lulc(dem, slope, rng, h, w)
    _save_raster(os.path.join(out_dir, "lulc.tif"), lulc, transform, crs, dtype="uint8", nodata=255)

    print("[4/9] Generating NDVI / fuel proxy...")
    ndvi = np.clip(gaussian_filter(rng.normal(0.55, 0.18, (h, w)), sigma=5), 0, 1).astype(np.float32)
    ndvi[lulc == 0] = -0.1
    ndvi[lulc == 1] = 0.05
    ndvi[lulc == 2] = 0.05
    _save_raster(os.path.join(out_dir, "ndvi.tif"), ndvi, transform, crs)

    fuel = _fuel_load_from_lulc_ndvi(lulc, np.clip(ndvi, 0, 1))
    _save_raster(os.path.join(out_dir, "fuel_load.tif"), fuel, transform, crs)

    print("[5/9] Generating weather fields (today)...")
    today_doy = 110  # late April, peak fire season - matches Fig 6.8 reference date style
    temperature, rh, wind_speed, wind_dir, rainfall = _generate_weather_fields(h, w, rng, today_doy)
    _save_raster(os.path.join(out_dir, "temperature.tif"), temperature, transform, crs)
    _save_raster(os.path.join(out_dir, "rh.tif"), rh, transform, crs)
    _save_raster(os.path.join(out_dir, "wind_speed.tif"), wind_speed, transform, crs)
    _save_raster(os.path.join(out_dir, "wind_dir.tif"), wind_dir, transform, crs)
    _save_raster(os.path.join(out_dir, "rainfall.tif"), rainfall, transform, crs)

    print("[6/9] Generating distance-to-road (human stressor)...")
    road_seed = np.zeros((h, w), dtype=bool)
    n_roads = 4
    for _ in range(n_roads):
        r0 = rng.integers(0, h)
        for c in range(w):
            r0 = int(np.clip(r0 + rng.integers(-2, 3), 0, h - 1))
            road_seed[r0, c] = True
    dist_to_road = distance_transform_edt(~road_seed) * res
    _save_raster(os.path.join(out_dir, "dist_to_road.tif"), dist_to_road.astype(np.float32), transform, crs)

    print(f"[7/9] Generating {n_days}-day VIIRS-style fire history time series...")
    fire_ts = _generate_fire_history(
        lulc, fuel, temperature, rh, wind_speed, slope, rng, h, w, n_days, season_range,
        config["synthetic_data"]["random_seed"],
    )
    np.save(os.path.join(out_dir, "fire_history_timeseries.npy"), fire_ts)

    print("[8/9] Deriving 'today' fire occurrence label from time series...")
    fire_today = fire_ts[-1]
    _save_raster(os.path.join(out_dir, "fire_history.tif"), fire_today, transform, crs, dtype="uint8", nodata=255)

    print("[9/9] Writing metadata...")
    meta = {
        "grid_shape": [h, w],
        "resolution_m": res,
        "crs": crs,
        "transform": list(transform)[:6],
        "n_historical_days": n_days,
        "today_doy": today_doy,
        "lulc_legend": {
            "0": "water", "1": "settlement", "2": "barren", "3": "agriculture",
            "4": "grassland", "5": "scrub", "6": "forest_open", "7": "forest_dense",
        },
    }
    import json
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Synthetic dataset written to: {out_dir}")
    return meta


if __name__ == "__main__":
    cfg = load_config()
    out = os.path.join(os.path.dirname(__file__), "..", "..", cfg["paths"]["synthetic_dir"])
    generate_all(cfg, out)
