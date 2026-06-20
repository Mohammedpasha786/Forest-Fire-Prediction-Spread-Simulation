Aligns all raster layers (LULC, DEM-derived slope/aspect, NDVI/fuel, weather,
human-stressor distance) onto the common 30m grid and assembles the
multi-channel feature tensor consumed by the U-NET model.

This module is data-source agnostic: it reads whatever GeoTIFFs exist in
data/synthetic/ (or data/processed/ once populated by real-data fetchers)
and produces a single stacked array + channel manifest, so swapping synthetic
for real data requires no changes downstream.

Output:
    data/processed/feature_stack.npy   - shape (C, H, W) float32
    data/processed/feature_manifest.json - channel order + normalization stats
    data/processed/label.npy           - shape (H, W) uint8, 4-class risk label

import os
import json
import numpy as np
import rasterio


def _read(path):
    with rasterio.open(path) as src:
        return src.read(1), src.transform, src.crs


def _normalize(arr, lo=None, hi=None):
    lo = np.nanmin(arr) if lo is None else lo
    hi = np.nanmax(arr) if hi is None else hi
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32), lo, hi
    return ((arr - lo) / (hi - lo)).astype(np.float32), lo, hi


def build_feature_stack(data_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    lulc, transform, crs = _read(os.path.join(data_dir, "lulc.tif"))
    dem, _, _ = _read(os.path.join(data_dir, "dem.tif"))
    slope, _, _ = _read(os.path.join(data_dir, "slope.tif"))
    aspect, _, _ = _read(os.path.join(data_dir, "aspect.tif"))
    ndvi, _, _ = _read(os.path.join(data_dir, "ndvi.tif"))
    fuel, _, _ = _read(os.path.join(data_dir, "fuel_load.tif"))
    temperature, _, _ = _read(os.path.join(data_dir, "temperature.tif"))
    rh, _, _ = _read(os.path.join(data_dir, "rh.tif"))
    wind_speed, _, _ = _read(os.path.join(data_dir, "wind_speed.tif"))
    wind_dir, _, _ = _read(os.path.join(data_dir, "wind_dir.tif"))
    rainfall, _, _ = _read(os.path.join(data_dir, "rainfall.tif"))
    dist_to_road, _, _ = _read(os.path.join(data_dir, "dist_to_road.tif"))
    fire_today, _, _ = _read(os.path.join(data_dir, "fire_history.tif"))

    channels = {}
    manifest = {"channel_order": [], "stats": {}}

    def add_channel(name, arr, lo=None, hi=None):
        norm, lo_, hi_ = _normalize(arr.astype(np.float32), lo, hi)
        channels[name] = norm
        manifest["channel_order"].append(name)
        manifest["stats"][name] = {"min": float(lo_), "max": float(hi_)}

    add_channel("lulc_norm", lulc, 0, 7)
    add_channel("slope", slope, 0, 90)
    aspect_rad = np.deg2rad(aspect)
    channels["aspect_sin"] = np.sin(aspect_rad).astype(np.float32)
    channels["aspect_cos"] = np.cos(aspect_rad).astype(np.float32)
    manifest["channel_order"] += ["aspect_sin", "aspect_cos"]
    add_channel("ndvi", ndvi, -0.2, 1.0)
    add_channel("fuel_load", fuel, 0, 1)
    add_channel("temperature", temperature)
    add_channel("rh", rh)
    add_channel("wind_speed", wind_speed)
    wind_rad = np.deg2rad(wind_dir)
    channels["wind_dir_sin"] = np.sin(wind_rad).astype(np.float32)
    channels["wind_dir_cos"] = np.cos(wind_rad).astype(np.float32)
    manifest["channel_order"] += ["wind_dir_sin", "wind_dir_cos"]
    add_channel("rainfall", rainfall)
    add_channel("dist_to_road", dist_to_road)

    ordered = [channels[name] for name in manifest["channel_order"]]
    stack = np.stack(ordered, axis=0).astype(np.float32)  # (C, H, W)

    # Build 4-class label (0 nil/very_less, 1 low/less, 2 moderate, 3 high+very_high)
    # from the fuel/weather-driven danger index that also generated fire_today,
    # discretized so the U-NET learns a graded risk map rather than only the
    # binary occurrence pixel.
    danger = (
        0.35 * channels["fuel_load"]
        + 0.25 * channels["temperature"]
        + 0.20 * (1 - channels["rh"])
        + 0.15 * channels["wind_speed"]
        + 0.05 * channels["slope"]
    )
    danger = np.clip(danger, 0, 1)
    label = np.zeros_like(danger, dtype=np.uint8)
    label[(danger >= 0.25) & (danger < 0.45)] = 1
    label[(danger >= 0.45) & (danger < 0.65)] = 2
    label[danger >= 0.65] = 3
    no_burn_mask = lulc.astype(np.uint8) <= 2  # water/settlement/barren -> force nil
    label[no_burn_mask] = 0

    np.save(os.path.join(out_dir, "feature_stack.npy"), stack)
    np.save(os.path.join(out_dir, "label.npy"), label)
    np.save(os.path.join(out_dir, "danger_index.npy"), danger.astype(np.float32))

    manifest["grid_shape"] = list(lulc.shape)
    manifest["n_channels"] = stack.shape[0]
    manifest["crs"] = str(crs)
    manifest["transform"] = list(transform)[:6]
    manifest["label_classes"] = {"0": "nil/very_less", "1": "low/less", "2": "moderate", "3": "high/very_high"}
    with open(os.path.join(out_dir, "feature_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Feature stack: {stack.shape} ({stack.shape[0]} channels)")
    print(f"Label distribution: {dict(zip(*np.unique(label, return_counts=True)))}")
    print(f"Saved to {out_dir}")
    return stack, label, manifest


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    data_dir = os.path.join(here, "..", "..", "data", "synthetic")
    out_dir = os.path.join(here, "..", "..", "data", "processed")
    build_feature_stack(data_dir, out_dir)
