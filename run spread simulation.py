Drives the Cellular Automata fire spread engine starting from high-risk
zones identified by the U-NET prediction map (Objective 1 -> Objective 2
handoff, exactly as specified: "Simulate the spread of fire within 1, 2, 3,
6, and 12 hours from high-risk zones identified in Objective 1.")

Produces, for each duration in config.spread_simulation.durations_hours:
    outputs/maps/fire_spread_<H>hr.tif    - 30m GeoTIFF (0 unburned/1 burning/2 burned)
    outputs/maps/fire_spread_<H>hr.png    - styled snapshot
And a combined:
    outputs/animations/fire_spread_animation.gif   - hourly animation, 1-12h
    outputs/maps/spread_growth_curve.png           - burned area (ha) vs time

import os
import json
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.animation as animation
import yaml

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulation.cellular_automata import FireSpreadCA, moisture_from_weather, UNBURNED, BURNING, BURNED, NONBURNABLE

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..")

STATE_COLORS = ["#3CB44B", "#E6342A", "#3A3A3A", "#BEBEBE"]  # unburned green, burning red, burned charcoal, nonburnable grey
STATE_LABELS = ["Unburned", "Burning", "Burned", "Non-burnable"]


def load_config():
    with open(os.path.join(ROOT, "configs", "config.yaml")) as f:
        return yaml.safe_load(f)


def _read(path):
    with rasterio.open(path) as src:
        return src.read(1), src.transform, src.crs


def select_ignition_points(risk_class, n_points=3, min_separation=20, seed=42):
    """Picks high-risk-zone ignition seeds (class >= 2, i.e. moderate/high),
    matching Objective 2's instruction to ignite 'from high-risk zones
    identified in Objective 1'. Mirrors the reference figure's pattern of
    3 simultaneous ignition points across the region."""
    rng = np.random.default_rng(seed)
    high_risk = np.argwhere(risk_class >= 2)
    if len(high_risk) == 0:
        high_risk = np.argwhere(risk_class >= 1)
    if len(high_risk) == 0:
        h, w = risk_class.shape
        return [(h // 2, w // 2)]

    chosen = []
    shuffled = high_risk[rng.permutation(len(high_risk))]
    for r, c in shuffled:
        if all(np.hypot(r - cr, c - cc) >= min_separation for cr, cc in chosen):
            chosen.append((int(r), int(c)))
        if len(chosen) >= n_points:
            break
    if not chosen:
        idx = rng.integers(0, len(high_risk))
        chosen = [tuple(high_risk[idx])]
    return chosen


def state_to_rgba(state):
    cmap = ListedColormap(STATE_COLORS)
    bounds_map = np.zeros_like(state, dtype=np.uint8)
    bounds_map[state == UNBURNED] = 0
    bounds_map[state == BURNING] = 1
    bounds_map[state == BURNED] = 2
    bounds_map[state == NONBURNABLE] = 3
    return bounds_map, cmap


def save_snapshot(state, out_path, title):
    bounds_map, cmap = state_to_rgba(state)
    norm = BoundaryNorm([0, 1, 2, 3, 4], cmap.N)
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(bounds_map, cmap=cmap, norm=norm)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, ticks=[0.5, 1.5, 2.5, 3.5], fraction=0.04, pad=0.03)
    cbar.ax.set_yticklabels(STATE_LABELS)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_geotiff(array, transform, crs, out_path):
    with rasterio.open(
        out_path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1],
        count=1, dtype="uint8", crs=crs, transform=transform, nodata=None, compress="lzw",
    ) as dst:
        dst.write(array.astype(np.uint8), 1)


def run_simulation(config):
    sim_cfg = config["spread_simulation"]
    synthetic_dir = os.path.join(ROOT, config["paths"]["synthetic_dir"])
    maps_dir = os.path.join(ROOT, config["paths"]["maps_dir"])
    anim_dir = os.path.join(ROOT, config["paths"]["animations_dir"])
    os.makedirs(maps_dir, exist_ok=True)
    os.makedirs(anim_dir, exist_ok=True)

    fuel, transform, crs = _read(os.path.join(synthetic_dir, "fuel_load.tif"))
    slope, _, _ = _read(os.path.join(synthetic_dir, "slope.tif"))
    aspect, _, _ = _read(os.path.join(synthetic_dir, "aspect.tif"))
    wind_speed, _, _ = _read(os.path.join(synthetic_dir, "wind_speed.tif"))
    wind_dir, _, _ = _read(os.path.join(synthetic_dir, "wind_dir.tif"))
    rh, _, _ = _read(os.path.join(synthetic_dir, "rh.tif"))
    rainfall, _, _ = _read(os.path.join(synthetic_dir, "rainfall.tif"))
    lulc, _, _ = _read(os.path.join(synthetic_dir, "lulc.tif"))

    risk_path = os.path.join(maps_dir, "fire_risk_prediction.tif")
    if os.path.exists(risk_path):
        risk_class, _, _ = _read(risk_path)
    else:
        print("WARNING: no trained-model prediction found; using danger_index.npy as a stand-in for ignition seeding.")
        danger = np.load(os.path.join(ROOT, config["paths"]["processed_dir"], "danger_index.npy"))
        risk_class = np.digitize(danger, bins=[0.25, 0.45, 0.65]).astype(np.uint8)

    moisture = moisture_from_weather(rh, rainfall)

    ca = FireSpreadCA(
        fuel_load=fuel, slope_deg=slope, aspect_deg=aspect,
        wind_speed_kmph=wind_speed, wind_dir_deg=wind_dir,
        moisture_suppression=moisture, lulc=lulc,
        cell_size_m=sim_cfg["cell_size_m"], config=sim_cfg,
        burn_duration_steps=sim_cfg.get("burn_duration_steps", 6),
    )

    ignition_points = select_ignition_points(risk_class, n_points=3, seed=config["synthetic_data"]["random_seed"])
    print(f"Ignition points (high-risk zones from Objective 1): {ignition_points}")
    ca.ignite(ignition_points)

    steps_per_hour = 60 // sim_cfg["time_step_minutes"]
    durations = sorted(sim_cfg["durations_hours"])
    max_hours = max(durations)
    total_steps = max_hours * steps_per_hour

    snapshots = {0: ca.state.copy()}
    burned_area_curve = [(0.0, ca.burned_area_ha())]
    rng = np.random.default_rng(config["synthetic_data"]["random_seed"])

    print(f"Running CA: {total_steps} steps ({steps_per_hour} steps/hour x {max_hours} hours)...")
    for step_i in range(1, total_steps + 1):
        ca.step(rng)
        hour = step_i / steps_per_hour
        if step_i % steps_per_hour == 0:
            snapshots[int(hour)] = ca.state.copy()
            burned_area_curve.append((hour, ca.burned_area_ha()))
            print(f"  Hour {int(hour):>2d}: burned/burning area = {ca.burned_area_ha():.1f} ha")
        else:
            burned_area_curve.append((hour, ca.burned_area_ha()))

    results = {}
    for h in durations:
        state = snapshots.get(h)
        if state is None:
            continue
        tif_path = os.path.join(maps_dir, f"fire_spread_{h}hr.tif")
        png_path = os.path.join(maps_dir, f"fire_spread_{h}hr.png")
        save_geotiff(state, transform, crs, tif_path)
        save_snapshot(state, png_path, f"Forest Fire Spread - Hour {h}")
        area_ha = np.isin(state, [BURNING, BURNED]).sum() * (sim_cfg["cell_size_m"] ** 2) / 10000.0
        results[h] = {"burned_area_ha": float(area_ha), "n_ignition_points": len(ignition_points)}

    # Growth curve
    hours_arr = np.array([t for t, _ in burned_area_curve])
    area_arr = np.array([a for _, a in burned_area_curve])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(hours_arr, area_arr, color="#E6342A", linewidth=2)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Burned + Burning Area (hectares)")
    ax.set_title("Simulated Fire Spread Growth Curve")
    ax.grid(alpha=0.3)
    for h in durations:
        ax.axvline(h, color="grey", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(maps_dir, "spread_growth_curve.png"), dpi=180)
    plt.close(fig)

    # Animation across all hourly snapshots (1..max_hours)
    print("Rendering hourly animation...")
    bounds_map_0, cmap = state_to_rgba(snapshots[0])
    norm = BoundaryNorm([0, 1, 2, 3, 4], cmap.N)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(bounds_map_0, cmap=cmap, norm=norm)
    title_text = ax.set_title("Forest Fire Spread - Hour 0")
    ax.set_xticks([]); ax.set_yticks([])

    hour_keys = sorted(snapshots.keys())

    def update(frame_idx):
        h = hour_keys[frame_idx]
        bmap, _ = state_to_rgba(snapshots[h])
        im.set_data(bmap)
        title_text.set_text(f"Forest Fire Spread - Hour {h}")
        return [im, title_text]

    anim = animation.FuncAnimation(fig, update, frames=len(hour_keys), interval=700, blit=False)
    gif_path = os.path.join(anim_dir, "fire_spread_animation.gif")
    anim.save(gif_path, writer="pillow", fps=1.4)
    plt.close(fig)

    with open(os.path.join(maps_dir, "spread_simulation_results.json"), "w") as f:
        json.dump({
            "ignition_points": ignition_points,
            "durations_results": results,
            "config_used": sim_cfg,
        }, f, indent=2)

    print(f"\nDone. GeoTIFFs + PNGs in {maps_dir}, animation at {gif_path}")
    return results


if __name__ == "__main__":
    cfg = load_config()
    run_simulation(cfg)
