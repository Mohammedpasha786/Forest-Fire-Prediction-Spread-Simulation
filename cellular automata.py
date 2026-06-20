Probabilistic Cellular Automata (CA) fire spread engine.

Each cell transitions {unburned -> burning -> burned} based on neighbor
state and a per-edge spread probability computed from:
    P(spread i->j) = P_base * P_wind(i,j) * P_slope(i,j) * P_fuel(j) * P_moisture(j)

This follows the widely-used Rothermel-inspired CA formulation (e.g. the
approach used in FARSITE-style simplified models): wind alignment and
upslope direction both bias spread probability multiplicatively, fuel
availability scales it, and moisture (from rainfall/RH) suppresses it.

A spotting (ember-jump) mechanism is included: burning cells have a small
probability of igniting a non-adjacent cell downwind, capturing long-range
spotting behavior relevant to wind-driven Himalayan forest fires.

Grid encoding: 0 = unburned, 1 = burning, 2 = burned, 255 = non-burnable
(water/settlement/barren, taken from LULC).

import numpy as np

UNBURNED, BURNING, BURNED, NONBURNABLE = 0, 1, 2, 255

# Moore-8 neighborhood offsets and their compass bearing (deg, meteorological: 0=N, 90=E)
NEIGHBOR_OFFSETS = [
    (-1, 0, 0), (-1, 1, 45), (0, 1, 90), (1, 1, 135),
    (1, 0, 180), (1, -1, 225), (0, -1, 270), (-1, -1, 315),
]


class FireSpreadCA:
    def __init__(self, fuel_load, slope_deg, aspect_deg, wind_speed_kmph, wind_dir_deg,
                 moisture_suppression, lulc, cell_size_m=30, config=None, burn_duration_steps=3):
        """
        fuel_load, slope_deg, aspect_deg, wind_speed_kmph, wind_dir_deg,
        moisture_suppression: all (H, W) arrays on the same grid.
        lulc: (H, W) uint8 LULC class array (used to mark non-burnable cells).
        burn_duration_steps: number of CA steps a cell stays BURNING before
            transitioning to BURNED (fuel consumption). Higher-fuel cells get
            a longer duration (scaled by fuel_load), so dense forest burns
            longer than scrub/grassland.
        """
        self.h, self.w = fuel_load.shape
        self.cell_size_m = cell_size_m
        cfg = config or {}
        self.base_p = cfg.get("base_spread_probability", 0.06)
        self.wind_factor = cfg.get("wind_effect_factor", 0.35)
        self.slope_factor = cfg.get("slope_effect_factor", 0.25)
        self.fuel_factor = cfg.get("fuel_effect_factor", 0.30)
        self.moisture_factor = cfg.get("moisture_suppression_factor", 0.40)
        self.spotting_enabled = cfg.get("spotting_enabled", True)
        self.spotting_p = cfg.get("spotting_probability", 0.015)
        self.spotting_max_dist = cfg.get("spotting_max_distance_cells", 4)
        self.base_burn_duration = burn_duration_steps

        self.fuel = fuel_load.astype(np.float32)
        self.slope = slope_deg.astype(np.float32)
        self.aspect = aspect_deg.astype(np.float32)
        self.wind_speed = wind_speed_kmph.astype(np.float32)
        self.wind_dir = wind_dir_deg.astype(np.float32)
        self.moisture = moisture_suppression.astype(np.float32)

        self.state = np.full((self.h, self.w), UNBURNED, dtype=np.uint8)
        non_burnable_mask = np.isin(lulc, [0, 1, 2])  # water, settlement, barren
        self.state[non_burnable_mask] = NONBURNABLE
        # Remaining burn timer per cell (0 = not burning / not yet ignited)
        self.burn_timer = np.zeros((self.h, self.w), dtype=np.int32)
        self.history = []

    def _initial_burn_duration(self, r, c):
        return max(1, int(round(self.base_burn_duration * (0.5 + 0.5 * self.fuel[r, c]))))

    def ignite(self, points):
        """points: list of (row, col) tuples to set as initially burning."""
        for r, c in points:
            if 0 <= r < self.h and 0 <= c < self.w and self.state[r, c] != NONBURNABLE:
                self.state[r, c] = BURNING
                self.burn_timer[r, c] = self._initial_burn_duration(r, c)
        self.history.append(self.state.copy())

    def _wind_alignment_factor(self, bearing_deg):
        """Returns a multiplier in roughly [1-wind_factor, 1+2*wind_factor]:
        spread direction aligned with wind -> boosted; against wind -> suppressed."""
        wind_to_dir = (self.wind_dir + 180) % 360  # direction wind is blowing TOWARD
        angle_diff = np.abs(((wind_to_dir - bearing_deg + 180) % 360) - 180)
        alignment = np.cos(np.deg2rad(angle_diff))  # 1 = aligned, -1 = opposite
        speed_norm = np.clip(self.wind_speed / 40.0, 0, 1.5)
        return 1.0 + self.wind_factor * alignment * speed_norm

    def _slope_alignment_factor(self, bearing_deg, dr, dc):
        """Fire spreads faster upslope. Aspect = downslope-facing direction by
        convention here (matches _compute_slope_aspect in synthetic_data_generator),
        so upslope direction = aspect + 180."""
        upslope_dir = (self.aspect + 180) % 360
        angle_diff = np.abs(((upslope_dir - bearing_deg + 180) % 360) - 180)
        upslope_alignment = np.cos(np.deg2rad(angle_diff))
        slope_norm = np.clip(self.slope / 45.0, 0, 1.5)
        return 1.0 + self.slope_factor * upslope_alignment * slope_norm

    def step(self, rng):
        """Advance the CA by one time step. Returns True if any cell changed state.

        Order of operations each step:
          1. Currently-burning cells attempt to ignite UNBURNED neighbors
             (8-connected) with probability P(spread), plus long-range spotting.
          2. Newly-ignited cells get a fresh burn_timer.
          3. All currently-burning cells (from before this step) decrement
             their timer; any reaching 0 transition to BURNED.
        """
        burning_mask = self.state == BURNING
        if not burning_mask.any():
            return False

        burning_coords = np.argwhere(burning_mask)
        new_ignitions = np.zeros((self.h, self.w), dtype=bool)
        changed = False

        for dr, dc, bearing in NEIGHBOR_OFFSETS:
            wind_mult = self._wind_alignment_factor(bearing)
            slope_mult = self._slope_alignment_factor(bearing, dr, dc)

            shifted_burning = np.zeros_like(burning_mask)
            valid = (
                (burning_coords[:, 0] + dr >= 0) & (burning_coords[:, 0] + dr < self.h) &
                (burning_coords[:, 1] + dc >= 0) & (burning_coords[:, 1] + dc < self.w)
            )
            tgt_rows = burning_coords[valid, 0] + dr
            tgt_cols = burning_coords[valid, 1] + dc
            shifted_burning[tgt_rows, tgt_cols] = True

            eligible = shifted_burning & (self.state == UNBURNED) & (~new_ignitions)
            if not eligible.any():
                continue

            p_spread = (
                self.base_p
                * wind_mult
                * slope_mult
                * (0.2 + 0.8 * self.fuel)
                * (1.0 - self.moisture_factor * self.moisture)
            )
            p_spread = np.clip(p_spread, 0, 1)

            draws = rng.random(self.state.shape)
            ignites = eligible & (draws < p_spread)
            if ignites.any():
                new_ignitions |= ignites
                changed = True

        # Spotting: burning cells can jump-ignite downwind cells beyond adjacency
        if self.spotting_enabled and burning_coords.shape[0] > 0:
            n_attempts = max(1, int(burning_coords.shape[0] * self.spotting_p * 10))
            idxs = rng.integers(0, burning_coords.shape[0], size=n_attempts)
            for idx in idxs:
                r, c = burning_coords[idx]
                wind_to = np.deg2rad((self.wind_dir[r, c] + 180) % 360)
                jump = rng.integers(2, self.spotting_max_dist + 1)
                nr = int(round(r - jump * np.cos(wind_to)))
                nc = int(round(c + jump * np.sin(wind_to)))
                if (0 <= nr < self.h and 0 <= nc < self.w
                        and self.state[nr, nc] == UNBURNED and not new_ignitions[nr, nc]):
                    if rng.random() < self.spotting_p:
                        new_ignitions[nr, nc] = True
                        changed = True

        # Apply new ignitions
        if new_ignitions.any():
            self.state[new_ignitions] = BURNING
            rows, cols = np.where(new_ignitions)
            for r, c in zip(rows, cols):
                self.burn_timer[r, c] = self._initial_burn_duration(r, c)

        # Decrement timers for cells that were already burning before this step;
        # cells reaching 0 become BURNED (fuel exhausted)
        self.burn_timer[burning_mask] -= 1
        burned_out = burning_mask & (self.burn_timer <= 0)
        if burned_out.any():
            self.state[burned_out] = BURNED
            changed = True

        self.history.append(self.state.copy())
        return changed

    def run(self, n_steps, seed=42):
        rng = np.random.default_rng(seed)
        for _ in range(n_steps):
            any_change = self.step(rng)
            if not any_change and not (self.state == BURNING).any():
                break
        return self.history

    def burned_area_ha(self):
        burned_or_burning = np.isin(self.state, [BURNING, BURNED])
        n_cells = burned_or_burning.sum()
        return n_cells * (self.cell_size_m ** 2) / 10000.0  # m^2 -> hectares


def moisture_from_weather(rh, rainfall):
    """Combines RH and antecedent rainfall into a single suppression scalar in [0,1].
    Higher RH/rainfall -> higher suppression -> lower spread probability."""
    rh_norm = np.clip(rh / 100.0, 0, 1)
    rain_norm = np.clip(rainfall / 20.0, 0, 1)
    return np.clip(0.6 * rh_norm + 0.4 * rain_norm, 0, 1).astype(np.float32)
