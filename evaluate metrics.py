Evaluation utilities for both the prediction map (classification metrics
live alongside training in train_unet.py::compute_metrics) and the spread
simulation, per the problem statement's evaluation parameters:
"Accuracy of prediction maps and fidelity of the spread simulation models
will be the key evaluation metrics."

Spread-fidelity metrics implemented here:
    - Jaccard similarity (IoU) between simulated burned-area mask and a
      reference/ground-truth mask (e.g. a held-out VIIRS detection mask for
      validation against real fire events).
    - Boundary displacement error (mean distance, in metres, between
      simulated and reference fire-front boundaries).
    - Area growth-rate error (%), comparing simulated vs reference burned
      area growth over time.
import numpy as np
from scipy.ndimage import binary_erosion


def jaccard_similarity(sim_mask, ref_mask):
    sim_mask = sim_mask.astype(bool)
    ref_mask = ref_mask.astype(bool)
    intersection = np.logical_and(sim_mask, ref_mask).sum()
    union = np.logical_or(sim_mask, ref_mask).sum()
    return float(intersection / union) if union > 0 else 1.0


def _boundary_pixels(mask):
    mask = mask.astype(bool)
    eroded = binary_erosion(mask)
    return mask & ~eroded


def boundary_displacement_error_m(sim_mask, ref_mask, cell_size_m=30):
    """Mean nearest-neighbor distance (metres) from each simulated boundary
    pixel to the nearest reference boundary pixel, and vice versa (symmetric)."""
    from scipy.ndimage import distance_transform_edt

    sim_b = _boundary_pixels(sim_mask)
    ref_b = _boundary_pixels(ref_mask)
    if not sim_b.any() or not ref_b.any():
        return float("nan")

    dist_to_ref = distance_transform_edt(~ref_b) * cell_size_m
    dist_to_sim = distance_transform_edt(~sim_b) * cell_size_m

    d1 = dist_to_ref[sim_b].mean()
    d2 = dist_to_sim[ref_b].mean()
    return float((d1 + d2) / 2.0)


def area_growth_rate_error_pct(sim_areas_ha, ref_areas_ha):
    """sim_areas_ha, ref_areas_ha: equal-length sequences of cumulative
    burned area (ha) at matching timestamps. Returns mean absolute
    percentage error of growth rate (finite differences)."""
    sim = np.asarray(sim_areas_ha, dtype=np.float64)
    ref = np.asarray(ref_areas_ha, dtype=np.float64)
    if len(sim) != len(ref) or len(sim) < 2:
        raise ValueError("sim_areas_ha and ref_areas_ha must be equal-length sequences of length >= 2")

    sim_rate = np.diff(sim)
    ref_rate = np.diff(ref)
    denom = np.where(np.abs(ref_rate) < 1e-6, 1e-6, ref_rate)
    pct_errors = np.abs((sim_rate - ref_rate) / denom) * 100.0
    return float(np.mean(pct_errors))


def evaluate_spread_fidelity(sim_mask, ref_mask, sim_area_series=None, ref_area_series=None, cell_size_m=30):
    result = {
        "jaccard_similarity": jaccard_similarity(sim_mask, ref_mask),
        "boundary_displacement_error_m": boundary_displacement_error_m(sim_mask, ref_mask, cell_size_m),
    }
    if sim_area_series is not None and ref_area_series is not None:
        result["area_growth_rate_error_pct"] = area_growth_rate_error_pct(sim_area_series, ref_area_series)
    return result
