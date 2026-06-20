End-to-end orchestrator. Running this single script reproduces the entire
project from scratch:

    1. Generate synthetic geospatial dataset (or skip if --use-existing-data)
    2. Build the aligned multi-channel feature stack + 4-class label
    3. Train the U-NET fire-risk prediction model
    4. Run full-scene inference -> next-day fire risk prediction map (Obj. 1)
    5. Seed the Cellular Automata from high-risk zones and simulate spread
       for 1/2/3/6/12 hours, with animation (Obj. 2)

Usage:
    python run_pipeline.py                  # full run
    python run_pipeline.py --skip-training  # reuse models/fire_unet_best.pt
    python run_pipeline.py --quick           # fewer epochs, for smoke-testing

import argparse
import os
import sys
import time
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from data_prep.synthetic_data_generator import load_config, generate_all
from data_prep.feature_stack import build_feature_stack
from prediction.train_unet import train as train_unet
from prediction.predict_risk_map import predict as predict_risk_map
from simulation.run_spread_simulation import run_simulation


def main():
    parser = argparse.ArgumentParser(description="Forest Fire Prediction & Spread Simulation - Full Pipeline")
    parser.add_argument("--skip-data-gen", action="store_true", help="Skip synthetic data generation (reuse data/synthetic/)")
    parser.add_argument("--skip-training", action="store_true", help="Skip U-NET training (reuse models/fire_unet_best.pt)")
    parser.add_argument("--quick", action="store_true", help="Smoke-test mode: smaller grid + fewer epochs")
    args = parser.parse_args()

    config = load_config()

    if args.quick:
        config["synthetic_data"]["grid_height"] = 256
        config["synthetic_data"]["grid_width"] = 256
        config["synthetic_data"]["n_historical_days"] = 20
        config["prediction_model"]["epochs"] = 3
        config["prediction_model"]["patch_size"] = 64
        config["prediction_model"]["patch_stride"] = 48
        config["spread_simulation"]["durations_hours"] = [1, 2]
        print(">>> QUICK MODE: reduced grid/epochs for smoke testing.\n")

    t0 = time.time()

    if not args.skip_data_gen:
        print("=" * 70)
        print("STAGE 1/4: Synthetic Data Generation")
        print("=" * 70)
        synth_dir = os.path.join(ROOT, config["paths"]["synthetic_dir"])
        generate_all(config, synth_dir)
    else:
        print("Skipping data generation (--skip-data-gen)")

    print("\n" + "=" * 70)
    print("STAGE 2/4: Feature Stacking")
    print("=" * 70)
    synth_dir = os.path.join(ROOT, config["paths"]["synthetic_dir"])
    processed_dir = os.path.join(ROOT, config["paths"]["processed_dir"])
    build_feature_stack(synth_dir, processed_dir)

    if not args.skip_training:
        print("\n" + "=" * 70)
        print("STAGE 3/4: U-NET Training (Fire Risk Prediction)")
        print("=" * 70)
        train_unet(config)
    else:
        print("Skipping training (--skip-training); reusing existing checkpoint.")

    print("\n" + "=" * 70)
    print("STAGE 4a/4: Full-Scene Inference -> Next-Day Risk Map (Objective 1)")
    print("=" * 70)
    predict_risk_map(config)

    print("\n" + "=" * 70)
    print("STAGE 4b/4: Cellular Automata Spread Simulation (Objective 2)")
    print("=" * 70)
    run_simulation(config)

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETE in {elapsed:.1f}s")
    print("=" * 70)
    print(f"Outputs:")
    print(f"  - Risk prediction map : outputs/maps/fire_risk_prediction.tif (+ .png)")
    print(f"  - Binary fire/no-fire : outputs/maps/fire_binary_prediction.tif")
    print(f"  - Spread snapshots    : outputs/maps/fire_spread_{{1,2,3,6,12}}hr.tif (+ .png)")
    print(f"  - Spread animation    : outputs/animations/fire_spread_animation.gif")
    print(f"  - Growth curve        : outputs/maps/spread_growth_curve.png")
    print(f"  - Test metrics        : outputs/maps/test_metrics.json")


if __name__ == "__main__":
    main()
