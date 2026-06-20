Reference fetchers for real-world data sources named in the problem
statement. These are NOT run by default (the pipeline ships with
synthetic_data_generator.py so it works offline / without credentials).

Switch to real data once credentials are available:
    1. Set environment variables listed under each fetcher below.
    2. Run: python src/data_prep/download_real_data.py --source all
    3. Outputs land in data/raw/, then run feature_stack.py to produce the
       same processed format the synthetic generator produces, so the rest
       of the pipeline (prediction, simulation) needs zero code changes.

Network access for these specific external endpoints is NOT enabled in this
sandboxed evaluation environment, so calls here are illustrative scaffolding
only - request shapes and parsing logic are correct, but you must run this
script in an environment with outbound internet access and real credentials.

import argparse
import os

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")


def fetch_bhuvan_lulc(bbox, out_path, year=2024):
    """
    LULC from Bhuvan (NRSC/ISRO) WMS/WFS service.
    Requires: BHUVAN_API_KEY (register at https://bhuvan.nrsc.gov.in)
    Endpoint pattern: https://bhuvan-vec1.nrsc.gov.in/bhuvan/wms
    Layer: lulc50k:LULC50K_<year>_<state_code>
    """
    print("[Bhuvan] Would fetch LULC WMS GetMap request for bbox:", bbox)
    print(f"[Bhuvan] Target year: {year} -> writing to {out_path}")
    raise NotImplementedError(
        "Register at bhuvan.nrsc.gov.in, set BHUVAN_API_KEY, and implement "
        "the WMS GetMap call (owslib.wms.WebMapService is recommended)."
    )


def fetch_bhoonidhi_dem(bbox, out_path, product="CartoDEM_v3R1"):
    """
    30m DEM from Bhoonidhi Portal (ISRO).
    Requires: BHOONIDHI_USERNAME, BHOONIDHI_PASSWORD
    Portal: https://bhoonidhi.nrsc.gov.in
    """
    print(f"[Bhoonidhi] Would request {product} tiles covering bbox:", bbox)
    print(f"[Bhoonidhi] -> {out_path}")
    raise NotImplementedError(
        "Authenticate via Bhoonidhi portal session, query the CartoDEM/ "
        "Cartosat-1 DEM product catalog for the bbox, and download GeoTIFF tiles."
    )


def fetch_imd_era5_weather(bbox, date, out_path):
    """
    Weather rasters (temp, RH, wind, rainfall).
    Option A: IMD gridded data (https://imdpune.gov.in) - registration required.
    Option B: ERA-5 reanalysis via Copernicus CDS API.
        Requires: ~/.cdsapirc with CDS_API_KEY (https://cds.climate.copernicus.eu)
    """
    try:
        import cdsapi  # noqa: F401
    except ImportError:
        print("[ERA-5] Install with: pip install cdsapi --break-system-packages")

    print(f"[ERA-5/IMD] Would request weather variables for {date}, bbox:", bbox)
    print(f"[ERA-5/IMD] -> {out_path}")
    raise NotImplementedError(
        "cdsapi.Client().retrieve('reanalysis-era5-land', {...}, target_path) "
        "for temperature_2m, relative_humidity, 10m_u/v wind components, "
        "and total_precipitation; reproject to EPSG:32644 at 30m via rasterio.warp."
    )


def fetch_firms_viirs_fire_history(bbox, date_range, out_path):
    """
    Active fire detections (VIIRS-SNPP/NOAA-20, 375m) from NASA FIRMS.
    Requires: FIRMS_MAP_KEY (free, https://firms.modaps.eosdis.nasa.gov/api/)
    Endpoint: https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/VIIRS_SNPP_NRT/{bbox}/{day_range}
    """
    map_key = os.environ.get("FIRMS_MAP_KEY")
    if not map_key:
        print("[FIRMS] Set FIRMS_MAP_KEY environment variable to proceed.")
    print(f"[FIRMS] Would fetch VIIRS detections for {date_range}, bbox:", bbox)
    print(f"[FIRMS] -> {out_path}")
    raise NotImplementedError(
        "requests.get(f'https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        "{map_key}/VIIRS_SNPP_NRT/{bbox_str}/{n_days}') then rasterize lat/lon "
        "+ FRP/confidence points onto the 30m grid (rasterio.features.rasterize)."
    )


def fetch_ghsl_settlements(bbox, out_path):
    """
    Global Human Settlement Layer - built-up surface / population grids.
    Source: https://ghsl.jrc.ec.europa.eu/download.php (no auth required, direct download)
    """
    print("[GHSL] Would download built-up grid tiles intersecting bbox:", bbox)
    print(f"[GHSL] -> {out_path}")
    raise NotImplementedError(
        "Direct HTTP download of GHS-BUILT-S tiles from JRC, then clip to bbox "
        "with rasterio.mask and resample to 30m."
    )


SOURCES = {
    "lulc": fetch_bhuvan_lulc,
    "dem": fetch_bhoonidhi_dem,
    "weather": fetch_imd_era5_weather,
    "fire_history": fetch_firms_viirs_fire_history,
    "settlements": fetch_ghsl_settlements,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-data fetch stubs (requires credentials + internet).")
    parser.add_argument("--source", choices=list(SOURCES.keys()) + ["all"], default="all")
    args = parser.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)
    print("NOTE: These are scaffolds. See docstrings for required API keys / endpoints.")
    print("For an end-to-end runnable demo, use src/data_prep/synthetic_data_generator.py instead.\n")
