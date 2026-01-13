# Mushroom Pipeline (Windows / VS Code)

Tools included:
- **inat**: download iNaturalist observations into per-taxon CSVs
- **nwm**: NOAA NWM LAND features (bbox geoparquet or enrich CSVs)
- **terrain**: AWS elevation tiles terrain *preview* (print-only; optional)

## Setup

PowerShell:
- `py -m venv .venv`
- `\.\.venv\Scripts\Activate.ps1`

Install one of:
- collector only: `pip install -r requirements/base.txt`
- NWM: `pip install -r requirements/nwm.txt`
- Terrain: `pip install -r requirements/terrain.txt`

## Run

### iNaturalist
- `python run.py inat --output-dir .\mushroom_data`

### NWM
- `python run.py nwm bbox --year 2023 --out-root .\nwm_land_out`
- `python run.py nwm enrich --mushroom-dir .\mushroom_data`



### Land cover (ESA WorldCover)
Install:
- `pip install -r requirements/landcover.txt`

Build WA 250 m land cover parquet (sharded):
- `set N_SHARDS=5`
- `set SHARD_ID=0`
- `python run.py landcover build-wa-250m --out-dir .\land_cover_data --step-m 250 --parquet-name wa_landcover_250m.parquet --year 2021`

Enrich mushroom CSVs in place (sharded):
- `set N_SHARDS=5`
- `set SHARD_ID=0`
- `python run.py landcover enrich-csv --mushroom-dir .\mushroom_data --year 2021`

Notes:
- ESA WorldCover is sampled from remote GeoTIFFs via HTTP range reads (no full-tile downloads).
- If you hit GDAL/HTTPS issues on Windows, try Conda for geo dependencies.

### Terrain previews (no disk writes)
WA bbox grid preview (~10 points):
- `python run.py terrain preview-wa --n-points 10 --step-m 20000`

Preview by sampling ~10 rows from first taxon CSV:
- `python run.py terrain preview-csv --mushroom-dir .\mushroom_data --n-rows 10`

Notes:
- Terrain uses AWS elevation tiles GeoTIFFs (Mapzen/tiles-prod endpoint).
- If Geo stack install is painful on Windows, use Conda for `requirements/nwm.txt` and `requirements/terrain.txt`.
