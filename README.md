# Mushroom Collector + NWM Land Featurizer (Windows / VS Code)

This repo contains two runnable tools:
- **inat**: downloads iNaturalist observation CSVs per mushroom taxon
- **nwm**: downloads NOAA NWM *land* analysis-assim files from Azure and either:
  - `bbox`: write daily WA-bbox geoparquet for a year (sharded by day)
  - `enrich`: enrich iNat CSV rows with NWM land features (sharded by rows)

## Setup (Windows / VS Code)

### 1) Create venv
PowerShell:
- `py -m venv .venv`
- `\.\.venv\Scripts\Activate.ps1`

### 2) Install deps

Collector only:
- `pip install -r requirements/base.txt`

NWM tool (heavier geo stack):
- `pip install -r requirements/nwm.txt`

> Geo wheels on Windows can be picky. If `pip` struggles with `rasterio/geopandas`,
> install the NWM stack in a Conda env. (I can give you a one-liner for that.)

## Run

### iNaturalist collector
- `python run.py inat --output-dir .\mushroom_data`

With sharding env vars:
- `$env:N_SHARDS="5"; $env:SHARD_ID="0"; $env:PARALLEL_WORKERS="4"; python run.py inat --output-dir .\mushroom_data`

### NWM land tool
BBOX mode (WA bbox, year 2023):
- `python run.py nwm bbox --year 2023 --out-root .\nwm_land_out`

Enrich mode (uses iNat output folder):
- `python run.py nwm enrich --mushroom-dir .\mushroom_data`

Disable progress bars:
- `python run.py nwm bbox --no-progress-bar --year 2023 --out-root .\nwm_land_out`

## Notes
- The original Colab `%pip` and Drive mount cells were removed; use venv + requirements instead.
- Azure NWM filesystem reads from the public `noaanwm` account.
