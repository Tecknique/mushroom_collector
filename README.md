# Mushroom Pipeline (Windows / VS Code)

Tools included:
- **inat**: download iNaturalist observations into per-taxon CSVs
- **nwm**: NOAA NWM LAND features (bbox geoparquet or enrich CSVs)
- **terrain**: AWS elevation tiles terrain (preview + build WA 250m parquet + enrich CSVs)

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

### Terrain previews (print-only)
- `python run.py terrain preview-wa --n-points 10 --step-m 20000`
- `python run.py terrain preview-csv --mushroom-dir .\mushroom_data --n-rows 10`

### Terrain production (writes output)
Build WA blanket 250m parquet (sharded by N_SHARDS/SHARD_ID):
- `set N_SHARDS=5` (PowerShell: `$env:N_SHARDS=5`)
- `set SHARD_ID=0`  (PowerShell: `$env:SHARD_ID=0`)
- `python run.py terrain build-wa-250m --out-dir .\terrain_data --step-m 250 --parquet-name wa_terrain_250m_v2.parquet`

Enrich mushroom CSVs in place (also sharded):
- `python run.py terrain enrich-csv --mushroom-dir .\mushroom_data`

Notes:
- Terrain uses AWS elevation tiles GeoTIFFs.
- Geo stack on Windows can be finicky; if pip wheels fail, use Conda.
