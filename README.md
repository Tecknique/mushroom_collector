# Mushroom Collector — Local (Windows / VS Code) Runner

This repo is a **data pipeline** that:
1) downloads **iNaturalist mushroom observations** into CSVs, then (optionally)
2) enriches those CSVs with geospatial features from:
   - **NOAA National Water Model (NWM) LAND** (daily soil/snow/ET features)
   - **Terrain** from AWS elevation tiles (DEM-derived variables)
   - **ESA WorldCover** land-cover class

Everything here runs **outside Colab** on any machine with Python + the geo stack installed.
All default paths are **relative to your repo folder**, so you can clone/run anywhere.

---

## Quick start (VS Code, Windows)

### 1) Create and activate a virtual environment
PowerShell:
```powershell
cd path\to\mushroom_collector
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Git Bash:
```bash
cd /c/Users/<you>/.../mushroom_collector
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
```

### 2) Install the dependencies you need
Choose the smallest set that matches what you’re running:

```bash
pip install -r requirements/base.txt        # iNaturalist collector only
pip install -r requirements/nwm.txt         # NWM LAND (xarray/netCDF + geo)
pip install -r requirements/terrain.txt     # DEM terrain features
pip install -r requirements/landcover.txt   # ESA WorldCover landcover
```

> If pip fails on Windows for `fiona/rasterio/geopandas`, use **Conda** (see “Troubleshooting”).

### 3) Run commands
All entrypoints are via:
```bash
python run.py --help
```

---

## Recommended folder layout (portable)

Inside the repo root:
```
mushroom_collector/
  mushroom_data/         # iNat outputs (default)
  nwm_land_out/          # NWM bbox outputs (default)
  terrain_data/          # terrain outputs (if you choose a custom dir)
  land_cover_data/       # landcover outputs (default)
  src/mushroom_collector/
  run.py
```

**Why this works anywhere:** all tools default to `Path.cwd()` + a relative folder.
If you clone this repo to a different computer, the pipeline still writes into these folders.

---

## Configuration (paths + sharding)

### Paths (override defaults)
You can pass paths as CLI flags, or set environment variables.

Common env vars:
- `OUTPUT_DIR` — iNat output dir (default: `./mushroom_data`)
- `MUSHROOM_DIR` — location of existing mushroom CSV folders (default: `./mushroom_data`)
- `NWM_OUT_ROOT` — NWM bbox output root (default: `./nwm_land_out`)

### Sharding (run multiple terminals)
Several steps are designed to be run in parallel using shards:
- `N_SHARDS` — how many shards total (e.g. `5`)
- `SHARD_ID` — which shard this process is running (`0..N_SHARDS-1`)

On Windows PowerShell:
```powershell
setx N_SHARDS 5
setx SHARD_ID 0
```

Or per-terminal (recommended):
```powershell
$env:N_SHARDS = 5
$env:SHARD_ID = 0
```

On Git Bash:
```bash
export N_SHARDS=5
export SHARD_ID=0
```

### VS Code env file (nice option)
If you use the VS Code Python extension, you can keep a `.env` file (local-only) and point your launch config to it.

Example `.env` (do not commit secrets):
```
N_SHARDS=5
SHARD_ID=0
PROGRESS_BAR=true
```

---

## 1) Download iNaturalist mushroom observations (CSV)

This creates one folder per taxon and writes `data.csv` inside each.

```bash
python run.py inat --help
python run.py inat
```

Common options:
```bash
python run.py inat --output-dir .\mushroom_data
python run.py inat --inat-token <OPTIONAL_TOKEN>
python run.py inat --progress-bar
```

**Output:**
- `mushroom_data/<taxon_name>/data.csv`
- `mushroom_data/MushroomDataCombined/data.csv`

---

## 2) NWM LAND features (NOAA NWM on Azure)

### A) Build a daily GeoParquet grid for a bbox (WA)
This downloads day-by-day NWM LAND netCDF files (cached locally) and writes a parquet per day.

```bash
python run.py nwm bbox --year 2023 --bbox -124.9 45.5 -117.0 49.05
```

Defaults:
- output root: `./nwm_land_out/geoparquet/date=YYYY-MM-DD/wa_nwm_daily.parquet`
- sharded by day using `N_SHARDS` / `SHARD_ID`

Run multiple shards in separate terminals:
```powershell
$env:N_SHARDS=5
$env:SHARD_ID=0
python run.py nwm bbox --year 2023
```

### B) Enrich iNat CSVs using NWM features (nearest grid cell, by date)
This reads each `mushroom_data/<taxon>/data.csv` and writes back NWM columns.

```bash
python run.py nwm enrich --mushroom-dir .\mushroom_data
```

Sharding here is by **row index** inside each taxon file; run multiple shards (`SHARD_ID=0..N_SHARDS-1`) and it auto-combines.

**What gets added:** columns like `soil_t_*`, `soil_m_*`, snow/ET variables, etc.

---

## 3) Terrain features (AWS elevation tiles)

Terrain uses Mapzen/AWS elevation GeoTIFF tiles, computes derivatives (slope/aspect/curvature/TRI/etc),
then samples values at points.

### Preview (print-only)
WA preview (samples a few random grid points, prints a table):
```bash
python run.py terrain preview-wa --n-points 10 --step-m 20000
```

CSV preview (samples ~10 rows from your mushroom CSVs and prints annotated rows):
```bash
python run.py terrain preview-csv --mushroom-dir .\mushroom_data --n-rows 10
```

> If you want “full WA 250m parquet build” + “CSV enrich in place”, those functions are implemented in
`src/mushroom_collector/terrain_wa.py` and can be exposed as CLI commands the same way as landcover if you decide to use them.

---

## 4) ESA WorldCover land cover (WA 250m parquet + CSV enrich)

### A) Build WA 250m land-cover parquet (sharded)
```bash
python run.py landcover build-wa-250m --out-dir .\land_cover_data --step-m 250 --year 2021
```

This writes shard parts:
- `land_cover_data/wa_landcover_250m.parquet.part0 ... part4`

When all parts exist, it auto-combines to:
- `land_cover_data/wa_landcover_250m.parquet`

### B) Enrich mushroom CSVs with land cover
```bash
python run.py landcover enrich-csv --mushroom-dir .\mushroom_data --year 2021
```

Adds columns:
- `lc_code, lc_name, lc_r, lc_g, lc_b, lc_source, lc_year`

---

## How the datasets “interact” (what joins to what)

Your mushroom observation rows are the **spine** dataset.

### Join keys / linkage
- **Location:** `lat` / `long`
- **Date:** `timestamp` (parsed to date for NWM matching)

Enrichers work like this:
- **NWM:** for each observation date, find the nearest NWM LAND grid cell and copy that day’s values onto the row.
- **Terrain:** sample terrain rasters at the observation point and copy values onto the row.
- **Landcover:** sample ESA WorldCover at the observation point and copy class info onto the row.

Because everything writes back into `mushroom_data/<taxon>/data.csv`, you can train models directly from those CSVs,
or combine them afterwards.

---

## Troubleshooting (Windows geo stack)

### If pip fails building geo packages
On Windows, `geopandas/rasterio/fiona` can be easiest via Conda.

Conda approach:
```bash
conda create -n mushrooms python=3.11 -y
conda activate mushrooms
conda install -c conda-forge geopandas rasterio fiona pyproj shapely pyarrow -y
pip install -r requirements/base.txt
pip install -r requirements/nwm.txt
```

### Slow downloads / HTTP issues
- Use fewer workers: `--workers 2`
- For NWM, the code caches netCDF files locally; reruns should be faster.

---

## Repo commands summary

```bash
python run.py inat
python run.py nwm bbox --year 2023
python run.py nwm enrich --mushroom-dir .\mushroom_data
python run.py terrain preview-wa
python run.py landcover build-wa-250m --year 2021
python run.py landcover enrich-csv --mushroom-dir .\mushroom_data --year 2021
```
