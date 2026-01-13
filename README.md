# Mushroom Collector + Environmental Featurizers (VS Code / Local)

This repo builds a **single, analysis-ready dataset** from multiple sources:

1) **iNaturalist mushroom observations** (point records)  
2) **NOAA National Water Model (NWM) LAND** daily land-surface features (≈1 km grid)  
3) **Terrain derivatives** from DEM tiles (≈10 m DEM → sampled onto a 250 m WA point grid)  
4) **ESA WorldCover land cover** (10 m categorical raster → sampled onto the same 250 m WA point grid or directly onto mushroom points)

You can run everything **outside Colab** on Windows/macOS/Linux in **VS Code** using standard Python tooling.

---

## Repository layout (portable paths)

This repo is designed to work on *any* computer by using paths relative to the repo root.

Recommended local folders (create as needed):

```
mushroom_collector/
  mushroom_data/          # output from iNat collector (CSV folders by taxon)
  nwm_data/               # output from NWM bbox mode (daily geoparquet)
  terrain_data/           # output from terrain 250m grid parquet
  land_cover_data/        # output from landcover 250m grid parquet
  .cache/                 # optional caches (tiles / netcdf)
  src/
  requirements/
  run.py
```

**Tip:** Put large outputs in these folders and keep them out of git via `.gitignore`.

---

## Quickstart (VS Code)

### 1) Create a virtual environment
Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

macOS/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2) Install dependencies (choose what you need)
Core collector only:
```bash
pip install -r requirements/base.txt
```

Add NWM pipeline:
```bash
pip install -r requirements/nwm.txt
```

Add Terrain pipeline:
```bash
pip install -r requirements/terrain.txt
```

Add LandCover pipeline:
```bash
pip install -r requirements/landcover.txt
```

> ⚠️ Windows note: `geopandas/rasterio/fiona` can be finicky with pip. If installation fails, use **Miniconda** and install geo libs from `conda-forge`, then `pip install -r requirements/*.txt` for the rest.

---

## How sharding works (recommended for big runs)

Most steps support **sharding** so you can run multiple terminals in parallel.

Environment variables:
- `N_SHARDS` (or `SHARD_COUNT` in some older scripts) — total shards, default 5
- `SHARD_ID` — which shard this terminal runs (0..N_SHARDS-1)
- `PARALLEL_WORKERS` / `WORKERS` — threads for downloads/computation

Example (Windows PowerShell):
```powershell
setx N_SHARDS 5
setx SHARD_ID 0
setx WORKERS 8
```

Then open 5 terminals and set `SHARD_ID` to 0,1,2,3,4.

Many pipelines write `part_*.csv` or `*.part*` outputs and **auto-combine** when all shards finish.

---

## What “pixel level” / “cell resolution” means in this project

This project mixes data at different native resolutions. **Unless stated otherwise, sampling is point-based**:
we take values at a grid cell center / nearest pixel — we do **not** compute area averages.

### A) iNaturalist observations
- **Type:** point records (lat/lon)
- **Resolution:** whatever the observation provides (not gridded)

### B) NWM LAND (NOAA NWM)
- **Type:** gridded model output
- **Native resolution:** approximately **1 km** (CONUS land analysis grid; varies slightly by projection/area)
- **What we compute:** for each day, we read up to 4 cycles (`00,06,12,18`) and compute daily stats.
- **How we sample:**
  - **bbox mode:** output a GeoParquet containing **one row per NWM grid cell center** inside a WA bbox per day
  - **enrich mode:** for each mushroom point on a date, attach values from the **nearest NWM grid cell**

**No interpolation; nearest/center sampling only.**

### C) Terrain features (DEM derivatives)
- **DEM source:** AWS/Mapzen terrain GeoTIFF tiles at zoom ~14 (≈10 m pixels mid-latitudes)
- **We compute:** elevation, slope/aspect, curvature, TRI, roughness, hillshade, etc. at DEM pixel scale
- **Final output grid:** **250 m spaced points** across WA (EPSG:5070)
- **Sampling meaning:** each 250 m point gets values sampled from the derived raster at that location.  
  It is **not** a 250 m averaged pixel.

### D) Land cover (ESA WorldCover)
- **Native resolution:** **10 m**
- **Sampling meaning:** we sample the land cover class at the point location (grid point or mushroom point).  
  It is **not** a majority class within a 250 m area (unless you extend the code to do that).

---

## Pipelines and how they interact

You can use either workflow:

### Workflow 1 (simple for ML): Enrich mushroom CSVs directly
Run:
1) Collect mushrooms → writes per-taxon `data.csv`
2) Enrich with NWM → writes additional `soil_*`, `snow_*`, `edir_*` columns into CSVs
3) Enrich with Terrain → writes terrain columns into CSVs
4) Enrich with LandCover → writes `lc_*` columns into CSVs

End result: each mushroom record becomes a single row with all environmental features.

### Workflow 2 (more scalable): Build statewide grids, join later
Run:
- Build WA terrain 250 m parquet
- Build WA landcover 250 m parquet
- Build NWM daily WA geoparquet (bbox mode)

Then spatially join mushroom points to nearest grid points (or use a GIS join).

---

## 1) iNaturalist Mushroom Collector

### What it does
Downloads iNaturalist observations for target taxa and writes:
- `mushroom_data/<taxon>/data.csv`
- `mushroom_data/MushroomDataCombined/data.csv`

### Run
```bash
python run.py collect
```

Configure via `.env` or environment variables (see `.env.example`):
- `OUTPUT_DIR` (default should be `./mushroom_data`)
- `INAT_API_TOKEN` (optional)
- `N_SHARDS`, `SHARD_ID`, `PARALLEL_WORKERS`

---

## 2) NWM LAND Featurizer (soil / snow / evap)

### Mode A: bbox mode (daily WA geoparquet)
Writes:
- `./nwm_data/geoparquet/date=YYYY-MM-DD/wa_nwm_daily.parquet`
plus progress logs per shard.

Run:
```bash
python run.py nwm bbox --year 2023 --out-root ./nwm_data
```

### Mode B: enrich mode (adds columns into mushroom CSVs)
Reads:
- `./mushroom_data/<taxon>/data.csv`

Writes:
- updated `data.csv` with NWM columns (sharded parts → auto-combine)

Run:
```bash
python run.py nwm enrich --mushroom-dir ./mushroom_data
```

---

## 3) Terrain (DEM-derived features)

### Option A: Build WA 250 m terrain parquet (recommended)
Writes:
- `./terrain_data/wa_terrain_250m_v2.parquet` (or parts until all shards complete)

Run:
```bash
python run.py terrain build-wa-250m --out-dir ./terrain_data --step-m 250 --parquet-name wa_terrain_250m_v2.parquet
```

### Option B: Enrich mushroom CSVs
Adds columns like:
- `elevation_m, slope_deg, aspect_deg, eastness, northness, hillshade_315_45, tri_3x3, roughness_3x3, curvature_profile, curvature_plan`

Run:
```bash
python run.py terrain enrich-csv --mushroom-dir ./mushroom_data
```

---

## 4) ESA WorldCover Land Cover

### Option A: Build WA 250 m landcover parquet
Writes:
- `./land_cover_data/wa_landcover_250m.parquet` (or parts until all shards complete)

Run:
```bash
python run.py landcover build-wa-250m --out-dir ./land_cover_data --step-m 250 --parquet-name wa_landcover_250m.parquet --year 2021
```

### Option B: Enrich mushroom CSVs
Adds columns:
- `lc_code, lc_name, lc_r, lc_g, lc_b, lc_source, lc_year`

Run:
```bash
python run.py landcover enrich-csv --mushroom-dir ./mushroom_data --year 2021
```

---

## Column dictionary (schemas)

### A) Mushroom CSV columns (collector output)
| Column | Meaning |
|---|---|
| `combined_uid` | Project ID (taxon prefix + iNat observation id) |
| `uid` | iNaturalist observation id |
| `timestamp` | Observation date (string; later parsed to date for joins) |
| `lat` | Latitude (WGS84) |
| `long` | Longitude (WGS84) |
| `picture_url` | Photo URL |
| `name` | Taxon label (e.g., `morel`) |

### B) NWM LAND features (bbox parquet + enrich columns)
| Column | Meaning / Units |
|---|---|
| `date` | Day (`YYYY-MM-DD`) |
| `soil_t_top_min_c` | Daily min soil temp top layer (°C) |
| `soil_t_top_max_c` | Daily max soil temp top layer (°C) |
| `soil_t_top_mean_c` | Daily mean soil temp top layer (°C) |
| `soil_t_l2_min_c` | Daily min soil temp layer2 (°C) |
| `soil_t_l2_max_c` | Daily max soil temp layer2 (°C) |
| `soil_t_l2_mean_c` | Daily mean soil temp layer2 (°C) |
| `soil_t_col_mean_c` | 0–2 m weighted mean soil temp (°C) |
| `soil_m_top_mean` | Daily mean soil moisture top layer (m³/m³) |
| `soil_m_col_mean` | 0–2 m weighted mean soil moisture (m³/m³) |
| `soilsat_top_mean` | Top-layer saturation (NWM variable) |
| `soilice_mean` | Soil ice (NWM variable) |
| `fsno_mean` | Snow cover fraction (0–1) |
| `isnow_mean` | Snow indicator/amount (NWM variable) |
| `sneqv_mean_kgm2` | Snow water equivalent (kg/m²) |
| `snowh_mean_m` | Snow height/depth (m) |
| `snowt_avg_mean_c` | Snow temperature (°C) |
| `acsnom_mean_mm` | Accumulated snowmelt (mm; NWM variable) |
| `qrain_mean_mms` | Rain rate (mm/s; NWM variable) |
| `qsnow_mean_mms` | Snow rate (mm/s; NWM variable) |
| `edir_mean_kgm2s` | Direct soil evaporation (kg/m²/s; NWM variable) |
| `accet_mean_mm` | Accumulated evapotranspiration (mm; NWM variable) |

### C) Terrain features (WA 250 m parquet + enrich columns)
| Column | Meaning / Units |
|---|---|
| `x_5070`, `y_5070` | Grid coordinates (EPSG:5070 meters) |
| `lon`, `lat` | WGS84 coordinates |
| `elevation_m` | Elevation (m) |
| `slope_deg` | Slope (degrees) |
| `aspect_deg` | Aspect (0–360 degrees) |
| `eastness` | sin(aspect) |
| `northness` | cos(aspect) |
| `hillshade_315_45` | Hillshade (0–255) |
| `tri_3x3` | Ruggedness (3×3 neighborhood) |
| `roughness_3x3` | Local relief range (3×3 neighborhood) |
| `curvature_profile` | Profile curvature |
| `curvature_plan` | Plan curvature |

### D) Land cover features (WA 250 m parquet + enrich columns)
| Column | Meaning |
|---|---|
| `lc_code` | ESA WorldCover class code |
| `lc_name` | Class name |
| `lc_r`, `lc_g`, `lc_b` | Class RGB color |
| `lc_source` | e.g., `ESA WorldCover v200` |
| `lc_year` | Product year sampled |

WorldCover classes (common codes):
- 10 Trees
- 20 Shrubland
- 30 Grassland
- 40 Cropland
- 50 Built-up
- 60 Bare/sparse
- 70 Snow/ice
- 80 Water
- 90 Herbaceous wetland
- 95 Mangroves
- 100 Moss/lichen

---

## Local path guidance (works on any computer)

- Use paths like `./mushroom_data`, `./terrain_data`, etc.
- Avoid absolute Colab paths like `/content/drive/...`
- In VS Code, open the repo folder and run commands from the integrated terminal so relative paths resolve correctly.

---

## Troubleshooting

### Geo stack on Windows
If you hit install errors for rasterio/fiona/geopandas:
- Install Miniconda
- Create env
- `conda install -c conda-forge geopandas rasterio fiona pyproj shapely pyarrow scipy`
- Then `pip install -r requirements/base.txt` etc.

### Performance
- Increase `WORKERS` / `PARALLEL_WORKERS`
- Run multiple shards in parallel terminals

---

## Suggested run order (most common)
1) `python run.py collect`
2) `python run.py nwm enrich --mushroom-dir ./mushroom_data`
3) `python run.py terrain enrich-csv --mushroom-dir ./mushroom_data`
4) `python run.py landcover enrich-csv --mushroom-dir ./mushroom_data --year 2021`

Done: your per-taxon CSVs contain all features.
