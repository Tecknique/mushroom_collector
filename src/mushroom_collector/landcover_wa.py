from __future__ import annotations

import csv
import math
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import geopandas as gpd
import rasterio
from pyproj import CRS, Transformer
from shapely.geometry import box
from tqdm.auto import tqdm


WC_CLASSES: Dict[int, Dict[str, object]] = {
    10: {"name": "Trees", "rgb": (0, 100, 0)},
    20: {"name": "Shrubland", "rgb": (255, 187, 34)},
    30: {"name": "Grassland", "rgb": (255, 255, 76)},
    40: {"name": "Cropland", "rgb": (240, 150, 255)},
    50: {"name": "Built-up", "rgb": (250, 0, 0)},
    60: {"name": "Bare / sparse vegetation", "rgb": (180, 180, 180)},
    70: {"name": "Snow and ice", "rgb": (240, 240, 240)},
    80: {"name": "Permanent water bodies", "rgb": (0, 100, 200)},
    90: {"name": "Herbaceous wetland", "rgb": (0, 150, 160)},
    95: {"name": "Mangroves", "rgb": (0, 207, 117)},
    100: {"name": "Moss and lichen", "rgb": (250, 230, 160)},
}

_WORLDCOVER_BASES = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com",
    "https://esa-worldcover.s3.amazonaws.com",
    "https://s3.eu-central-1.amazonaws.com/esa-worldcover",
)


def _env_flag(name: str, default: bool) -> bool:
    v = os.getenv(name, "true" if default else "false").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _version_for_year(year: int, requested: str) -> str:
    req = (requested or "auto").strip().lower()
    if req in ("auto", ""):
        return "v200" if year >= 2021 else "v100"
    return req


def _clamp_lon(lon: float) -> float:
    lon = ((lon + 180.0) % 360.0) - 180.0
    return min(179.9999, max(-179.9999, lon))


def _clamp_lat(lat: float) -> float:
    return min(89.9999, max(-89.9999, lat))


def wc_tile_code(lat: float, lon: float) -> str:
    lat0 = math.floor(_clamp_lat(lat) / 3.0) * 3
    lon0 = math.floor(_clamp_lon(lon) / 3.0) * 3
    ns = "N" if lat0 >= 0 else "S"
    ew = "E" if lon0 >= 0 else "W"
    return f"{ns}{abs(int(lat0)):02d}{ew}{abs(int(lon0)):03d}"


def wc_href_candidates(tile: str, year: int, version: str) -> List[str]:
    rel = f"/{version}/{year}/map/ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif"
    return [b + rel for b in _WORLDCOVER_BASES]


def resolve_wc_href(session: requests.Session, tile: str, year: int, version: str) -> Optional[str]:
    for url in wc_href_candidates(tile, year, version):
        try:
            r = session.head(url, timeout=20, allow_redirects=True)
            if r.status_code == 200:
                return url
        except requests.RequestException:
            continue
    return None


def range_for_shard(n: int, shard_count: int, shard_id: int) -> Tuple[int, int]:
    base = n // shard_count
    rem = n % shard_count
    start = shard_id * base + min(shard_id, rem)
    length = base + (1 if shard_id < rem else 0)
    return start, start + length


def load_washington_boundary_5070(*, grid_crs: CRS, progress: bool) -> gpd.GeoDataFrame:
    """
    Robustly load Washington State boundary as EPSG:5070 (or provided CRS).
    Downloads the Census ZIP to a temp file to avoid vsicurl issues.
    Falls back to WA bbox if download/read fails.
    """
    url = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_20m.zip"
    wa_bbox_wgs84 = (-124.9, 45.5, -117.0, 49.05)

    try:
        tmpdir = Path(tempfile.gettempdir()) / "mushroom_collector_cache"
        tmpdir.mkdir(parents=True, exist_ok=True)
        local_zip = tmpdir / "cb_2023_us_state_20m.zip"

        if (not local_zip.exists()) or local_zip.stat().st_size == 0:
            if progress:
                print(f"[landcover] downloading boundary → {local_zip}")
            with requests.get(url, stream=True, timeout=180) as r:
                r.raise_for_status()
                with open(local_zip, "wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)

        states = gpd.read_file(f"zip://{local_zip}")
        wa = states[states["STATEFP"] == "53"].to_crs(grid_crs)
        if wa.empty:
            raise RuntimeError("WA polygon not found in TIGER states dataset.")
        return wa
    except Exception as e:
        if progress:
            print(f"[landcover] WARN boundary load failed ({e}); using bbox fallback.")
        wa_poly = gpd.GeoSeries([box(*wa_bbox_wgs84)], crs="EPSG:4326").to_crs(grid_crs)
        return gpd.GeoDataFrame(geometry=wa_poly, crs=grid_crs)


def generate_grid_within(geom_5070: gpd.GeoDataFrame, *, step_m: float, grid_crs: CRS) -> pd.DataFrame:
    minx, miny, maxx, maxy = geom_5070.total_bounds
    xs = np.arange(minx, maxx + step_m, step_m, dtype="float64")
    ys = np.arange(miny, maxy + step_m, step_m, dtype="float64")

    xx, yy = np.meshgrid(xs, ys)
    df = pd.DataFrame({"x_5070": xx.ravel(), "y_5070": yy.ravel()})
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["x_5070"], df["y_5070"], crs=grid_crs))

    inside = gdf.within(geom_5070.geometry.union_all())
    gdf = gdf[inside].copy()

    lon, lat = Transformer.from_crs(grid_crs, "EPSG:4326", always_xy=True).transform(
        gdf["x_5070"].values, gdf["y_5070"].values
    )
    gdf["lon"] = lon
    gdf["lat"] = lat
    return pd.DataFrame(gdf.drop(columns="geometry"))


@dataclass(frozen=True)
class LandCoverConfig:
    out_dir: Path
    mushroom_dir: Path
    shard_count: int
    shard_id: int
    workers: int
    progress: bool
    year: int
    version: str
    step_m: float


class WashingtonLandCover:
    """
    ESA WorldCover sampler for WA 250m grid and mushroom CSV enrichment.

    Outputs:
      - Parquet parts: <parquet_name>.part{shard_id} in out_dir, auto-combines when all shards present.
      - CSV parts per taxon: <taxon>/.landcover_parts/part_{shard_id}.csv, auto-combines.
    """

    def __init__(self, *, workers: int, year: int, version: str, progress: bool) -> None:
        self.workers = max(1, int(workers))
        self.year = int(year)
        self.version = _version_for_year(self.year, version)
        self.progress = bool(progress)

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "MushroomCollector-LandCover/1.0"})
        self.grid_crs = CRS.from_epsg(5070)

    def _bucket_by_tile(self, lats: np.ndarray, lons: np.ndarray) -> Dict[str, List[int]]:
        buckets: Dict[str, List[int]] = {}
        for i in range(len(lats)):
            lat = float(lats[i])
            lon = float(lons[i])
            if not np.isfinite(lat) or not np.isfinite(lon):
                continue
            buckets.setdefault(wc_tile_code(lat, lon), []).append(i)
        return buckets

    def _sample_tile(self, tile: str, idxs: List[int], lats: np.ndarray, lons: np.ndarray) -> Dict[int, Tuple[int, str, int, int, int]]:
        url = resolve_wc_href(self.session, tile, self.year, self.version)
        if url is None:
            return {}

        coords = [(_clamp_lon(float(lons[i])), _clamp_lat(float(lats[i]))) for i in idxs]

        # Use /vsicurl/ (HTTP range reads) when possible.
        gdal_opts = {
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": "tif",
        }

        out: Dict[int, Tuple[int, str, int, int, int]] = {}
        try:
            with rasterio.Env(**gdal_opts):
                with rasterio.open(url) as ds:
                    for ridx, val in zip(idxs, ds.sample(coords, indexes=1)):
                        try:
                            code = int(val[0])
                        except Exception:
                            continue
                        meta = WC_CLASSES.get(code)
                        if not meta:
                            continue
                        name = str(meta["name"])
                        r, g, b = meta["rgb"]  # type: ignore[misc]
                        out[ridx] = (code, name, int(r), int(g), int(b))
        except Exception:
            return {}
        return out

    def sample_points(self, lats: np.ndarray, lons: np.ndarray) -> pd.DataFrame:
        n = len(lats)
        result = pd.DataFrame(
            {
                "lc_code": pd.Series([pd.NA] * n, dtype="Int64"),
                "lc_name": pd.Series([""] * n, dtype="string"),
                "lc_r": pd.Series([pd.NA] * n, dtype="Int64"),
                "lc_g": pd.Series([pd.NA] * n, dtype="Int64"),
                "lc_b": pd.Series([pd.NA] * n, dtype="Int64"),
            }
        )

        buckets = self._bucket_by_tile(lats, lons)
        tiles = sorted(buckets.keys())

        pbar = tqdm(total=len(tiles), desc="worldcover tiles", leave=False) if self.progress else None

        def _work(tile: str) -> Dict[int, Tuple[int, str, int, int, int]]:
            return self._sample_tile(tile, buckets[tile], lats, lons)

        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futs = {ex.submit(_work, t): t for t in tiles}
            for fut in as_completed(futs):
                tile = futs[fut]
                try:
                    hits = fut.result()
                    for i, (code, name, r, g, b) in hits.items():
                        result.at[i, "lc_code"] = code
                        result.at[i, "lc_name"] = name
                        result.at[i, "lc_r"] = r
                        result.at[i, "lc_g"] = g
                        result.at[i, "lc_b"] = b
                finally:
                    if pbar:
                        pbar.update(1)

        if pbar:
            pbar.close()

        result["lc_source"] = f"ESA WorldCover {self.version}"
        result["lc_year"] = int(self.year)
        return result

    def build_wa_parquet_250m(
        self,
        *,
        out_dir: Path,
        step_m: float,
        parquet_name: str,
        shard_count: int,
        shard_id: int,
    ) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        wa = load_washington_boundary_5070(grid_crs=self.grid_crs, progress=self.progress)
        grid_df = generate_grid_within(wa, step_m=step_m, grid_crs=self.grid_crs)

        s, e = range_for_shard(len(grid_df), shard_count, shard_id)
        shard = grid_df.iloc[s:e].reset_index(drop=True)

        lc = self.sample_points(shard["lat"].to_numpy(), shard["lon"].to_numpy())
        shard = pd.concat([shard.reset_index(drop=True), lc.reset_index(drop=True)], axis=1)

        part_path = out_dir / f"{parquet_name}.part{shard_id}"
        shard.to_parquet(part_path, compression="zstd", index=False)

        all_parts = [out_dir / f"{parquet_name}.part{i}" for i in range(shard_count)]
        if all(p.exists() for p in all_parts):
            full_path = out_dir / parquet_name
            pd.concat([pd.read_parquet(p) for p in all_parts], ignore_index=True).to_parquet(
                full_path, compression="zstd", index=False
            )
            for p in all_parts:
                try:
                    p.unlink()
                except Exception:
                    pass
            return full_path

        return part_path

    def enrich_mushroom_csvs(
        self,
        *,
        mushroom_dir: Path,
        shard_count: int,
        shard_id: int,
        input_name: str = "data.csv",
        lat_col: str = "lat",
        lon_col: str = "long",
    ) -> None:
        mushroom_dir = Path(mushroom_dir)
        taxa_dirs = sorted(
            [p for p in mushroom_dir.iterdir() if p.is_dir() and p.name != "MushroomDataCombined"],
            key=lambda p: p.name.lower(),
        )

        for td in taxa_dirs:
            csv_path = td / input_name
            if not csv_path.exists():
                continue

            with open(csv_path, "r", encoding="utf-8") as f:
                rdr = csv.reader(f)
                header = next(rdr)
                nrows = sum(1 for _ in rdr)

            s, e = range_for_shard(nrows, shard_count, shard_id)
            rows: List[Dict[str, str]] = []
            with open(csv_path, "r", encoding="utf-8") as f:
                rdr = csv.DictReader(f)
                for i, r in enumerate(rdr):
                    if s <= i < e:
                        rows.append(r)

            if not rows:
                continue

            lats = np.array([pd.to_numeric(r.get(lat_col, ""), errors="coerce") for r in rows], dtype="float64")
            lons = np.array([pd.to_numeric(r.get(lon_col, ""), errors="coerce") for r in rows], dtype="float64")
            lc = self.sample_points(lats, lons)

            # attach to row dicts
            for i, r in enumerate(rows):
                if pd.isna(lc.at[i, "lc_code"]):
                    continue
                r["lc_code"] = str(int(lc.at[i, "lc_code"]))
                r["lc_name"] = str(lc.at[i, "lc_name"])
                r["lc_r"] = "" if pd.isna(lc.at[i, "lc_r"]) else str(int(lc.at[i, "lc_r"]))
                r["lc_g"] = "" if pd.isna(lc.at[i, "lc_g"]) else str(int(lc.at[i, "lc_g"]))
                r["lc_b"] = "" if pd.isna(lc.at[i, "lc_b"]) else str(int(lc.at[i, "lc_b"]))
                r["lc_source"] = str(lc.at[i, "lc_source"])
                r["lc_year"] = str(int(lc.at[i, "lc_year"]))

            out_cols = list(header)
            for k in ["lc_code", "lc_name", "lc_r", "lc_g", "lc_b", "lc_source", "lc_year"]:
                if k not in out_cols:
                    out_cols.append(k)

            parts_dir = td / ".landcover_parts"
            parts_dir.mkdir(parents=True, exist_ok=True)
            part_path = parts_dir / f"part_{shard_id}.csv"
            done_path = parts_dir / f"part_{shard_id}.done"

            with open(part_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=out_cols)
                w.writeheader()
                for r in rows:
                    for k in out_cols:
                        r.setdefault(k, "")
                    w.writerow(r)

            done_path.write_text("ok", encoding="utf-8")

            all_parts = [parts_dir / f"part_{i}.csv" for i in range(shard_count)]
            all_done = [parts_dir / f"part_{i}.done" for i in range(shard_count)]
            if all(p.exists() for p in all_parts) and all(p.exists() for p in all_done):
                tmp_out = td / f"{input_name}.new"
                with open(tmp_out, "w", newline="", encoding="utf-8") as fout:
                    w = csv.writer(fout)
                    w.writerow(out_cols)
                    for p in all_parts:
                        with open(p, "r", encoding="utf-8") as fin:
                            rdr = csv.reader(fin)
                            next(rdr, None)
                            for row in rdr:
                                if len(row) < len(out_cols):
                                    row = row + [""] * (len(out_cols) - len(row))
                                w.writerow(row[: len(out_cols)])

                ts = pd.Timestamp.utcnow().strftime("%Y%m%d%H%M%S")
                bak = td / f"{input_name}.lc.{ts}.bak"
                try:
                    os.replace(csv_path, bak)
                except Exception:
                    pass
                os.replace(tmp_out, csv_path)

                for p in all_parts + all_done:
                    try:
                        p.unlink()
                    except Exception:
                        pass
                try:
                    parts_dir.rmdir()
                except Exception:
                    pass
