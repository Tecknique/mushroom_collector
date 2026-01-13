from __future__ import annotations

import csv
import math
import os
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import rasterio
from pyproj import CRS, Transformer
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import MemoryFile
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform as rio_transform
from scipy.ndimage import generic_filter, uniform_filter
from tqdm.auto import tqdm

import geopandas as gpd
from shapely.geometry import box

from .logging_utils import get_logger

logger = get_logger("terrain")

TERRAIN_URL = "https://s3.amazonaws.com/elevation-tiles-prod/geotiff/{z}/{x}/{y}.tif"
GRID_CRS = CRS.from_epsg(5070)
WGS84 = CRS.from_epsg(4326)


def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or not str(v).strip():
        return default
    try:
        return int(v)
    except Exception:
        return default


def lonlat_to_tile(lon: float, lat: float, z: int) -> Tuple[int, int]:
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = 2.0**z
    xtile = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def looks_like_tiff(body: bytes, ctype: str) -> bool:
    if not body or len(body) < 4:
        return False
    if ctype and "tiff" in ctype.lower():
        return True
    sig = body[:4]
    return sig in (b"II*\x00", b"MM\x00*")


@dataclass(frozen=True)
class ShardSpec:
    count: int
    shard_id: int

    @staticmethod
    def from_env(default_count: int = 5, default_id: int = 0) -> "ShardSpec":
        # accept either SHARD_COUNT or N_SHARDS
        count = env_int("SHARD_COUNT", env_int("N_SHARDS", default_count))
        sid = env_int("SHARD_ID", default_id)
        return ShardSpec(count=max(1, int(count)), shard_id=max(0, int(sid)))


class TileCache:
    def __init__(self, cache_dir: Path, max_workers: int) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = int(max_workers)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "WA-Terrain/2.1"})

    def _path(self, z: int, x: int, y: int) -> Path:
        return self.cache_dir / f"aws_geo_{z}_{x}_{y}.tif"

    def fetch_one(self, z: int, x: int, y: int) -> Optional[Path]:
        path = self._path(z, x, y)
        if path.exists() and path.stat().st_size > 0:
            return path

        url = TERRAIN_URL.format(z=z, x=x, y=y)
        try:
            r = self.session.get(url, timeout=120)
            if r.status_code == 200 and looks_like_tiff(r.content, r.headers.get("Content-Type", "")):
                path.write_bytes(r.content)
                return path
            return None
        except requests.RequestException:
            return None

    def fetch_many(self, tiles: Iterable[Tuple[int, int, int]], desc: str) -> Dict[Tuple[int, int, int], Optional[Path]]:
        tiles = list(tiles)
        results: Dict[Tuple[int, int, int], Optional[Path]] = {}
        pbar = tqdm(total=len(tiles), desc=f"download {desc}", leave=False)
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = {ex.submit(self.fetch_one, z, x, y): (z, x, y) for (z, x, y) in tiles}
            for fut in as_completed(futs):
                z, x, y = futs[fut]
                try:
                    results[(z, x, y)] = fut.result()
                except Exception:
                    results[(z, x, y)] = None
                pbar.update(1)
        pbar.close()
        return results


def tri_3x3(arr: np.ndarray) -> np.ndarray:
    def func(win):
        c = win[4]
        nb = np.delete(win, 4)
        return float(np.mean(np.abs(nb - c)))

    return generic_filter(arr, func, size=3, mode="nearest").astype("float32")


def roughness_3x3(arr: np.ndarray) -> np.ndarray:
    def func(win) -> float:
        return float(np.max(win) - np.min(win))

    return generic_filter(arr, func, size=3, mode="nearest").astype("float32")


def tpi(arr: np.ndarray, px_radius: int) -> np.ndarray:
    k = 2 * px_radius + 1
    mean_nb = uniform_filter(arr, size=k, mode="nearest")
    return (arr - mean_nb).astype("float32")


def slope_aspect_from_dem(elev: np.ndarray, xres_m: float, yres_m: float) -> Tuple[np.ndarray, np.ndarray]:
    with np.errstate(all="ignore"):
        gy, gx = np.gradient(elev.astype("float32"), yres_m, xres_m)
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))
        aspect = np.degrees(np.arctan2(gy, -gx))
        aspect = (aspect + 360.0) % 360.0
    return slope.astype("float32"), aspect.astype("float32")


def curvature_profile_plan(elev: np.ndarray, xres_m: float, yres_m: float) -> Tuple[np.ndarray, np.ndarray]:
    with np.errstate(all="ignore"):
        gy, gx = np.gradient(elev.astype("float32"), yres_m, xres_m)
        gyy, gyx = np.gradient(gy, yres_m, xres_m)
        gxy, gxx = np.gradient(gx, yres_m, xres_m)
        p = gx
        q = gy
        pp = gxx
        qq = gyy
        pq = gxy
        eps = 1e-6
        g2 = p * p + q * q
        g = np.sqrt(g2) + eps
        k_prof = -((pp * p * p) + (2 * pq * p * q) + (qq * q * q)) / (g2 * g + eps)
        k_plan = -((pp * q * q) - (2 * pq * p * q) + (qq * p * p)) / (g2 * g + eps)
        k_prof = np.where(np.isfinite(k_prof), k_prof, np.nan)
        k_plan = np.where(np.isfinite(k_plan), k_plan, np.nan)
    return k_prof.astype("float32"), k_plan.astype("float32")


def hillshade_from_slope_aspect(
    slope_deg: np.ndarray,
    aspect_deg: np.ndarray,
    azimuth_deg: float = 315.0,
    altitude_deg: float = 45.0,
) -> np.ndarray:
    with np.errstate(all="ignore"):
        az = math.radians(azimuth_deg)
        alt = math.radians(altitude_deg)
        sl = np.radians(slope_deg.astype("float32"))
        asp = np.radians(aspect_deg.astype("float32"))
        hs = (np.cos(alt) * np.cos(sl)) + (np.sin(alt) * np.sin(sl) * np.cos(az - asp))
        hs = np.clip(hs, 0.0, 1.0)
    return (hs * 255.0).astype("float32")


def eastness_northness(aspect_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ar = np.radians(aspect_deg.astype("float32"))
    return np.sin(ar).astype("float32"), np.cos(ar).astype("float32")


class TerrainEngine:
    def __init__(self, *, zoom: int, cache_dir: Path, workers: int) -> None:
        self.zoom = int(zoom)
        self.cache = TileCache(cache_dir=cache_dir, max_workers=int(workers))

    def derive_tile(self, tif_path: Path, dst_crs: CRS) -> Tuple[Dict[str, np.ndarray], dict, str]:
        with rasterio.open(tif_path) as src:
            if src.crs is None:
                raise ValueError("Source tile has no CRS.")
            dst_transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
            elev = np.empty((height, width), dtype="float32")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", NotGeoreferencedWarning)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=elev,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )

        xres_m, yres_m = abs(dst_transform.a), abs(dst_transform.e)
        slope, aspect = slope_aspect_from_dem(elev, xres_m, yres_m)
        k_prof, k_plan = curvature_profile_plan(elev, xres_m, yres_m)
        east, north = eastness_northness(aspect)
        hill = hillshade_from_slope_aspect(slope, aspect, 315.0, 45.0)
        tri = tri_3x3(elev)
        rough = roughness_3x3(elev)

        px_300m = max(1, int(round(300.0 / xres_m)))
        px_1km = max(1, int(round(1000.0 / xres_m)))
        tpi300 = tpi(elev, px_300m)
        tpi1k = tpi(elev, px_1km)

        rasters = dict(
            elevation_m=elev,
            slope_deg=slope,
            aspect_deg=aspect,
            eastness=east,
            northness=north,
            hillshade_315_45=hill,
            tri_3x3=tri,
            roughness_3x3=rough,
            tpi_300m=tpi300,
            tpi_1km=tpi1k,
            curvature_profile=k_prof,
            curvature_plan=k_plan,
        )
        meta = {
            "transform": dst_transform,
            "crs": dst_crs,
            "width": elev.shape[1],
            "height": elev.shape[0],
            "xres_m": xres_m,
            "yres_m": yres_m,
        }
        return rasters, meta, dst_crs.to_string()

    def sample_many(
        self,
        lon_vals: np.ndarray,
        lat_vals: np.ndarray,
        rasters: Dict[str, np.ndarray],
        meta: dict,
        dst_crs_str: str,
    ) -> Dict[str, np.ndarray]:
        xs, ys = rio_transform("EPSG:4326", dst_crs_str, lon_vals.tolist(), lat_vals.tolist())
        rows = ((np.array(ys) - meta["transform"].f) / meta["transform"].e).astype(int)
        cols = ((np.array(xs) - meta["transform"].c) / meta["transform"].a).astype(int)
        h, w = next(iter(rasters.values())).shape
        mask = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)

        out: Dict[str, np.ndarray] = {}
        for name, arr in rasters.items():
            vals = np.full(len(rows), np.nan, dtype="float32")
            vals[mask] = arr[rows[mask], cols[mask]]
            out[name] = vals
        return out


def bucket_points_by_tile(latlon_iter: Iterable[Tuple[float, float]], z: int) -> Dict[Tuple[int, int, int], List[int]]:
    buckets: Dict[Tuple[int, int, int], List[int]] = {}
    for idx, (lat, lon) in enumerate(latlon_iter):
        x, y = lonlat_to_tile(lon, lat, z)
        buckets.setdefault((z, x, y), []).append(idx)
    return buckets


def load_washington_boundary_5070() -> gpd.GeoDataFrame:
    """Load WA boundary; download TIGER/GENZ to a temp zip; fallback to bbox."""
    url = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_20m.zip"
    local_zip = Path(tempfile.gettempdir()) / "cb_2023_us_state_20m.zip"

    try:
        if (not local_zip.exists()) or local_zip.stat().st_size == 0:
            with requests.get(url, stream=True, timeout=180) as r:
                r.raise_for_status()
                with local_zip.open("wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)
        states = gpd.read_file(f"zip://{local_zip}")
        wa = states[states["STATEFP"] == "53"].to_crs(GRID_CRS)
        if wa.empty:
            raise RuntimeError("WA polygon not found.")
        return wa
    except Exception as e:
        logger.warning("WA boundary download/read failed (%s); using bbox fallback.", e)
        wa_bbox_wgs84 = (-124.9, 45.5, -117.0, 49.05)
        wa_poly = gpd.GeoSeries([box(*wa_bbox_wgs84)], crs="EPSG:4326").to_crs(GRID_CRS)
        return gpd.GeoDataFrame(geometry=wa_poly, crs=GRID_CRS)


def generate_grid_within(geom_5070, step_m: float) -> pd.DataFrame:
    minx, miny, maxx, maxy = geom_5070.total_bounds
    xs = np.arange(minx, maxx + step_m, step_m, dtype="float64")
    ys = np.arange(miny, maxy + step_m, step_m, dtype="float64")
    xx, yy = np.meshgrid(xs, ys)
    df = pd.DataFrame({"x_5070": xx.ravel(), "y_5070": yy.ravel()})
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["x_5070"], df["y_5070"], crs=GRID_CRS))
    inside = gdf.within(geom_5070.unary_union)
    gdf = gdf[inside].copy()
    lon, lat = Transformer.from_crs(GRID_CRS, WGS84, always_xy=True).transform(gdf["x_5070"].values, gdf["y_5070"].values)
    gdf["lon"] = lon
    gdf["lat"] = lat
    return pd.DataFrame(gdf.drop(columns="geometry"))


class WashingtonTerrain:
    """Preview + production terrain pipeline using AWS elevation tiles."""

    VARS_PREVIEW = [
        "elevation_m",
        "slope_deg",
        "aspect_deg",
        "eastness",
        "northness",
        "hillshade_315_45",
        "tri_3x3",
        "roughness_3x3",
        "curvature_profile",
        "curvature_plan",
    ]

    VARS_FULL = [
        "elevation_m",
        "slope_deg",
        "aspect_deg",
        "eastness",
        "northness",
        "hillshade_315_45",
        "tri_3x3",
        "roughness_3x3",
        "tpi_300m",
        "tpi_1km",
        "curvature_profile",
        "curvature_plan",
    ]

    def __init__(self, *, workers: int, zoom: int, cache_dir: Path) -> None:
        self.workers = int(workers)
        self.zoom = int(zoom)
        self.engine = TerrainEngine(zoom=self.zoom, cache_dir=cache_dir, workers=self.workers)

    # ---------- preview ----------
    def preview_wa(self, *, n_points: int, step_m: float) -> pd.DataFrame:
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        wa = gpd.GeoDataFrame(geometry=gpd.GeoSeries([box(-124.9, 45.5, -117.0, 49.05)], crs="EPSG:4326")).to_crs(GRID_CRS)
        grid_df = generate_grid_within(wa.geometry, step_m=float(step_m))
        sample_df = grid_df.sample(n=min(int(n_points), len(grid_df)), random_state=42).reset_index(drop=True)

        latlon = list(zip(sample_df["lat"].to_numpy(), sample_df["lon"].to_numpy()))
        buckets = bucket_points_by_tile(latlon, self.zoom)
        tiles = sorted(buckets.keys())
        path_map = self.engine.cache.fetch_many(tiles, desc="DEM tiles (preview-wa)")

        out_arrays: Dict[str, np.ndarray] = {k: np.full(len(sample_df), np.nan, dtype="float32") for k in self.VARS_PREVIEW}

        def process_tile(tile):
            tif = path_map.get(tile)
            if not tif:
                return tile, None
            try:
                rasters, meta, dst = self.engine.derive_tile(tif, GRID_CRS)
                idxs = buckets[tile]
                lon_vals = sample_df.loc[idxs, "lon"].values
                lat_vals = sample_df.loc[idxs, "lat"].values
                sampled = self.engine.sample_many(lon_vals, lat_vals, rasters, meta, dst)
                return tile, (idxs, sampled)
            except Exception:
                return tile, None

        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futs = {ex.submit(process_tile, t): t for t in tiles}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="derive+sample (preview-wa)", leave=False):
                _tile, payload = fut.result()
                if payload is None:
                    continue
                idxs, sampled = payload
                for name, vals in sampled.items():
                    if name in out_arrays:
                        out_arrays[name][idxs] = vals

        for k, arr in out_arrays.items():
            sample_df[k] = arr

        cols = ["lon", "lat", "x_5070", "y_5070"] + self.VARS_PREVIEW
        print("\n=== TERRAIN preview-wa • WA grid sample (print-only) ===")
        print(sample_df[cols].round(3).to_string(index=False, max_rows=20))
        return sample_df

    def preview_csv(self, *, mushroom_dir: Path, n_rows: int) -> Optional[pd.DataFrame]:
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        csv_path: Optional[Path] = None
        for d in sorted(mushroom_dir.iterdir(), key=lambda p: p.name.lower()):
            if d.is_dir() and d.name != "MushroomDataCombined":
                p = d / "data.csv"
                if p.exists():
                    csv_path = p
                    break

        if not csv_path:
            print(f"⚠️  No mushroom data.csv under {mushroom_dir}.")
            return None

        df = pd.read_csv(csv_path)
        df = df.rename(columns={"longitude": "long"})
        df = df[pd.to_numeric(df.get("lat"), errors="coerce").notna() & pd.to_numeric(df.get("long"), errors="coerce").notna()].copy()
        if df.empty:
            print(f"⚠️  {csv_path} has no valid lat/long rows.")
            return None

        df = df.head(int(n_rows)).copy()
        rows = df.to_dict(orient="records")

        latlon: List[Tuple[float, float]] = []
        for r in rows:
            try:
                lat = float(r.get("lat", ""))
                lon = float(r.get("long", ""))
            except Exception:
                lat, lon = float("nan"), float("nan")
            latlon.append((lat, lon))

        buckets = bucket_points_by_tile(latlon, self.zoom)
        tiles = sorted(buckets.keys())
        path_map = self.engine.cache.fetch_many(tiles, desc="DEM tiles (preview-csv)")

        sampled_store: Dict[int, Dict[str, float]] = {i: {} for i in range(len(rows))}

        def process_tile(tile):
            tif = path_map.get(tile)
            if not tif:
                return tile, None
            try:
                rasters, meta, dst = self.engine.derive_tile(tif, GRID_CRS)
                idxs = buckets[tile]
                lon_vals = np.array([float(rows[i].get("long", "nan")) for i in idxs], dtype="float64")
                lat_vals = np.array([float(rows[i].get("lat", "nan")) for i in idxs], dtype="float64")
                sampled = self.engine.sample_many(lon_vals, lat_vals, rasters, meta, dst)
                return tile, (idxs, sampled)
            except Exception:
                return tile, None

        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futs = {ex.submit(process_tile, t): t for t in tiles}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="derive+sample (preview-csv)", leave=False):
                _tile, payload = fut.result()
                if payload is None:
                    continue
                idxs, sampled = payload
                for j, i_row in enumerate(idxs):
                    for k in self.VARS_PREVIEW:
                        val = float(sampled.get(k, np.array([np.nan]))[j])
                        if not np.isnan(val):
                            sampled_store[i_row][k] = val

        out_df = pd.DataFrame(rows).copy()
        for k in self.VARS_PREVIEW:
            out_df[k] = [sampled_store[i].get(k, np.nan) for i in range(len(rows))]

        front = [c for c in ["id", "lat", "long"] if c in out_df.columns]
        cols = front + [c for c in self.VARS_PREVIEW if c in out_df.columns]
        print("\n=== TERRAIN preview-csv • Mushroom CSV sample (print-only) ===")
        print(f"Source CSV: {csv_path}")
        print(out_df[cols].round(3).to_string(index=False, max_rows=len(out_df)))
        return out_df

    # ---------- production ----------
    def build_wa_parquet_250m(self, *, out_dir: Path, step_m: float, parquet_name: str, shard: ShardSpec) -> Path:
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        t_all0 = time.time()
        out_dir.mkdir(parents=True, exist_ok=True)

        wa = load_washington_boundary_5070()
        grid_df = generate_grid_within(wa.geometry, step_m=float(step_m))

        n = len(grid_df)
        base = n // shard.count
        rem = n % shard.count
        start = shard.shard_id * base + min(shard.shard_id, rem)
        length = base + (1 if shard.shard_id < rem else 0)
        end = start + length
        shard_df = grid_df.iloc[start:end].reset_index(drop=True)

        latlon = list(zip(shard_df["lat"].to_numpy(), shard_df["lon"].to_numpy()))
        buckets = bucket_points_by_tile(latlon, self.zoom)
        tiles = sorted(buckets.keys())

        path_map = self.engine.cache.fetch_many(tiles, desc=f"DEM tiles (WA 250m shard {shard.shard_id})")

        out_arrays: Dict[str, np.ndarray] = {k: np.full(len(shard_df), np.nan, dtype="float32") for k in self.VARS_FULL}

        def process_tile(tile):
            tif = path_map.get(tile)
            if not tif:
                return tile, None
            try:
                rasters, meta, dst = self.engine.derive_tile(tif, GRID_CRS)
                idxs = buckets[tile]
                lon_vals = shard_df.loc[idxs, "lon"].values
                lat_vals = shard_df.loc[idxs, "lat"].values
                sampled = self.engine.sample_many(lon_vals, lat_vals, rasters, meta, dst)
                return tile, (idxs, sampled)
            except Exception:
                return tile, None

        pbar = tqdm(total=len(tiles), desc="derive+sample tiles (parallel)", leave=False)
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futs = {ex.submit(process_tile, t): t for t in tiles}
            for fut in as_completed(futs):
                try:
                    _tile, payload = fut.result()
                    if payload is None:
                        continue
                    idxs, sampled = payload
                    for name, vals in sampled.items():
                        if name in out_arrays:
                            out_arrays[name][idxs] = vals
                except Exception:
                    pass
                pbar.update(1)
        pbar.close()

        for k, arr in out_arrays.items():
            shard_df[k] = arr

        part_path = out_dir / f"{parquet_name}.part{shard.shard_id}"
        shard_df.to_parquet(part_path, compression="snappy", index=False)

        all_parts = [out_dir / f"{parquet_name}.part{i}" for i in range(shard.count)]
        out_path = part_path
        if all(p.exists() for p in all_parts):
            full = out_dir / parquet_name
            pd.concat([pd.read_parquet(p) for p in all_parts], ignore_index=True).to_parquet(full, compression="snappy", index=False)
            for p in all_parts:
                try:
                    p.unlink()
                except Exception:
                    pass
            out_path = full

        logger.info("[DONE] shard %s/%s rows=%s time=%.1fs -> %s", shard.shard_id, shard.count, len(shard_df), time.time() - t_all0, out_path)
        return out_path

    def enrich_mushroom_csvs(self, *, mushroom_dir: Path, shard: ShardSpec) -> None:
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        taxa_dirs = sorted([d for d in mushroom_dir.iterdir() if d.is_dir() and d.name != "MushroomDataCombined"], key=lambda p: p.name.lower())

        for td in taxa_dirs:
            csv_path = td / "data.csv"
            if not csv_path.exists():
                continue

            # Count rows and header
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                rdr = csv.reader(f)
                try:
                    header = next(rdr)
                except StopIteration:
                    continue
                nrows = sum(1 for _ in rdr)

            base = nrows // shard.count
            rem = nrows % shard.count
            start = shard.shard_id * base + min(shard.shard_id, rem)
            length = base + (1 if shard.shard_id < rem else 0)
            end = start + length

            rows: List[Dict[str, str]] = []
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                rdr = csv.DictReader(f)
                for i, r in enumerate(rdr):
                    if start <= i < end:
                        rows.append(dict(r))

            if not rows:
                continue

            out_cols = list(header)
            for k in self.VARS_FULL:
                if k not in out_cols:
                    out_cols.append(k)

            latlon: List[Tuple[float, float]] = []
            for r in rows:
                try:
                    lat = float(r.get("lat", ""))
                    lon = float(r.get("long", ""))
                except Exception:
                    lat, lon = float("nan"), float("nan")
                latlon.append((lat, lon))

            buckets = bucket_points_by_tile(latlon, self.zoom)
            tiles = sorted(buckets.keys())
            path_map = self.engine.cache.fetch_many(tiles, desc=f"DEM tiles ({td.name} shard {shard.shard_id})")

            def process_tile(tile):
                tif = path_map.get(tile)
                if not tif:
                    return tile, None
                try:
                    rasters, meta, dst = self.engine.derive_tile(tif, GRID_CRS)
                    idxs = buckets[tile]
                    lon_vals = np.array([float(rows[i].get("long", "nan")) for i in idxs], dtype="float64")
                    lat_vals = np.array([float(rows[i].get("lat", "nan")) for i in idxs], dtype="float64")
                    sampled = self.engine.sample_many(lon_vals, lat_vals, rasters, meta, dst)
                    return tile, (idxs, sampled)
                except Exception:
                    return tile, None

            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                futs = {ex.submit(process_tile, t): t for t in tiles}
                for fut in tqdm(as_completed(futs), total=len(futs), desc=f"{td.name} • shard {shard.shard_id}", leave=False):
                    _tile, payload = fut.result()
                    if payload is None:
                        continue
                    idxs, sampled = payload
                    for j, i_row in enumerate(idxs):
                        for k in self.VARS_FULL:
                            val = float(sampled.get(k, np.array([np.nan]))[j])
                            if not np.isnan(val):
                                rows[i_row][k] = f"{val:.8g}"

            parts_dir = td / ".terrain_parts_wa_v2"
            parts_dir.mkdir(parents=True, exist_ok=True)
            part_path = parts_dir / f"part_{shard.shard_id}.csv"

            with part_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=out_cols)
                w.writeheader()
                for r in rows:
                    for k in out_cols:
                        r.setdefault(k, "")
                    w.writerow(r)

            done_path = part_path.with_suffix(part_path.suffix + ".done")
            done_path.write_text("ok", encoding="utf-8")

            all_parts = [parts_dir / f"part_{i}.csv" for i in range(shard.count)]
            all_done = [p.with_suffix(p.suffix + ".done") for p in all_parts]

            if all(p.exists() for p in all_parts) and all(p.exists() for p in all_done):
                tmp_out = csv_path.with_suffix(".new")
                with tmp_out.open("w", newline="", encoding="utf-8") as fout:
                    w = csv.writer(fout)
                    w.writerow(out_cols)
                    for p in all_parts:
                        with p.open("r", encoding="utf-8", newline="") as fin:
                            rdr = csv.reader(fin)
                            next(rdr, None)
                            for row in rdr:
                                if len(row) != len(out_cols):
                                    row = (row + [""] * len(out_cols))[: len(out_cols)]
                                w.writerow(row)

                ts = time.strftime("%Y%m%d%H%M%S", time.gmtime())
                bak = csv_path.with_suffix(csv_path.suffix + f".{ts}.bak")
                try:
                    os.replace(csv_path, bak)
                except Exception:
                    pass
                os.replace(tmp_out, csv_path)

                for p in all_parts:
                    try:
                        p.unlink()
                    except Exception:
                        pass
                    d = p.with_suffix(p.suffix + ".done")
                    try:
                        d.unlink()
                    except Exception:
                        pass
                try:
                    parts_dir.rmdir()
                except Exception:
                    pass
