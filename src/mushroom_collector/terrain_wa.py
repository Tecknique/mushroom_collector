from __future__ import annotations

import math
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import MemoryFile
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform as rio_transform
from scipy.ndimage import generic_filter
from tqdm.auto import tqdm

import geopandas as gpd
from shapely.geometry import box
from pyproj import CRS, Transformer

from .logging_utils import get_logger

logger = get_logger("terrain")

TERRAIN_URL = "https://s3.amazonaws.com/elevation-tiles-prod/geotiff/{z}/{x}/{y}.tif"
GRID_CRS = CRS.from_epsg(5070)  # metric (CONUS Albers)
WGS84 = CRS.from_epsg(4326)


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


def nan_mask(a: np.ndarray) -> np.ndarray:
    m = ~np.isfinite(a) | (a <= -3.4028235e38)
    out = a.astype("float32", copy=True)
    out[m] = np.nan
    return out


def tri_3x3_nan(a: np.ndarray) -> np.ndarray:
    a = nan_mask(a)

    def func(win):
        if not np.isfinite(win).any():
            return np.nan
        c = win[4]
        nb = np.delete(win, 4)
        return float(np.nanmean(np.abs(nb - c)))

    return generic_filter(a, func, size=3, mode="nearest").astype("float32")


def roughness_3x3_nan(a: np.ndarray) -> np.ndarray:
    a = nan_mask(a)

    def func(win):
        w = win[np.isfinite(win)]
        if w.size == 0:
            return np.nan
        return float(np.nanmax(w) - np.nanmin(w))

    return generic_filter(a, func, size=3, mode="nearest").astype("float32")


def slope_aspect_from_dem(elev: np.ndarray, xres_m: float, yres_m: float) -> Tuple[np.ndarray, np.ndarray]:
    with np.errstate(all="ignore"):
        gy, gx = np.gradient(elev.astype("float32"), yres_m, xres_m)
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))
        aspect = np.degrees(np.arctan2(gy, -gx))
        aspect = (aspect + 360.0) % 360.0
    slope[~np.isfinite(elev)] = np.nan
    aspect[~np.isfinite(elev)] = np.nan
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
    hs[~np.isfinite(slope_deg)] = np.nan
    return (hs * 255.0).astype("float32")


def eastness_northness(aspect_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ar = np.radians(aspect_deg.astype("float32"))
    east = np.sin(ar).astype("float32")
    north = np.cos(ar).astype("float32")
    east[~np.isfinite(aspect_deg)] = np.nan
    north[~np.isfinite(aspect_deg)] = np.nan
    return east, north


class MemTileFetcher:
    def __init__(self, max_workers: int = 8) -> None:
        self.max_workers = int(max_workers)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "WA-Terrain-Preview/1.0"})
        self._memcache: Dict[Tuple[int, int, int], bytes] = {}

    def fetch_one(self, z: int, x: int, y: int) -> Optional[bytes]:
        key = (z, x, y)
        if key in self._memcache:
            return self._memcache[key]

        url = TERRAIN_URL.format(z=z, x=x, y=y)
        try:
            r = self.session.get(url, timeout=60)
            if r.status_code == 200 and looks_like_tiff(r.content, r.headers.get("Content-Type", "")):
                self._memcache[key] = r.content
                return r.content
            return None
        except requests.RequestException:
            return None

    def fetch_many(self, tiles: Iterable[Tuple[int, int, int]], desc: str = "tiles") -> Dict[Tuple[int, int, int], Optional[bytes]]:
        tiles = list(tiles)
        out: Dict[Tuple[int, int, int], Optional[bytes]] = {}
        pbar = tqdm(total=len(tiles), desc=f"download {desc}", leave=False)
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = {ex.submit(self.fetch_one, z, x, y): (z, x, y) for (z, x, y) in tiles}
            for fut in as_completed(futs):
                z, x, y = futs[fut]
                try:
                    out[(z, x, y)] = fut.result()
                except Exception:
                    out[(z, x, y)] = None
                pbar.update(1)
        pbar.close()
        return out


class TerrainEngine:
    def __init__(self, zoom: int = 14, workers: int = 8) -> None:
        self.zoom = int(zoom)
        self.fetcher = MemTileFetcher(max_workers=workers)
        self.workers = int(workers)

    def derive_tile_from_bytes(self, tif_bytes: bytes, dst_crs: CRS) -> Tuple[Dict[str, np.ndarray], dict, str]:
        with MemoryFile(tif_bytes) as memfile, memfile.open() as src:
            if (src.crs is None) or (src.transform is None):
                raise ValueError("tile not georeferenced")
            dst_transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
            elev = np.full((height, width), np.nan, dtype="float32")
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
                    src_nodata=src.nodata if src.nodata is not None else -3.4028235e38,
                    dst_nodata=np.nan,
                    init_dest_nodata=True,
                )

        elev = nan_mask(elev)
        xres_m, yres_m = abs(dst_transform.a), abs(dst_transform.e)

        slope, aspect = slope_aspect_from_dem(elev, xres_m, yres_m)
        east, north = eastness_northness(aspect)
        hill = hillshade_from_slope_aspect(slope, aspect, 315.0, 45.0)
        tri = tri_3x3_nan(elev)
        rough = roughness_3x3_nan(elev)
        k_prof, k_plan = curvature_profile_plan(elev, xres_m, yres_m)

        rasters = dict(
            elevation_m=elev,
            slope_deg=slope,
            aspect_deg=aspect,
            eastness=east,
            northness=north,
            hillshade_315_45=hill,
            tri_3x3=tri,
            roughness_3x3=rough,
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


def load_washington_bbox_5070() -> gpd.GeoDataFrame:
    wa_bbox_wgs84 = (-124.9, 45.5, -117.0, 49.05)
    wa_poly = gpd.GeoSeries([box(*wa_bbox_wgs84)], crs="EPSG:4326").to_crs(GRID_CRS)
    return gpd.GeoDataFrame(geometry=wa_poly, crs=GRID_CRS)


def generate_grid_within(geom_5070: gpd.GeoDataFrame, step_m: float) -> pd.DataFrame:
    minx, miny, maxx, maxy = geom_5070.total_bounds
    xs = np.arange(minx, maxx + step_m, step_m, dtype="float64")
    ys = np.arange(miny, maxy + step_m, step_m, dtype="float64")
    xx, yy = np.meshgrid(xs, ys)
    df = pd.DataFrame({"x_5070": xx.ravel(), "y_5070": yy.ravel()})
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["x_5070"], df["y_5070"], crs=GRID_CRS))
    inside = gdf.within(geom_5070.union_all())
    gdf = gdf[inside].copy()
    lon, lat = Transformer.from_crs(GRID_CRS, WGS84, always_xy=True).transform(gdf["x_5070"].values, gdf["y_5070"].values)
    gdf["lon"] = lon
    gdf["lat"] = lat
    return pd.DataFrame(gdf.drop(columns="geometry"))


def bucket_points_by_tile(latlon_iter: Iterable[Tuple[float, float]], z: int) -> Dict[Tuple[int, int, int], List[int]]:
    buckets: Dict[Tuple[int, int, int], List[int]] = {}
    for idx, (lat, lon) in enumerate(latlon_iter):
        x, y = lonlat_to_tile(lon, lat, z)
        buckets.setdefault((z, x, y), []).append(idx)
    return buckets


class WashingtonTerrain:
    VARS = [
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

    def __init__(self, *, workers: int = 8, zoom: int = 14) -> None:
        self.zoom = int(zoom)
        self.workers = int(workers)
        self.engine = TerrainEngine(zoom=self.zoom, workers=self.workers)

    def build_wa_sample_preview(self, *, n_points: int = 10, step_m: float = 20_000.0) -> pd.DataFrame:
        wa = load_washington_bbox_5070()
        grid_df = generate_grid_within(wa.geometry, step_m=float(step_m))
        if len(grid_df) == 0:
            raise RuntimeError("No grid points generated.")
        sample_df = grid_df.sample(n=min(int(n_points), len(grid_df)), random_state=42).reset_index(drop=True)

        latlon = list(zip(sample_df["lat"].to_numpy(), sample_df["lon"].to_numpy()))
        buckets = bucket_points_by_tile(latlon, self.zoom)
        tiles = sorted(buckets.keys())
        tile_bytes = self.engine.fetcher.fetch_many(tiles, desc="DEM tiles (preview-wa)")

        out_arrays: Dict[str, np.ndarray] = {k: np.full(len(sample_df), np.nan, dtype="float32") for k in self.VARS}

        def process_tile(tile):
            tif = tile_bytes.get(tile)
            if not tif:
                return tile, None
            try:
                rasters, meta, dst = self.engine.derive_tile_from_bytes(tif, GRID_CRS)
            except Exception:
                return tile, None
            idxs = buckets[tile]
            lon_vals = sample_df.loc[idxs, "lon"].values
            lat_vals = sample_df.loc[idxs, "lat"].values
            sampled = self.engine.sample_many(lon_vals, lat_vals, rasters, meta, dst)
            return tile, (idxs, sampled)

        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futs = {ex.submit(process_tile, t): t for t in tiles}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="derive+sample (preview-wa)", leave=False):
                try:
                    _tile, payload = fut.result()
                    if payload is None:
                        continue
                    idxs, sampled = payload
                    for name, vals in sampled.items():
                        if name in out_arrays:
                            out_arrays[name][idxs] = vals
                except Exception:
                    continue

        for k, arr in out_arrays.items():
            sample_df[k] = arr

        cols = ["lon", "lat", "x_5070", "y_5070"] + self.VARS
        print("\n=== TERRAIN preview-wa • WA grid sample from AWS (print-only) ===")
        print(sample_df[cols].round(3).to_string(index=False, max_rows=20))
        return sample_df

    def annotate_sample_preview(self, mushroom_dir: str, n_rows: int = 10) -> Optional[pd.DataFrame]:
        csv_path: Optional[str] = None
        if os.path.isdir(mushroom_dir):
            for d in sorted(os.scandir(mushroom_dir), key=lambda e: e.name.lower()):
                if d.is_dir() and d.name != "MushroomDataCombined":
                    p = os.path.join(d.path, "data.csv")
                    if os.path.exists(p):
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
        tile_bytes = self.engine.fetcher.fetch_many(tiles, desc="DEM tiles (preview-csv)")

        sampled_store: Dict[int, Dict[str, float]] = {i: {} for i in range(len(rows))}

        def process_tile(tile):
            tif = tile_bytes.get(tile)
            if not tif:
                return tile, None
            try:
                rasters, meta, dst = self.engine.derive_tile_from_bytes(tif, GRID_CRS)
            except Exception:
                return tile, None
            idxs = buckets[tile]
            lon_vals = np.array([float(rows[i].get("long", "nan")) for i in idxs], dtype="float64")
            lat_vals = np.array([float(rows[i].get("lat", "nan")) for i in idxs], dtype="float64")
            sampled = self.engine.sample_many(lon_vals, lat_vals, rasters, meta, dst)
            return tile, (idxs, sampled)

        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futs = {ex.submit(process_tile, t): t for t in tiles}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="derive+sample (preview-csv)", leave=False):
                try:
                    _tile, payload = fut.result()
                    if payload is None:
                        continue
                    idxs, sampled = payload
                    for j, i_row in enumerate(idxs):
                        for k in self.VARS:
                            val = float(sampled[k][j])
                            if not np.isnan(val):
                                sampled_store[i_row][k] = val
                except Exception:
                    continue

        out_df = pd.DataFrame(rows).copy()
        for k in self.VARS:
            out_df[k] = [sampled_store[i].get(k, np.nan) for i in range(len(rows))]

        front = [c for c in ["id", "lat", "long"] if c in out_df.columns]
        cols = front + [c for c in self.VARS if c in out_df.columns]
        print("\n=== TERRAIN preview-csv • Mushroom CSV sample annotated from AWS (print-only) ===")
        print(f"Source CSV: {csv_path}")
        print(out_df[cols].round(3).to_string(index=False, max_rows=len(out_df)))
        return out_df
