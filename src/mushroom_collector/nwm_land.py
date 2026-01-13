from __future__ import annotations

import hashlib
import math
import os
import pathlib
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm

import geopandas as gpd
from shapely.geometry import Point, box, mapping
from pyproj import CRS
from rasterio.features import geometry_mask
from rasterio.transform import from_origin

import adlfs

from .logging_utils import get_logger

logger = get_logger("nwm")

W_SOIL = xr.DataArray(np.array([0.10, 0.30, 0.60, 1.00]) / 2.0, dims=["soil_layers_stag"])


def crs_from_ds(ds: xr.Dataset) -> CRS:
    gm_var = None
    for v in ds.data_vars:
        gm = ds[v].attrs.get("grid_mapping")
        if gm in ds.variables:
            gm_var = gm
            break
    if gm_var is None and "crs" in ds.variables:
        gm_var = "crs"
    if gm_var:
        try:
            return CRS.from_cf(ds[gm_var].attrs)
        except Exception:
            wkt = ds[gm_var].attrs.get("spatial_ref") or ds[gm_var].attrs.get("crs_wkt")
            if wkt:
                return CRS.from_wkt(wkt)
    raise RuntimeError("Could not determine CRS from dataset.")


def grid_window_and_transform(ds: xr.Dataset, geom_proj) -> Tuple[Tuple[slice, slice], object]:
    x = ds["x"].values
    y = ds["y"].values
    dx = float(np.nanmedian(np.diff(x)))
    dy = float(np.nanmedian(np.diff(y)))
    ascending_y = (y.size < 2) or (y[1] > y[0])

    minx, miny, maxx, maxy = geom_proj.bounds
    west = float(x.min() - abs(dx) / 2.0)
    north = float((y.max() + abs(dy) / 2.0) if ascending_y else (y.min() + abs(dy) / 2.0))

    col0 = max(0, int(math.floor((minx - west) / abs(dx))))
    col1 = min(len(x), int(math.ceil((maxx - west) / abs(dx))))

    if ascending_y:
        row0 = max(0, int(math.floor((north - maxy) / abs(dy))))
        row1 = min(len(y), int(math.ceil((north - miny) / abs(dy))))
    else:
        row0 = max(0, int(math.floor((north - miny) / abs(dy))))
        row1 = min(len(y), int(math.ceil((north - maxy) / abs(dy))))

    transform = from_origin(west + col0 * abs(dx), north - row0 * abs(dy), abs(dx), abs(dy))
    return (slice(row0, row1), slice(col0, col1)), transform


def geometry_mask_ok(transform, shape_rc: Tuple[int, int], geom_proj) -> np.ndarray:
    return ~geometry_mask([mapping(geom_proj)], out_shape=shape_rc, transform=transform, invert=True)


def cell_centers(ds: xr.Dataset, rwin: slice, cwin: slice) -> Tuple[np.ndarray, np.ndarray]:
    x = ds["x"].values[cwin]
    y = ds["y"].values[rwin]
    return np.meshgrid(x, y)


def t0(da: xr.DataArray) -> xr.DataArray:
    return da.isel(time=0) if "time" in da.dims else da


class NWMLandFeaturizer:
    """Reads NOAA NWM 'analysis_assim land' netCDFs from Azure public blob (account: noaanwm)."""

    def __init__(self, *, workers: int = 4, cache_dir: Optional[pathlib.Path] = None) -> None:
        self.fs = adlfs.AzureBlobFileSystem(account_name="noaanwm")
        self.cycles = ["00", "06", "12", "18"]
        self.cache_dir = cache_dir or pathlib.Path(os.getenv("NWM_CACHE_DIR", ".nwm_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.workers = int(workers)

    @staticmethod
    def land_key(ymd: str, cc: str) -> str:
        return f"nwm/nwm.{ymd}/analysis_assim/nwm.t{cc}z.analysis_assim.land.tm00.conus.nc"

    def exists(self, key: str) -> bool:
        return self.fs.exists(key)

    def cache_path(self, key: str) -> pathlib.Path:
        h = hashlib.md5(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{h}.nc"

    def ensure_local(self, key: str) -> pathlib.Path:
        local = self.cache_path(key)
        if local.exists() and local.stat().st_size > 0:
            return local

        tmp = local.with_suffix(".nc.__part__")
        with self.fs.open(key, "rb") as src, tmp.open("wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
        tmp.replace(local)
        return local

    def open_ds(self, key: str) -> xr.Dataset:
        return xr.open_dataset(self.ensure_local(key), engine="netcdf4", chunks="auto", cache=False)

    @staticmethod
    def concat_da(seq: List[xr.DataArray]) -> xr.DataArray:
        return xr.concat(seq, dim="cycle", join="override")

    @staticmethod
    def stack_mean(seq: List[xr.DataArray], like: xr.DataArray) -> xr.DataArray:
        if not seq:
            return xr.full_like(like, np.nan)
        return xr.concat(seq, dim="cycle", join="override").mean("cycle")

    def land_features_window(self, ymd: str, geom_proj):
        land_key_for_window = None
        for cc in self.cycles:
            k = self.land_key(ymd, cc)
            if self.exists(k):
                land_key_for_window = k
                break
        if not land_key_for_window:
            raise RuntimeError(f"No land file found for {ymd}")

        ds0 = self.open_ds(land_key_for_window)
        crs = crs_from_ds(ds0)
        (rwin, cwin), transform = grid_window_and_transform(ds0, geom_proj)
        if (rwin.stop - rwin.start) <= 0 or (cwin.stop - cwin.start) <= 0:
            ds0.close()
            raise RuntimeError("Empty window (bbox out of grid?)")

        st_top, st_l2, st_col, sm_top, sm_col = [], [], [], [], []
        soilsat, soilice, sneqv, snowt, snowh, fsno, isnow, qrain, qsnow, acsnom, edir, accet = [], [], [], [], [], [], [], [], [], [], [], []

        for cc in self.cycles:
            k = self.land_key(ymd, cc)
            if not self.exists(k):
                continue
            dsl = self.open_ds(k).isel(y=rwin, x=cwin)

            if "SOIL_T" in dsl:
                st_top.append(t0(dsl["SOIL_T"].isel(soil_layers_stag=0) - 273.15).load())
                st_l2.append(t0(dsl["SOIL_T"].isel(soil_layers_stag=1) - 273.15).load())
                st_col.append(t0((dsl["SOIL_T"] * W_SOIL).sum("soil_layers_stag") - 273.15).load())

            if "SOIL_M" in dsl:
                sm_top.append(t0(dsl["SOIL_M"].isel(soil_layers_stag=0)).load())
                sm_col.append(t0((dsl["SOIL_M"] * W_SOIL).sum("soil_layers_stag")).load())

            for name, lst in [
                ("SOILSAT_TOP", soilsat),
                ("SOILICE", soilice),
                ("SNEQV", sneqv),
                ("SNOWT_AVG", snowt),
                ("SNOWH", snowh),
                ("FSNO", fsno),
                ("ISNOW", isnow),
                ("QRAIN", qrain),
                ("QSNOW", qsnow),
                ("ACSNOM", acsnom),
                ("EDIR", edir),
                ("ACCET", accet),
            ]:
                if name in dsl:
                    lst.append(t0(dsl[name]).load())
            dsl.close()

        if not st_top:
            ds0.close()
            raise RuntimeError("SOIL_T missing for chosen day/cycles")

        st_top_stack = self.concat_da(st_top)
        st_l2_stack = self.concat_da(st_l2)
        st_col_stack = self.concat_da(st_col)
        sm_top_stack = self.concat_da(sm_top)
        sm_col_stack = self.concat_da(sm_col)

        st_top_min, st_top_max, st_top_mean = st_top_stack.min("cycle"), st_top_stack.max("cycle"), st_top_stack.mean("cycle")
        st_l2_min, st_l2_max, st_l2_mean = st_l2_stack.min("cycle"), st_l2_stack.max("cycle"), st_l2_stack.mean("cycle")
        st_col_mean = st_col_stack.mean("cycle")
        sm_top_mean = sm_top_stack.mean("cycle")
        sm_col_mean = sm_col_stack.mean("cycle")

        edir_mean = self.stack_mean(edir, st_top_mean)
        accet_mean = self.stack_mean(accet, st_top_mean)
        soilsat_mean = self.stack_mean(soilsat, st_top_mean)
        soilice_mean = self.stack_mean(soilice, st_top_mean)
        sneqv_mean = self.stack_mean(sneqv, st_top_mean)
        snowt_mean_c = self.stack_mean(snowt, st_top_mean) - 273.15
        snowh_mean_m = self.stack_mean(snowh, st_top_mean)
        fsno_mean = self.stack_mean(fsno, st_top_mean)
        isnow_mean = self.stack_mean(isnow, st_top_mean)
        qrain_mean = self.stack_mean(qrain, st_top_mean)
        qsnow_mean = self.stack_mean(qsnow, st_top_mean)
        acsnom_mean = self.stack_mean(acsnom, st_top_mean)

        xx, yy = cell_centers(ds0, rwin, cwin)
        mask = geometry_mask_ok(transform, st_top_mean.shape, geom_proj)
        ds0.close()

        sel = mask
        if int(sel.sum()) == 0:
            raise RuntimeError("Mask selected 0 cells (unexpected)")

        def F(a: xr.DataArray) -> np.ndarray:
            return a.values[sel]

        data = dict(
            soil_t_top_min_c=F(st_top_min),
            soil_t_top_max_c=F(st_top_max),
            soil_t_top_mean_c=F(st_top_mean),
            soil_t_l2_min_c=F(st_l2_min),
            soil_t_l2_max_c=F(st_l2_max),
            soil_t_l2_mean_c=F(st_l2_mean),
            soil_t_col_mean_c=F(st_col_mean),
            soil_m_top_mean=F(sm_top_mean),
            soil_m_col_mean=F(sm_col_mean),
            soilsat_top_mean=F(soilsat_mean),
            soilice_mean=F(soilice_mean),
            fsno_mean=F(fsno_mean),
            isnow_mean=F(isnow_mean),
            sneqv_mean_kgm2=F(sneqv_mean),
            snowh_mean_m=F(snowh_mean_m),
            snowt_avg_mean_c=F(snowt_mean_c),
            acsnom_mean_mm=F(acsnom_mean),
            qrain_mean_mms=F(qrain_mean),
            qsnow_mean_mms=F(qsnow_mean),
            edir_mean_kgm2s=F(edir_mean),
            accet_mean_mm=F(accet_mean),
        )

        geom = [Point(float(x), float(y)) for x, y in zip(xx[sel], yy[sel])]
        return data, geom, crs

    def run_bbox_year(
        self,
        *,
        year: int,
        bbox_wgs84: Tuple[float, float, float, float],
        out_root: pathlib.Path,
        n_shards: int,
        shard_id: int,
        max_retries: int = 3,
        heartbeat_seconds: int = 30,
        progress_bar: bool = True,
    ) -> None:
        out_root = pathlib.Path(out_root)
        out_root.mkdir(parents=True, exist_ok=True)

        progress_csv = out_root / f"_progress_land_shard{shard_id}.csv"
        if not progress_csv.exists():
            progress_csv.write_text("date,status,rows,path,attempts,elapsed_sec\n", encoding="utf-8")

        sample_ymd = None
        for m in (1, 3, 6, 9, 12):
            ymd = f"{year}{m:02d}15"
            if self.exists(self.land_key(ymd, "00")):
                sample_ymd = ymd
                break
        if sample_ymd is None:
            raise SystemExit(f"No NWM land files found for {year} on Azure 'noaanwm'.")

        ds0 = self.open_ds(self.land_key(sample_ymd, "00"))
        crs = crs_from_ds(ds0)
        ds0.close()

        minx, miny, maxx, maxy = bbox_wgs84
        geom_proj = gpd.GeoSeries([box(minx, miny, maxx, maxy)], crs="EPSG:4326").to_crs(crs).iloc[0]

        all_days = [date(year, 1, 1) + timedelta(days=i) for i in range(366) if (date(year, 1, 1) + timedelta(days=i)).year == year]
        shard_days = [d for i, d in enumerate(all_days) if (i % n_shards) == shard_id]
        logger.info("[DAYS] total=%s this_shard=%s", len(all_days), len(shard_days))

        root = out_root / "geoparquet"
        root.mkdir(parents=True, exist_ok=True)

        def day_paths(d: date):
            ddir = root / f"date={d.isoformat()}"
            return ddir, ddir / "wa_nwm_daily.parquet", ddir / "_SUCCESS", ddir / "_STARTED", ddir / "_FAIL"

        hb = out_root / f"_heartbeat_land_shard{shard_id}.txt"
        ok = 0
        fails: List[Tuple[str, str]] = []
        last_hb = 0.0

        it = tqdm(shard_days, desc=f"{year} shard {shard_id} (land)") if progress_bar else shard_days
        for d in it:
            ddir, outp, succ, started, failm = day_paths(d)

            if succ.exists() and outp.exists():
                try:
                    rows = len(gpd.read_parquet(outp))
                except Exception:
                    rows = -1
                with progress_csv.open("a", encoding="utf-8") as f:
                    f.write(f"{d.isoformat()},ok,{rows},{outp},0,0\n")
                continue

            if outp.exists() and outp.stat().st_size > 0:
                succ.write_text("ok", encoding="utf-8")
                try:
                    if failm.exists():
                        failm.unlink()
                except Exception:
                    pass
                try:
                    rows = len(gpd.read_parquet(outp))
                except Exception:
                    rows = -1
                with progress_csv.open("a", encoding="utf-8") as f:
                    f.write(f"{d.isoformat()},ok,{rows},{outp},0,0\n")
                continue

            ddir.mkdir(parents=True, exist_ok=True)
            started.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
            try:
                if failm.exists():
                    failm.unlink()
            except Exception:
                pass

            err: Optional[str] = None
            t0 = time.time()
            attempts = 0

            for attempt in range(1, max_retries + 1):
                attempts = attempt
                try:
                    ymd = d.strftime("%Y%m%d")
                    data, geom, crs2 = self.land_features_window(ymd, geom_proj)
                    gdf = gpd.GeoDataFrame(pd.DataFrame({"date": [d.isoformat()] * len(geom), **data}), geometry=geom, crs=crs2)
                    if len(gdf):
                        outp.parent.mkdir(parents=True, exist_ok=True)
                        gdf.to_parquet(outp, index=False)
                        succ.write_text("ok", encoding="utf-8")
                        try:
                            if failm.exists():
                                failm.unlink()
                        except Exception:
                            pass
                        rows = len(gdf)
                        elapsed = round(time.time() - t0, 2)
                        with progress_csv.open("a", encoding="utf-8") as f:
                            f.write(f"{d.isoformat()},ok,{rows},{outp},{attempt},{elapsed}\n")
                    else:
                        (ddir / "_EMPTY").write_text("", encoding="utf-8")
                        elapsed = round(time.time() - t0, 2)
                        with progress_csv.open("a", encoding="utf-8") as f:
                            f.write(f"{d.isoformat()},empty,0,{ddir},{attempt},{elapsed}\n")
                    ok += 1
                    err = None
                    break
                except Exception as e:
                    err = repr(e)
                    time.sleep(5 * attempt)

            if err:
                elapsed = round(time.time() - t0, 2)
                fails.append((d.isoformat(), err))
                failm.write_text(err, encoding="utf-8")
                with progress_csv.open("a", encoding="utf-8") as f:
                    f.write(f"{d.isoformat()},fail,,-,{attempts},{elapsed}\n")

            tnow = time.time()
            if tnow - last_hb >= heartbeat_seconds:
                hb.write_text(
                    f"last_day={d.isoformat()} ok={ok} fail={len(fails)} ts={time.strftime('%Y-%m-%d %H:%M:%S')}",
                    encoding="utf-8",
                )
                last_hb = tnow

        (out_root / f"_summary_land_shard{shard_id}.txt").write_text(
            "OK: %d\nFAILS: %d\n%s" % (ok, len(fails), "\n".join(f"{d}: {m}" for d, m in fails)),
            encoding="utf-8",
        )

        logger.info("[DONE] shard %s: ok=%s fails=%s → %s", shard_id, ok, len(fails), root)
        logger.info("[LOG] Progress CSV: %s", progress_csv)

    def enrich_inat_folder(
        self,
        *,
        root_dir: pathlib.Path,
        n_shards: int,
        shard_id: int,
        input_name: str = "data.csv",
        lat_col: str = "lat",
        lon_col: str = "long",
        date_col: str = "timestamp",
        progress_bar: bool = True,
    ) -> None:
        root = pathlib.Path(root_dir)
        subdirs = [p for p in root.iterdir() if p.is_dir() and p.name != "MushroomDataCombined"]
        logger.info("[enrich] taxa folders=%s shard %s/%s", len(subdirs), shard_id, n_shards)
        if not subdirs:
            raise SystemExit("No taxon subfolders found.")

        def process_one(sub: pathlib.Path):
            csv_path = sub / input_name
            if not csv_path.exists():
                return None

            df = pd.read_csv(csv_path)
            if df.empty:
                return None

            df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date
            df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
            df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")

            keep = df.dropna(subset=[lat_col, lon_col, date_col]).reset_index(drop=True)
            if keep.empty:
                return None

            n = len(keep)
            base = n // n_shards
            rem = n % n_shards
            start = shard_id * base + min(shard_id, rem)
            length = base + (1 if shard_id < rem else 0)
            end = start + length

            shard_df = keep.iloc[start:end].copy()
            if shard_df.empty:
                return (sub.name, 0, 0)

            crs_cache: Dict[int, CRS] = {}
            filled = 0
            days = sorted(shard_df[date_col].unique())

            day_it = tqdm(days, desc=f"{sub.name} • shard {shard_id}", leave=False) if progress_bar else days
            for d in day_it:
                ymd = d.strftime("%Y%m%d")
                land_key = None
                for cc in self.cycles:
                    k = self.land_key(ymd, cc)
                    if self.exists(k):
                        land_key = k
                        break
                if land_key is None:
                    continue

                yr = int(ymd[:4])
                if yr not in crs_cache:
                    ds = self.open_ds(land_key)
                    crs_cache[yr] = crs_from_ds(ds)
                    ds.close()
                crs = crs_cache[yr]

                subd = shard_df[shard_df[date_col] == d]
                pad = 0.1
                b = box(
                    subd[lon_col].min() - pad,
                    subd[lat_col].min() - pad,
                    subd[lon_col].max() + pad,
                    subd[lat_col].max() + pad,
                )
                geom_proj = gpd.GeoSeries([b], crs="EPSG:4326").to_crs(crs).iloc[0]

                data, geom, _ = self.land_features_window(ymd, geom_proj)
                gdf_grid = gpd.GeoDataFrame(pd.DataFrame(data), geometry=geom, crs=crs)
                if gdf_grid.empty:
                    continue

                pts = gpd.GeoDataFrame(
                    subd[[lat_col, lon_col]],
                    geometry=gpd.points_from_xy(subd[lon_col], subd[lat_col], crs="EPSG:4326"),
                ).to_crs(crs)

                gx = np.vstack([gdf_grid.geometry.x.values, gdf_grid.geometry.y.values]).T
                q = np.vstack([pts.geometry.x.values, pts.geometry.y.values]).T

                idx = np.empty(len(q), dtype=np.int64)
                ch = 4000
                for i0 in range(0, len(q), ch):
                    qq = q[i0 : i0 + ch]
                    d2 = ((qq[:, None, :] - gx[None, :, :]) ** 2).sum(axis=2)
                    idx[i0 : i0 + ch] = np.argmin(d2, axis=1)

                feat_cols = [c for c in gdf_grid.columns if c != "geometry"]
                sampled = gdf_grid.iloc[idx][feat_cols].set_index(subd.index)

                for c in feat_cols:
                    if c not in df.columns:
                        df[c] = np.nan
                df.loc[sampled.index, sampled.columns] = sampled.values
                filled += len(sampled)

            parts_dir = sub / ".nwm_land_parts"
            parts_dir.mkdir(parents=True, exist_ok=True)

            part_path = parts_dir / f"part_{shard_id}.csv"
            df_part = df.iloc[keep.index[start:end]]
            df_part.to_csv(part_path, index=False)
            (parts_dir / f"part_{shard_id}.done").write_text("ok", encoding="utf-8")

            all_parts = [parts_dir / f"part_{i}.csv" for i in range(n_shards)]
            all_done = [parts_dir / f"part_{i}.done" for i in range(n_shards)]
            if all(p.exists() for p in all_parts) and all(p.exists() for p in all_done):
                frames = [pd.read_csv(p) for p in all_parts]
                combined = pd.concat(frames, ignore_index=True)
                bak = sub / f"{input_name}.bak"
                if not bak.exists():
                    try:
                        os.replace(sub / input_name, bak)
                    except Exception:
                        pass
                combined.to_csv(sub / input_name, index=False)
                for p in all_parts + all_done:
                    try:
                        p.unlink()
                    except Exception:
                        pass
                try:
                    parts_dir.rmdir()
                except Exception:
                    pass

            return (sub.name, len(shard_df), filled)

        results = []
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futs = {ex.submit(process_one, sub): sub.name for sub in subdirs}
            it = tqdm(as_completed(futs), total=len(futs), desc=f"taxa shard {shard_id}") if progress_bar else as_completed(futs)
            for fut in it:
                try:
                    r = fut.result()
                    if r:
                        results.append(r)
                except Exception as e:
                    logger.error("[%s] failed: %s", futs[fut], e)
                    traceback.print_exc()

        logger.info("[enrich] summary (taxon, shard_rows, filled_rows):")
        for r in results:
            logger.info("  %s", r)
