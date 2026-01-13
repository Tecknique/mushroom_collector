from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

from .collector import MushroomDataCollector, env_flag
from .logging_utils import get_logger

logger = get_logger("mushrooms")

DEFAULT_TAXA: Dict[int, str] = {
    56830: "morel",
    47348: "chanterelle",
    53713: "chickenOfTheWoods",
    48215: "lobster",
    48702: "boletus",
    48496: "oyster",
    49158: "lionsMane",
}

DEFAULT_WA_BBOX_WGS84: Tuple[float, float, float, float] = (-124.9, 45.5, -117.0, 49.05)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Mushroom pipeline tools")
    sub = p.add_subparsers(dest="tool", required=True)

    # -------- inat --------
    pin = sub.add_parser("inat", help="Collect iNaturalist observations CSVs")
    pin.add_argument("--output-dir", default=None, help="Local output directory (preferred).")
    pin.add_argument("--gcs-bucket", default=None, help="GCS bucket name (optional).")
    pin.add_argument("--inat-url", default=None)
    pin.add_argument("--inat-token", default=None)
    pin.add_argument("--start-date", default=None, help="YYYY-MM-DD")
    pin.add_argument("--end-date", default=None, help="YYYY-MM-DD")
    pin.add_argument("--per-page", type=int, default=200)
    pin.add_argument("--max-retries", type=int, default=5)
    pin.add_argument("--retry-base-delay", type=float, default=1.0)
    pin.add_argument("--sleep-between-pages", type=float, default=0.1)
    pin.add_argument("--workers", type=int, default=4)
    pin.add_argument("--n-shards", type=int, default=5)
    pin.add_argument("--shard-id", type=int, default=0)
    pin.add_argument("--progress-bar", action=argparse.BooleanOptionalAction, default=None)

    # -------- nwm --------
    pnwm = sub.add_parser("nwm", help="NWM land features")
    nsub = pnwm.add_subparsers(dest="mode", required=True)

    pb = nsub.add_parser("bbox", help="Build daily bbox geoparquet for a year (sharded by day)")
    pb.add_argument("--year", type=int, required=True)
    pb.add_argument("--out-root", default=None, help="Output root dir")
    pb.add_argument("--bbox", nargs=4, type=float, default=list(DEFAULT_WA_BBOX_WGS84), metavar=("MINLON","MINLAT","MAXLON","MAXLAT"))
    pb.add_argument("--max-retries", type=int, default=3)
    pb.add_argument("--heartbeat-seconds", type=int, default=30)
    pb.add_argument("--workers", type=int, default=4)
    pb.add_argument("--n-shards", type=int, default=5)
    pb.add_argument("--shard-id", type=int, default=0)
    pb.add_argument("--progress-bar", action=argparse.BooleanOptionalAction, default=None)

    pe = nsub.add_parser("enrich", help="Enrich iNat CSVs with NWM features (sharded by row index)")
    pe.add_argument("--mushroom-dir", default=None, help="Root dir containing taxon subfolders with data.csv")
    pe.add_argument("--workers", type=int, default=4)
    pe.add_argument("--n-shards", type=int, default=5)
    pe.add_argument("--shard-id", type=int, default=0)
    pe.add_argument("--progress-bar", action=argparse.BooleanOptionalAction, default=None)

    # -------- terrain --------
    pter = sub.add_parser("terrain", help="Terrain features from AWS elevation tiles")
    tsub = pter.add_subparsers(dest="mode", required=True)

    twa = tsub.add_parser("preview-wa", help="Sample a few grid points over WA bbox (print-only)")
    twa.add_argument("--n-points", type=int, default=10)
    twa.add_argument("--step-m", type=float, default=20_000.0)
    twa.add_argument("--zoom", type=int, default=int(os.getenv("TERRAIN_ZOOM", "14")))
    twa.add_argument("--workers", type=int, default=int(os.getenv("TERRAIN_WORKERS", "8")))
    twa.add_argument("--tile-cache", default=None, help="Tile cache directory")

    tcs = tsub.add_parser("preview-csv", help="Sample first taxon CSV rows and annotate (print-only)")
    tcs.add_argument("--mushroom-dir", default=None)
    tcs.add_argument("--n-rows", type=int, default=10)
    tcs.add_argument("--zoom", type=int, default=int(os.getenv("TERRAIN_ZOOM", "14")))
    tcs.add_argument("--workers", type=int, default=int(os.getenv("TERRAIN_WORKERS", "8")))
    tcs.add_argument("--tile-cache", default=None, help="Tile cache directory")

    tb = tsub.add_parser("build-wa-250m", help="Build WA 250m terrain parquet (sharded)")
    tb.add_argument("--out-dir", default=None, help="Output directory")
    tb.add_argument("--step-m", type=float, default=250.0)
    tb.add_argument("--parquet-name", default="wa_terrain_250m_v2.parquet")
    tb.add_argument("--zoom", type=int, default=int(os.getenv("TERRAIN_ZOOM", "14")))
    tb.add_argument("--workers", type=int, default=int(os.getenv("TERRAIN_WORKERS", "8")))
    tb.add_argument("--tile-cache", default=None, help="Tile cache directory")

    te = tsub.add_parser("enrich-csv", help="Enrich mushroom data.csv files in place (sharded)")
    te.add_argument("--mushroom-dir", default=None)
    te.add_argument("--zoom", type=int, default=int(os.getenv("TERRAIN_ZOOM", "14")))
    te.add_argument("--workers", type=int, default=int(os.getenv("TERRAIN_WORKERS", "8")))
    te.add_argument("--tile-cache", default=None, help="Tile cache directory")

    return p


def run_inat(args: argparse.Namespace) -> int:
    progress_bar = env_flag("PROGRESS_BAR", True) if args.progress_bar is None else args.progress_bar

    output_dir = args.output_dir or os.getenv("OUTPUT_DIR")
    gcs_bucket = args.gcs_bucket or os.getenv("GCS_BUCKET")
    output_path = Path(output_dir).expanduser() if output_dir else None

    if not output_path and not gcs_bucket:
        output_path = Path.cwd() / "mushroom_data"

    collector = MushroomDataCollector(
        inat_api_token=args.inat_token or os.getenv("INAT_API_TOKEN"),
        inat_url=args.inat_url or os.getenv("INAT_URL", "https://api.inaturalist.org/v1/observations"),
        user_agent=os.getenv("INAT_USER_AGENT", "MushroomCollector/1.0 (contact: you@example.com)"),
        output_dir=output_path,
        bucket_name=gcs_bucket if not output_path else None,
        start_date=args.start_date,
        end_date=args.end_date,
        per_page=args.per_page,
        max_retries=args.max_retries,
        retry_base_delay=args.retry_base_delay,
        progress_bar=progress_bar,
    )

    try:
        collector.collect_data_for_all_taxa(
            DEFAULT_TAXA,
            sleep_between_pages=args.sleep_between_pages,
            parallel_workers=int(os.getenv("PARALLEL_WORKERS", args.workers)),
            n_shards=int(os.getenv("N_SHARDS", args.n_shards)),
            shard_id=int(os.getenv("SHARD_ID", args.shard_id)),
        )
    finally:
        collector.close()

    logger.info("inat done.")
    return 0


def run_nwm_bbox(args: argparse.Namespace) -> int:
    from .nwm_land import NWMLandFeaturizer

    progress_bar = env_flag("PROGRESS_BAR", True) if args.progress_bar is None else args.progress_bar
    out_root = Path(args.out_root or os.getenv("NWM_OUT_ROOT") or (Path.cwd() / "nwm_land_out")).expanduser()

    nwm = NWMLandFeaturizer(workers=int(os.getenv("PARALLEL_WORKERS", args.workers)))
    bbox = tuple(args.bbox)
    nwm.run_bbox_year(
        year=args.year,
        bbox_wgs84=bbox,
        out_root=out_root,
        n_shards=int(os.getenv("N_SHARDS", args.n_shards)),
        shard_id=int(os.getenv("SHARD_ID", args.shard_id)),
        max_retries=args.max_retries,
        heartbeat_seconds=args.heartbeat_seconds,
        progress_bar=progress_bar,
    )
    return 0


def run_nwm_enrich(args: argparse.Namespace) -> int:
    from .nwm_land import NWMLandFeaturizer

    progress_bar = env_flag("PROGRESS_BAR", True) if args.progress_bar is None else args.progress_bar
    mushroom_dir = Path(args.mushroom_dir or os.getenv("MUSHROOM_DIR") or (Path.cwd() / "mushroom_data")).expanduser()

    nwm = NWMLandFeaturizer(workers=int(os.getenv("PARALLEL_WORKERS", args.workers)))
    nwm.enrich_inat_folder(
        root_dir=mushroom_dir,
        n_shards=int(os.getenv("N_SHARDS", args.n_shards)),
        shard_id=int(os.getenv("SHARD_ID", args.shard_id)),
        progress_bar=progress_bar,
    )
    return 0


def _terrain_cache_dir(arg: str | None) -> Path:
    return Path(arg or os.getenv("TERRAIN_TILE_CACHE") or ".terrain_tile_cache").expanduser()


def run_terrain_preview_wa(args: argparse.Namespace) -> int:
    from .terrain_wa import ShardSpec, WashingtonTerrain

    shard = ShardSpec.from_env()
    pipeline = WashingtonTerrain(workers=args.workers, zoom=args.zoom, cache_dir=_terrain_cache_dir(args.tile_cache))
    pipeline.preview_wa(n_points=args.n_points, step_m=args.step_m)
    return 0


def run_terrain_preview_csv(args: argparse.Namespace) -> int:
    from .terrain_wa import WashingtonTerrain

    mushroom_dir = Path(args.mushroom_dir or os.getenv("MUSHROOM_DIR") or (Path.cwd() / "mushroom_data")).expanduser()
    pipeline = WashingtonTerrain(workers=args.workers, zoom=args.zoom, cache_dir=_terrain_cache_dir(args.tile_cache))
    pipeline.preview_csv(mushroom_dir=mushroom_dir, n_rows=args.n_rows)
    return 0


def run_terrain_build_wa_250m(args: argparse.Namespace) -> int:
    from .terrain_wa import ShardSpec, WashingtonTerrain

    shard = ShardSpec.from_env()
    out_dir = Path(args.out_dir or os.getenv("TERRAIN_OUT_DIR") or (Path.cwd() / "terrain_data")).expanduser()
    pipeline = WashingtonTerrain(workers=args.workers, zoom=args.zoom, cache_dir=_terrain_cache_dir(args.tile_cache))
    pipeline.build_wa_parquet_250m(out_dir=out_dir, step_m=args.step_m, parquet_name=args.parquet_name, shard=shard)
    return 0


def run_terrain_enrich_csv(args: argparse.Namespace) -> int:
    from .terrain_wa import ShardSpec, WashingtonTerrain

    shard = ShardSpec.from_env()
    mushroom_dir = Path(args.mushroom_dir or os.getenv("MUSHROOM_DIR") or (Path.cwd() / "mushroom_data")).expanduser()
    pipeline = WashingtonTerrain(workers=args.workers, zoom=args.zoom, cache_dir=_terrain_cache_dir(args.tile_cache))
    pipeline.enrich_mushroom_csvs(mushroom_dir=mushroom_dir, shard=shard)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)

    if args.tool == "inat":
        return run_inat(args)

    if args.tool == "nwm":
        if args.mode == "bbox":
            return run_nwm_bbox(args)
        if args.mode == "enrich":
            return run_nwm_enrich(args)

    if args.tool == "terrain":
        if args.mode == "preview-wa":
            return run_terrain_preview_wa(args)
        if args.mode == "preview-csv":
            return run_terrain_preview_csv(args)
        if args.mode == "build-wa-250m":
            return run_terrain_build_wa_250m(args)
        if args.mode == "enrich-csv":
            return run_terrain_enrich_csv(args)

    raise SystemExit("Invalid arguments")
