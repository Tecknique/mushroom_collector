from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict

from .collector import MushroomDataCollector, _env_flag
from .config import CollectorConfig
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


def build_config_from_env_and_args(args: argparse.Namespace) -> CollectorConfig:
    progress_bar = _env_flag("PROGRESS_BAR", True) if args.progress_bar is None else args.progress_bar

    output_dir = args.output_dir or os.getenv("OUTPUT_DIR")
    gcs_bucket = args.gcs_bucket or os.getenv("GCS_BUCKET")

    output_path = Path(output_dir).expanduser() if output_dir else None

    return CollectorConfig(
        inat_url=args.inat_url or os.getenv("INAT_URL", "https://api.inaturalist.org/v1/observations"),
        inat_api_token=args.inat_token or os.getenv("INAT_API_TOKEN"),
        user_agent=os.getenv("INAT_USER_AGENT", "MushroomCollector/1.0 (contact: you@example.com)"),
        output_dir=output_path,
        gcs_bucket=gcs_bucket if not output_path else None,
        start_date=args.start_date,
        end_date=args.end_date,
        per_page=args.per_page,
        max_retries=args.max_retries,
        retry_base_delay=args.retry_base_delay,
        sleep_between_pages=args.sleep_between_pages,
        parallel_workers=int(os.getenv("PARALLEL_WORKERS", args.workers)),
        n_shards=int(os.getenv("N_SHARDS", args.n_shards)),
        shard_id=int(os.getenv("SHARD_ID", args.shard_id)),
        progress_bar=progress_bar,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="iNaturalist Mushroom Data Collector")
    p.add_argument("--output-dir", default=None, help="Local output directory (preferred).")
    p.add_argument("--gcs-bucket", default=None, help="GCS bucket name (optional).")

    p.add_argument("--inat-url", default=None)
    p.add_argument("--inat-token", default=None)

    p.add_argument("--start-date", default=None, help="YYYY-MM-DD")
    p.add_argument("--end-date", default=None, help="YYYY-MM-DD")

    p.add_argument("--per-page", type=int, default=200)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--retry-base-delay", type=float, default=1.0)

    p.add_argument("--sleep-between-pages", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--n-shards", type=int, default=5)
    p.add_argument("--shard-id", type=int, default=0)

    p.add_argument("--progress-bar", action=argparse.BooleanOptionalAction, default=None)

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = build_config_from_env_and_args(args)

    if not cfg.output_dir and not cfg.gcs_bucket:
        cfg = CollectorConfig(**{**cfg.__dict__, "output_dir": Path.cwd() / "mushroom_data"})

    collector = MushroomDataCollector(
        inat_api_token=cfg.inat_api_token,
        inat_url=cfg.inat_url,
        user_agent=cfg.user_agent,
        output_dir=cfg.output_dir,
        bucket_name=cfg.gcs_bucket,
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        per_page=cfg.per_page,
        max_retries=cfg.max_retries,
        retry_base_delay=cfg.retry_base_delay,
        progress_bar=cfg.progress_bar,
    )

    try:
        collector.collect_data_for_all_taxa(
            DEFAULT_TAXA,
            sleep_between_pages=cfg.sleep_between_pages,
            parallel_workers=cfg.parallel_workers,
            n_shards=cfg.n_shards,
            shard_id=cfg.shard_id,
        )
    finally:
        collector.close()

    logger.info("Done.")
    return 0
