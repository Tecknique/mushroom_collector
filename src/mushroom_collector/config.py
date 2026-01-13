from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class CollectorConfig:
    inat_url: str = "https://api.inaturalist.org/v1/observations"
    inat_api_token: Optional[str] = None
    user_agent: str = "MushroomCollector/1.0 (contact: you@example.com)"

    output_dir: Optional[Path] = None
    gcs_bucket: Optional[str] = None

    nelat: Optional[float] = None
    nelng: Optional[float] = None
    swlat: Optional[float] = None
    swlng: Optional[float] = None

    start_date: Optional[str] = None  # YYYY-MM-DD
    end_date: Optional[str] = None    # YYYY-MM-DD

    per_page: int = 200
    max_retries: int = 5
    retry_base_delay: float = 1.0

    sleep_between_pages: float = 0.1
    parallel_workers: int = 4

    n_shards: int = 5
    shard_id: int = 0
    progress_bar: bool = True
