from __future__ import annotations

import csv
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from tqdm.auto import tqdm

from .logging_utils import get_logger

logger = get_logger("mushrooms")

try:
    from google.cloud import storage as gcs_storage  # type: ignore
except Exception:
    gcs_storage = None  # type: ignore


def env_flag(name: str, default: bool) -> bool:
    v = os.getenv(name, "true" if default else "false").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


@dataclass(frozen=True)
class OutputTarget:
    mode: str  # "local" | "gcs"
    output_dir: Optional[Path] = None
    bucket_name: Optional[str] = None


class MushroomDataCollector:
    """
    Collect iNaturalist observations into per-taxon CSVs + a combined CSV.
    Supports local filesystem or GCS (optional dependency).
    """

    def __init__(
        self,
        *,
        inat_api_token: Optional[str],
        inat_url: str,
        user_agent: str,
        output_dir: Optional[Path] = None,
        bucket_name: Optional[str] = None,
        nelat: Optional[float] = None,
        nelng: Optional[float] = None,
        swlat: Optional[float] = None,
        swlng: Optional[float] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        per_page: int = 200,
        max_retries: int = 5,
        retry_base_delay: float = 1.0,
        progress_bar: bool = True,
    ) -> None:
        self.inat_url = inat_url.rstrip("/")
        self.per_page = max(1, min(200, per_page))
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.progress_bar = progress_bar

        self.nelat, self.nelng, self.swlat, self.swlng = nelat, nelng, swlat, swlng
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

        self.headers = {"User-Agent": user_agent}
        if inat_api_token:
            self.headers["Authorization"] = f"Bearer {inat_api_token}"

        self.session = requests.Session()
        self.session.headers.update(self.headers)

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            self.target = OutputTarget(mode="local", output_dir=output_dir)
            logger.info("Output: local directory → %s", str(output_dir))
            self.bucket = None
            self.storage_client = None
        else:
            if not bucket_name:
                raise ValueError("Provide output_dir (local) or bucket_name (GCS).")
            if gcs_storage is None:
                raise RuntimeError("google-cloud-storage not installed but bucket_name provided.")
            self.target = OutputTarget(mode="gcs", bucket_name=bucket_name)
            self.storage_client = gcs_storage.Client()
            self.bucket = self.storage_client.bucket(bucket_name)
            logger.info("Output: GCS bucket → gs://%s", bucket_name)

    def close(self) -> None:
        try:
            self.session.close()
        except Exception:
            pass

    def _request(self, params: Dict) -> Optional[requests.Response]:
        attempt = 0
        while True:
            try:
                resp = self.session.get(self.inat_url, params=params, timeout=60)
                if resp.status_code == 200:
                    return resp

                if resp.status_code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                    delay = self.retry_base_delay * (2**attempt) + random.random()
                    if not self.progress_bar:
                        logger.warning("Transient %s; backoff %.2fs", resp.status_code, delay)
                    time.sleep(delay)
                    attempt += 1
                    continue

                if not self.progress_bar:
                    logger.error("HTTP %s: %s", resp.status_code, getattr(resp, "text", ""))
                return resp
            except requests.RequestException:
                if attempt < self.max_retries:
                    delay = self.retry_base_delay * (2**attempt) + random.random()
                    if not self.progress_bar:
                        logger.warning("Request error; retry in %.2fs", delay)
                    time.sleep(delay)
                    attempt += 1
                    continue
                if not self.progress_bar:
                    logger.exception("Request failed after retries")
                return None

    def _fetch_observations(self, taxon_id: int, last_id: Optional[int]) -> List[Dict]:
        params: Dict = {
            "taxon_id": taxon_id,
            "photos": True,
            "per_page": self.per_page,
            "order_by": "id",
            "order": "desc",
            "quality_grade": "research",
        }
        if self.start_date:
            params["d1"] = self.start_date.strftime("%Y-%m-%d")
        if self.end_date:
            params["d2"] = self.end_date.strftime("%Y-%m-%d")
        if None not in (self.nelat, self.nelng, self.swlat, self.swlng):
            params.update({"nelat": self.nelat, "nelng": self.nelng, "swlat": self.swlat, "swlng": self.swlng})
        if last_id:
            params["id_below"] = last_id

        resp = self._request(params)
        if not resp or resp.status_code != 200:
            return []
        try:
            payload = resp.json()
        except json.JSONDecodeError:
            return []
        return payload.get("results", []) or []

    @staticmethod
    def process_observations(observations: List[Dict], taxon_name: str) -> List[Dict]:
        rows: List[Dict] = []
        prefix = (taxon_name[:3] or "tax").replace(" ", "_")
        for obs in observations:
            try:
                location = obs.get("location", "") or ""
                lat, lon = ("", "")
                if "," in location:
                    lat, lon = location.split(",", 1)

                observed_date = (
                    (obs.get("observed_on_details") or {}).get("date")
                    or (obs.get("time_observed_at") or "").split("T")[0]
                    or (obs.get("observed_on") or "")
                )

                photos = obs.get("photos") or []
                photo_url = ""
                if photos:
                    url = photos[0].get("url") or ""
                    photo_url = url.replace("square", "original") if url else ""

                rows.append(
                    {
                        "combined_uid": f"{prefix}_{obs['id']}",
                        "uid": obs["id"],
                        "timestamp": observed_date,
                        "lat": lat.strip(),
                        "long": lon.strip(),
                        "picture_url": photo_url,
                        "name": taxon_name,
                    }
                )
            except Exception:
                continue
        return rows

    def _save_csv_rows_to_local(self, rows: List[Dict], taxon_name: str) -> None:
        if not rows or not self.target.output_dir:
            return
        taxon_dir = self.target.output_dir / taxon_name
        taxon_dir.mkdir(parents=True, exist_ok=True)
        file_path = taxon_dir / "data.csv"
        with file_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _combine_local_csvs(self) -> None:
        if not self.target.output_dir:
            return
        combined_dir = self.target.output_dir / "MushroomDataCombined"
        combined_dir.mkdir(parents=True, exist_ok=True)
        combined_path = combined_dir / "data.csv"

        headers: Optional[List[str]] = None
        combined_rows: List[List[str]] = []

        for entry in self.target.output_dir.iterdir():
            if not entry.is_dir() or entry.name == "MushroomDataCombined":
                continue
            csv_path = entry / "data.csv"
            if not csv_path.exists():
                continue

            with csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                try:
                    file_headers = next(reader)
                except StopIteration:
                    continue
                if headers is None:
                    headers = file_headers
                    combined_rows.append(headers)
                elif headers != file_headers:
                    continue
                combined_rows.extend(list(reader))

        if combined_rows:
            with combined_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(combined_rows)

    def _save_csv_rows_to_gcs(self, rows: List[Dict], taxon_name: str) -> None:
        if not rows or not self.bucket:
            return
        csv_buffer = StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        blob_name = f"MushroomLocationData/{taxon_name}/data.csv"
        self.bucket.blob(blob_name).upload_from_string(csv_buffer.getvalue(), content_type="text/csv")

    def _combine_gcs_csvs(self) -> None:
        if not self.storage_client or not self.target.bucket_name or not self.bucket:
            return
        combined_blob_name = "MushroomLocationData/MushroomDataCombined/data.csv"
        prefix = "MushroomLocationData/"

        combined_rows: List[List[str]] = []
        headers: Optional[List[str]] = None

        for blob in self.storage_client.list_blobs(self.target.bucket_name, prefix=prefix):
            if blob.name == combined_blob_name or not blob.name.endswith("data.csv"):
                continue
            content = blob.download_as_text()
            f = StringIO(content)
            reader = csv.reader(f)
            try:
                file_headers = next(reader)
            except StopIteration:
                continue
            if headers is None:
                headers = file_headers
                combined_rows.append(headers)
            elif headers != file_headers:
                continue
            combined_rows.extend(list(reader))

        if combined_rows:
            out = StringIO()
            writer = csv.writer(out)
            writer.writerows(combined_rows)
            self.bucket.blob(combined_blob_name).upload_from_string(out.getvalue(), content_type="text/csv")

    def collect_data_for_all_taxa(
        self,
        taxon_ids_names: Dict[int, str],
        *,
        sleep_between_pages: float,
        parallel_workers: int,
        n_shards: int,
        shard_id: int,
    ) -> None:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        items = sorted(taxon_ids_names.items(), key=lambda kv: kv[1])
        my_items = [kv for i, kv in enumerate(items) if i % n_shards == shard_id]
        if not my_items:
            logger.warning("No taxa assigned to this shard (check SHARD_ID/N_SHARDS).")
            return

        taxa_bar = (
            tqdm(total=len(my_items), desc=f"shard {shard_id} • taxa", unit="taxon", leave=True)
            if self.progress_bar
            else None
        )

        taxon_bars: Dict[str, tqdm] = {}
        if self.progress_bar:
            for pos, (_, name) in enumerate(my_items, start=1):
                taxon_bars[name] = tqdm(total=0, position=pos, desc=name, unit="obs", leave=False)

        def _close_bars() -> None:
            if taxa_bar:
                taxa_bar.close()
            for b in taxon_bars.values():
                try:
                    b.close()
                except Exception:
                    pass

        def run_one(taxon_id: int, taxon_name: str) -> Tuple[str, int]:
            last_uid: Optional[int] = None
            processed = 0
            rows_all: List[Dict] = []

            while True:
                observations = self._fetch_observations(taxon_id, last_uid)
                if not observations:
                    break
                rows = self.process_observations(observations, taxon_name)
                rows_all.extend(rows)
                processed += len(rows)

                if self.progress_bar:
                    taxon_bars[taxon_name].update(len(rows))

                last_uid = observations[-1]["id"] if observations else None
                time.sleep(sleep_between_pages)

            if self.target.mode == "local":
                self._save_csv_rows_to_local(rows_all, taxon_name)
            else:
                self._save_csv_rows_to_gcs(rows_all, taxon_name)

            if self.progress_bar:
                taxon_bars[taxon_name].set_description_str(f"{taxon_name} ✓")
                if taxa_bar:
                    taxa_bar.update(1)

            return taxon_name, processed

        try:
            with ThreadPoolExecutor(max_workers=parallel_workers) as ex:
                futs = {ex.submit(run_one, tid, tname): tname for tid, tname in my_items}
                for fut in as_completed(futs):
                    tname = futs[fut]
                    try:
                        fut.result()
                    except Exception as e:
                        if self.progress_bar and tname in taxon_bars:
                            taxon_bars[tname].set_description_str(f"{tname} ✗")
                        logger.exception("[%s] failed: %s", tname, e)
        finally:
            _close_bars()

        if self.target.mode == "local":
            self._combine_local_csvs()
        else:
            self._combine_gcs_csvs()
