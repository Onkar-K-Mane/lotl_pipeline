"""
src/ingestion/download.py
─────────────────────────────────────────────────────────────────────────────
Downloads attack datasets from OTRF Security Datasets and Splunk attack_data.
Each downloaded file is tagged with its MITRE TTP codes so the labeller
can assign ground-truth labels without heuristics.

Usage:
    python -m src.ingestion.download --config configs/pipeline.yaml
"""

import os
import json
import zipfile
import hashlib
import logging
import argparse
import requests
import yaml
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


@dataclass
class DownloadedFile:
    local_path: Path
    source:     str          # "otrf" | "splunk"
    ttps:       list[str]    # e.g. ["T1059.001", "T1027"]
    family:     str          # human-readable name derived from filename
    checksum:   str          # sha256 of downloaded bytes


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, timeout: int = 60) -> bool:
    """Stream-download url → dest. Returns True on success."""
    if dest.exists():
        log.info(f"  SKIP (cached): {dest.name}")
        return True
    log.info(f"  GET  {url}")
    try:
        r = requests.get(url, stream=True, timeout=timeout)
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
        log.info(f"  SAVED → {dest} ({dest.stat().st_size // 1024} KB)")
        return True
    except Exception as e:
        log.error(f"  FAILED {url}: {e}")
        return False


def _unzip(zip_path: Path, out_dir: Path) -> list[Path]:
    """Unzip archive, return list of extracted .json or .log files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith((".json", ".log", ".jsonl")):
                dest = out_dir / Path(name).name
                if not dest.exists():
                    zf.extract(name, out_dir)
                    # flatten — move to out_dir root if nested
                    nested = out_dir / name
                    if nested != dest and nested.exists():
                        nested.rename(dest)
                extracted.append(dest)
    return extracted


def download_otrf(cfg: dict, raw_dir: Path) -> list[DownloadedFile]:
    """Fetch all OTRF datasets listed in config."""
    base   = cfg["sources"]["otrf"]["base_url"].rstrip("/")
    result = []

    for entry in cfg["sources"]["otrf"]["datasets"]:
        url    = f"{base}/{entry['path']}"
        fname  = Path(entry["path"]).name
        family = fname.replace(".zip", "")
        dest   = raw_dir / "otrf" / fname

        ok = _download(url, dest)
        if not ok:
            continue

        # Unzip if needed
        if fname.endswith(".zip"):
            json_files = _unzip(dest, raw_dir / "otrf" / family)
        else:
            json_files = [dest]

        for jf in json_files:
            result.append(DownloadedFile(
                local_path = jf,
                source     = "otrf",
                ttps       = entry["ttp"],
                family     = family,
                checksum   = _sha256(jf),
            ))

    return result


def download_splunk(cfg: dict, raw_dir: Path) -> list[DownloadedFile]:
    """Fetch all Splunk attack_data datasets listed in config."""
    base   = cfg["sources"]["splunk_attack_data"]["base_url"].rstrip("/")
    result = []

    for entry in cfg["sources"]["splunk_attack_data"]["datasets"]:
        url    = f"{base}/{entry['path']}"
        # e.g. T1059.001/atomic_red_team/windows-sysmon.log
        parts  = entry["path"].split("/")
        ttp_id = parts[0]
        fname  = parts[-1]
        family = f"splunk_{ttp_id}_{fname.replace('.log','')}"
        dest   = raw_dir / "splunk" / ttp_id / fname

        ok = _download(url, dest)
        if not ok:
            continue

        result.append(DownloadedFile(
            local_path = dest,
            source     = "splunk",
            ttps       = entry["ttp"],
            family     = family,
            checksum   = _sha256(dest),
        ))

    return result


def save_manifest(files: list[DownloadedFile], out_path: Path) -> None:
    """Write a JSON manifest recording what was downloaded and its TTP tags."""
    manifest = [
        {
            "path":     str(f.local_path),
            "source":   f.source,
            "ttps":     f.ttps,
            "family":   f.family,
            "checksum": f.checksum,
        }
        for f in files
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fp:
        json.dump(manifest, fp, indent=2)
    log.info(f"Manifest → {out_path}  ({len(manifest)} files)")


def run(config_path: str) -> list[DownloadedFile]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== Downloading OTRF datasets ===")
    otrf_files = download_otrf(cfg, raw_dir)

    log.info("=== Downloading Splunk attack_data datasets ===")
    splunk_files = download_splunk(cfg, raw_dir)

    all_files = otrf_files + splunk_files
    save_manifest(all_files, raw_dir / "manifest.json")

    log.info(f"Download complete: {len(all_files)} files total")
    log.info(f"  OTRF:   {len(otrf_files)}")
    log.info(f"  Splunk: {len(splunk_files)}")

    return all_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()
    run(args.config)
