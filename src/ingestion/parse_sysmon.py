"""
src/ingestion/parse_sysmon.py
─────────────────────────────────────────────────────────────────────────────
Parses raw Sysmon event files (OTRF JSON and Splunk .log format) into a
unified, normalised Pandas DataFrame — one row per event.

Unified schema
──────────────
Field               Type        Description
─────────────────── ──────────  ──────────────────────────────────────────────
event_id            int         Sysmon EventID (1, 3, 7, 10, 11, 17, 18)
timestamp           datetime    UTC event time
session_id          str         Assigned by session_splitter (PID lineage key)
pid                 int         Process ID of subject process
ppid                int         Parent process ID
image               str         Lowercase basename of subject process image
parent_image        str         Lowercase basename of parent image
command_line        str         Raw CommandLine field (preserved for NLP)
user                str         Domain\\User executing the process
integrity_level     str         Low / Medium / High / System
hostname            str         Source machine hostname
source_file         str         Origin file path (for traceability)
family              str         Attack family name (from manifest)
ttps                list[str]   MITRE ATT&CK TTP codes (from manifest)
label               int         0 = benign, 1 = malicious (set by labeller)
"""

import json
import re
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# ── Event IDs we care about ───────────────────────────────────────────────────
RELEVANT_EVENT_IDS = {1, 3, 7, 10, 11, 17, 18}

# ── PowerShell image names ────────────────────────────────────────────────────
PS_IMAGES = {"powershell.exe", "pwsh.exe"}

# ── Unified column order ──────────────────────────────────────────────────────
COLUMNS = [
    "event_id", "timestamp", "pid", "ppid",
    "image", "parent_image", "command_line",
    "user", "integrity_level", "hostname",
    "source_file", "family", "ttps", "label",
]


# ─────────────────────────────────────────────────────────────────────────────
# OTRF parser — events are JSON lines, each a dict with nested EventData
# ─────────────────────────────────────────────────────────────────────────────

def _parse_otrf_record(rec: dict) -> Optional[dict]:
    """Parse one OTRF JSON record → normalised dict or None."""
    try:
        event_id = int(rec.get("EventID", rec.get("event_id", 0)))
        if event_id not in RELEVANT_EVENT_IDS:
            return None

        # OTRF stores fields at top level after normalisation
        def g(key, default=""):
            return str(rec.get(key, default)).strip()

        ts_raw = g("UtcTime") or g("@timestamp") or g("TimeCreated")
        try:
            ts = pd.to_datetime(ts_raw, utc=True)
        except Exception:
            ts = pd.NaT

        raw_img = g("Image").replace("\\", "/"); image = raw_img.split("/")[-1].split("\\")[-1].lower()
        raw_par = g("ParentImage").replace("\\", "/"); parent_image = raw_par.split("/")[-1].split("\\")[-1].lower()
        cmd          = g("CommandLine")

        return {
            "event_id":       event_id,
            "timestamp":      ts,
            "pid":            int(g("ProcessId") or 0),
            "ppid":           int(g("ParentProcessId") or 0),
            "image":          image,
            "parent_image":   parent_image,
            "command_line":   cmd,
            "user":           g("User"),
            "integrity_level": g("IntegrityLevel"),
            "hostname":       g("Computer") or g("host"),
        }
    except Exception as e:
        log.debug(f"Skipped OTRF record: {e}")
        return None


def parse_otrf_file(path: Path) -> list[dict]:
    """Parse an OTRF JSON file (one JSON object per line)."""
    records = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                # Handle both list-wrapped and bare records
                if isinstance(rec, list):
                    for r in rec:
                        parsed = _parse_otrf_record(r)
                        if parsed:
                            records.append(parsed)
                elif isinstance(rec, dict):
                    parsed = _parse_otrf_record(rec)
                    if parsed:
                        records.append(parsed)
            except json.JSONDecodeError:
                continue
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Splunk parser — .log files are JSON lines with Splunk field names
# ─────────────────────────────────────────────────────────────────────────────

def _parse_splunk_record(rec: dict) -> Optional[dict]:
    """Parse one Splunk attack_data JSON record → normalised dict or None."""
    try:
        # Splunk format uses "EventCode" or "event_id"
        event_id = int(
            rec.get("EventCode",
            rec.get("event_id",
            rec.get("EventID", 0)))
        )
        if event_id not in RELEVANT_EVENT_IDS:
            return None

        def g(key, default=""):
            return str(rec.get(key, default)).strip()

        ts_raw = g("_time") or g("date_time") or g("timestamp")
        try:
            ts = pd.to_datetime(ts_raw, utc=True)
        except Exception:
            ts = pd.NaT

        # Splunk flattens EventData fields to top level
        image        = Path(g("Image") or g("process_path")).name.lower()
        parent_image = Path(g("ParentImage") or g("parent_process")).name.lower()
        cmd          = g("CommandLine") or g("process")

        return {
            "event_id":        event_id,
            "timestamp":       ts,
            "pid":             int(g("ProcessId") or g("process_id") or 0),
            "ppid":            int(g("ParentProcessId") or g("parent_process_id") or 0),
            "image":           image,
            "parent_image":    parent_image,
            "command_line":    cmd,
            "user":            g("User") or g("user"),
            "integrity_level": g("IntegrityLevel") or "",
            "hostname":        g("ComputerName") or g("host") or g("dvc"),
        }
    except Exception as e:
        log.debug(f"Skipped Splunk record: {e}")
        return None


def parse_splunk_file(path: Path) -> list[dict]:
    """Parse a Splunk attack_data .log file."""
    records = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                parsed = _parse_splunk_record(rec)
                if parsed:
                    records.append(parsed)
            except json.JSONDecodeError:
                continue
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Session splitter — groups events into PowerShell sessions
# ─────────────────────────────────────────────────────────────────────────────

def assign_sessions(df: pd.DataFrame, gap_secs: int = 300) -> pd.DataFrame:
    """
    Assign a session_id to each event.

    A 'session' is defined as a contiguous run of events rooted in the
    same PowerShell process (by PID lineage), where no gap between
    consecutive events exceeds gap_secs seconds.

    Strategy:
      1. Find all Event ID 1 records where image is powershell.exe or pwsh.exe
         → these are PS session roots.
      2. For each root PID, collect all events where the PID matches or
         whose PPID traces back to that root PID (within the same host).
      3. Split on time gaps > gap_secs.
    """
    df = df.sort_values(["hostname", "timestamp"]).reset_index(drop=True)
    df["session_id"] = ""

    session_counter = 0

    for hostname, hdf in df.groupby("hostname"):
        hdf = hdf.sort_values("timestamp")

        # Build PID → parent PID map for this host
        pid_to_ppid: dict[int, int] = {}
        for _, row in hdf[hdf["event_id"] == 1].iterrows():
            pid_to_ppid[row["pid"]] = row["ppid"]

        # Find PS roots (Event ID 1, image is PS)
        ps_roots = hdf[
            (hdf["event_id"] == 1) &
            (hdf["image"].isin(PS_IMAGES))
        ]["pid"].tolist()

        def is_descendant_of_ps(pid: int, depth: int = 0) -> bool:
            if depth > 5:
                return False
            if pid in ps_roots:
                return True
            ppid = pid_to_ppid.get(pid)
            if ppid is None or ppid == pid:
                return False
            return is_descendant_of_ps(ppid, depth + 1)

        # Tag events belonging to a PS session
        session_events = hdf[
            hdf["pid"].apply(is_descendant_of_ps) |
            hdf["ppid"].apply(is_descendant_of_ps)
        ].copy()

        if session_events.empty:
            continue

        # Split into sub-sessions on time gaps
        session_events = session_events.sort_values("timestamp")
        times = session_events["timestamp"]
        gaps  = (times.diff().dt.total_seconds() > gap_secs)
        sub_session = gaps.cumsum()

        for sub_id, sub_df in session_events.groupby(sub_session):
            sid = f"sess_{session_counter:06d}"
            df.loc[sub_df.index, "session_id"] = sid
            session_counter += 1

    # Drop events that are not part of any PS session
    df = df[df["session_id"] != ""].reset_index(drop=True)
    log.info(f"Sessions identified: {df['session_id'].nunique()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main parse entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_manifest(manifest_path: Path, gap_secs: int = 300) -> pd.DataFrame:
    """
    Given a download manifest, parse all files and return a single
    normalised, session-segmented DataFrame.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    all_records = []
    for entry in manifest:
        path   = Path(entry["path"])
        source = entry["source"]
        ttps   = entry["ttps"]
        family = entry["family"]

        if not path.exists():
            log.warning(f"Missing file (skipped): {path}")
            continue

        log.info(f"Parsing [{source}] {path.name}")
        if source == "otrf":
            records = parse_otrf_file(path)
        elif source == "splunk":
            records = parse_splunk_file(path)
        else:
            log.warning(f"Unknown source '{source}', skipping.")
            continue

        for r in records:
            r["family"] = family
            r["ttps"]   = ttps
            r["label"]  = 1   # all attack dataset records are malicious

        all_records.extend(records)
        log.info(f"  → {len(records)} events parsed")

    if not all_records:
        log.warning("No records parsed from manifest.")
        return pd.DataFrame(columns=COLUMNS + ["session_id"])

    df = pd.DataFrame(all_records)

    # Ensure correct dtypes
    df["pid"]  = pd.to_numeric(df["pid"],  errors="coerce").fillna(0).astype(int)
    df["ppid"] = pd.to_numeric(df["ppid"], errors="coerce").fillna(0).astype(int)
    df["command_line"] = df["command_line"].fillna("")
    df["image"]        = df["image"].fillna("").str.lower()
    df["parent_image"] = df["parent_image"].fillna("").str.lower()

    log.info(f"Total events before session split: {len(df)}")
    df = assign_sessions(df, gap_secs=gap_secs)
    log.info(f"Total events after session split:  {len(df)}")

    return df
