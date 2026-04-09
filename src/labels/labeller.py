"""
src/labels/labeller.py
─────────────────────────────────────────────────────────────────────────────
Session-level labeller and TTP mapper.

Two responsibilities:
  1. Final label assignment  — session-level binary (0/1)
  2. TTP annotation          — maps each malicious session to its MITRE ATT&CK
                               TTP sequence for reconstruction ground truth

The labeller also performs a quality check:
  - Sessions with < min_events are dropped
  - Sessions where TTP source is "heuristic" (not dataset_metadata) are
    flagged with lower confidence
"""

import logging
import json
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# MITRE ATT&CK TTP → human-readable name and kill-chain stage
# ─────────────────────────────────────────────────────────────────────────────
TTP_META: dict[str, dict] = {
    # Initial Access
    "T1566":      {"name": "Phishing",                 "tactic": "Initial Access",     "stage": 1},
    "T1566.001":  {"name": "Spearphishing Attachment",  "tactic": "Initial Access",     "stage": 1},
    # Execution
    "T1059":      {"name": "Command & Scripting Interpreter", "tactic": "Execution",   "stage": 2},
    "T1059.001":  {"name": "PowerShell",               "tactic": "Execution",          "stage": 2},
    "T1059.003":  {"name": "Windows Command Shell",    "tactic": "Execution",          "stage": 2},
    "T1204":      {"name": "User Execution",           "tactic": "Execution",          "stage": 2},
    # Persistence
    "T1053":      {"name": "Scheduled Task/Job",       "tactic": "Persistence",        "stage": 3},
    "T1053.005":  {"name": "Scheduled Task",           "tactic": "Persistence",        "stage": 3},
    "T1547":      {"name": "Boot/Logon Autostart",     "tactic": "Persistence",        "stage": 3},
    # Privilege Escalation
    "T1055":      {"name": "Process Injection",        "tactic": "Privilege Escalation", "stage": 4},
    "T1548":      {"name": "Abuse Elevation Control",  "tactic": "Privilege Escalation", "stage": 4},
    # Defense Evasion
    "T1027":      {"name": "Obfuscated Files/Info",    "tactic": "Defense Evasion",    "stage": 5},
    "T1055.001":  {"name": "DLL Injection",            "tactic": "Defense Evasion",    "stage": 5},
    "T1218":      {"name": "System Binary Proxy Exec", "tactic": "Defense Evasion",    "stage": 5},
    "T1574":      {"name": "Hijack Execution Flow",    "tactic": "Defense Evasion",    "stage": 5},
    # Credential Access
    "T1003":      {"name": "OS Credential Dumping",    "tactic": "Credential Access",  "stage": 6},
    "T1003.001":  {"name": "LSASS Memory",             "tactic": "Credential Access",  "stage": 6},
    "T1555":      {"name": "Credentials from PWD Store","tactic": "Credential Access", "stage": 6},
    "T1552":      {"name": "Unsecured Credentials",    "tactic": "Credential Access",  "stage": 6},
    # Discovery
    "T1069":      {"name": "Permission Groups Discovery","tactic": "Discovery",         "stage": 7},
    "T1069.001":  {"name": "Local Groups",             "tactic": "Discovery",          "stage": 7},
    "T1082":      {"name": "System Info Discovery",    "tactic": "Discovery",          "stage": 7},
    "T1083":      {"name": "File & Dir Discovery",     "tactic": "Discovery",          "stage": 7},
    # Lateral Movement
    "T1021":      {"name": "Remote Services",          "tactic": "Lateral Movement",   "stage": 8},
    "T1021.002":  {"name": "SMB/Windows Admin Shares", "tactic": "Lateral Movement",   "stage": 8},
    "T1021.006":  {"name": "WinRM",                    "tactic": "Lateral Movement",   "stage": 8},
    "T1570":      {"name": "Lateral Tool Transfer",    "tactic": "Lateral Movement",   "stage": 8},
    # Collection
    "T1005":      {"name": "Data from Local System",   "tactic": "Collection",         "stage": 9},
    "T1056":      {"name": "Input Capture",            "tactic": "Collection",         "stage": 9},
    # Command & Control
    "T1071":      {"name": "Application Layer Protocol","tactic": "Command & Control",  "stage": 10},
    "T1071.001":  {"name": "Web Protocols",            "tactic": "Command & Control",  "stage": 10},
    "T1105":      {"name": "Ingress Tool Transfer",    "tactic": "Command & Control",  "stage": 10},
    # Exfiltration
    "T1041":      {"name": "Exfiltration Over C2",     "tactic": "Exfiltration",       "stage": 11},
    "T1567":      {"name": "Exfiltration Over Web",    "tactic": "Exfiltration",       "stage": 11},
}


def resolve_ttp(ttp_code: str) -> dict:
    """Return TTP metadata for a given TTP code, with fallback."""
    code = ttp_code.strip().upper()
    if code in TTP_META:
        return TTP_META[code]
    # Try parent TTP (e.g. T1059.001 → T1059)
    parent = code.split(".")[0]
    if parent in TTP_META:
        return TTP_META[parent]
    return {"name": code, "tactic": "Unknown", "stage": 0}


def build_ttp_chain(ttps: list[str]) -> list[dict]:
    """
    Given a list of TTP codes, return an ordered kill-chain sequence
    sorted by ATT&CK stage number.
    """
    resolved = [{"code": t, **resolve_ttp(t)} for t in ttps]
    # Sort by kill-chain stage, then alphabetically within stage
    resolved.sort(key=lambda x: (x["stage"], x["code"]))
    return resolved


# ─────────────────────────────────────────────────────────────────────────────
# Session-level labeller
# ─────────────────────────────────────────────────────────────────────────────

def label_sessions(
    tabular_df: pd.DataFrame,
    min_events: int = 3,
) -> pd.DataFrame:
    """
    Finalise session labels and add TTP chain for malicious sessions.

    Input:  tabular feature DataFrame (output of tabular.py)
    Output: same DataFrame with additional columns:
            - label           (int 0/1) — confirmed
            - ttp_chain       (list[dict]) — ordered kill chain
            - label_source    (str) — "dataset_metadata" or "heuristic"
            - confidence      (float) — 1.0 for dataset_metadata, 0.7 for heuristic
    """
    df = tabular_df.copy()

    # Drop undersized sessions
    before = len(df)
    df = df[df["session_event_count"] >= min_events].reset_index(drop=True)
    log.info(f"Dropped {before - len(df)} sessions with < {min_events} events")

    # Ensure label column exists
    if "label" not in df.columns:
        df["label"] = 0

    # Build TTP chain for each malicious session
    def _build_chain(row):
        ttps = row.get("ttps", [])
        if isinstance(ttps, str):
            # Might have been serialised as pipe-joined string
            ttps = [t for t in ttps.split("|") if t]
        if not ttps or row["label"] == 0:
            return []
        return build_ttp_chain(ttps)

    df["ttp_chain"]    = df.apply(_build_chain, axis=1)
    df["label_source"] = df["label"].apply(
        lambda l: "dataset_metadata" if l == 1 else "heuristic"
    )
    df["confidence"]   = df["label_source"].map(
        {"dataset_metadata": 1.0, "heuristic": 0.7}
    )

    log.info(
        f"Labels finalised: {df['label'].sum()} malicious / "
        f"{(df['label']==0).sum()} benign  ({len(df)} sessions total)"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Train / Val / Test splitter
# ─────────────────────────────────────────────────────────────────────────────

def split_dataset(
    df: pd.DataFrame,
    train_frac:    float = 0.70,
    val_frac:      float = 0.10,
    test_frac:     float = 0.20,
    ood_families:  list[str] | None = None,
    random_seed:   int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Stratified split by label.

    OOD families (if specified) are held out entirely in the OOD test set.
    The remaining data is split train/val/in-distribution-test.

    Returns
    ───────
    {
      "train": DataFrame,
      "val":   DataFrame,
      "test":  DataFrame,   # in-distribution
      "ood":   DataFrame,   # out-of-distribution (for generalisability)
    }
    """
    from sklearn.model_selection import train_test_split

    ood_families = ood_families or []

    # Separate OOD
    ood_mask = df["family"].isin(ood_families)
    ood_df   = df[ood_mask].copy()
    iid_df   = df[~ood_mask].copy()

    if ood_df.empty:
        log.info("No OOD families found — OOD split will be empty.")

    # Validate fractions
    total = train_frac + val_frac + test_frac
    assert abs(total - 1.0) < 1e-6, f"Fractions must sum to 1.0 (got {total})"

    # Stratified split: IID → train+val+test
    relative_test  = test_frac / (train_frac + val_frac + test_frac)
    relative_val   = val_frac  / (train_frac + val_frac)

    train_val, test = train_test_split(
        iid_df,
        test_size    = relative_test,
        stratify     = iid_df["label"],
        random_state = random_seed,
    )
    train, val = train_test_split(
        train_val,
        test_size    = relative_val,
        stratify     = train_val["label"],
        random_state = random_seed,
    )

    splits = {"train": train, "val": val, "test": test, "ood": ood_df}

    for name, sdf in splits.items():
        pos = sdf["label"].sum()
        log.info(
            f"  {name:6s}: {len(sdf):5d} sessions  "
            f"({pos} malicious / {len(sdf)-pos} benign)"
        )

    return splits


def save_splits(splits: dict[str, pd.DataFrame], splits_dir: Path) -> None:
    """Save each split to parquet."""
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    for name, df in splits.items():
        out = splits_dir / f"{name}.parquet"
        # ttp_chain (list of dicts) → JSON string for parquet compatibility
        df = df.copy()
        if "ttp_chain" in df.columns:
            df["ttp_chain"] = df["ttp_chain"].apply(json.dumps)
        df.to_parquet(out, index=False)
        log.info(f"  Saved {name} → {out}")
