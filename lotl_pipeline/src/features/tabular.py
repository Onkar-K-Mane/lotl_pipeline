"""
src/features/tabular.py
─────────────────────────────────────────────────────────────────────────────
Extracts the 42 handcrafted tabular features used by the Random Forest
triage layer (Tier 1).

Input:  session-level DataFrame (output of parse_sysmon.py)
Output: feature DataFrame — one row per session_id

Feature categories
──────────────────
A. Command-line patterns   (regex-based, per event)
B. Session structural      (aggregated over all events in session)
C. Process ancestry        (parent image analysis)
D. Network & I/O signals   (from Event IDs 3, 11, 17, 18)
E. Entropy & complexity    (obfuscation indicators)
"""

import re
import math
import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Known LOLBins (subset most relevant to PS chains)
# ─────────────────────────────────────────────────────────────────────────────
LOLBINS = {
    "certutil.exe", "mshta.exe", "regsvr32.exe", "rundll32.exe",
    "wscript.exe", "cscript.exe", "bitsadmin.exe", "wmic.exe",
    "msiexec.exe", "installutil.exe", "regasm.exe", "regsvcs.exe",
    "ieexec.exe", "cmstp.exe", "msdeploy.exe", "diskshadow.exe",
    "esentutl.exe", "expand.exe", "extrac32.exe", "findstr.exe",
    "hh.exe", "makecab.exe", "mavinject.exe", "microsoft.workflow.compiler.exe",
    "msbuild.exe", "msconfig.exe", "msdt.exe", "odbcconf.exe",
    "pcalua.exe", "pcwrun.exe", "replace.exe", "rpcping.exe",
    "runscripthelper.exe", "sfc.exe", "syncappvpublishingserver.exe",
    "tttracer.exe", "verclsid.exe", "wab.exe", "winrm.cmd",
    "xwizard.exe", "appsyncpublishingserver.exe",
}

OFFICE_IMAGES = {
    "winword.exe", "excel.exe", "outlook.exe",
    "powerpnt.exe", "msaccess.exe", "onenote.exe",
}

BROWSER_IMAGES = {
    "chrome.exe", "firefox.exe", "msedge.exe",
    "iexplore.exe", "opera.exe", "safari.exe",
}

SCRIPT_HOSTS = {"wscript.exe", "cscript.exe", "mshta.exe", "wmic.exe"}
SERVICE_IMAGES = {"services.exe", "svchost.exe", "taskhost.exe", "taskhostw.exe"}
PS_IMAGES = {"powershell.exe", "pwsh.exe"}


# ─────────────────────────────────────────────────────────────────────────────
# Compiled regexes  (compile once, reuse)
# ─────────────────────────────────────────────────────────────────────────────
RE_ENCODED     = re.compile(r"-e(nc(odedcommand)?)?\b", re.IGNORECASE)
RE_NOPROFILE   = re.compile(r"-no(p(rofile)?)?\b", re.IGNORECASE)
RE_HIDDEN      = re.compile(r"-w(indowstyle)?\s+h(idden)?\b", re.IGNORECASE)
RE_BYPASS      = re.compile(r"-exec(utionpolicy)?\s+bypass\b", re.IGNORECASE)
RE_IEX         = re.compile(r"\biex\b|invoke-expression\b", re.IGNORECASE)
RE_DOWNLOAD    = re.compile(
    r"downloadstring|downloadfile|new-object\s+net\.webclient|"
    r"invoke-webrequest|iwr\b|webclient|bits|start-bitstransfer",
    re.IGNORECASE
)
RE_REFLECTION  = re.compile(
    r"\[reflection\.assembly\]|add-type|system\.reflection|"
    r"\.getmethod\(|\.invoke\(",
    re.IGNORECASE
)
RE_WMI         = re.compile(r"\bwmi\b|get-wmiobject|gwmi\b|invoke-wmimethod|"
                             r"[Ww][Mm][Ii][Cc]\b", re.IGNORECASE)
RE_SCHTASK     = re.compile(r"schtasks|register-scheduledtask|"
                             r"new-scheduledtask\b", re.IGNORECASE)
RE_REGISTRY    = re.compile(r"set-itemproperty|reg\s+add|hklm:|hkcu:|"
                             r"new-itemproperty", re.IGNORECASE)
RE_CRED        = re.compile(r"mimikatz|sekurlsa|lsass|credential|"
                             r"invoke-mimikatz|dump.*pass", re.IGNORECASE)
RE_LATERAL     = re.compile(r"psexec|enter-pssession|invoke-command.*"
                             r"computername|new-pssession", re.IGNORECASE)
RE_BASE64      = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")


def _shannon_entropy(s: str) -> float:
    """Shannon entropy of a string in bits per character."""
    if not s:
        return 0.0
    counts = {}
    for c in s:
        counts[c] = counts.get(c, 0) + 1
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in counts.values())


def _obfuscation_score(cmd: str) -> float:
    """
    Ratio of 'unusual' characters in the command string.
    High values indicate obfuscation (tick insertion, char ops, etc.)
    """
    if not cmd:
        return 0.0
    unusual = sum(1 for c in cmd if not (c.isalnum() or c in " .-_/\\:\"'=@"))
    return unusual / len(cmd)


def _base64_token_count(cmd: str) -> int:
    return len(RE_BASE64.findall(cmd))


def _count_cmdlets(cmd: str) -> int:
    """Count distinct PS cmdlet-like tokens (Verb-Noun pattern)."""
    pattern = re.compile(r"\b[A-Z][a-z]+-[A-Z][A-Za-z]+\b")
    return len(set(pattern.findall(cmd)))


def _has_double_extension(cmd: str) -> int:
    return int(bool(re.search(r"\.\w{2,4}\.\w{2,4}\b", cmd)))


# ─────────────────────────────────────────────────────────────────────────────
# Per-event feature extraction (Event ID 1 — Process Creation)
# ─────────────────────────────────────────────────────────────────────────────

def extract_event_features(row: pd.Series) -> dict:
    """Extract per-event features from a single Event ID 1 row."""
    cmd = str(row.get("command_line", ""))
    img = str(row.get("image", "")).lower()
    par = str(row.get("parent_image", "")).lower()
    il  = str(row.get("integrity_level", "")).lower()

    return {
        "has_encoded_arg":       int(bool(RE_ENCODED.search(cmd))),
        "has_noprofile":         int(bool(RE_NOPROFILE.search(cmd))),
        "has_hidden_window":     int(bool(RE_HIDDEN.search(cmd))),
        "has_bypass":            int(bool(RE_BYPASS.search(cmd))),
        "has_iex":               int(bool(RE_IEX.search(cmd))),
        "has_downloadstring":    int(bool(RE_DOWNLOAD.search(cmd))),
        "has_reflection":        int(bool(RE_REFLECTION.search(cmd))),
        "has_wmi":               int(bool(RE_WMI.search(cmd))),
        "has_scheduled_task":    int(bool(RE_SCHTASK.search(cmd))),
        "has_registry_write":    int(bool(RE_REGISTRY.search(cmd))),
        "has_credential_access": int(bool(RE_CRED.search(cmd))),
        "has_lateral_movement":  int(bool(RE_LATERAL.search(cmd))),
        "cmd_length":            len(cmd),
        "entropy_score":         _shannon_entropy(cmd),
        "parent_is_office":      int(par in OFFICE_IMAGES),
        "parent_is_browser":     int(par in BROWSER_IMAGES),
        "parent_is_script_host": int(par in SCRIPT_HOSTS),
        "parent_is_service":     int(par in SERVICE_IMAGES),
        "parent_is_powershell":  int(par in PS_IMAGES),
        "user_is_system":        int(il == "system"),
        "user_is_high":          int(il == "high"),
        "base64_token_count":    _base64_token_count(cmd),
        "obfuscation_score":     _obfuscation_score(cmd),
        "token_count":           len(cmd.split()),
        "unique_cmdlet_count":   _count_cmdlets(cmd),
        "has_double_extension":  _has_double_extension(cmd),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Session-level feature aggregation
# ─────────────────────────────────────────────────────────────────────────────

def extract_session_features(session_df: pd.DataFrame) -> dict:
    """
    Aggregate event-level features into one feature vector per session.

    For binary features: max (if ANY event triggers → session triggers)
    For count features:  sum
    For scalar features: max (worst case in session)
    """
    proc_events = session_df[session_df["event_id"] == 1]
    net_events  = session_df[session_df["event_id"] == 3]
    file_events = session_df[session_df["event_id"] == 11]
    pipe_events = session_df[session_df["event_id"].isin([17, 18])]

    feats = {}

    # ── A. Command-line pattern features (max over all proc events) ──────────
    if not proc_events.empty:
        event_feats = proc_events.apply(extract_event_features, axis=1)
        ef_df = pd.DataFrame(list(event_feats))

        binary_cols = [c for c in ef_df.columns if c.startswith("has_") or
                       c.startswith("parent_is_") or c.startswith("user_is_")]
        scalar_cols = ["cmd_length", "entropy_score", "obfuscation_score",
                       "base64_token_count", "token_count",
                       "unique_cmdlet_count", "has_double_extension"]

        for col in binary_cols:
            feats[col] = int(ef_df[col].max())
        for col in scalar_cols:
            feats[col] = float(ef_df[col].max())
    else:
        # No process events — zero out
        dummy = extract_event_features(pd.Series({
            "command_line": "", "image": "", "parent_image": "", "integrity_level": ""
        }))
        for k, v in dummy.items():
            feats[k] = type(v)(0)

    # ── B. Session structural features ───────────────────────────────────────
    feats["session_event_count"] = len(session_df)
    feats["child_count"]         = len(proc_events)
    feats["unique_child_images"] = proc_events["image"].nunique() if not proc_events.empty else 0

    if not proc_events.empty:
        feats["lolbin_child_count"] = int(
            proc_events["image"].isin(LOLBINS).sum()
        )
    else:
        feats["lolbin_child_count"] = 0

    # Session duration
    if len(session_df) >= 2 and not session_df["timestamp"].isna().all():
        ts = session_df["timestamp"].dropna()
        duration = (ts.max() - ts.min()).total_seconds()
        feats["session_duration_secs"] = max(0.0, duration)
    else:
        feats["session_duration_secs"] = 0.0

    # ── C. Network signals ───────────────────────────────────────────────────
    feats["has_network_call"] = int(not net_events.empty)

    if not net_events.empty and "destination_port" in net_events.columns:
        ports = net_events["destination_port"].dropna().astype(int)
        feats["has_outbound_443"]      = int((ports == 443).any())
        feats["has_outbound_80"]       = int((ports == 80).any())
        feats["has_nonstandard_port"]  = int(
            (~ports.isin([80, 443, 53, 8080, 8443])).any()
        )
    else:
        feats["has_outbound_443"]     = 0
        feats["has_outbound_80"]      = 0
        feats["has_nonstandard_port"] = 0

    # ── D. File & pipe signals ───────────────────────────────────────────────
    feats["has_pipe_activity"] = int(not pipe_events.empty)
    feats["file_write_count"]  = len(file_events)

    if not file_events.empty and "target_filename" in file_events.columns:
        fnames = file_events["target_filename"].fillna("").str.lower()
        feats["has_temp_write"] = int(
            fnames.str.contains(r"\\temp\\|\\appdata\\|\\tmp\\").any()
        )
    else:
        feats["has_temp_write"] = 0

    # ── E. Depth ──────────────────────────────────────────────────────────────
    # Proxy for tree depth: number of unique PPIDs seen
    feats["depth_in_tree"] = proc_events["ppid"].nunique() if not proc_events.empty else 0

    # Script block length (Event ID 4104 if present in session)
    feats["script_block_length"] = 0  # populated downstream if 4104 available

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_tabular_features(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given the normalised events DataFrame, return a session-level
    feature DataFrame suitable for training the Random Forest triage model.

    Returns
    ───────
    DataFrame with columns:
        session_id | feature_0 ... feature_41 | label | family | ttps
    """
    records = []

    for session_id, sdf in events_df.groupby("session_id"):
        feats = extract_session_features(sdf)
        feats["session_id"] = session_id

        # Carry through metadata from first event in session
        first = sdf.iloc[0]
        feats["label"]  = int(first.get("label", 0))
        feats["family"] = first.get("family", "")
        feats["ttps"]   = first.get("ttps", [])

        records.append(feats)

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records)

    # Ensure no NaNs leak through
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    result[numeric_cols] = result[numeric_cols].fillna(0)

    log.info(f"Tabular features built: {len(result)} sessions × {len(result.columns)} columns")
    return result
