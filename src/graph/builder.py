"""
src/graph/builder.py
─────────────────────────────────────────────────────────────────────────────
Converts a normalised session DataFrame into PyTorch Geometric Data objects.

One graph = one PowerShell session.

Node schema (18 features)        — see configs/pipeline.yaml graph.node_features
Edge types                       — parent_spawn, pipe_connect, net_connect, file_write

The graph is heterogeneous by edge type but stored as a simple directed
graph with an edge_type attribute (int 0-3) so standard GAT layers work
without modification.  You can upgrade to HeteroData later.

Output
──────
list[torch_geometric.data.Data]  — one per session
Saved as .pt files in data/graphs/
"""

import math
import logging
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Lazy import — only needed when actually building graphs
try:
    import torch
    from torch_geometric.data import Data
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    log.warning("torch / torch_geometric not installed. Graph building disabled.")


# ─────────────────────────────────────────────────────────────────────────────
# Constants matching schema.py
# ─────────────────────────────────────────────────────────────────────────────
LOLBINS = {
    "certutil.exe", "mshta.exe", "regsvr32.exe", "rundll32.exe",
    "wscript.exe", "cscript.exe", "bitsadmin.exe", "wmic.exe",
    "msiexec.exe", "installutil.exe", "regasm.exe", "regsvcs.exe",
    "diskshadow.exe", "esentutl.exe", "findstr.exe", "msbuild.exe",
}

EDGE_TYPE = {
    "parent_spawn": 0,
    "pipe_connect": 1,
    "net_connect":  2,
    "file_write":   3,
}

INTEGRITY_MAP = {"low": 0, "medium": 1, "high": 2, "system": 3}


def _image_hash(image: str) -> float:
    """Map image name → stable float in [0, 1] via MD5 mod."""
    h = int(hashlib.md5(image.encode()).hexdigest(), 16) % 1000
    return h / 1000.0


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = {}
    for c in s:
        counts[c] = counts.get(c, 0) + 1
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in counts.values())


# ─────────────────────────────────────────────────────────────────────────────
# Node feature vector
# ─────────────────────────────────────────────────────────────────────────────

def _node_features(row: pd.Series, depth: int, child_count: int,
                   net_pids: set, pipe_pids: set, file_pids: set) -> list[float]:
    """
    Build the 18-dim node feature vector for one process node.

    Features (in order, matching schema.py):
      0  is_powershell
      1  is_lolbin
      2  integrity_level   (0-3)
      3  has_network_connection
      4  has_pipe_activity
      5  has_file_write
      6  cmd_entropy
      7  cmd_length_norm   log(len+1)/10
      8  has_encoded_cmd
      9  has_download_pattern
      10 has_credential_pattern
      11 parent_depth
      12 child_count
      13 user_privilege    (same as integrity_level)
      14 time_delta_secs   (normalised, filled later)
      15 image_hash
      16 has_wmi_activity
      17 has_lateral_pattern
    """
    import re
    cmd  = str(row.get("command_line", ""))
    img  = str(row.get("image", "")).lower()
    il   = str(row.get("integrity_level", "")).lower()
    pid  = int(row.get("pid", 0))

    il_val = float(INTEGRITY_MAP.get(il, 1))

    return [
        float(img in {"powershell.exe", "pwsh.exe"}),       # 0
        float(img in LOLBINS),                               # 1
        il_val,                                              # 2
        float(pid in net_pids),                              # 3
        float(pid in pipe_pids),                             # 4
        float(pid in file_pids),                             # 5
        _shannon_entropy(cmd),                               # 6
        math.log(len(cmd) + 1) / 10.0,                      # 7
        float(bool(re.search(r"-enc", cmd, re.I))),          # 8
        float(bool(re.search(r"downloadstring|iwr\b", cmd, re.I))),  # 9
        float(bool(re.search(r"mimikatz|lsass|sekurlsa", cmd, re.I))),  # 10
        float(depth),                                        # 11
        float(child_count),                                  # 12
        il_val,                                              # 13  (same as IL)
        0.0,                                                 # 14  time_delta filled later
        _image_hash(img),                                    # 15
        float(bool(re.search(r"\bwmi\b|gwmi\b|wmic", cmd, re.I))),  # 16
        float(bool(re.search(r"psexec|enter-pssession|invoke-command", cmd, re.I))),  # 17
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_session_graph(session_df: pd.DataFrame) -> Optional["Data"]:
    """
    Convert one session's events into a PyG Data object.

    Returns None if the session has fewer than 2 process nodes
    (degenerate graph — no edges possible).
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch_geometric is required for graph building.")

    proc_events = session_df[session_df["event_id"] == 1].copy()
    if len(proc_events) < 2:
        return None

    proc_events = proc_events.sort_values("timestamp").reset_index(drop=True)

    # ── Build auxiliary PID sets for cross-event signals ─────────────────────
    net_pids  = set(session_df[session_df["event_id"] == 3]["pid"].dropna().astype(int))
    pipe_pids = set(session_df[session_df["event_id"].isin([17, 18])]["pid"].dropna().astype(int))
    file_pids = set(session_df[session_df["event_id"] == 11]["pid"].dropna().astype(int))

    # ── Build PID → node index map ────────────────────────────────────────────
    pids = proc_events["pid"].tolist()
    pid_to_idx = {pid: i for i, pid in enumerate(pids)}

    # ── Compute depths via BFS from root ─────────────────────────────────────
    pid_to_ppid: dict[int, int] = {}
    for _, row in proc_events.iterrows():
        pid_to_ppid[row["pid"]] = row["ppid"]

    depths: dict[int, int] = {}
    for pid in pids:
        depth = 0
        cur   = pid
        seen  = set()
        while cur in pid_to_ppid and cur not in seen:
            seen.add(cur)
            par = pid_to_ppid[cur]
            if par not in pid_to_idx:
                break
            cur = par
            depth += 1
        depths[pid] = depth

    # ── Compute child counts ──────────────────────────────────────────────────
    child_counts: dict[int, int] = {pid: 0 for pid in pids}
    for pid in pids:
        ppid = pid_to_ppid.get(pid)
        if ppid in child_counts:
            child_counts[ppid] += 1

    # ── Build node feature matrix ─────────────────────────────────────────────
    node_feats = []
    for _, row in proc_events.iterrows():
        pid = row["pid"]
        nf  = _node_features(
            row,
            depth       = depths.get(pid, 0),
            child_count = child_counts.get(pid, 0),
            net_pids    = net_pids,
            pipe_pids   = pipe_pids,
            file_pids   = file_pids,
        )
        node_feats.append(nf)

    # Fill time_delta (feature index 14)
    if not proc_events["timestamp"].isna().all():
        times = proc_events["timestamp"].dt.tz_localize(None) \
                if proc_events["timestamp"].dt.tz else proc_events["timestamp"]
        deltas = times.diff().dt.total_seconds().fillna(0).clip(upper=3600)
        for i, delta in enumerate(deltas):
            node_feats[i][14] = float(delta)

    x = torch.tensor(node_feats, dtype=torch.float)

    # ── Build edge index ──────────────────────────────────────────────────────
    edge_src, edge_dst, edge_type = [], [], []

    for _, row in proc_events.iterrows():
        child_pid  = row["pid"]
        parent_pid = row["ppid"]
        child_idx  = pid_to_idx.get(child_pid)
        parent_idx = pid_to_idx.get(parent_pid)

        if child_idx is not None and parent_idx is not None and child_idx != parent_idx:
            # parent → child (spawn edge)
            edge_src.append(parent_idx)
            edge_dst.append(child_idx)
            edge_type.append(EDGE_TYPE["parent_spawn"])

    # Network connection edges: PS process → any child that made net call
    for net_pid in net_pids:
        if net_pid in pid_to_idx:
            ppid = pid_to_ppid.get(net_pid)
            if ppid and ppid in pid_to_idx:
                edge_src.append(pid_to_idx[ppid])
                edge_dst.append(pid_to_idx[net_pid])
                edge_type.append(EDGE_TYPE["net_connect"])

    # Pipe connection edges (bidirectional)
    pipe_node_idxs = [pid_to_idx[p] for p in pipe_pids if p in pid_to_idx]
    for i in range(len(pipe_node_idxs)):
        for j in range(i + 1, len(pipe_node_idxs)):
            edge_src.extend([pipe_node_idxs[i], pipe_node_idxs[j]])
            edge_dst.extend([pipe_node_idxs[j], pipe_node_idxs[i]])
            edge_type.extend([EDGE_TYPE["pipe_connect"]] * 2)

    if not edge_src:
        # No edges — degenerate
        return None

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr  = torch.tensor(edge_type, dtype=torch.long)

    # ── Graph-level label ─────────────────────────────────────────────────────
    label = int(session_df["label"].max())   # 1 if any event is malicious

    # ── TTP tags (stored as string for post-processing) ───────────────────────
    ttps_raw = session_df["ttps"].iloc[0]
    ttps_str = "|".join(ttps_raw) if isinstance(ttps_raw, list) else str(ttps_raw)

    data = Data(
        x          = x,
        edge_index = edge_index,
        edge_attr  = edge_attr,
        y          = torch.tensor([label], dtype=torch.long),
        session_id = session_df["session_id"].iloc[0],
        ttps       = ttps_str,
        family     = session_df["family"].iloc[0],
        num_nodes  = len(node_feats),
    )
    return data


def build_all_graphs(events_df: pd.DataFrame,
                     graphs_dir: Path,
                     min_events: int = 3) -> tuple[list, int, int]:
    """
    Build and save all session graphs.

    Returns
    ───────
    (graphs, n_saved, n_skipped)
    """
    graphs_dir = Path(graphs_dir)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    graphs   = []
    n_saved  = 0
    n_skipped = 0

    for session_id, sdf in events_df.groupby("session_id"):
        if len(sdf) < min_events:
            n_skipped += 1
            continue

        try:
            g = build_session_graph(sdf)
        except Exception as e:
            log.debug(f"Session {session_id} skipped: {e}")
            n_skipped += 1
            continue

        if g is None:
            n_skipped += 1
            continue

        out_path = graphs_dir / f"{session_id}.pt"
        if TORCH_AVAILABLE:
            import torch
            torch.save(g, out_path)

        graphs.append(g)
        n_saved += 1

    log.info(f"Graphs: {n_saved} saved, {n_skipped} skipped → {graphs_dir}")
    return graphs, n_saved, n_skipped
