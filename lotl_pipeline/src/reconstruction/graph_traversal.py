"""
src/reconstruction/graph_traversal.py
─────────────────────────────────────────────────────────────────────────────
Attack path extraction from a trained GAT model.

After the fusion layer declares a session malicious, this module:

  1. Runs the GAT in attention-return mode on the session graph
  2. Builds an attention-weighted directed graph over the process tree
  3. Extracts the highest-attention path from root node to leaf
     using a modified Dijkstra traversal (weights = 1 - attention,
     so we find the MAX-attention path)
  4. Returns an ordered list of ProcessNode objects representing
     the reconstructed attack chain

The path is the sequence of processes the attacker moved through,
from the initial PowerShell spawn to the most suspicious leaf process.
This path is then handed to the TTP mapper.

Key design choice — why attention-based path extraction?
─────────────────────────────────────────────────────────
The GAT's attention weights tell us which edges it considered most
important when making the malicious classification. High-attention
edges connect processes that co-vary with the malicious label in
training — i.e., the structural backbone of known attack chains.
Re-traversing those edges gives us a forensically meaningful path
rather than an arbitrary BFS/DFS traversal.
"""

import logging
import heapq
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProcessNode:
    """One node in the reconstructed attack chain."""
    node_idx:        int
    image:           str          # process image name (e.g. "powershell.exe")
    command_line:    str          # raw CLI string
    pid:             int
    ppid:            int
    timestamp:       Optional[str]
    integrity_level: str
    depth:           int          # hops from session root
    attention_score: float        # max attention weight on edges from this node
    # Feature flags (derived from node feature vector)
    is_powershell:   bool
    is_lolbin:       bool
    has_network:     bool
    has_encoded_cmd: bool
    has_download:    bool
    has_credential:  bool
    has_lateral:     bool


@dataclass
class AttackPath:
    """The full reconstructed attack chain for one session."""
    session_id:     str
    nodes:          list[ProcessNode]    # ordered root → leaf
    edge_attentions: list[float]         # attention weight for each step
    total_attention: float               # sum of edge attentions (chain confidence)
    root_image:     str                  # first process (entry point)
    leaf_image:     str                  # last process (terminal action)
    n_hops:         int                  # number of steps in chain
    p_malicious:    float                # fusion model final verdict score


# ─────────────────────────────────────────────────────────────────────────────
# Attention extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_attention_graph(
    model,
    graph_data,
    device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the GAT model with return_attention=True on one graph.

    Returns
    ───────
    edge_index   : (2, E) int array   — [src_nodes, dst_nodes]
    attn_weights : (E,)   float array — layer-3 attention weights per edge
    node_features: (N, F) float array — raw node feature matrix
    """
    import torch

    model.eval()
    with torch.no_grad():
        graph_data = graph_data.to(device)

        # Build a batch of 1 for the model's batch argument
        batch_vec = torch.zeros(graph_data.num_nodes, dtype=torch.long).to(device)

        try:
            out, attn_dict = model(
                graph_data.x,
                graph_data.edge_index,
                batch_vec,
                edge_attr       = getattr(graph_data, "edge_attr", None),
                return_attention = True,
            )
            # Use layer 3 (most aggregated, single-head)
            ei, aw = attn_dict["layer3"]
            ei_np  = ei.cpu().numpy()                      # (2, E)
            aw_np  = aw.squeeze().cpu().numpy()            # (E,)
            if aw_np.ndim == 0:
                aw_np = np.array([float(aw_np)])
        except TypeError:
            # Model doesn't support return_attention — use uniform weights
            log.debug("Model doesn't support return_attention; using uniform weights.")
            ei_np = graph_data.edge_index.cpu().numpy()
            aw_np = np.ones(ei_np.shape[1], dtype=np.float32)

    x_np = graph_data.x.cpu().numpy()
    return ei_np, aw_np, x_np


def find_root_node(
    edge_index: np.ndarray,
    x: np.ndarray,
    n_nodes: int,
) -> int:
    """
    Find the root of the process tree.

    Strategy: the root node has no incoming parent_spawn edges
    (it is not a child of any other node in the session graph).
    If multiple candidates exist, prefer the one with is_powershell=1
    or the shallowest depth (feature index 11).
    """
    # Nodes that appear as dst (have a parent in the graph)
    has_parent = set(edge_index[1].tolist())

    # Candidates: nodes with no parent in graph
    candidates = [i for i in range(n_nodes) if i not in has_parent]

    if not candidates:
        # Degenerate — all nodes have parents (cycle). Fall back to node 0.
        return 0

    # Prefer PowerShell nodes (feature 0) among candidates
    ps_candidates = [i for i in candidates if x[i, 0] > 0.5]
    if ps_candidates:
        # Among PS roots, take the one with smallest depth (feature 11)
        return min(ps_candidates, key=lambda i: x[i, 11])

    # Otherwise take shallowest
    return min(candidates, key=lambda i: x[i, 11])


def max_attention_path(
    root:        int,
    edge_index:  np.ndarray,
    attn_weights: np.ndarray,
    n_nodes:     int,
) -> tuple[list[int], list[float]]:
    """
    Find the path from root to the leaf with the highest total attention.

    Uses a max-heap (negated for Python's min-heap) over cumulative
    attention scores. Equivalent to finding the highest-weight path
    in a DAG — this gives us the chain the GAT paid most attention to.

    Returns
    ───────
    path         : list[int]   — node indices in order
    edge_weights : list[float] — attention weight at each step
    """
    # Build adjacency: src → list of (dst, attention_weight)
    adj: dict[int, list[tuple[int, float]]] = {i: [] for i in range(n_nodes)}
    for k in range(edge_index.shape[1]):
        src = int(edge_index[0, k])
        dst = int(edge_index[1, k])
        w   = float(attn_weights[k]) if k < len(attn_weights) else 1.0
        adj[src].append((dst, w))

    # Max-heap: (-cumulative_attention, node, path_so_far, edge_weights_so_far)
    heap = [(-0.0, root, [root], [])]
    best: dict[int, float] = {root: 0.0}

    best_path        = [root]
    best_edge_weights = []
    best_score        = -1.0

    while heap:
        neg_score, node, path, ew = heapq.heappop(heap)
        score = -neg_score

        # If this node is a leaf (no outgoing edges) or we already found better
        if not adj[node]:
            if score > best_score:
                best_score        = score
                best_path         = path
                best_edge_weights = ew
            continue

        for dst, w in adj[node]:
            if dst in path:          # break cycles
                continue
            new_score = score + w
            if new_score <= best.get(dst, -1.0):
                continue             # already found a better path to dst
            best[dst] = new_score
            heapq.heappush(heap, (-new_score, dst, path + [dst], ew + [w]))

    return best_path, best_edge_weights


# ─────────────────────────────────────────────────────────────────────────────
# Node metadata extraction
# ─────────────────────────────────────────────────────────────────────────────

# Known LOLBins (subset for display)
LOLBINS_SET = {
    "certutil.exe", "mshta.exe", "regsvr32.exe", "rundll32.exe",
    "wscript.exe", "cscript.exe", "bitsadmin.exe", "wmic.exe",
    "msiexec.exe", "installutil.exe", "regasm.exe", "msbuild.exe",
    "diskshadow.exe", "esentutl.exe", "findstr.exe", "odbcconf.exe",
}

INTEGRITY_NAMES = {0: "Low", 1: "Medium", 2: "High", 3: "System"}

# Image hash reverse-lookup built from builder.py's _image_hash
# We can't perfectly invert MD5 mod 1000, so we store known mappings
_IMAGE_HASH_CACHE: dict[float, str] = {}


def _image_from_features(x_row: np.ndarray) -> str:
    """
    Recover image name from node features.
    Feature 15 is image_hash = MD5(image)%1000 / 1000.
    We use is_powershell (feat 0) and is_lolbin (feat 1) for fallback labels.
    """
    if x_row[0] > 0.5:
        return "powershell.exe"
    if x_row[1] > 0.5:
        return "lolbin_process"
    return "unknown_process"


def build_process_nodes(
    path:         list[int],
    edge_weights: list[float],
    x:            np.ndarray,
    session_events: "pd.DataFrame | None" = None,
) -> list[ProcessNode]:
    """
    Build ProcessNode objects for each node in the attack path.

    When session_events is provided, we enrich with actual CommandLine,
    PID, PPID, timestamp from the raw event DataFrame.
    """
    nodes = []

    for step, node_idx in enumerate(path):
        row = x[node_idx]

        # Attention score for this node = weight of incoming edge
        att = edge_weights[step - 1] if step > 0 else 0.0

        # Integrity level
        il_val = int(round(row[2]))
        il_str = INTEGRITY_NAMES.get(il_val, "Medium")

        # Enrich from raw events if available
        cmd, pid, ppid, ts, image = "", 0, 0, None, _image_from_features(row)

        if session_events is not None:
            # Match by node position (assumes events sorted by timestamp,
            # matching the order in builder.py)
            proc_events = session_events[session_events["event_id"] == 1]
            if node_idx < len(proc_events):
                ev_row = proc_events.iloc[node_idx]
                cmd    = str(ev_row.get("command_line", ""))
                pid    = int(ev_row.get("pid",  0))
                ppid   = int(ev_row.get("ppid", 0))
                ts     = str(ev_row.get("timestamp", ""))
                img    = str(ev_row.get("image", ""))
                image  = img if img else image

        # Derive feature flags from command string when available
        import re as _re
        if cmd:
            _has_enc  = bool(_re.search(r"-e(nc(odedcommand)?)?", cmd, _re.I))
            _has_dl   = bool(_re.search(r"downloadstring|iwr\b|webclient|new-object net", cmd, _re.I))
            _has_cred = bool(_re.search(r"mimikatz|lsass|sekurlsa", cmd, _re.I))
            _has_lat  = bool(_re.search(r"psexec|enter-pssession|invoke-command", cmd, _re.I))
        else:
            _has_enc  = bool(row[8]  > 0.5)
            _has_dl   = bool(row[9]  > 0.5)
            _has_cred = bool(row[10] > 0.5)
            _has_lat  = bool(row[17] > 0.5)

        nodes.append(ProcessNode(
            node_idx        = node_idx,
            image           = image,
            command_line    = cmd,
            pid             = pid,
            ppid            = ppid,
            timestamp       = ts,
            integrity_level = il_str,
            depth           = int(row[11]),
            attention_score = float(att),
            is_powershell   = bool(row[0] > 0.5) or image in {"powershell.exe", "pwsh.exe"},
            is_lolbin       = bool(row[1] > 0.5) or image in LOLBINS_SET,
            has_network     = bool(row[3] > 0.5),
            has_encoded_cmd = _has_enc,
            has_download    = _has_dl,
            has_credential  = _has_cred,
            has_lateral     = _has_lat,
        ))

    return nodes


# ─────────────────────────────────────────────────────────────────────────────
# Main reconstruction entry point
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_attack_path(
    model,
    graph_data,
    device,
    session_id:     str,
    p_malicious:    float,
    session_events: "pd.DataFrame | None" = None,
) -> AttackPath:
    """
    Full pipeline: GAT attention → path extraction → ProcessNode list.

    Parameters
    ──────────
    model          : trained GAT model (LogitWrapper)
    graph_data     : PyG Data object for this session
    device         : torch device
    session_id     : string identifier
    p_malicious    : fusion model final score
    session_events : raw events DataFrame filtered to this session (optional)

    Returns
    ───────
    AttackPath with ordered ProcessNode list
    """
    n_nodes = graph_data.num_nodes

    # Step 1: extract attention graph
    edge_index, attn_weights, x = extract_attention_graph(model, graph_data, device)

    # Step 2: find root
    root = find_root_node(edge_index, x, n_nodes)

    # Step 3: max-attention path
    path, edge_weights = max_attention_path(root, edge_index, attn_weights, n_nodes)

    # Step 4: build ProcessNode objects
    nodes = build_process_nodes(path, edge_weights, x, session_events)

    total_att   = float(sum(edge_weights)) if edge_weights else 0.0
    root_image  = nodes[0].image if nodes else "unknown"
    leaf_image  = nodes[-1].image if nodes else "unknown"

    return AttackPath(
        session_id      = session_id,
        nodes           = nodes,
        edge_attentions = edge_weights,
        total_attention = total_att,
        root_image      = root_image,
        leaf_image      = leaf_image,
        n_hops          = len(nodes) - 1,
        p_malicious     = p_malicious,
    )
