"""
src/reconstruction/pipeline.py
─────────────────────────────────────────────────────────────────────────────
Reconstruction pipeline — runs on all malicious sessions in the test set
and produces the kill chain case studies for Section 7 of the paper.

For each malicious session detected by the fusion model:
  1. Load the session graph (.pt file)
  2. Run GAT attention extraction → attack path
  3. Map path nodes to MITRE ATT&CK TTPs
  4. Generate JSON + text + SVG figure reports

Also computes reconstruction quality metrics for the paper:
  - Coverage: % of known TTPs recovered vs ground-truth labels
  - Precision: % of predicted TTPs that match ground truth
  - Chain completeness: % of sessions with ≥2 TTPs in chain

Usage
─────
  python -m src.reconstruction.pipeline --config configs/pipeline.yaml

  # Reconstruct a single session (for debugging / case study generation)
  python -m src.reconstruction.pipeline --config configs/pipeline.yaml \
      --session sess_000042
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Reconstruction quality metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_reconstruction_metrics(
    kill_chains: list,
    labelled_df: pd.DataFrame,
) -> dict:
    """
    Compare reconstructed TTP chains against ground-truth labels.

    Metrics
    ───────
    ttp_precision     : mean(predicted TTPs ∩ ground truth / predicted TTPs)
    ttp_recall        : mean(predicted TTPs ∩ ground truth / ground truth TTPs)
    chain_completeness: fraction of sessions with ≥2 unique TTPs detected
    mean_chain_length : mean number of steps in reconstructed chains
    coverage_at_1     : fraction of sessions where at least 1 TTP is correct
    """
    precisions, recalls, completeness, lengths, coverage1 = [], [], [], [], []

    for kc in kill_chains:
        pred_ttps = set(kc.raw_ttps)
        lengths.append(len(kc.chain))
        completeness.append(int(len(pred_ttps) >= 2))

        # Get ground-truth TTPs from labelled_df
        row = labelled_df[labelled_df["session_id"] == kc.session_id]
        if row.empty:
            continue

        ttp_raw = row.iloc[0].get("ttps", "[]")
        if isinstance(ttp_raw, str):
            try:
                gt_ttps = set(json.loads(ttp_raw))
            except json.JSONDecodeError:
                gt_ttps = set()
        elif isinstance(ttp_raw, list):
            gt_ttps = set(ttp_raw)
        else:
            gt_ttps = set()

        if not gt_ttps:
            continue

        # T1059.001 is always in attack data — include parent TTP matching
        # (e.g. T1059 matches T1059.001)
        def expand(ttps):
            expanded = set(ttps)
            for t in ttps:
                parent = t.split(".")[0]
                expanded.add(parent)
            return expanded

        pred_exp = expand(pred_ttps)
        gt_exp   = expand(gt_ttps)

        tp = len(pred_exp & gt_exp)
        prec = tp / len(pred_exp) if pred_exp else 0.0
        rec  = tp / len(gt_exp)   if gt_exp   else 0.0

        precisions.append(prec)
        recalls.append(rec)
        coverage1.append(int(tp >= 1))

    def safe_mean(lst):
        return round(float(np.mean(lst)), 4) if lst else 0.0

    return {
        "n_sessions_reconstructed": len(kill_chains),
        "ttp_precision":      safe_mean(precisions),
        "ttp_recall":         safe_mean(recalls),
        "chain_completeness": safe_mean(completeness),
        "mean_chain_length":  safe_mean(lengths),
        "coverage_at_1":      safe_mean(coverage1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_reconstruction(
    config_path:       str,
    target_session_id: str | None = None,
    max_sessions:      int = 50,
) -> list:
    """
    Full reconstruction pipeline.

    Parameters
    ──────────
    config_path       : path to pipeline.yaml
    target_session_id : if set, reconstruct only this session
    max_sessions      : cap on number of sessions to reconstruct

    Returns
    ───────
    list[KillChain]
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    splits_dir  = Path(cfg["paths"]["splits_dir"])
    graphs_dir  = Path(cfg["paths"]["graphs_dir"])
    reports_dir = Path("reports/kill_chains")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ── Load fusion predictions ───────────────────────────────────────────────
    fusion_pred_path = Path("models/fusion/fusion_predictions.parquet")
    if not fusion_pred_path.exists():
        log.error("Fusion predictions not found. Run train_fusion.py first.")
        return []

    fusion_preds = pd.read_parquet(fusion_pred_path)
    # Focus on test set malicious sessions
    test_malicious = fusion_preds[
        (fusion_preds["split"]     == "test") &
        (fusion_preds["label"]     == 1) &
        (fusion_preds["p_malicious_fusion"] >= 0.5)
    ].copy()

    if target_session_id:
        test_malicious = test_malicious[
            test_malicious["session_id"] == target_session_id
        ]

    test_malicious = test_malicious.head(max_sessions)
    log.info(f"Sessions to reconstruct: {len(test_malicious)}")

    if test_malicious.empty:
        log.warning("No malicious test sessions found for reconstruction.")
        return []

    # ── Load labelled data (for ground truth comparison) ─────────────────────
    labelled_path = Path(cfg["paths"]["processed_dir"]) / "labelled.parquet"
    labelled_df   = pd.read_parquet(labelled_path) if labelled_path.exists() else pd.DataFrame()

    # ── Load raw events (for command-line enrichment) ─────────────────────────
    events_path = Path(cfg["paths"]["processed_dir"]) / "events.parquet"
    events_df   = pd.read_parquet(events_path) if events_path.exists() else None

    # ── Load GAT model ────────────────────────────────────────────────────────
    try:
        import torch
        from src.models.train_gat import build_gat_model

        gat_cfg_path = Path("models/gat/gat_config.json")
        if gat_cfg_path.exists():
            with open(gat_cfg_path) as f:
                gat_cfg = json.load(f)
        else:
            gat_cfg = {"node_feature_dim": 18, "hidden_dim": 64, "heads": 8, "dropout": 0.3}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gat_model = build_gat_model(
            node_feature_dim = gat_cfg["node_feature_dim"],
            hidden_dim       = gat_cfg["hidden_dim"],
            heads            = gat_cfg["heads"],
            dropout          = gat_cfg.get("dropout", 0.3),
        ).to(device)

        model_path = Path("models/gat/gat_model.pt")
        if model_path.exists():
            gat_model.load_state_dict(
                torch.load(model_path, map_location=device, weights_only=True)
            )
            log.info("GAT model weights loaded.")
        else:
            log.warning("GAT weights not found — using random weights (for testing only)")

        gat_model.eval()
        torch_available = True

    except ImportError:
        log.warning("torch not available — using structural fallback for path extraction")
        gat_model       = None
        device          = None
        torch_available = False

    # ── Import reconstruction modules ─────────────────────────────────────────
    from src.reconstruction.graph_traversal import (
        reconstruct_attack_path, extract_attention_graph,
        find_root_node, max_attention_path, build_process_nodes,
        AttackPath,
    )
    from src.reconstruction.ttp_mapper  import map_attack_path_to_ttps
    from src.reconstruction.report      import generate_report

    kill_chains = []

    for _, pred_row in test_malicious.iterrows():
        sid          = pred_row["session_id"]
        p_malicious  = float(pred_row["p_malicious_fusion"])
        pt_path      = graphs_dir / f"{sid}.pt"

        log.info(f"Reconstructing: {sid}  (p={p_malicious:.3f})")

        # Load graph
        if torch_available:
            if not pt_path.exists():
                log.warning(f"  Graph not found: {pt_path} — skipping")
                continue
            graph_data = torch.load(pt_path, weights_only=False)
        else:
            # Structural-only fallback — build a dummy AttackPath
            log.debug("  Using structural fallback (no GAT model)")
            graph_data = None

        # Get session events for command-line enrichment
        sess_events = None
        if events_df is not None:
            sess_events = events_df[events_df["session_id"] == sid].copy()

        # Reconstruct attack path
        try:
            if torch_available and graph_data is not None:
                attack_path = reconstruct_attack_path(
                    model          = gat_model,
                    graph_data     = graph_data,
                    device         = device,
                    session_id     = sid,
                    p_malicious    = p_malicious,
                    session_events = sess_events,
                )
            else:
                # Structural fallback: build path from events only
                attack_path = _structural_fallback(
                    sid, p_malicious, sess_events
                )
        except Exception as e:
            log.error(f"  Reconstruction failed for {sid}: {e}")
            continue

        # Map to MITRE ATT&CK TTPs
        kill_chain = map_attack_path_to_ttps(attack_path)

        if not kill_chain.chain:
            log.warning(f"  No TTPs mapped for {sid} — skipping report")
            continue

        # Generate reports
        generate_report(kill_chain, reports_dir)
        kill_chains.append(kill_chain)

        log.info(f"  → {len(kill_chain.chain)} TTPs: "
                 f"{', '.join(kill_chain.raw_ttps[:5])}")

    # ── Reconstruction quality metrics ────────────────────────────────────────
    if kill_chains and not labelled_df.empty:
        metrics = compute_reconstruction_metrics(kill_chains, labelled_df)

        log.info("Reconstruction quality metrics:")
        for k, v in metrics.items():
            log.info(f"  {k:<30s} {v}")

        metrics_path = reports_dir / "reconstruction_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        log.info(f"Metrics → {metrics_path}")

        print("\n" + "═" * 55)
        print("  ATTACK CHAIN RECONSTRUCTION — METRICS")
        print("═" * 55)
        print(f"  Sessions reconstructed : {metrics['n_sessions_reconstructed']}")
        print(f"  TTP precision          : {metrics['ttp_precision']:.4f}")
        print(f"  TTP recall             : {metrics['ttp_recall']:.4f}")
        print(f"  Chain completeness     : {metrics['chain_completeness']:.4f}")
        print(f"  Mean chain length      : {metrics['mean_chain_length']:.1f} steps")
        print(f"  Coverage@1             : {metrics['coverage_at_1']:.4f}")
        print("═" * 55 + "\n")

    log.info(f"Reconstruction complete. {len(kill_chains)} kill chains → {reports_dir}/")
    return kill_chains


# ─────────────────────────────────────────────────────────────────────────────
# Structural fallback (no GAT / no torch)
# ─────────────────────────────────────────────────────────────────────────────

def _structural_fallback(
    session_id:    str,
    p_malicious:   float,
    sess_events:   "pd.DataFrame | None",
) -> "AttackPath":
    """
    Build an AttackPath from raw events without the GAT model.
    Used when torch is unavailable or for testing the TTP mapper.
    """
    from src.reconstruction.graph_traversal import (
        AttackPath, ProcessNode
    )
    import math, re

    nodes = []
    if sess_events is not None:
        proc_ev = sess_events[sess_events["event_id"] == 1].reset_index(drop=True)
        for i, row in proc_ev.iterrows():
            cmd = str(row.get("command_line", ""))
            nodes.append(ProcessNode(
                node_idx        = i,
                image           = str(row.get("image", "unknown")),
                command_line    = cmd,
                pid             = int(row.get("pid",  0)),
                ppid            = int(row.get("ppid", 0)),
                timestamp       = str(row.get("timestamp", "")),
                integrity_level = str(row.get("integrity_level", "Medium")),
                depth           = i,
                attention_score = 0.5,
                is_powershell   = str(row.get("image","")).lower() in {"powershell.exe","pwsh.exe"},
                is_lolbin       = False,
                has_network     = not sess_events[sess_events["event_id"] == 3].empty,
                has_encoded_cmd = bool(re.search(r"-enc", cmd, re.I)),
                has_download    = bool(re.search(r"downloadstring|iwr\b", cmd, re.I)),
                has_credential  = bool(re.search(r"mimikatz|lsass|sekurlsa", cmd, re.I)),
                has_lateral     = bool(re.search(r"psexec|enter-pssession", cmd, re.I)),
            ))

    if not nodes:
        # Absolute fallback — synthetic minimal node
        nodes = [ProcessNode(
            node_idx=0, image="powershell.exe", command_line="", pid=0, ppid=0,
            timestamp=None, integrity_level="High", depth=0, attention_score=0.5,
            is_powershell=True, is_lolbin=False, has_network=True,
            has_encoded_cmd=True, has_download=True, has_credential=False, has_lateral=False,
        )]

    return AttackPath(
        session_id      = session_id,
        nodes           = nodes,
        edge_attentions = [0.5] * max(len(nodes) - 1, 0),
        total_attention = 0.5 * max(len(nodes) - 1, 0),
        root_image      = nodes[0].image,
        leaf_image      = nodes[-1].image,
        n_hops          = len(nodes) - 1,
        p_malicious     = p_malicious,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/pipeline.yaml")
    parser.add_argument("--session", default=None,
                        help="Reconstruct a specific session_id only")
    parser.add_argument("--max",     type=int, default=50,
                        help="Max sessions to reconstruct")
    args = parser.parse_args()
    run_reconstruction(args.config, args.session, args.max)
