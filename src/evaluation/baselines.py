"""
src/evaluation/baselines.py
─────────────────────────────────────────────────────────────────────────────
Baseline comparison — Section 6.3 of the paper.

We implement or replicate the core detection logic of four prior systems
using the same test set, so all comparisons are on identical data:

  B1: Signature/rule-based  — simulates LOTLDetector rule-tag approach
      (keyword matching + known-bad pattern flags, no ML)

  B2: Random Forest alone   — our Tier 1, treated as a standalone detector
      at default threshold (represents traditional ML baselines)

  B3: NLP-only (BERT-style) — our fine-tuned DistilBERT with no graph context
      (represents Hendler et al. 2020 / Yang et al. 2023 style systems)

  B4: GNN-only (GAT)        — our GAT with no NLP content analysis
      (represents Choi 2021 / ThreaTrace style systems)

  Ours: Full hierarchical system (RF + DistilBERT + GAT + Learned Fusion)

For each baseline we report: Recall, Precision, F1, AUC-ROC, FNR, FN.
This becomes Table 2 in the paper (the main comparison table).

Outputs saved to  evaluation/
  baseline_results.json        — all baseline metrics
  baseline_comparison_table.tex — LaTeX comparison table (Table 2)
  baseline_roc_comparison.png  — multi-ROC plot (paper Figure)
"""

import json
import logging
import re
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
# B1: Rule/signature baseline
# ─────────────────────────────────────────────────────────────────────────────

# Malicious keyword patterns from LOLBAS + known PowerShell attack signatures
_MALICIOUS_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"-enc(odedcommand)?",
        r"-noprofile|-nop\b",
        r"-windowstyle\s+hidden|-w\s+h",
        r"-executionpolicy\s+bypass",
        r"invoke-expression|iex\b",
        r"downloadstring|downloadfile|new-object\s+net\.webclient",
        r"invoke-webrequest|iwr\b",
        r"start-bitstransfer|bitsadmin",
        r"invoke-mimikatz|sekurlsa|lsass",
        r"certutil\s+-urlcache|-decode",
        r"regsvr32\s+/s\s+/n\s+/u\s+/i:",
        r"mshta\s+vbscript:|mshta\s+javascript:",
        r"msbuild\s+.+\.xml",
        r"installutil\s+/logfile",
        r"\[system\.reflection\.assembly\]",
        r"add-type\s+-assemblyname",
        r"powershell\s+-c\s+iex|powershell\.exe\s+.*iex",
        r"invoke-command\s+-computername",
        r"new-pssession|enter-pssession",
        r"wmic\s+process\s+call\s+create",
    ]
]

_SUSPICIOUS_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"base64",
        r"frombase64string",
        r"\[convert\]::",
        r"system\.net\.",
        r"shellexecute",
        r"createobject",
    ]
]


def rule_based_score(command_line: str) -> float:
    """
    Simulate a rule-tag system (LOTLDetector-style).
    Returns a score in [0, 1] based on how many rules fire.
    Deterministic — no training required.
    """
    if not command_line:
        return 0.0
    malicious_hits  = sum(1 for p in _MALICIOUS_PATTERNS  if p.search(command_line))
    suspicious_hits = sum(1 for p in _SUSPICIOUS_PATTERNS if p.search(command_line))

    # Normalise: each malicious hit = 0.15, suspicious = 0.05, cap at 1.0
    score = min(1.0, malicious_hits * 0.15 + suspicious_hits * 0.05)
    return score


def evaluate_rule_baseline(test_df: pd.DataFrame,
                            events_df: pd.DataFrame | None) -> tuple[np.ndarray, np.ndarray]:
    """Apply rule scoring to each session (max score across session events)."""
    scores = []
    for sid in test_df["session_id"]:
        if events_df is not None and "session_id" in events_df.columns:
            sess_ev = events_df[
                (events_df["session_id"] == sid) &
                (events_df["event_id"] == 1)
            ]
            if not sess_ev.empty:
                max_score = sess_ev["command_line"].fillna("").apply(rule_based_score).max()
                scores.append(max_score)
                continue
        # Fallback — use tabular features if available
        row = test_df[test_df["session_id"] == sid]
        if not row.empty:
            heuristic = float(row.iloc[0].get("has_encoded_arg", 0)) * 0.3 + \
                        float(row.iloc[0].get("has_iex", 0)) * 0.2 + \
                        float(row.iloc[0].get("has_downloadstring", 0)) * 0.2 + \
                        float(row.iloc[0].get("has_credential_access", 0)) * 0.3
            scores.append(min(1.0, heuristic))
        else:
            scores.append(0.0)

    y_prob = np.array(scores)
    y_true = test_df["label"].values.astype(int)
    return y_prob, y_true


# ─────────────────────────────────────────────────────────────────────────────
# Metric calculation
# ─────────────────────────────────────────────────────────────────────────────

def metrics(y_prob: np.ndarray, y_true: np.ndarray,
            threshold: float = 0.5, name: str = "") -> dict:
    from sklearn.metrics import (
        recall_score, precision_score, f1_score,
        roc_auc_score, confusion_matrix,
    )
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return {
        "system":    name,
        "recall":    round(float(recall_score(y_true,    y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_true,        y_pred, zero_division=0)), 4),
        "auc_roc":   round(float(roc_auc_score(y_true, y_prob)
                           if len(np.unique(y_true)) > 1 else 0.0), 4),
        "fnr":       round(float(fnr), 6),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "threshold": threshold,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX and plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(results: list[dict]) -> str:
    print("\n" + "═" * 95)
    print("  BASELINE COMPARISON — TEST SET  (Table 2 in paper)")
    print("═" * 95)
    header = (f"  {'System':<40s}  {'Recall':>7}  {'Precision':>9}  "
              f"{'F1':>6}  {'AUC':>6}  {'FNR':>8}  {'FN':>4}")
    print(header)
    print("  " + "─" * 91)
    for r in results:
        marker = "  ◀ OURS" if "Proposed" in r["system"] or "ours" in r["system"].lower() else ""
        print(
            f"  {r['system']:<40s}  "
            f"{r['recall']:>7.4f}  {r['precision']:>9.4f}  "
            f"{r['f1']:>6.4f}  {r['auc_roc']:>6.4f}  "
            f"{r['fnr']:>8.6f}  {r['fn']:>4d}{marker}"
        )
    print("═" * 95 + "\n")

    # LaTeX
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Comparison with baseline systems on the test set. "
        r"FNR = False Negative Rate. "
        r"$\dagger$ Rule-based: LOTLDetector-style keyword matching. "
        r"$\ddagger$ NLP-only: fine-tuned DistilBERT without graph context. "
        r"Best values \textbf{bolded}.}",
        r"\label{tab:comparison}",
        r"\begin{tabular}{lcccccc}",
        r"\hline",
        r"\textbf{System} & \textbf{Recall} & \textbf{Precision} & "
        r"\textbf{F1} & \textbf{AUC} & \textbf{FNR} & \textbf{FN} \\",
        r"\hline",
    ]

    # Find best values for bolding
    best_recall = max(r["recall"]    for r in results)
    best_prec   = max(r["precision"] for r in results)
    best_f1     = max(r["f1"]        for r in results)
    best_auc    = max(r["auc_roc"]   for r in results)
    best_fnr    = min(r["fnr"]       for r in results)
    best_fn     = min(r["fn"]        for r in results)

    def b(val, best, fmt):
        s = fmt.format(val)
        return f"\\textbf{{{s}}}" if abs(val - best) < 1e-6 else s

    for r in results:
        name  = r["system"].replace("&", r"\&").replace("#", r"\#")
        lines.append(
            f"{name} & "
            f"{b(r['recall'],    best_recall, '{:.4f}')} & "
            f"{b(r['precision'], best_prec,   '{:.4f}')} & "
            f"{b(r['f1'],        best_f1,     '{:.4f}')} & "
            f"{b(r['auc_roc'],   best_auc,    '{:.4f}')} & "
            f"{b(r['fnr'],       best_fnr,    '{:.6f}')} & "
            f"{b(r['fn'],        best_fn,     '{:d}')} \\\\"
        )

    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def plot_roc_comparison(roc_data: dict, out_path: Path) -> None:
    """Multi-ROC curve comparing all baselines + proposed system."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc

        style_map = {
            "Rule-based$^\\dagger$":       ("#888780", "--", 1.2),
            "RF only (B2)":                ("#B4B2A9", "--", 1.2),
            "NLP only (B3)$^\\ddagger$":   ("#534AB7", ":",  1.5),
            "GAT only (B4)":               ("#D85A30", "-.", 1.5),
            "Proposed system (ours)":      ("#1D9E75", "-",  2.5),
        }

        fig, ax = plt.subplots(figsize=(6, 5))

        for name, (y_prob, y_true) in roc_data.items():
            if len(np.unique(y_true)) < 2:
                continue
            fpr_arr, tpr_arr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr_arr, tpr_arr)
            color, ls, lw = style_map.get(name, ("#378ADD", "-", 1.5))
            ax.plot(fpr_arr, tpr_arr, color=color, ls=ls, lw=lw,
                    label=f"{name} (AUC={roc_auc:.4f})")

        ax.plot([0, 1], [0, 1], "k:", lw=1, label="Random (AUC=0.5000)")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate (Recall)")
        ax.set_title("ROC curves — baseline comparison (test set)")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        log.info(f"ROC comparison → {out_path}")
    except Exception as e:
        log.warning(f"Could not save ROC plot: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(config_path: str = "configs/pipeline.yaml") -> list[dict]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    splits_dir  = Path(cfg["paths"]["splits_dir"])
    proc_dir    = Path(cfg["paths"]["processed_dir"])
    eval_dir    = Path("evaluation")
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Load test split
    test_path = splits_dir / "test.parquet"
    if not test_path.exists():
        log.error("test.parquet not found. Run pipeline first.")
        return []
    test_df = pd.read_parquet(test_path)

    # Load raw events for rule baseline
    events_path = proc_dir / "events.parquet"
    events_df   = pd.read_parquet(events_path) if events_path.exists() else None

    # Load RF threshold
    rf_thresh_path = Path("models/rf/rf_threshold.json")
    rf_threshold   = 0.5
    if rf_thresh_path.exists():
        with open(rf_thresh_path) as f:
            rf_threshold = json.load(f).get("threshold", 0.5)

    all_results = []
    roc_data    = {}

    # ── B1: Rule-based ──────────────────────────────────────────────────────
    log.info("Evaluating B1: rule-based baseline …")
    y_prob_rule, y_true = evaluate_rule_baseline(test_df, events_df)
    r1 = metrics(y_prob_rule, y_true, threshold=0.3, name="Rule-based$^\\dagger$")
    all_results.append(r1)
    roc_data["Rule-based$^\\dagger$"] = (y_prob_rule, y_true)

    # ── B2–B4 and Ours: from saved model predictions ─────────────────────────
    from src.evaluation.ablation import load_predictions

    preds = load_predictions()

    def _get(key):
        if key + "_from_fusion" in preds:
            return preds[key + "_from_fusion"]
        return preds.get(key)

    rf_data  = _get("rf")
    nlp_data = _get("nlp")
    gat_data = _get("gat")

    if rf_data is not None:
        log.info("Evaluating B2: RF standalone …")
        y_prob, y_true_rf = rf_data
        r2 = metrics(y_prob, y_true_rf, rf_threshold, "RF only (B2)")
        all_results.append(r2)
        roc_data["RF only (B2)"] = (y_prob, y_true_rf)

    if nlp_data is not None:
        log.info("Evaluating B3: NLP only …")
        y_prob, y_true_nlp = nlp_data
        r3 = metrics(y_prob, y_true_nlp, 0.5,
                     "NLP only (B3)$^\\ddagger$")
        all_results.append(r3)
        roc_data["NLP only (B3)$^\\ddagger$"] = (y_prob, y_true_nlp)

    if gat_data is not None:
        log.info("Evaluating B4: GAT only …")
        y_prob, y_true_gat = gat_data
        r4 = metrics(y_prob, y_true_gat, 0.5, "GAT only (B4)")
        all_results.append(r4)
        roc_data["GAT only (B4)"] = (y_prob, y_true_gat)

    if "fusion" in preds:
        log.info("Evaluating proposed system …")
        y_prob, y_true_fus = preds["fusion"]
        r5 = metrics(y_prob, y_true_fus, 0.5,
                     "Proposed system (ours)")
        all_results.append(r5)
        roc_data["Proposed system (ours)"] = (y_prob, y_true_fus)

    if not all_results:
        log.warning("No results — train models and run pipeline first.")
        return []

    # Save + display
    with open(eval_dir / "baseline_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    latex = print_comparison_table(all_results)
    with open(eval_dir / "baseline_comparison_table.tex", "w") as f:
        f.write(latex)

    plot_roc_comparison(roc_data, eval_dir / "baseline_roc_comparison.png")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()
    run(args.config)
