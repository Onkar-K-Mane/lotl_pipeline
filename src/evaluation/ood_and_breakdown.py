"""
src/evaluation/ood_and_breakdown.py
─────────────────────────────────────────────────────────────────────────────
Out-of-distribution (OOD) generalisation + per-family breakdown.
Sections 6.4 and 6.5 of the paper.

OOD evaluation (6.4)
─────────────────────
The OOD test set contains attack families held out entirely during
training (empire_mimikatz_extract_keys by default). Performance on OOD
data demonstrates that the model generalises to unseen attack patterns,
not just memorising training examples. This is the most critical
credibility test for a Scopus reviewer.

Per-family breakdown (6.5)
───────────────────────────
FNR broken down by attack family (Execution, CredentialAccess,
LateralMovement, DefenseEvasion, Persistence) — shows which attack
types the system handles best and worst.

Outputs saved to  evaluation/
  ood_results.json            — OOD split metrics for all models
  ood_comparison_table.tex    — LaTeX OOD table
  family_breakdown.json       — per-family FNR breakdown
  family_fnr_heatmap.png      — per-family × per-model FNR heatmap
"""

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


def _metrics(y_prob, y_true, threshold=0.5, name="", n_total=None):
    from sklearn.metrics import (
        recall_score, precision_score, f1_score,
        roc_auc_score, confusion_matrix,
    )
    if len(y_prob) == 0:
        return {"split": name, "n": 0, "recall": 0, "precision": 0,
                "f1": 0, "auc_roc": 0, "fnr": 1.0, "fn": 0}
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return {
        "split":     name,
        "n":         int(n_total or len(y_prob)),
        "n_pos":     int(y_true.sum()),
        "recall":    round(float(recall_score(y_true,    y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_true,        y_pred, zero_division=0)), 4),
        "auc_roc":   round(float(roc_auc_score(y_true, y_prob)
                           if len(np.unique(y_true)) > 1 else 0.0), 4),
        "fnr":       round(float(fnr), 6),
        "fn":        int(fn),
        "tp":        int(tp),
    }


# ─────────────────────────────────────────────────────────────────────────────
# OOD evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_ood_evaluation(config_path: str = "configs/pipeline.yaml") -> list[dict]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    eval_dir   = Path("evaluation")
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Load OOD predictions from each model
    model_files = {
        "RF (Tier 1)":            ("models/rf/rf_predictions.parquet",       "p_malicious_rf"),
        "DistilBERT (Tier 2A)":   ("models/distilbert/nlp_predictions.parquet", "p_malicious_nlp"),
        "GAT (Tier 2B)":          ("models/gat/gat_predictions.parquet",     "p_malicious_gat"),
        "Full system (proposed)": ("models/fusion/fusion_predictions.parquet","p_malicious_fusion"),
    }

    results = []
    roc_data = {}

    # Load RF threshold
    rf_thresh_path = Path("models/rf/rf_threshold.json")
    rf_threshold   = 0.5
    if rf_thresh_path.exists():
        with open(rf_thresh_path) as f2:
            rf_threshold = json.load(f2).get("threshold", 0.5)

    for model_name, (pred_path, score_col) in model_files.items():
        path = Path(pred_path)
        if not path.exists():
            log.warning(f"OOD predictions not found for {model_name}")
            continue

        df   = pd.read_parquet(path)
        ood  = df[df["split"] == "ood"]

        if ood.empty:
            log.warning(f"No OOD rows in {pred_path}")
            continue

        # Also get IID test for comparison
        iid  = df[df["split"] == "test"]

        thresh = rf_threshold if "RF" in model_name else 0.5

        m_iid = _metrics(
            iid[score_col].values, iid["label"].values,
            threshold=thresh, name=f"{model_name} [IID test]"
        )
        m_ood = _metrics(
            ood[score_col].values, ood["label"].values,
            threshold=thresh, name=f"{model_name} [OOD test]"
        )

        results.append({"model": model_name, "iid": m_iid, "ood": m_ood})
        roc_data[f"{model_name} (OOD)"] = (
            ood[score_col].values, ood["label"].values
        )

        log.info(
            f"[{model_name}]  "
            f"IID FNR={m_iid['fnr']:.4f}  "
            f"OOD FNR={m_ood['fnr']:.4f}  "
            f"(Δ={m_ood['fnr']-m_iid['fnr']:+.4f})"
        )

    if not results:
        log.warning("No OOD results — train models first.")
        return []

    with open(eval_dir / "ood_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print and save LaTeX table
    _print_ood_table(results)
    latex = _ood_latex(results)
    with open(eval_dir / "ood_comparison_table.tex", "w") as f:
        f.write(latex)

    return results


def _print_ood_table(results: list[dict]) -> None:
    print("\n" + "═" * 85)
    print("  OOD GENERALISATION — IID vs OOD FNR  (Table 4 in paper)")
    print("═" * 85)
    print(f"  {'Model':<30s}  {'IID Recall':>10}  {'OOD Recall':>10}  "
          f"{'IID FNR':>8}  {'OOD FNR':>8}  {'ΔFNR':>7}")
    print("  " + "─" * 81)
    for r in results:
        iid, ood = r["iid"], r["ood"]
        delta_fnr = ood["fnr"] - iid["fnr"]
        print(
            f"  {r['model']:<30s}  "
            f"{iid['recall']:>10.4f}  {ood['recall']:>10.4f}  "
            f"{iid['fnr']:>8.6f}  {ood['fnr']:>8.6f}  "
            f"{delta_fnr:>+7.4f}"
        )
    print("═" * 85 + "\n")


def _ood_latex(results: list[dict]) -> str:
    lines = [
        r"\begin{table}[h]", r"\centering",
        r"\caption{Generalisation to out-of-distribution (OOD) attack families. "
        r"$\Delta$FNR = OOD FNR $-$ IID FNR; lower is better. "
        r"Smaller $\Delta$FNR indicates stronger generalisation.}",
        r"\label{tab:ood}",
        r"\begin{tabular}{lcccccc}",
        r"\hline",
        r"\textbf{Model} & \textbf{IID Recall} & \textbf{OOD Recall} & "
        r"\textbf{IID FNR} & \textbf{OOD FNR} & \textbf{$\Delta$FNR} & "
        r"\textbf{OOD FN} \\", r"\hline",
    ]
    for r in results:
        iid, ood = r["iid"], r["ood"]
        delta = ood["fnr"] - iid["fnr"]
        lines.append(
            f"{r['model']} & {iid['recall']:.4f} & {ood['recall']:.4f} & "
            f"{iid['fnr']:.6f} & {ood['fnr']:.6f} & "
            f"{delta:+.4f} & {ood['fn']} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Per-family breakdown
# ─────────────────────────────────────────────────────────────────────────────

# Map attack families to MITRE tactic groups for cleaner display
FAMILY_TO_TACTIC = {
    "empire_psinject":                        "Defense Evasion",
    "empire_find_local_admin_access":         "Discovery",
    "empire_schtasks":                        "Persistence",
    "empire_mimikatz_extract_keys":           "Credential Access",
    "empire_psexec_pth_dcerpc_svcctl":        "Lateral Movement",
    "splunk_T1059.001_windows-sysmon":        "Execution (PS)",
    "splunk_T1027_windows-sysmon":            "Defense Evasion",
    "splunk_T1003.001_windows-sysmon":        "Credential Access",
}


def run_family_breakdown(config_path: str = "configs/pipeline.yaml") -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    eval_dir = Path("evaluation")
    eval_dir.mkdir(parents=True, exist_ok=True)

    fusion_path = Path("models/fusion/fusion_predictions.parquet")
    if not fusion_path.exists():
        log.warning("Fusion predictions not found.")
        return {}

    df = pd.read_parquet(fusion_path)

    # We need the family column — merge with labelled data
    labelled_path = Path(cfg["paths"]["processed_dir"]) / "labelled.parquet"
    if labelled_path.exists():
        lab_df = pd.read_parquet(labelled_path)[["session_id", "family"]]
        df = df.merge(lab_df, on="session_id", how="left")

    if "family" not in df.columns:
        log.warning("family column not available — skipping family breakdown")
        return {}

    test_df = df[df["split"] == "test"].copy()
    test_df["tactic"] = test_df["family"].map(FAMILY_TO_TACTIC).fillna("Other")

    breakdown = {}
    print("\n" + "═" * 70)
    print("  PER-FAMILY BREAKDOWN — FNR by attack family  (Section 6.5)")
    print("═" * 70)

    for family, fdf in test_df.groupby("family"):
        if fdf["label"].sum() == 0:
            continue   # no positive examples in this family on test set
        m = _metrics(
            fdf["p_malicious_fusion"].values,
            fdf["label"].values,
            threshold=0.5,
            name=family,
        )
        tactic = FAMILY_TO_TACTIC.get(family, "Other")
        breakdown[family] = {**m, "tactic": tactic}
        log.info(f"  [{tactic:<22s}] {family[:35]:<35s}  "
                 f"recall={m['recall']:.4f}  FNR={m['fnr']:.4f}  "
                 f"n={m['n']}  n_pos={m['n_pos']}")

        print(f"  {tactic:<22s}  {family[:32]:<32s}  "
              f"recall={m['recall']:.4f}  FNR={m['fnr']:.6f}  FN={m['fn']}")

    print("═" * 70 + "\n")

    with open(eval_dir / "family_breakdown.json", "w") as f:
        json.dump(breakdown, f, indent=2)

    _plot_family_heatmap(breakdown, eval_dir / "family_fnr_heatmap.png")
    return breakdown


def _plot_family_heatmap(breakdown: dict, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not breakdown:
            return

        families = list(breakdown.keys())
        fnrs     = [breakdown[f]["fnr"]     for f in families]
        recalls  = [breakdown[f]["recall"]  for f in families]
        tactics  = [breakdown[f]["tactic"]  for f in families]

        labels = [f"{t}\n({f[:20]})" for f, t in zip(families, tactics)]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, max(4, len(families) * 0.7)))

        # FNR bar (horizontal)
        colors = ["#D85A30" if v > 0.05 else "#1D9E75" for v in fnrs]
        ax1.barh(range(len(labels)), fnrs, color=colors)
        ax1.set_yticks(range(len(labels)))
        ax1.set_yticklabels(labels, fontsize=8)
        ax1.set_xlabel("False Negative Rate")
        ax1.set_title("FNR by attack family\n(red = FNR > 5%)")
        ax1.axvline(0.05, color="#BA7517", ls="--", lw=1, label="5% FNR threshold")
        ax1.legend(fontsize=8)

        # Recall bar
        ax2.barh(range(len(labels)), recalls, color="#534AB7")
        ax2.set_yticks(range(len(labels)))
        ax2.set_yticklabels([""] * len(labels))
        ax2.set_xlabel("Recall")
        ax2.set_title("Recall by attack family")

        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        log.info(f"Family heatmap → {out_path}")
    except Exception as e:
        log.warning(f"Could not save family heatmap: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(config_path: str = "configs/pipeline.yaml") -> dict:
    ood_results    = run_ood_evaluation(config_path)
    family_results = run_family_breakdown(config_path)
    return {"ood": ood_results, "family": family_results}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()
    run(args.config)
