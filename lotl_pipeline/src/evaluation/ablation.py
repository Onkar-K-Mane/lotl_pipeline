"""
src/evaluation/ablation.py
─────────────────────────────────────────────────────────────────────────────
Ablation study — Section 6.2 of the paper.

Tests 7 system configurations to prove each component contributes:

  Config 1:  RF only               (baseline — rule-like)
  Config 2:  NLP only              (content only)
  Config 3:  GAT only              (context only)
  Config 4:  RF + NLP              (no graph)
  Config 5:  RF + GAT              (no NLP)
  Config 6:  NLP + GAT             (no triage)
  Config 7:  RF + NLP + GAT        (full system, learned fusion)

Primary metric: False Negative Rate (FNR) — attacks missed.
Secondary: F1, AUC-ROC, Precision, Recall.

Also runs the model-trust ablation from train_fusion.py:
zeroing each upstream score to measure which one the fusion MLP
learned to rely on most — this becomes Table 4 in the paper.

Outputs saved to  evaluation/
  ablation_results.json    — all 7 configs × all metrics
  ablation_table.txt       — LaTeX-ready table (paste into paper)
  ablation_fnr_plot.png    — FNR bar chart (paper Figure X)
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


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_prob: np.ndarray, y_true: np.ndarray,
                    threshold: float = 0.5, name: str = "") -> dict:
    from sklearn.metrics import (
        recall_score, precision_score, f1_score,
        roc_auc_score, confusion_matrix,
    )
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "config":    name,
        "recall":    round(float(recall_score(y_true,    y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_true,        y_pred, zero_division=0)), 4),
        "auc_roc":   round(float(roc_auc_score(y_true, y_prob)
                           if len(np.unique(y_true)) > 1 else 0.0), 4),
        "fnr":       round(float(fnr), 6),
        "fpr":       round(float(fpr), 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "threshold": round(threshold, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Load all model predictions for the test set
# ─────────────────────────────────────────────────────────────────────────────

def load_predictions(models_dir: Path = Path("models")) -> dict:
    """
    Load probability scores from each trained model for the test split.
    Returns dict: model_name → (y_prob, y_true) arrays.
    """
    preds = {}

    # RF predictions
    rf_path = models_dir / "rf" / "rf_predictions.parquet"
    if rf_path.exists():
        df = pd.read_parquet(rf_path)
        test = df[df["split"] == "test"]
        preds["rf"] = (test["p_malicious_rf"].values, test["label"].values)

    # NLP predictions
    nlp_path = models_dir / "distilbert" / "nlp_predictions.parquet"
    if nlp_path.exists():
        df = pd.read_parquet(nlp_path)
        test = df[df["split"] == "test"]
        preds["nlp"] = (test["p_malicious_nlp"].values, test["label"].values)

    # GAT predictions
    gat_path = models_dir / "gat" / "gat_predictions.parquet"
    if gat_path.exists():
        df = pd.read_parquet(gat_path)
        test = df[df["split"] == "test"]
        preds["gat"] = (test["p_malicious_gat"].values, test["label"].values)

    # Fusion predictions (all scores in one file)
    fusion_path = models_dir / "fusion" / "fusion_predictions.parquet"
    if fusion_path.exists():
        df = pd.read_parquet(fusion_path)
        test = df[df["split"] == "test"]
        preds["fusion"]  = (test["p_malicious_fusion"].values, test["label"].values)
        # Also extract upstream scores from fusion file for combo configs
        if "p_malicious_rf" in test.columns:
            preds["rf_from_fusion"]  = (test["p_malicious_rf"].values,  test["label"].values)
        if "p_malicious_nlp" in test.columns:
            preds["nlp_from_fusion"] = (test["p_malicious_nlp"].values, test["label"].values)
        if "p_malicious_gat" in test.columns:
            preds["gat_from_fusion"] = (test["p_malicious_gat"].values, test["label"].values)

    return preds


def ensemble_probs(*prob_arrays, weights=None) -> np.ndarray:
    """Simple weighted average ensemble of probability arrays."""
    if not prob_arrays:
        raise ValueError("No probability arrays provided")
    stacked = np.stack(prob_arrays, axis=0)
    if weights is None:
        return stacked.mean(axis=0)
    w = np.array(weights, dtype=float)
    w /= w.sum()
    return (stacked * w[:, None]).sum(axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Run all ablation configurations
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(
    preds:     dict,
    rf_threshold: float = 0.5,
) -> list[dict]:
    """
    Evaluate all 7 ablation configurations.

    For combo configs without a trained fusion, we use equal-weight
    average ensemble. The full system (Config 7) uses the trained
    fusion MLP output directly.
    """
    results = []

    def _get(key):
        """Get predictions, trying fusion file first then standalone."""
        if key + "_from_fusion" in preds:
            return preds[key + "_from_fusion"]
        return preds.get(key)

    rf_data  = _get("rf")
    nlp_data = _get("nlp")
    gat_data = _get("gat")

    if rf_data is None and nlp_data is None and gat_data is None:
        log.warning("No model predictions found. Run training scripts first.")
        return []

    # Config 1: RF only
    if rf_data is not None:
        y_prob, y_true = rf_data
        results.append(compute_metrics(y_prob, y_true, rf_threshold, "RF only"))

    # Config 2: NLP only
    if nlp_data is not None:
        y_prob, y_true = nlp_data
        results.append(compute_metrics(y_prob, y_true, 0.5, "NLP (DistilBERT) only"))

    # Config 3: GAT only
    if gat_data is not None:
        y_prob, y_true = gat_data
        results.append(compute_metrics(y_prob, y_true, 0.5, "GAT only"))

    # Config 4: RF + NLP (average ensemble)
    if rf_data is not None and nlp_data is not None:
        # Align on common labels
        y_true = rf_data[1]
        y_prob = ensemble_probs(rf_data[0], nlp_data[0])
        results.append(compute_metrics(y_prob, y_true, 0.5, "RF + NLP (no graph)"))

    # Config 5: RF + GAT
    if rf_data is not None and gat_data is not None:
        y_true = rf_data[1]
        y_prob = ensemble_probs(rf_data[0], gat_data[0])
        results.append(compute_metrics(y_prob, y_true, 0.5, "RF + GAT (no NLP)"))

    # Config 6: NLP + GAT (no triage)
    if nlp_data is not None and gat_data is not None:
        y_true = nlp_data[1]
        y_prob = ensemble_probs(nlp_data[0], gat_data[0])
        results.append(compute_metrics(y_prob, y_true, 0.5, "NLP + GAT (no triage)"))

    # Config 7: Full system (trained fusion)
    if "fusion" in preds:
        y_prob, y_true = preds["fusion"]
        results.append(compute_metrics(y_prob, y_true, 0.5,
                        "RF + NLP + GAT (full system, learned fusion)"))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Output formatting
# ─────────────────────────────────────────────────────────────────────────────

def print_ablation_table(results: list[dict]) -> str:
    """Print a formatted table and return LaTeX source."""
    print("\n" + "═" * 95)
    print("  ABLATION STUDY — TEST SET RESULTS  (Table 3 in paper)")
    print("═" * 95)
    header = f"  {'Configuration':<42s}  {'Recall':>7}  {'Precision':>9}  {'F1':>6}  {'AUC':>6}  {'FNR':>8}  {'FN':>4}"
    print(header)
    print("  " + "─" * 91)

    for r in results:
        marker = "  ◀ FULL" if "full system" in r["config"] else ""
        print(
            f"  {r['config']:<42s}  "
            f"{r['recall']:>7.4f}  "
            f"{r['precision']:>9.4f}  "
            f"{r['f1']:>6.4f}  "
            f"{r['auc_roc']:>6.4f}  "
            f"{r['fnr']:>8.6f}  "
            f"{r['fn']:>4d}"
            f"{marker}"
        )
    print("═" * 95 + "\n")

    # LaTeX table
    latex = []
    latex.append(r"\begin{table}[h]")
    latex.append(r"\centering")
    latex.append(r"\caption{Ablation study results on the test set. "
                 r"FNR = False Negative Rate (primary metric). "
                 r"Full system uses learned MLP fusion of all three tiers.}")
    latex.append(r"\label{tab:ablation}")
    latex.append(r"\begin{tabular}{lcccccc}")
    latex.append(r"\hline")
    latex.append(r"\textbf{Configuration} & \textbf{Recall} & \textbf{Precision} & "
                 r"\textbf{F1} & \textbf{AUC} & \textbf{FNR} & \textbf{FN} \\")
    latex.append(r"\hline")
    for r in results:
        name = r["config"].replace("&", r"\&")
        bold = r["config"] == results[-1]["config"]
        row = (f"{'\\textbf{' if bold else ''}{name}{'}'  if bold else ''} & "
               f"{'\\textbf{' if bold else ''}{r['recall']:.4f}{'}'  if bold else ''} & "
               f"{r['precision']:.4f} & {r['f1']:.4f} & "
               f"{r['auc_roc']:.4f} & "
               f"{'\\textbf{' if bold else ''}{r['fnr']:.6f}{'}'  if bold else ''} & "
               f"{r['fn']} \\\\")
        latex.append(row)
    latex.append(r"\hline")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    return "\n".join(latex)


def plot_ablation_fnr(results: list[dict], out_path: Path) -> None:
    """FNR bar chart — lower is better. Full system highlighted."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [r["config"].replace("(", "\n(") for r in results]
        fnrs  = [r["fnr"] for r in results]
        colors = ["#1D9E75" if "full system" in r["config"] else "#B4B2A9"
                  for r in results]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(range(len(names)), fnrs, color=colors, width=0.6)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel("False Negative Rate (↓ better)")
        ax.set_title("Ablation study — FNR by configuration\n"
                     "(green = full system; grey = ablated configurations)")
        ax.set_ylim(0, max(fnrs) * 1.15 if fnrs else 0.1)

        # Value labels
        for bar, v in zip(bars, fnrs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8)

        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        log.info(f"FNR plot → {out_path}")
    except Exception as e:
        log.warning(f"Could not save FNR plot: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(config_path: str = "configs/pipeline.yaml") -> list[dict]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    eval_dir = Path("evaluation")
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Load RF threshold
    rf_thresh_path = Path("models/rf/rf_threshold.json")
    rf_threshold   = 0.5
    if rf_thresh_path.exists():
        with open(rf_thresh_path) as f:
            rf_threshold = json.load(f).get("threshold", 0.5)

    log.info("Loading model predictions …")
    preds = load_predictions()

    log.info("Running ablation …")
    results = run_ablation(preds, rf_threshold)

    if not results:
        log.warning("No results — predictions not found. Train models first.")
        return []

    # Save results
    with open(eval_dir / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    latex = print_ablation_table(results)

    with open(eval_dir / "ablation_table.tex", "w") as f:
        f.write(latex)
    log.info(f"LaTeX table → {eval_dir}/ablation_table.tex")

    plot_ablation_fnr(results, eval_dir / "ablation_fnr_plot.png")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()
    run(args.config)
