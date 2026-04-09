"""
src/models/train_rf.py
─────────────────────────────────────────────────────────────────────────────
Tier 1 — Random Forest triage model training.

Design goal: NEAR-ZERO false negatives (missed attacks).
We deliberately sacrifice precision to maximise recall. A false positive
(benign session passed to Tier 2) costs compute. A false negative
(attack session dropped) costs the whole detection chain.

Training strategy
─────────────────
1. Load tabular features from data/splits/train.parquet
2. Tune recall threshold (not accuracy) via validation set
3. Save the trained model + optimal threshold + feature importances
4. Evaluate on test + OOD sets

Key hyperparameter: class_weight="balanced" + threshold sweep (not 0.5 default)

Outputs saved to  models/rf/
  rf_model.joblib       — trained RandomForestClassifier
  rf_threshold.json     — optimal P(malicious) threshold for 98%+ recall
  rf_features.json      — ordered feature importance dict
  rf_metrics.json       — full evaluation metrics on all splits
  rf_confusion.png      — confusion matrix figure

Usage
─────
  python -m src.models.train_rf --config configs/pipeline.yaml
  python -m src.models.train_rf --config configs/pipeline.yaml --threshold-recall 0.99
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

# ── Feature columns (must match tabular.py output) ───────────────────────────
FEATURE_COLS = [
    "has_encoded_arg", "has_noprofile", "has_hidden_window", "has_bypass",
    "has_iex", "has_downloadstring", "has_reflection", "has_wmi",
    "has_scheduled_task", "has_registry_write", "has_credential_access",
    "has_lateral_movement", "cmd_length", "entropy_score",
    "parent_is_office", "parent_is_browser", "parent_is_script_host",
    "parent_is_service", "parent_is_powershell",
    "user_is_system", "user_is_high",
    "child_count", "unique_child_images", "session_event_count",
    "session_duration_secs", "has_network_call", "has_outbound_443",
    "has_outbound_80", "has_nonstandard_port", "file_write_count",
    "has_temp_write", "lolbin_child_count", "base64_token_count",
    "obfuscation_score", "token_count", "unique_cmdlet_count",
    "has_double_extension", "has_pipe_activity", "depth_in_tree",
    "script_block_length",
]


def load_split(path: Path, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Load parquet split → (X, y) arrays."""
    df = pd.read_parquet(path)
    # Fill any missing feature columns with 0
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    X = df[feature_cols].fillna(0).values.astype(np.float32)
    y = df["label"].values.astype(int)
    return X, y


def find_recall_threshold(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    target_recall: float = 0.98,
) -> tuple[float, float, float]:
    """
    Sweep probability thresholds from 0.01 → 0.99.
    Return the highest threshold that still achieves target_recall.
    Returns (threshold, achieved_recall, precision_at_threshold).
    """
    from sklearn.metrics import recall_score, precision_score

    best_thresh     = 0.5
    best_precision  = 0.0
    achieved_recall = 0.0

    for t in np.arange(0.01, 1.00, 0.01):
        preds   = (y_prob >= t).astype(int)
        recall  = recall_score(y_true, preds, zero_division=0)
        if recall >= target_recall:
            prec = precision_score(y_true, preds, zero_division=0)
            if prec >= best_precision:
                best_precision  = prec
                best_thresh     = t
                achieved_recall = recall

    return float(best_thresh), float(achieved_recall), float(best_precision)


def evaluate(
    model,
    threshold: float,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str,
) -> dict:
    """Full evaluation of a split at a given threshold."""
    from sklearn.metrics import (
        recall_score, precision_score, f1_score,
        roc_auc_score, confusion_matrix,
    )

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0   # False Negative Rate

    metrics = {
        "split":     split_name,
        "n_samples": int(len(y)),
        "n_pos":     int(y.sum()),
        "threshold": round(threshold, 3),
        "recall":    round(recall_score(y, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y, y_pred, zero_division=0), 4),
        "auc_roc":   round(roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else 0.0, 4),
        "fnr":       round(fnr, 6),   # KEY metric — must be near 0
        "fpr":       round(fp / (fp + tn) if (fp + tn) > 0 else 0.0, 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    return metrics


def plot_confusion(
    model,
    threshold: float,
    X: np.ndarray,
    y: np.ndarray,
    out_path: Path,
) -> None:
    """Save confusion matrix heatmap."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        y_pred = (model.predict_proba(X)[:, 1] >= threshold).astype(int)
        cm = confusion_matrix(y, y_pred, labels=[0, 1])

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Benign", "Malicious"],
            yticklabels=["Benign", "Malicious"],
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"RF Triage — Confusion Matrix (threshold={threshold:.2f})")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        log.info(f"Confusion matrix → {out_path}")
    except Exception as e:
        log.warning(f"Could not save confusion matrix: {e}")


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    out_path: Path,
    top_n: int = 20,
) -> None:
    """Bar chart of top-N feature importances."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        idx     = np.argsort(importances)[::-1][:top_n]
        names   = [feature_names[i] for i in idx]
        vals    = importances[idx]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(names[::-1], vals[::-1], color="#378ADD")
        ax.set_xlabel("Mean decrease in impurity")
        ax.set_title(f"Top {top_n} RF feature importances")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        log.info(f"Feature importance plot → {out_path}")
    except Exception as e:
        log.warning(f"Could not save feature importance plot: {e}")


def train(config_path: str, target_recall: float = 0.98) -> None:
    # ── Load config ───────────────────────────────────────────────────────────
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    splits_dir = Path(cfg["paths"]["splits_dir"])
    models_dir = Path("models/rf")
    models_dir.mkdir(parents=True, exist_ok=True)

    # ── Load splits ───────────────────────────────────────────────────────────
    log.info("Loading splits …")
    X_train, y_train = load_split(splits_dir / "train.parquet", FEATURE_COLS)
    X_val,   y_val   = load_split(splits_dir / "val.parquet",   FEATURE_COLS)
    X_test,  y_test  = load_split(splits_dir / "test.parquet",  FEATURE_COLS)

    ood_path = splits_dir / "ood.parquet"
    X_ood, y_ood = (load_split(ood_path, FEATURE_COLS)
                    if ood_path.exists() else (None, None))

    log.info(f"Train: {len(X_train)} samples ({y_train.sum()} malicious)")
    log.info(f"Val:   {len(X_val)} samples")
    log.info(f"Test:  {len(X_test)} samples")

    # ── Train ─────────────────────────────────────────────────────────────────
    from sklearn.ensemble import RandomForestClassifier

    log.info("Training Random Forest …")
    model = RandomForestClassifier(
        n_estimators      = 500,
        max_depth         = None,       # grow full trees
        min_samples_leaf  = 2,
        class_weight      = "balanced", # compensate for imbalance
        max_features      = "sqrt",
        n_jobs            = -1,
        random_state      = 42,
        oob_score         = True,
    )
    model.fit(X_train, y_train)
    log.info(f"OOB score: {model.oob_score_:.4f}")

    # ── Threshold tuning on validation set ───────────────────────────────────
    log.info(f"Tuning threshold for ≥{target_recall:.0%} recall on val set …")
    y_val_prob = model.predict_proba(X_val)[:, 1]
    threshold, val_recall, val_prec = find_recall_threshold(
        y_val_prob, y_val, target_recall
    )
    log.info(
        f"Optimal threshold: {threshold:.2f}  "
        f"→ recall={val_recall:.4f}  precision={val_prec:.4f}"
    )

    # ── Evaluate all splits ───────────────────────────────────────────────────
    all_metrics = []
    for name, X, y in [
        ("train", X_train, y_train),
        ("val",   X_val,   y_val),
        ("test",  X_test,  y_test),
    ]:
        m = evaluate(model, threshold, X, y, name)
        all_metrics.append(m)
        log.info(
            f"[{name:5s}] recall={m['recall']:.4f}  "
            f"precision={m['precision']:.4f}  "
            f"F1={m['f1']:.4f}  AUC={m['auc_roc']:.4f}  "
            f"FNR={m['fnr']:.6f}  FN={m['fn']}"
        )

    if X_ood is not None:
        m = evaluate(model, threshold, X_ood, y_ood, "ood")
        all_metrics.append(m)
        log.info(
            f"[ood  ] recall={m['recall']:.4f}  "
            f"precision={m['precision']:.4f}  FNR={m['fnr']:.6f}"
        )

    # ── Feature importances ───────────────────────────────────────────────────
    importances = model.feature_importances_
    feat_imp    = dict(zip(FEATURE_COLS, importances.tolist()))
    feat_imp_sorted = dict(
        sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
    )

    log.info("Top 10 features:")
    for i, (k, v) in enumerate(list(feat_imp_sorted.items())[:10]):
        log.info(f"  {i+1:2d}. {k:<35s} {v:.4f}")

    # ── Save artefacts ────────────────────────────────────────────────────────
    import joblib
    joblib.dump(model, models_dir / "rf_model.joblib")

    with open(models_dir / "rf_threshold.json", "w") as f:
        json.dump({"threshold": threshold,
                   "target_recall": target_recall,
                   "val_recall": val_recall,
                   "val_precision": val_prec}, f, indent=2)

    with open(models_dir / "rf_features.json", "w") as f:
        json.dump(feat_imp_sorted, f, indent=2)

    with open(models_dir / "rf_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    plot_confusion(model, threshold, X_test, y_test,
                   models_dir / "rf_confusion.png")
    plot_feature_importance(FEATURE_COLS, importances,
                            models_dir / "rf_feature_importance.png")

    log.info(f"All RF artefacts saved → {models_dir}/")

    # ── Paper-ready summary ───────────────────────────────────────────────────
    test_m = next(m for m in all_metrics if m["split"] == "test")
    print("\n" + "═" * 55)
    print("  RF TRIAGE — FINAL TEST METRICS (Table 2 in paper)")
    print("═" * 55)
    print(f"  Recall (sensitivity) : {test_m['recall']:.4f}")
    print(f"  Precision            : {test_m['precision']:.4f}")
    print(f"  F1 score             : {test_m['f1']:.4f}")
    print(f"  AUC-ROC              : {test_m['auc_roc']:.4f}")
    print(f"  False Negative Rate  : {test_m['fnr']:.6f}  ← primary metric")
    print(f"  False Negatives (FN) : {test_m['fn']}  ← attacks missed")
    print(f"  Threshold used       : {threshold:.2f}")
    print("═" * 55 + "\n")

    return model, threshold, all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",           default="configs/pipeline.yaml")
    parser.add_argument("--threshold-recall", type=float, default=0.98,
                        help="Minimum recall target for threshold tuning")
    args = parser.parse_args()
    train(args.config, args.threshold_recall)
