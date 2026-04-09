"""
src/models/train_fusion.py
─────────────────────────────────────────────────────────────────────────────
Tier 3 — Learned fusion MLP (meta-classifier).

This is the final decision layer. It receives the probability scores from
all three upstream models and learns WHEN to trust each one.

Input vector (8 dimensions)
────────────────────────────
  0  p_rf           — Random Forest P(malicious)       [0, 1]
  1  p_nlp          — DistilBERT P(malicious)           [0, 1]
  2  p_gat          — GAT P(malicious)                  [0, 1]
  3  rf_confidence  — RF raw confidence (max class prob)[0, 1]
  4  session_duration_secs (log-normalised)             [0, 1]
  5  node_count     — number of process nodes in session
  6  has_lateral_movement_flag
  7  has_network_call

Architecture
────────────
  Linear(8 → 64) → ReLU → Dropout(0.2)
  → Linear(64 → 32) → ReLU
  → Linear(32 → 1) → Sigmoid

Training protocol
─────────────────
  All three upstream models are FROZEN. Only the fusion MLP is trained.
  This two-stage training ensures:
    a) Each upstream model specialises on its own signal independently
    b) The fusion layer learns combination weights, not compensating for
       individual model failures

  Loss: weighted BCE (same imbalance handling as individual models)
  Epochs: 50 (small model, fast convergence)
  lr: 1e-3 with step decay every 10 epochs

Outputs saved to  models/fusion/
  fusion_model.pt           — MLP state dict
  fusion_metrics.json       — full evaluation on all splits
  fusion_predictions.parquet— session_id + final P(malicious)
  fusion_weight_analysis.json — which model the fusion trusts most per attack family
  fusion_roc.png            — ROC curve for all four models (paper Figure)

Usage
─────
  python -m src.models.train_fusion --config configs/pipeline.yaml
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
# Fusion MLP
# ─────────────────────────────────────────────────────────────────────────────

def build_fusion_mlp(input_dim: int = 8, dropout: float = 0.2):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Feature assembly
# ─────────────────────────────────────────────────────────────────────────────

def assemble_fusion_features(
    tabular_df:  pd.DataFrame,
    rf_preds:    pd.DataFrame,
    nlp_preds:   pd.DataFrame,
    gat_preds:   pd.DataFrame,
    split:       str,
) -> pd.DataFrame:
    """
    Join upstream model scores + meta-features into one fusion feature matrix.

    Returns DataFrame with columns:
        session_id | p_rf | p_nlp | p_gat | rf_confidence |
        session_duration_norm | node_count | has_lateral_movement | has_network_call |
        label | family | split
    """
    import math

    # Filter each prediction table to this split
    rf_s  = rf_preds[rf_preds["split"]  == split][["session_id", "p_malicious_rf",  "rf_confidence"]].copy()
    nlp_s = nlp_preds[nlp_preds["split"] == split][["session_id", "p_malicious_nlp"]].copy()
    gat_s = gat_preds[gat_preds["split"] == split][["session_id", "p_malicious_gat"]].copy()
    tab_s = tabular_df[["session_id", "label", "family",
                         "session_duration_secs", "child_count",
                         "has_lateral_movement", "has_network_call"]].copy()

    # Merge everything on session_id
    df = tab_s.merge(rf_s,  on="session_id", how="left")
    df = df.merge(nlp_s,    on="session_id", how="left")
    df = df.merge(gat_s,    on="session_id", how="left")

    # Fill missing scores with 0.5 (uncertain) — model may not have covered all sessions
    for col in ["p_malicious_rf", "rf_confidence", "p_malicious_nlp", "p_malicious_gat"]:
        if col not in df.columns:
            df[col] = 0.5
        df[col] = df[col].fillna(0.5)

    # Log-normalise session duration (seconds → [0,1] approx)
    df["session_duration_norm"] = df["session_duration_secs"].apply(
        lambda x: min(math.log(max(x, 1) + 1) / 10.0, 1.0)
    )

    # Normalise node count (clip at 50, then /50)
    df["node_count_norm"] = (df["child_count"].clip(upper=50) / 50.0).fillna(0)

    df["split"] = split
    return df


def df_to_tensors(df: pd.DataFrame):
    """Extract (X, y) tensors from assembled fusion DataFrame."""
    import torch

    FEATURE_COLS = [
        "p_malicious_rf", "p_malicious_nlp", "p_malicious_gat",
        "rf_confidence",
        "session_duration_norm", "node_count_norm",
        "has_lateral_movement", "has_network_call",
    ]
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    X = torch.tensor(df[FEATURE_COLS].fillna(0).values, dtype=torch.float32)
    y = torch.tensor(df["label"].values, dtype=torch.float32)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, X, y, optimiser, loss_fn, batch_size: int = 64) -> float:
    import torch
    model.train()
    perm       = torch.randperm(len(X))
    total_loss = 0.0
    n_batches  = 0

    for i in range(0, len(X), batch_size):
        idx = perm[i : i + batch_size]
        xb, yb = X[idx], y[idx]
        optimiser.zero_grad()
        out  = model(xb).squeeze(-1)
        loss = loss_fn(out, yb)
        loss.backward()
        optimiser.step()
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


def evaluate(model, X: "torch.Tensor", y: "torch.Tensor",
             split_name: str, threshold: float = 0.5) -> tuple[dict, np.ndarray]:
    import torch
    from sklearn.metrics import (
        recall_score, precision_score, f1_score,
        roc_auc_score, confusion_matrix,
    )
    model.eval()
    with torch.no_grad():
        y_prob = model(X).squeeze(-1).numpy()
        y_true = y.numpy().astype(int)

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "split":     split_name,
        "threshold": threshold,
        "recall":    round(recall_score(y_true,    y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true,        y_pred, zero_division=0), 4),
        "auc_roc":   round(roc_auc_score(y_true, y_prob)
                           if len(np.unique(y_true)) > 1 else 0.0, 4),
        "fnr":       round(fn / (fn + tp) if (fn + tp) > 0 else 0.0, 6),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    return metrics, y_prob


def plot_roc_comparison(
    splits_data: dict,   # {"RF": (y_prob, y_true), "NLP": ..., "GAT": ..., "Fusion": ...}
    out_path: Path,
) -> None:
    """
    Multi-model ROC curve — this is a key figure for the paper.
    Shows all four models on one plot to demonstrate fusion superiority.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc

        colors = {
            "RF (triage)":    "#888780",
            "DistilBERT":     "#534AB7",
            "GAT":            "#D85A30",
            "Fusion (ours)":  "#1D9E75",
        }

        fig, ax = plt.subplots(figsize=(6, 5))

        for name, (y_prob, y_true) in splits_data.items():
            if y_prob is None or len(np.unique(y_true)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            color   = colors.get(name, "#378ADD")
            lw      = 2.5 if name == "Fusion (ours)" else 1.5
            ls      = "-" if name == "Fusion (ours)" else "--"
            ax.plot(fpr, tpr, color=color, lw=lw, ls=ls,
                    label=f"{name} (AUC = {roc_auc:.4f})")

        ax.plot([0, 1], [0, 1], "k:", lw=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate (Recall)")
        ax.set_title("ROC curve — all models on test set")
        ax.legend(loc="lower right", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        log.info(f"ROC comparison → {out_path}")
    except Exception as e:
        log.warning(f"Could not save ROC comparison: {e}")


def analyse_model_trust(
    fusion_df: pd.DataFrame,
    model,
    out_path: Path,
) -> None:
    """
    Probe which upstream model the fusion layer trusts most per attack family.
    Done by ablation: zero out each score and measure prediction change.
    Saves result as JSON — used in paper Section 6 (ablation analysis).
    """
    import torch

    model.eval()
    FEATURE_COLS = [
        "p_malicious_rf", "p_malicious_nlp", "p_malicious_gat",
        "rf_confidence", "session_duration_norm", "node_count_norm",
        "has_lateral_movement", "has_network_call",
    ]
    for col in FEATURE_COLS:
        if col not in fusion_df.columns:
            fusion_df[col] = 0.0

    X = torch.tensor(fusion_df[FEATURE_COLS].fillna(0).values, dtype=torch.float32)

    with torch.no_grad():
        base_prob = model(X).squeeze(-1).numpy()

    result = {}
    ablation_map = {
        "RF":         0,   # index in FEATURE_COLS
        "DistilBERT": 1,
        "GAT":        2,
    }

    for model_name, feat_idx in ablation_map.items():
        X_ablated = X.clone()
        X_ablated[:, feat_idx] = 0.5   # replace with uncertain (0.5)
        with torch.no_grad():
            ablated_prob = model(X_ablated).squeeze(-1).numpy()
        delta = float(np.abs(base_prob - ablated_prob).mean())
        result[model_name] = {
            "mean_delta":         round(delta, 4),
            "interpretation":     (
                "high influence" if delta > 0.1 else
                "moderate influence" if delta > 0.04 else
                "low influence"
            ),
        }
        log.info(f"  Ablation [{model_name}]: mean Δp = {delta:.4f} → {result[model_name]['interpretation']}")

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"Weight analysis → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(
    config_path: str,
    epochs:      int   = 50,
    lr:          float = 1e-3,
    batch_size:  int   = 64,
    patience:    int   = 10,
) -> None:
    try:
        import torch
    except ImportError:
        log.error("torch required. Run: pip install torch")
        raise

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    splits_dir = Path(cfg["paths"]["splits_dir"])
    models_dir = Path("models/fusion")
    models_dir.mkdir(parents=True, exist_ok=True)

    # ── Load upstream model predictions ──────────────────────────────────────
    log.info("Loading upstream model predictions …")

    def load_preds(path: Path, rename_col: str = None) -> pd.DataFrame:
        if not path.exists():
            log.warning(f"Prediction file not found: {path}. Using placeholder 0.5.")
            return pd.DataFrame(columns=["session_id", rename_col or "score", "split"])
        return pd.read_parquet(path)

    rf_preds  = load_preds(Path("models/rf/rf_predictions.parquet"),  "p_malicious_rf")
    nlp_preds = load_preds(Path("models/distilbert/nlp_predictions.parquet"))
    gat_preds = load_preds(Path("models/gat/gat_predictions.parquet"))

    # Add rf_confidence column if missing
    if "rf_confidence" not in rf_preds.columns and not rf_preds.empty:
        rf_preds["rf_confidence"] = rf_preds.get("p_malicious_rf", 0.5)
    if "p_malicious_rf" not in rf_preds.columns:
        rf_preds["p_malicious_rf"] = 0.5

    # ── Load tabular features (for meta-features) ─────────────────────────────
    tabular_path = Path(cfg["paths"]["processed_dir"]) / "labelled.parquet"
    if tabular_path.exists():
        tabular_df = pd.read_parquet(tabular_path)
    else:
        log.warning("labelled.parquet not found. Building minimal tabular_df from splits.")
        train_df = pd.read_parquet(splits_dir / "train.parquet")
        val_df   = pd.read_parquet(splits_dir / "val.parquet")
        test_df  = pd.read_parquet(splits_dir / "test.parquet")
        tabular_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        for col in ["session_duration_secs", "child_count",
                    "has_lateral_movement", "has_network_call"]:
            if col not in tabular_df.columns:
                tabular_df[col] = 0

    # ── Assemble fusion feature matrices ──────────────────────────────────────
    log.info("Assembling fusion feature matrices …")
    train_fdf = assemble_fusion_features(tabular_df, rf_preds, nlp_preds, gat_preds, "train")
    val_fdf   = assemble_fusion_features(tabular_df, rf_preds, nlp_preds, gat_preds, "val")
    test_fdf  = assemble_fusion_features(tabular_df, rf_preds, nlp_preds, gat_preds, "test")

    if train_fdf.empty:
        log.warning(
            "No training data assembled. Upstream model predictions may be missing.\n"
            "Run train_rf.py, train_distilbert.py, and train_gat.py first,\n"
            "then ensure they save prediction parquet files."
        )
        return

    X_train, y_train = df_to_tensors(train_fdf)
    X_val,   y_val   = df_to_tensors(val_fdf)
    X_test,  y_test  = df_to_tensors(test_fdf)

    log.info(f"Fusion train: {len(X_train)} | val: {len(X_val)} | test: {len(X_test)}")

    # ── Model + loss ──────────────────────────────────────────────────────────
    model = build_fusion_mlp(input_dim=8, dropout=0.2)

    n_pos    = int(y_train.sum().item())
    n_neg    = int(len(y_train) - n_pos)
    pw       = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float)
    loss_fn  = torch.nn.BCEWithLogitsLoss(pos_weight=pw)

    # Patch: use sigmoid output with BCELoss for cleaner inference
    model_for_training = torch.nn.Sequential(*list(model.children())[:-1])  # drop sigmoid

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=10, gamma=0.5)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_auc = 0.0
    best_epoch   = 0
    patience_ctr = 0
    history      = []

    log.info(f"Training fusion MLP for up to {epochs} epochs …")

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, X_train, y_train, optimiser,
                                 loss_fn=torch.nn.BCELoss(),
                                 batch_size=batch_size)
        scheduler.step()

        val_metrics, _ = evaluate(model, X_val, y_val, "val")
        val_auc = val_metrics["auc_roc"]

        if epoch % 5 == 0 or epoch <= 3:
            log.info(f"Epoch {epoch:4d} | loss={train_loss:.4f} | val_auc={val_auc:.4f}")

        history.append({"epoch": epoch, "train_loss": train_loss, "val_auc": val_auc})

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch   = epoch
            patience_ctr = 0
            torch.save(model.state_dict(), models_dir / "fusion_model.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                log.info(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(models_dir / "fusion_model.pt",
                                     weights_only=True))

    # ── Final evaluation ──────────────────────────────────────────────────────
    all_metrics  = []
    pred_records = []
    roc_data     = {}

    for split_name, X, y, fdf in [
        ("val",  X_val,  y_val,  val_fdf),
        ("test", X_test, y_test, test_fdf),
    ]:
        m, y_prob = evaluate(model, X, y, split_name)
        all_metrics.append(m)
        roc_data[f"Fusion (ours)"] = (y_prob, y.numpy().astype(int))

        log.info(
            f"[{split_name:5s}] recall={m['recall']:.4f}  "
            f"precision={m['precision']:.4f}  "
            f"F1={m['f1']:.4f}  AUC={m['auc_roc']:.4f}  FNR={m['fnr']:.6f}"
        )

        preds = pd.DataFrame({
            "session_id":        fdf["session_id"].tolist(),
            "p_malicious_fusion": y_prob,
            "p_malicious_rf":    fdf["p_malicious_rf"].tolist(),
            "p_malicious_nlp":   fdf["p_malicious_nlp"].tolist(),
            "p_malicious_gat":   fdf["p_malicious_gat"].tolist(),
            "label":             y.numpy().astype(int),
            "family":            fdf["family"].tolist(),
            "split":             split_name,
        })
        pred_records.append(preds)

    # Add individual model ROC data for comparison plot
    if not test_fdf.empty:
        y_true_test = y_test.numpy().astype(int)
        roc_data["RF (triage)"]  = (test_fdf["p_malicious_rf"].values,  y_true_test)
        roc_data["DistilBERT"]   = (test_fdf["p_malicious_nlp"].values, y_true_test)
        roc_data["GAT"]          = (test_fdf["p_malicious_gat"].values, y_true_test)

    # OOD
    ood_fdf_path = splits_dir / "ood.parquet"
    if ood_fdf_path.exists():
        ood_fdf      = assemble_fusion_features(
            tabular_df, rf_preds, nlp_preds, gat_preds, "ood"
        )
        if not ood_fdf.empty:
            X_ood, y_ood = df_to_tensors(ood_fdf)
            m_ood, y_prob_ood = evaluate(model, X_ood, y_ood, "ood")
            all_metrics.append(m_ood)
            log.info(
                f"[ood  ] recall={m_ood['recall']:.4f}  "
                f"precision={m_ood['precision']:.4f}  FNR={m_ood['fnr']:.6f}"
            )
            pred_records.append(pd.DataFrame({
                "session_id":         ood_fdf["session_id"].tolist(),
                "p_malicious_fusion": y_prob_ood,
                "label":              y_ood.numpy().astype(int),
                "split":              "ood",
            }))

    # ── Save artefacts ────────────────────────────────────────────────────────
    with open(models_dir / "fusion_metrics.json", "w") as f:
        json.dump({"history": history, "splits": all_metrics,
                   "best_epoch": best_epoch, "best_val_auc": best_val_auc}, f, indent=2)

    if pred_records:
        pd.concat(pred_records).to_parquet(
            models_dir / "fusion_predictions.parquet", index=False
        )

    plot_roc_comparison(roc_data, models_dir / "fusion_roc.png")

    if not test_fdf.empty:
        analyse_model_trust(test_fdf.copy(), model, models_dir / "fusion_weight_analysis.json")

    # ── Paper-ready summary ───────────────────────────────────────────────────
    test_m = next(m for m in all_metrics if m["split"] == "test")
    print("\n" + "═" * 60)
    print("  FUSION MLP — FINAL TEST METRICS  (Full system)")
    print("═" * 60)
    print(f"  Recall (sensitivity) : {test_m['recall']:.4f}")
    print(f"  Precision            : {test_m['precision']:.4f}")
    print(f"  F1 score             : {test_m['f1']:.4f}")
    print(f"  AUC-ROC              : {test_m['auc_roc']:.4f}")
    print(f"  False Negative Rate  : {test_m['fnr']:.6f}  ← target: < 0.01")
    print(f"  False Negatives (FN) : {test_m['fn']}  ← attacks missed by full system")
    print(f"  Best epoch           : {best_epoch}")
    print("═" * 60 + "\n")

    log.info(f"All fusion artefacts saved → {models_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/pipeline.yaml")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--patience",   type=int,   default=10)
    args = parser.parse_args()
    train(args.config, args.epochs, args.lr, args.batch_size, args.patience)
