"""
src/models/train_distilbert.py
─────────────────────────────────────────────────────────────────────────────
Tier 2A — DistilBERT fine-tuning for PowerShell command-line classification.

Model: distilbert-base-uncased (66M params, 2× faster than BERT-base)
Task:  Binary classification — malicious (1) vs benign (0) command string
Input: Raw CommandLine string of the HIGHEST-ENTROPY event in the session
       (we take the most suspicious single command per session, not the whole session)

Why DistilBERT over BERT-base?
  - 40% fewer parameters, 60% faster at inference — critical for real-time detection
  - Retains 97% of BERT's language understanding on downstream tasks
  - Knowledge-distilled: captures obfuscation patterns well without overfitting

Fine-tuning strategy
────────────────────
  - 3 epochs (sufficient for domain adaptation; more risks overfitting)
  - lr = 2e-5 with linear warmup + cosine decay
  - Weighted Binary Cross-Entropy — weight malicious class by imbalance ratio
  - Gradient clipping at 1.0 to stabilise training on adversarial text
  - Early stopping on validation AUC (patience = 2)

Outputs saved to  models/distilbert/
  best_model/           — HuggingFace model directory (config + weights)
  tokenizer/            — saved tokenizer
  nlp_metrics.json      — per-epoch + final eval metrics
  nlp_predictions.parquet — val/test session_id + P(malicious) scores

Usage
─────
  python -m src.models.train_distilbert --config configs/pipeline.yaml
  python -m src.models.train_distilbert --config configs/pipeline.yaml --epochs 5 --lr 1e-5
"""

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

MAX_LENGTH = 512     # DistilBERT context window
MODEL_NAME = "distilbert-base-uncased"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PowerShellDataset:
    """
    PyTorch Dataset wrapping PowerShell session records.

    For each session we take the single command with the highest Shannon
    entropy — this is typically the most obfuscated / suspicious command,
    and gives the NLP model the hardest signal to classify. Feeding the
    entire session as one concatenated string would exceed 512 tokens and
    dilute the malicious signal.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        events_df: Optional[pd.DataFrame] = None,
        max_length: int = MAX_LENGTH,
    ):
        import torch
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.labels     = df["label"].values.astype(np.int64)
        self.session_ids = df["session_id"].tolist()

        # Extract the most-entropic command per session
        self.texts = self._extract_commands(df, events_df)

    def _shannon_entropy(self, s: str) -> float:
        import math
        if not s:
            return 0.0
        counts = {}
        for c in s:
            counts[c] = counts.get(c, 0) + 1
        n = len(s)
        return -sum((v / n) * math.log2(v / n) for v in counts.values())

    def _extract_commands(
        self,
        df: pd.DataFrame,
        events_df: Optional[pd.DataFrame],
    ) -> list[str]:
        """
        For each session, find the command string with the highest entropy.
        Falls back to the 'command_line' column in the tabular df if events
        are not provided.
        """
        texts = []
        for sid in self.session_ids:
            if events_df is not None and "session_id" in events_df.columns:
                sess_events = events_df[
                    (events_df["session_id"] == sid) &
                    (events_df["event_id"] == 1)
                ]
                if not sess_events.empty:
                    cmds = sess_events["command_line"].fillna("").tolist()
                    best = max(cmds, key=self._shannon_entropy)
                    texts.append(best[:2000])   # truncate before tokenising
                    continue
            # Fallback: use command_line from tabular df if available
            row = df[df["session_id"] == sid]
            if not row.empty and "command_line" in row.columns:
                texts.append(str(row.iloc[0]["command_line"])[:2000])
            else:
                texts.append("")
        return texts

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        import torch
        enc = self.tokenizer(
            self.texts[idx],
            max_length      = self.max_length,
            padding         = "max_length",
            truncation      = True,
            return_tensors  = "pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(y: np.ndarray) -> "torch.Tensor":
    """Compute inverse-frequency class weights for imbalanced binary data."""
    import torch
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    total = len(y)
    w_neg = total / (2 * n_neg) if n_neg > 0 else 1.0
    w_pos = total / (2 * n_pos) if n_pos > 0 else 1.0
    log.info(f"Class weights: benign={w_neg:.3f}, malicious={w_pos:.3f}")
    return torch.tensor([w_neg, w_pos], dtype=torch.float)


def train_epoch(model, loader, optimiser, scheduler, loss_fn, device) -> float:
    """One training epoch. Returns mean loss."""
    import torch
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimiser.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits  = outputs.logits

        loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_epoch(model, loader, device) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate on a dataloader. Returns (AUC, loss, y_prob, y_true)."""
    import torch
    from sklearn.metrics import roc_auc_score
    import torch.nn.functional as F

    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
            probs          = F.softmax(outputs.logits, dim=-1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_prob = np.array(all_probs)
    y_true = np.array(all_labels)
    auc    = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    return float(auc), y_prob, y_true


def final_metrics(y_prob: np.ndarray, y_true: np.ndarray,
                  threshold: float = 0.5, split_name: str = "test") -> dict:
    from sklearn.metrics import (
        recall_score, precision_score, f1_score,
        roc_auc_score, confusion_matrix,
    )
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "split":     split_name,
        "threshold": threshold,
        "recall":    round(recall_score(y_true, y_pred,    zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred,        zero_division=0), 4),
        "auc_roc":   round(roc_auc_score(y_true, y_prob)
                           if len(np.unique(y_true)) > 1 else 0.0, 4),
        "fnr":       round(fn / (fn + tp) if (fn + tp) > 0 else 0.0, 6),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def plot_training_curve(history: list[dict], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs  = [h["epoch"] for h in history]
        tr_loss = [h["train_loss"] for h in history]
        val_auc = [h["val_auc"]   for h in history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(epochs, tr_loss, marker="o", color="#534AB7")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Training loss")
        ax1.set_title("DistilBERT — training loss")

        ax2.plot(epochs, val_auc, marker="o", color="#1D9E75")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Validation AUC-ROC")
        ax2.set_title("DistilBERT — validation AUC")

        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        log.info(f"Training curve → {out_path}")
    except Exception as e:
        log.warning(f"Could not save training curve: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(
    config_path: str,
    epochs:      int   = 3,
    lr:          float = 2e-5,
    batch_size:  int   = 16,
    patience:    int   = 2,
) -> None:
    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import (
            DistilBertTokenizerFast,
            DistilBertForSequenceClassification,
            get_cosine_schedule_with_warmup,
        )
    except ImportError:
        log.error("torch and transformers are required. Run: pip install torch transformers")
        raise

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    splits_dir = Path(cfg["paths"]["splits_dir"])
    models_dir = Path("models/distilbert")
    models_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Tokeniser ─────────────────────────────────────────────────────────────
    log.info(f"Loading tokeniser: {MODEL_NAME}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    # ── Load splits ───────────────────────────────────────────────────────────
    log.info("Loading splits …")
    train_df = pd.read_parquet(splits_dir / "train.parquet")
    val_df   = pd.read_parquet(splits_dir / "val.parquet")
    test_df  = pd.read_parquet(splits_dir / "test.parquet")

    log.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Load raw events for command extraction (optional — gracefully absent)
    events_path = Path(cfg["paths"]["processed_dir"]) / "events.parquet"
    events_df   = pd.read_parquet(events_path) if events_path.exists() else None

    # ── Datasets & loaders ────────────────────────────────────────────────────
    train_ds = PowerShellDataset(train_df, tokenizer, events_df)
    val_ds   = PowerShellDataset(val_df,   tokenizer, events_df)
    test_ds  = PowerShellDataset(test_df,  tokenizer, events_df)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)

    # ── Model ─────────────────────────────────────────────────────────────────
    log.info(f"Loading model: {MODEL_NAME}")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(device)

    # ── Class-weighted loss ───────────────────────────────────────────────────
    class_weights = compute_class_weights(train_df["label"].values).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = max(1, total_steps // 10)

    scheduler = get_cosine_schedule_with_warmup(
        optimiser,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps,
    )

    # ── Training loop with early stopping ────────────────────────────────────
    best_val_auc  = 0.0
    best_epoch    = 0
    patience_ctr  = 0
    history       = []

    for epoch in range(1, epochs + 1):
        log.info(f"── Epoch {epoch}/{epochs} ──")
        train_loss = train_epoch(model, train_loader, optimiser, scheduler,
                                 loss_fn, device)
        val_auc, val_prob, val_true = evaluate_epoch(model, val_loader, device)

        log.info(f"  train_loss={train_loss:.4f}  val_auc={val_auc:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_auc": val_auc})

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch   = epoch
            patience_ctr = 0
            # Save best model
            model.save_pretrained(models_dir / "best_model")
            tokenizer.save_pretrained(models_dir / "tokenizer")
            log.info(f"  ✓ New best model saved (val_auc={val_auc:.4f})")
        else:
            patience_ctr += 1
            log.info(f"  No improvement. Patience: {patience_ctr}/{patience}")
            if patience_ctr >= patience:
                log.info(f"  Early stopping at epoch {epoch}")
                break

    log.info(f"Best epoch: {best_epoch} | Best val AUC: {best_val_auc:.4f}")

    # ── Load best model for final evaluation ─────────────────────────────────
    model = DistilBertForSequenceClassification.from_pretrained(
        models_dir / "best_model"
    ).to(device)

    # ── Final metrics ─────────────────────────────────────────────────────────
    all_metrics = []
    pred_records = []

    for split_name, loader, df_split in [
        ("val",  val_loader,  val_df),
        ("test", test_loader, test_df),
    ]:
        _, y_prob, y_true = evaluate_epoch(model, loader, device)
        m = final_metrics(y_prob, y_true, threshold=0.5, split_name=split_name)
        all_metrics.append(m)
        log.info(
            f"[{split_name:5s}] recall={m['recall']:.4f}  "
            f"precision={m['precision']:.4f}  "
            f"F1={m['f1']:.4f}  AUC={m['auc_roc']:.4f}  FNR={m['fnr']:.6f}"
        )

        # Save probability scores for fusion layer
        preds = pd.DataFrame({
            "session_id": df_split["session_id"].tolist(),
            "p_malicious_nlp": y_prob,
            "label": y_true,
            "split": split_name,
        })
        pred_records.append(preds)

    # OOD evaluation
    ood_path = splits_dir / "ood.parquet"
    if ood_path.exists():
        ood_df  = pd.read_parquet(ood_path)
        ood_ds  = PowerShellDataset(ood_df, tokenizer, events_df)
        ood_ldr = DataLoader(ood_ds, batch_size=batch_size, shuffle=False)
        _, y_prob_ood, y_true_ood = evaluate_epoch(model, ood_ldr, device)
        m_ood = final_metrics(y_prob_ood, y_true_ood, threshold=0.5, split_name="ood")
        all_metrics.append(m_ood)
        log.info(
            f"[ood  ] recall={m_ood['recall']:.4f}  "
            f"precision={m_ood['precision']:.4f}  FNR={m_ood['fnr']:.6f}"
        )
        pred_records.append(pd.DataFrame({
            "session_id": ood_df["session_id"].tolist(),
            "p_malicious_nlp": y_prob_ood,
            "label": y_true_ood,
            "split": "ood",
        }))

    # ── Save artefacts ────────────────────────────────────────────────────────
    with open(models_dir / "nlp_metrics.json", "w") as f:
        json.dump({"history": history, "splits": all_metrics,
                   "best_epoch": best_epoch, "best_val_auc": best_val_auc}, f, indent=2)

    pd.concat(pred_records).to_parquet(
        models_dir / "nlp_predictions.parquet", index=False
    )

    plot_training_curve(history, models_dir / "nlp_training_curve.png")

    # ── Paper-ready summary ───────────────────────────────────────────────────
    test_m = next(m for m in all_metrics if m["split"] == "test")
    print("\n" + "═" * 55)
    print("  DISTILBERT NLP — FINAL TEST METRICS")
    print("═" * 55)
    print(f"  Recall (sensitivity) : {test_m['recall']:.4f}")
    print(f"  Precision            : {test_m['precision']:.4f}")
    print(f"  F1 score             : {test_m['f1']:.4f}")
    print(f"  AUC-ROC              : {test_m['auc_roc']:.4f}")
    print(f"  False Negative Rate  : {test_m['fnr']:.6f}")
    print(f"  False Negatives (FN) : {test_m['fn']}")
    print(f"  Best epoch           : {best_epoch}")
    print("═" * 55 + "\n")

    log.info(f"All NLP artefacts saved → {models_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/pipeline.yaml")
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--patience",   type=int,   default=2)
    args = parser.parse_args()
    train(args.config, args.epochs, args.lr, args.batch_size, args.patience)
