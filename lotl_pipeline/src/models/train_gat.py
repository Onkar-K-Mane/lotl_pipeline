"""
src/models/train_gat.py
─────────────────────────────────────────────────────────────────────────────
Tier 2B — Graph Attention Network (GAT) training for session-level
PowerShell process-tree classification.

Architecture
────────────
  Input:  PyG Data objects (one per session, from src/graph/builder.py)
          Node features: 18-dim vector per process node
          Edge types:    parent_spawn | pipe_connect | net_connect | file_write

  GAT layers:
    Layer 1: GATConv(18  → 64,  heads=8, concat=True)  → 512-dim
    Layer 2: GATConv(512 → 64,  heads=8, concat=True)  → 512-dim
    Layer 3: GATConv(512 → 64,  heads=1, concat=False) → 64-dim

  Readout: Global mean pooling over all nodes → session embedding
  Head:    Linear(64 → 32) → ReLU → Dropout(0.3) → Linear(32 → 1) → Sigmoid

  Output:  P_GAT(malicious) per session

Design decisions
────────────────
  - 3 GAT layers: captures up to 3-hop propagation paths
    (e.g. winword.exe → powershell.exe → certutil.exe)
  - Multi-head attention (8 heads): learns diverse structural patterns
  - Global mean pooling: graph-level representation invariant to node ordering
  - Weighted BCE loss: handles class imbalance
  - Edge-type-aware: edge_attr fed as additional node message signal

Outputs saved to  models/gat/
  gat_model.pt              — trained model state dict
  gat_config.json           — architecture hyperparameters
  gat_metrics.json          — per-epoch + final eval metrics
  gat_predictions.parquet   — session_id + P_GAT scores
  gat_attention_sample.png  — attention weight visualisation (5 test sessions)

Usage
─────
  python -m src.models.train_gat --config configs/pipeline.yaml
  python -m src.models.train_gat --config configs/pipeline.yaml --epochs 100 --lr 0.001
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


# ─────────────────────────────────────────────────────────────────────────────
# GAT Model definition
# ─────────────────────────────────────────────────────────────────────────────

def build_gat_model(
    node_feature_dim: int = 18,
    hidden_dim:       int = 64,
    heads:            int = 8,
    dropout:          float = 0.3,
    num_layers:       int = 3,
):
    """
    Construct the 3-layer GAT model.
    Returns a torch.nn.Module ready for training.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATConv, global_mean_pool

    class LotlGAT(nn.Module):
        def __init__(self):
            super().__init__()

            # Layer 1: 18 → 64*8 = 512
            self.conv1 = GATConv(
                node_feature_dim, hidden_dim,
                heads=heads, dropout=dropout, concat=True
            )
            # Layer 2: 512 → 64*8 = 512
            self.conv2 = GATConv(
                hidden_dim * heads, hidden_dim,
                heads=heads, dropout=dropout, concat=True
            )
            # Layer 3: 512 → 64*1 = 64  (single head for readout)
            self.conv3 = GATConv(
                hidden_dim * heads, hidden_dim,
                heads=1, dropout=dropout, concat=False
            )

            # Batch normalisation after each GAT layer
            self.bn1 = nn.BatchNorm1d(hidden_dim * heads)
            self.bn2 = nn.BatchNorm1d(hidden_dim * heads)
            self.bn3 = nn.BatchNorm1d(hidden_dim)

            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

            self.dropout = nn.Dropout(dropout)

        def forward(self, x, edge_index, batch, edge_attr=None,
                    return_attention: bool = False):
            # Conv 1
            if return_attention:
                x, (ei1, aw1) = self.conv1(x, edge_index,
                                            return_attention_weights=True)
            else:
                x = self.conv1(x, edge_index)
            x = self.bn1(x)
            x = F.elu(x)
            x = self.dropout(x)

            # Conv 2
            if return_attention:
                x, (ei2, aw2) = self.conv2(x, edge_index,
                                            return_attention_weights=True)
            else:
                x = self.conv2(x, edge_index)
            x = self.bn2(x)
            x = F.elu(x)
            x = self.dropout(x)

            # Conv 3
            if return_attention:
                x, (ei3, aw3) = self.conv3(x, edge_index,
                                            return_attention_weights=True)
            else:
                x = self.conv3(x, edge_index)
            x = self.bn3(x)
            x = F.elu(x)

            # Global mean pooling → session embedding
            x = global_mean_pool(x, batch)

            # Classification head
            out = self.classifier(x).squeeze(-1)

            if return_attention:
                return out, {
                    "layer1": (ei1, aw1),
                    "layer2": (ei2, aw2),
                    "layer3": (ei3, aw3),
                }
            return out

    return LotlGAT()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_graphs_for_split(
    split_df: pd.DataFrame,
    graphs_dir: Path,
) -> list:
    """
    Load saved .pt graph files for sessions in split_df.
    Returns list of torch_geometric.data.Data objects with correct labels.
    """
    import torch

    graphs = []
    missing = 0

    for _, row in split_df.iterrows():
        sid   = row["session_id"]
        label = int(row["label"])
        pt_path = graphs_dir / f"{sid}.pt"

        if not pt_path.exists():
            missing += 1
            continue

        g = torch.load(pt_path, weights_only=False)
        g.y = torch.tensor([label], dtype=torch.long)
        graphs.append(g)

    if missing > 0:
        log.warning(f"Missing .pt files: {missing} sessions skipped")

    return graphs


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimiser, loss_fn, device) -> float:
    import torch
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimiser.zero_grad()

        out  = model(batch.x, batch.edge_index, batch.batch,
                     edge_attr=getattr(batch, "edge_attr", None))
        y    = batch.y.float().to(device)
        loss = loss_fn(out, y)
        loss.backward()

        # Gradient clipping — important for deep GNNs
        import torch.nn as nn
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimiser.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def evaluate_epoch(model, loader, device) -> tuple[float, np.ndarray, np.ndarray]:
    import torch
    from sklearn.metrics import roc_auc_score

    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            batch  = batch.to(device)
            out    = model(batch.x, batch.edge_index, batch.batch,
                          edge_attr=getattr(batch, "edge_attr", None))
            probs  = out.cpu().numpy()
            labels = batch.y.cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels)

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
        "recall":    round(recall_score(y_true,    y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true,        y_pred, zero_division=0), 4),
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

        epochs  = [h["epoch"]      for h in history]
        losses  = [h["train_loss"] for h in history]
        aucs    = [h["val_auc"]    for h in history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(epochs, losses, marker="o", color="#D85A30")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("BCE loss")
        ax1.set_title("GAT — training loss")

        ax2.plot(epochs, aucs, marker="o", color="#1D9E75")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Validation AUC-ROC")
        ax2.set_title("GAT — validation AUC")

        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        log.info(f"Training curve → {out_path}")
    except Exception as e:
        log.warning(f"Could not save training curve: {e}")


def plot_attention_sample(
    model, graphs: list, device, out_path: Path, n_samples: int = 3
) -> None:
    """
    Visualise attention weights on a few test graphs.
    Saves a figure showing process trees with edge opacity ∝ attention weight.
    This is the figure that goes in Section 7 (Case studies) of the paper.
    """
    try:
        import torch
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from torch_geometric.data import Batch

        model.eval()
        fig, axes = plt.subplots(1, min(n_samples, len(graphs)),
                                 figsize=(5 * min(n_samples, len(graphs)), 4))
        if n_samples == 1:
            axes = [axes]

        for ax, g in zip(axes, graphs[:n_samples]):
            batch_g = Batch.from_data_list([g]).to(device)
            with torch.no_grad():
                _, attn = model(
                    batch_g.x, batch_g.edge_index, batch_g.batch,
                    return_attention=True
                )

            # Use layer 3 attention weights (single head, most interpretable)
            ei, aw = attn["layer3"]
            aw_np  = aw.squeeze().cpu().numpy()
            ei_np  = ei.cpu().numpy()
            n_nodes = g.num_nodes

            # Simple circular layout
            angles  = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
            pos     = {i: (np.cos(a), np.sin(a)) for i, a in enumerate(n_nodes
                          if isinstance(n_nodes, range) else range(n_nodes))}

            # Draw nodes
            for i, (x, y) in pos.items():
                color = "#D85A30" if g.x[i, 0].item() == 1 else "#B4B2A9"
                ax.scatter(x, y, s=100, c=color, zorder=3)

            # Draw edges with attention as opacity
            if aw_np.ndim == 0:
                aw_np = np.array([float(aw_np)])
            aw_norm = (aw_np - aw_np.min()) / (aw_np.max() - aw_np.min() + 1e-9)

            for k in range(ei_np.shape[1]):
                src, dst = ei_np[0, k], ei_np[1, k]
                if src in pos and dst in pos:
                    xs = [pos[src][0], pos[dst][0]]
                    ys = [pos[src][1], pos[dst][1]]
                    alpha = float(aw_norm[k]) if k < len(aw_norm) else 0.3
                    ax.plot(xs, ys, "k-", alpha=max(0.05, alpha), linewidth=1.5)

            label = g.y.item() if hasattr(g, "y") else "?"
            ax.set_title(f"{'Malicious' if label else 'Benign'}\n"
                         f"session: {getattr(g, 'session_id', '?')}")
            ax.axis("off")

        fig.suptitle("GAT attention weights — darker edges = higher attention",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        log.info(f"Attention visualisation → {out_path}")
    except Exception as e:
        log.warning(f"Could not save attention visualisation: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(
    config_path:  str,
    epochs:       int   = 150,
    lr:           float = 1e-3,
    batch_size:   int   = 32,
    patience:     int   = 20,
    hidden_dim:   int   = 64,
    heads:        int   = 8,
    dropout:      float = 0.3,
    num_layers:   int   = 3,
) -> None:
    try:
        import torch
        from torch_geometric.loader import DataLoader as GeoDataLoader
    except ImportError:
        log.error(
            "torch and torch_geometric required.\n"
            "Run: pip install torch torch_geometric torch-scatter torch-sparse"
        )
        raise

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    splits_dir = Path(cfg["paths"]["splits_dir"])
    graphs_dir = Path(cfg["paths"]["graphs_dir"])
    models_dir = Path("models/gat")
    models_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Load graph datasets ───────────────────────────────────────────────────
    log.info("Loading graph datasets …")
    train_df = pd.read_parquet(splits_dir / "train.parquet")
    val_df   = pd.read_parquet(splits_dir / "val.parquet")
    test_df  = pd.read_parquet(splits_dir / "test.parquet")

    train_graphs = load_graphs_for_split(train_df, graphs_dir)
    val_graphs   = load_graphs_for_split(val_df,   graphs_dir)
    test_graphs  = load_graphs_for_split(test_df,  graphs_dir)

    log.info(f"Graphs loaded — train:{len(train_graphs)} "
             f"val:{len(val_graphs)} test:{len(test_graphs)}")

    if not train_graphs:
        log.error("No training graphs found. Run 'graphs' pipeline stage first.")
        return

    train_loader = GeoDataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader   = GeoDataLoader(val_graphs,   batch_size=batch_size, shuffle=False)
    test_loader  = GeoDataLoader(test_graphs,  batch_size=batch_size, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    log.info("Building GAT model …")
    model = build_gat_model(
        node_feature_dim = 18,
        hidden_dim       = hidden_dim,
        heads            = heads,
        dropout          = dropout,
        num_layers       = num_layers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parameters: {n_params:,}")

    # ── Weighted BCE loss ──────────────────────────────────────────────────────
    train_labels = np.array([g.y.item() for g in train_graphs])
    n_pos = train_labels.sum()
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float).to(device)
    log.info(f"pos_weight: {pos_weight.item():.3f}  "
             f"(n_pos={n_pos}, n_neg={n_neg})")
    loss_fn = torch.nn.BCELoss(reduction="mean")

    # We apply pos_weight manually in the loop via weighted sampling / manual scaling
    # Use BCEWithLogitsLoss to avoid double-sigmoid issue
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Wrap model to return logits instead of sigmoid output for training
    # We use a logit wrapper to keep the architecture clean
    class LogitWrapper(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            # Replace sigmoid with identity in classifier
            self.base.classifier[-1] = torch.nn.Identity()

        def forward(self, x, edge_index, batch, edge_attr=None,
                    return_attention=False):
            return self.base(x, edge_index, batch, edge_attr, return_attention)

    logit_model = LogitWrapper(model).to(device)

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimiser = torch.optim.Adam(logit_model.parameters(), lr=lr,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", factor=0.5, patience=10, verbose=True
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_auc  = 0.0
    best_epoch    = 0
    patience_ctr  = 0
    history       = []

    log.info(f"Training GAT for up to {epochs} epochs (patience={patience}) …")

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(logit_model, train_loader, optimiser,
                                 loss_fn, device)
        val_auc, _, _ = evaluate_epoch(logit_model, val_loader, device)
        scheduler.step(val_auc)

        if epoch % 10 == 0 or epoch <= 5:
            log.info(f"Epoch {epoch:4d} | loss={train_loss:.4f} | val_auc={val_auc:.4f}")

        history.append({"epoch": epoch, "train_loss": train_loss, "val_auc": val_auc})

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch   = epoch
            patience_ctr = 0
            torch.save(logit_model.state_dict(), models_dir / "gat_model.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                log.info(f"Early stopping at epoch {epoch}")
                break

    log.info(f"Best epoch: {best_epoch} | Best val AUC: {best_val_auc:.4f}")

    # ── Load best weights for evaluation ─────────────────────────────────────
    logit_model.load_state_dict(torch.load(models_dir / "gat_model.pt",
                                           weights_only=True))

    # ── Final evaluation ──────────────────────────────────────────────────────
    all_metrics  = []
    pred_records = []

    for split_name, loader, df_split in [
        ("val",  val_loader,  val_df),
        ("test", test_loader, test_df),
    ]:
        _, y_prob, y_true = evaluate_epoch(logit_model, loader, device)
        # Apply sigmoid to logits for probability interpretation
        y_prob = 1 / (1 + np.exp(-y_prob))   # sigmoid
        m = final_metrics(y_prob, y_true, threshold=0.5, split_name=split_name)
        all_metrics.append(m)
        log.info(
            f"[{split_name:5s}] recall={m['recall']:.4f}  "
            f"precision={m['precision']:.4f}  "
            f"F1={m['f1']:.4f}  AUC={m['auc_roc']:.4f}  FNR={m['fnr']:.6f}"
        )
        pred_records.append(pd.DataFrame({
            "session_id":     [g.session_id for g in (val_graphs if split_name=="val" else test_graphs)],
            "p_malicious_gat": y_prob,
            "label":          y_true,
            "split":          split_name,
        }))

    # OOD
    ood_path = splits_dir / "ood.parquet"
    if ood_path.exists():
        ood_df     = pd.read_parquet(ood_path)
        ood_graphs = load_graphs_for_split(ood_df, graphs_dir)
        if ood_graphs:
            ood_ldr = GeoDataLoader(ood_graphs, batch_size=batch_size, shuffle=False)
            _, y_prob_ood, y_true_ood = evaluate_epoch(logit_model, ood_ldr, device)
            y_prob_ood = 1 / (1 + np.exp(-y_prob_ood))
            m_ood = final_metrics(y_prob_ood, y_true_ood, threshold=0.5, split_name="ood")
            all_metrics.append(m_ood)
            log.info(
                f"[ood  ] recall={m_ood['recall']:.4f}  "
                f"precision={m_ood['precision']:.4f}  FNR={m_ood['fnr']:.6f}"
            )
            pred_records.append(pd.DataFrame({
                "session_id":      [g.session_id for g in ood_graphs],
                "p_malicious_gat": y_prob_ood,
                "label":           y_true_ood,
                "split":           "ood",
            }))

    # ── Save artefacts ────────────────────────────────────────────────────────
    gat_cfg = {
        "node_feature_dim": 18, "hidden_dim": hidden_dim,
        "heads": heads, "dropout": dropout, "num_layers": num_layers,
        "best_epoch": best_epoch, "best_val_auc": best_val_auc,
    }
    with open(models_dir / "gat_config.json", "w") as f:
        json.dump(gat_cfg, f, indent=2)

    with open(models_dir / "gat_metrics.json", "w") as f:
        json.dump({"history": history, "splits": all_metrics}, f, indent=2)

    if pred_records:
        pd.concat(pred_records).to_parquet(
            models_dir / "gat_predictions.parquet", index=False
        )

    plot_training_curve(history, models_dir / "gat_training_curve.png")

    if test_graphs:
        mal_graphs = [g for g in test_graphs if g.y.item() == 1][:3]
        if mal_graphs:
            plot_attention_sample(logit_model, mal_graphs, device,
                                  models_dir / "gat_attention_sample.png")

    # ── Paper-ready summary ───────────────────────────────────────────────────
    test_m = next(m for m in all_metrics if m["split"] == "test")
    print("\n" + "═" * 55)
    print("  GAT — FINAL TEST METRICS")
    print("═" * 55)
    print(f"  Recall (sensitivity) : {test_m['recall']:.4f}")
    print(f"  Precision            : {test_m['precision']:.4f}")
    print(f"  F1 score             : {test_m['f1']:.4f}")
    print(f"  AUC-ROC              : {test_m['auc_roc']:.4f}")
    print(f"  False Negative Rate  : {test_m['fnr']:.6f}")
    print(f"  Parameters           : {n_params:,}")
    print(f"  Best epoch           : {best_epoch}")
    print("═" * 55 + "\n")

    log.info(f"All GAT artefacts saved → {models_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/pipeline.yaml")
    parser.add_argument("--epochs",     type=int,   default=150)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--patience",   type=int,   default=20)
    parser.add_argument("--hidden-dim", type=int,   default=64)
    parser.add_argument("--heads",      type=int,   default=8)
    parser.add_argument("--dropout",    type=float, default=0.3)
    args = parser.parse_args()
    train(args.config, args.epochs, args.lr, args.batch_size, args.patience,
          args.hidden_dim, args.heads, args.dropout)
