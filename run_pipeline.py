"""
run_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Single entry point for the full LOLBin detection dataset pipeline.

Usage
─────
  # Run all stages end-to-end
  python run_pipeline.py --config configs/pipeline.yaml --stage all

  # Run individual stages
  python run_pipeline.py --config configs/pipeline.yaml --stage download
  python run_pipeline.py --config configs/pipeline.yaml --stage parse
  python run_pipeline.py --config configs/pipeline.yaml --stage features
  python run_pipeline.py --config configs/pipeline.yaml --stage graphs
  python run_pipeline.py --config configs/pipeline.yaml --stage label
  python run_pipeline.py --config configs/pipeline.yaml --stage split

  # Print dataset statistics after pipeline runs
  python run_pipeline.py --config configs/pipeline.yaml --stage stats
"""

import argparse
import logging
import sys
import json
from pathlib import Path

import yaml
import pandas as pd

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger(__name__)

STAGES = ["download", "parse", "features", "graphs", "label", "split", "stats", "all"]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Stage runners
# ─────────────────────────────────────────────────────────────────────────────

def stage_download(cfg: dict) -> None:
    log.info("━━━ STAGE: download ━━━")
    from src.ingestion.download import run
    run(args.config)


def stage_parse(cfg: dict) -> None:
    log.info("━━━ STAGE: parse ━━━")
    from src.ingestion.parse_sysmon import parse_manifest

    raw_dir       = Path(cfg["paths"]["raw_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    manifest_path = raw_dir / "manifest.json"

    if not manifest_path.exists():
        log.error("manifest.json not found — run 'download' stage first.")
        sys.exit(1)

    gap_secs = cfg["labelling"]["session_gap_secs"]
    df = parse_manifest(manifest_path, gap_secs=gap_secs)

    if df.empty:
        log.error("No events parsed. Check data sources.")
        sys.exit(1)

    processed_dir.mkdir(parents=True, exist_ok=True)
    out = processed_dir / "events.parquet"
    # ttps (list) → json string for parquet
    df["ttps"] = df["ttps"].apply(
        lambda x: json.dumps(x) if isinstance(x, list) else str(x)
    )
    df.to_parquet(out, index=False)
    log.info(f"Events saved → {out}  ({len(df)} rows)")


def stage_features(cfg: dict) -> None:
    log.info("━━━ STAGE: features ━━━")
    from src.features.tabular import build_tabular_features

    processed_dir = Path(cfg["paths"]["processed_dir"])
    events_path   = processed_dir / "events.parquet"

    if not events_path.exists():
        log.error("events.parquet not found — run 'parse' stage first.")
        sys.exit(1)

    df = pd.read_parquet(events_path)
    df["ttps"] = df["ttps"].apply(
        lambda x: json.loads(x) if isinstance(x, str) and x.startswith("[") else x
    )

    tabular = build_tabular_features(df)
    out = processed_dir / "tabular.parquet"
    tabular["ttps"] = tabular["ttps"].apply(
        lambda x: json.dumps(x) if isinstance(x, list) else str(x)
    )
    tabular.to_parquet(out, index=False)
    log.info(f"Tabular features saved → {out}  ({len(tabular)} sessions)")


def stage_graphs(cfg: dict) -> None:
    log.info("━━━ STAGE: graphs ━━━")
    from src.graph.builder import build_all_graphs

    processed_dir = Path(cfg["paths"]["processed_dir"])
    graphs_dir    = Path(cfg["paths"]["graphs_dir"])
    events_path   = processed_dir / "events.parquet"

    if not events_path.exists():
        log.error("events.parquet not found — run 'parse' stage first.")
        sys.exit(1)

    df = pd.read_parquet(events_path)
    df["ttps"] = df["ttps"].apply(
        lambda x: json.loads(x) if isinstance(x, str) and x.startswith("[") else x
    )

    min_ev = cfg["labelling"]["min_session_events"]
    _, n_saved, n_skipped = build_all_graphs(df, graphs_dir, min_events=min_ev)
    log.info(f"Graph build complete: {n_saved} graphs saved, {n_skipped} skipped")


def stage_label(cfg: dict) -> None:
    log.info("━━━ STAGE: label ━━━")
    from src.labels.labeller import label_sessions

    processed_dir = Path(cfg["paths"]["processed_dir"])
    tabular_path  = processed_dir / "tabular.parquet"

    if not tabular_path.exists():
        log.error("tabular.parquet not found — run 'features' stage first.")
        sys.exit(1)

    df = pd.read_parquet(tabular_path)
    df["ttps"] = df["ttps"].apply(
        lambda x: json.loads(x) if isinstance(x, str) and x.startswith("[") else x
    )

    min_ev = cfg["labelling"]["min_session_events"]
    labelled = label_sessions(df, min_events=min_ev)

    out = processed_dir / "labelled.parquet"
    labelled["ttps"]      = labelled["ttps"].apply(
        lambda x: json.dumps(x) if isinstance(x, list) else str(x)
    )
    labelled["ttp_chain"] = labelled["ttp_chain"].apply(json.dumps)
    labelled.to_parquet(out, index=False)
    log.info(f"Labelled dataset saved → {out}")


def stage_split(cfg: dict) -> None:
    log.info("━━━ STAGE: split ━━━")
    from src.labels.labeller import split_dataset, save_splits

    processed_dir = Path(cfg["paths"]["processed_dir"])
    splits_dir    = Path(cfg["paths"]["splits_dir"])
    labelled_path = processed_dir / "labelled.parquet"

    if not labelled_path.exists():
        log.error("labelled.parquet not found — run 'label' stage first.")
        sys.exit(1)

    df = pd.read_parquet(labelled_path)

    split_cfg = cfg["split"]
    splits = split_dataset(
        df,
        train_frac   = split_cfg["train"],
        val_frac     = split_cfg["val"],
        test_frac    = split_cfg["test"],
        ood_families = split_cfg.get("ood_families", []),
        random_seed  = split_cfg["random_seed"],
    )
    save_splits(splits, splits_dir)
    log.info("Split complete.")


def stage_stats(cfg: dict) -> None:
    log.info("━━━ STAGE: stats ━━━")
    splits_dir    = Path(cfg["paths"]["splits_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])

    labelled_path = processed_dir / "labelled.parquet"
    if not labelled_path.exists():
        log.warning("labelled.parquet not found. Run pipeline first.")
        return

    df = pd.read_parquet(labelled_path)

    print("\n" + "═" * 60)
    print("  DATASET STATISTICS")
    print("═" * 60)
    print(f"  Total sessions:    {len(df):,}")
    print(f"  Malicious:         {df['label'].sum():,}  ({df['label'].mean()*100:.1f}%)")
    print(f"  Benign:            {(df['label']==0).sum():,}  ({(1-df['label'].mean())*100:.1f}%)")
    print(f"  Unique families:   {df['family'].nunique()}")
    print()

    if "family" in df.columns:
        print("  Attack families:")
        for fam, count in df[df["label"]==1]["family"].value_counts().items():
            print(f"    {fam:<50s} {count:>5,}")

    print()

    for split_name in ["train", "val", "test", "ood"]:
        p = splits_dir / f"{split_name}.parquet"
        if p.exists():
            sdf = pd.read_parquet(p)
            pos = sdf["label"].sum()
            print(f"  {split_name:6s}: {len(sdf):5,} sessions  "
                  f"({pos} malicious / {len(sdf)-pos} benign)")

    print("═" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LOLBin Detection Dataset Pipeline")
    parser.add_argument("--config", default="configs/pipeline.yaml",
                        help="Path to pipeline configuration YAML")
    parser.add_argument("--stage", default="all", choices=STAGES,
                        help="Pipeline stage to run")
    args = parser.parse_args()

    cfg = load_config(args.config)

    stage_map = {
        "download": stage_download,
        "parse":    stage_parse,
        "features": stage_features,
        "graphs":   stage_graphs,
        "label":    stage_label,
        "split":    stage_split,
        "stats":    stage_stats,
    }

    if args.stage == "all":
        for name, fn in stage_map.items():
            if name == "stats":
                continue
            fn(cfg)
        stage_stats(cfg)
    else:
        stage_map[args.stage](cfg)

    log.info("Pipeline complete.")
