"""
src/reconstruction/report.py
─────────────────────────────────────────────────────────────────────────────
Generates three output formats from a KillChain object:

  1. JSON report   — machine-readable, complete, used by downstream tools
  2. Text report   — human-readable analyst briefing
  3. Figure        — SVG/PNG kill chain diagram for the paper (Section 7)

The figure is the visual showpiece of the paper. It renders:
  - A horizontal process chain (root → leaf) with coloured nodes
  - Each node labelled with process image + PID
  - Each step annotated with TTP code + tactic name
  - Colour coding by kill-chain stage (green→amber→red progression)
  - Confidence score badge

Usage
─────
  from src.reconstruction.report import generate_report
  generate_report(kill_chain, output_dir=Path("reports/"))

  # or run all malicious test sessions:
  python -m src.reconstruction.report --config configs/pipeline.yaml
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette for kill chain stages
# ─────────────────────────────────────────────────────────────────────────────

STAGE_COLORS = {
    1:  "#E6F1FB",   # Initial Access      — blue-50
    2:  "#EAF3DE",   # Execution           — green-50
    3:  "#EEEDFE",   # Persistence         — purple-50
    4:  "#FAEEDA",   # Privilege Escalation— amber-50
    5:  "#FAECE7",   # Defense Evasion     — coral-50
    6:  "#FCEBEB",   # Credential Access   — red-50
    7:  "#E1F5EE",   # Discovery           — teal-50
    8:  "#FAEEDA",   # Lateral Movement    — amber-50
    9:  "#FBEAF0",   # Collection          — pink-50
    10: "#FAECE7",   # C2                  — coral-50
    11: "#FCEBEB",   # Exfiltration        — red-50
}

STAGE_STROKE = {
    1:  "#185FA5", 2:  "#3B6D11", 3:  "#534AB7",
    4:  "#BA7517", 5:  "#993C1D", 6:  "#A32D2D",
    7:  "#0F6E56", 8:  "#BA7517", 9:  "#993556",
    10: "#993C1D", 11: "#A32D2D",
}

STAGE_TEXT = {
    1:  "#0C447C", 2:  "#27500A", 3:  "#3C3489",
    4:  "#633806", 5:  "#4A1B0C", 6:  "#501313",
    7:  "#085041", 8:  "#633806", 9:  "#4B1528",
    10: "#4A1B0C", 11: "#501313",
}


# ─────────────────────────────────────────────────────────────────────────────
# JSON report
# ─────────────────────────────────────────────────────────────────────────────

def generate_json_report(kill_chain, out_path: Path) -> dict:
    """Save complete kill chain as JSON. Returns the dict."""
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "system":       "LOLBin Hierarchical Detection v1.0",
        **kill_chain.to_dict(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"JSON report → {out_path}")
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Text report (analyst briefing)
# ─────────────────────────────────────────────────────────────────────────────

def generate_text_report(kill_chain, out_path: Optional[Path] = None) -> str:
    """
    Generate a human-readable analyst briefing.
    Suitable for inclusion in a SOC ticket or paper appendix.
    """
    kc    = kill_chain
    lines = []

    lines.append("=" * 70)
    lines.append("  POWERSHELL LOTL ATTACK — ANALYST BRIEFING")
    lines.append("=" * 70)
    lines.append(f"  Session ID    : {kc.session_id}")
    lines.append(f"  Confidence    : {kc.p_malicious:.1%}")
    lines.append(f"  Attack hops   : {kc.n_hops}")
    lines.append(f"  Entry point   : {kc.root_image}")
    lines.append(f"  Terminal proc : {kc.leaf_image}")
    lines.append(f"  TTPs detected : {', '.join(kc.raw_ttps)}")
    lines.append("")
    lines.append("  SUMMARY")
    lines.append(f"  {kc.summary}")
    lines.append("")
    lines.append("  KILL CHAIN (ordered by ATT&CK stage)")
    lines.append("  " + "─" * 66)

    for step in kc.chain:
        lines.append(f"  [{step.stage_num:2d}] {step.tactic:<22s}  "
                     f"{step.technique_id:<12s}  {step.technique_name}")
        lines.append(f"       Process: {step.process_image}")
        for ev in step.evidence:
            lines.append(f"       Evidence: {ev}")
        lines.append("")

    lines.append("  ATT&CK Navigator")
    lines.append(f"  {kc.mitre_url[:80]}{'...' if len(kc.mitre_url)>80 else ''}")
    lines.append("=" * 70)

    text = "\n".join(lines)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(text)
        log.info(f"Text report  → {out_path}")

    return text


# ─────────────────────────────────────────────────────────────────────────────
# Kill chain figure (SVG — paper quality)
# ─────────────────────────────────────────────────────────────────────────────

def _wrap(text: str, max_chars: int = 18) -> str:
    """Truncate a label at max_chars."""
    return text if len(text) <= max_chars else text[:max_chars - 1] + "…"


def generate_figure(kill_chain, out_path: Path) -> None:
    """
    Render the kill chain as a publication-quality SVG figure.

    Layout
    ──────
    Horizontal chain of process boxes connected by arrows.
    Below each box: TTP badge (technique_id + tactic name).
    Colour coded by kill-chain stage.
    Header bar shows session_id and confidence score.
    """
    kc     = kill_chain
    steps  = kc.chain
    n      = max(len(steps), 1)

    # Layout constants
    BOX_W      = 140
    BOX_H      = 60
    BOX_GAP    = 50
    TTP_H      = 44
    MARGIN_X   = 40
    MARGIN_TOP = 80
    ARROW_LEN  = BOX_GAP
    SVG_W      = MARGIN_X * 2 + n * BOX_W + (n - 1) * BOX_GAP
    SVG_H      = MARGIN_TOP + BOX_H + TTP_H + 60

    # Ensure minimum width
    SVG_W = max(SVG_W, 500)

    def x_center(i):
        return MARGIN_X + i * (BOX_W + BOX_GAP) + BOX_W // 2

    def box_x(i):
        return MARGIN_X + i * (BOX_W + BOX_GAP)

    lines = []
    lines.append(f'<svg width="100%" viewBox="0 0 {SVG_W} {SVG_H}" '
                 f'xmlns="http://www.w3.org/2000/svg" '
                 f'font-family="system-ui, sans-serif">')

    # ── Arrow marker ──────────────────────────────────────────────────────────
    lines.append("""<defs>
  <marker id="arr" viewBox="0 0 10 10" refX="8" refY="5"
          markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M2 1L8 5L2 9" fill="none" stroke="#888780"
          stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
  </marker>
</defs>""")

    # ── Header ────────────────────────────────────────────────────────────────
    conf_color = "#1D9E75" if kc.p_malicious >= 0.8 else "#BA7517"
    lines.append(f'<text x="{SVG_W//2}" y="28" text-anchor="middle" '
                 f'font-size="15" font-weight="500" fill="#2C2C2A">'
                 f'PowerShell LOLBin — Attack Chain Reconstruction</text>')
    lines.append(f'<text x="{SVG_W//2}" y="48" text-anchor="middle" '
                 f'font-size="12" fill="#5F5E5A">'
                 f'Session: {kc.session_id}  |  '
                 f'Confidence: {kc.p_malicious:.0%}  |  '
                 f'{kc.n_hops} hops  |  {len(steps)} TTPs detected</text>')

    # ── Process chain nodes ───────────────────────────────────────────────────
    y_box = MARGIN_TOP
    y_ttp = MARGIN_TOP + BOX_H + 8

    for i, step in enumerate(steps):
        bx    = box_x(i)
        cx    = x_center(i)
        stage = step.stage_num
        fill  = STAGE_COLORS.get(stage, "#F1EFE8")
        strk  = STAGE_STROKE.get(stage, "#5F5E5A")
        txt   = STAGE_TEXT.get(stage, "#2C2C2A")

        # Process box
        lines.append(
            f'<rect x="{bx}" y="{y_box}" width="{BOX_W}" height="{BOX_H}" '
            f'rx="8" fill="{fill}" stroke="{strk}" stroke-width="1"/>'
        )
        # Process name (line 1)
        proc = _wrap(step.process_image, 18)
        lines.append(
            f'<text x="{cx}" y="{y_box + 22}" text-anchor="middle" '
            f'font-size="12" font-weight="500" fill="{txt}">{proc}</text>'
        )
        # Technique ID (line 2)
        lines.append(
            f'<text x="{cx}" y="{y_box + 40}" text-anchor="middle" '
            f'font-size="11" fill="{strk}">{step.technique_id}</text>'
        )

        # TTP badge below
        tactic_short = step.tactic.replace("Privilege ", "Priv. ")
        tactic_label = _wrap(tactic_short, 20)
        lines.append(
            f'<text x="{cx}" y="{y_ttp + 14}" text-anchor="middle" '
            f'font-size="10" fill="{strk}" font-weight="500">'
            f'{step.technique_name[:18]}</text>'
        )
        lines.append(
            f'<text x="{cx}" y="{y_ttp + 28}" text-anchor="middle" '
            f'font-size="9" fill="#888780">{tactic_label}</text>'
        )

        # Arrow to next box
        if i < len(steps) - 1:
            ax1 = bx + BOX_W + 4
            ax2 = bx + BOX_W + BOX_GAP - 4
            ay  = y_box + BOX_H // 2
            lines.append(
                f'<line x1="{ax1}" y1="{ay}" x2="{ax2}" y2="{ay}" '
                f'stroke="#888780" stroke-width="1.5" '
                f'marker-end="url(#arr)"/>'
            )

    # ── Confidence badge (bottom right) ───────────────────────────────────────
    badge_x = SVG_W - 10
    badge_y = SVG_H - 16
    lines.append(
        f'<text x="{badge_x}" y="{badge_y}" text-anchor="end" '
        f'font-size="10" fill="{conf_color}" font-weight="500">'
        f'p(malicious) = {kc.p_malicious:.4f}</text>'
    )

    lines.append("</svg>")

    svg_content = "\n".join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(svg_content)
    log.info(f"Figure (SVG) → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Master report generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    kill_chain,
    output_dir: Path,
    formats:    list[str] = ("json", "text", "figure"),
) -> dict[str, Path]:
    """
    Generate all report formats for one kill chain.

    Returns dict mapping format name → output file path.
    """
    output_dir = Path(output_dir)
    sid        = kill_chain.session_id
    outputs    = {}

    if "json" in formats:
        p = output_dir / f"{sid}_kill_chain.json"
        generate_json_report(kill_chain, p)
        outputs["json"] = p

    if "text" in formats:
        p = output_dir / f"{sid}_briefing.txt"
        generate_text_report(kill_chain, p)
        outputs["text"] = p

    if "figure" in formats:
        p = output_dir / f"{sid}_chain.svg"
        generate_figure(kill_chain, p)
        outputs["figure"] = p

    return outputs
