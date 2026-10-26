"""Render comparison figures from the M2 reproduction results.

Reads ``results/results.json`` and writes three PNGs into ``results/figures/``:

  1. asr_by_defense.png   - attack success rate per defense config,
                            reproduced vs. original (the defense ladder).
  2. per_technique.png    - naive vs. full-defense ASR for each of the nine
                            concealment techniques, with detector misses flagged.
  3. detection.png        - the hidden-prompt detector's confusion matrix,
                            annotated with precision / recall / F1.

Colours use the colourblind-safe categorical pair blue (#2a78d6) and
orange (#eb6834). Matplotlib is the only non-stdlib dependency and is needed
for plotting alone; the M1/M2 pipeline itself stays dependency-free.

Usage:  python scripts/make_plots.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt

# --- paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "results.json"
FIGDIR = ROOT / "results" / "figures"

# --- palette (validated colourblind-safe slots) ------------------------------
BLUE = "#2a78d6"    # series 1 -> reproduced / naive
ORANGE = "#eb6834"  # series 2 -> original / full defense
RED = "#e34948"     # status: detector miss
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d9d8d4"
SURFACE = "#fcfcfb"


def _style() -> None:
    """Apply a clean, recessive house style to every figure."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _bar_labels(ax, bars, fmt="{:.0%}", pad=0.012, horizontal=True) -> None:
    """Write a value label at the end of each bar."""
    for b in bars:
        if horizontal:
            w = b.get_width()
            ax.text(w + pad, b.get_y() + b.get_height() / 2, fmt.format(w),
                    va="center", ha="left", fontsize=9, color=INK)
        else:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + pad, fmt.format(h),
                    va="bottom", ha="center", fontsize=9, color=INK)


def plot_asr_by_defense(data: dict) -> Path:
    """Grouped horizontal bars: reproduced vs. original ASR per config."""
    asr = data["asr"]
    configs = list(asr.keys())  # already ordered strongest-attack -> full-defense
    repro = [asr[c]["reproduced"] for c in configs]
    orig = [asr[c]["original"] for c in configs]

    y = range(len(configs))
    h = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.2))
    b1 = ax.barh([i + h / 2 for i in y], repro, height=h, color=BLUE,
                 label="Reproduced", zorder=3)
    b2 = ax.barh([i - h / 2 for i in y], orig, height=h, color=ORANGE,
                 label="Original (anchor paper)", zorder=3)
    _bar_labels(ax, b1)
    _bar_labels(ax, b2)

    ax.set_yticks(list(y))
    ax.set_yticklabels(configs)
    ax.invert_yaxis()  # naive at top, full defense at bottom
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Attack success rate  (REJECT → SELECT, n=72 injected)")
    ax.set_title("Attack success falls as defenses are layered on",
                 fontsize=13, fontweight="bold", pad=10)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False)

    fig.tight_layout()
    out = FIGDIR / "asr_by_defense.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_per_technique(data: dict) -> Path:
    """Stacked bars: detector catches vs. misses per technique (out of 8).

    Naive ASR is a flat 1.0 and full-defense ASR a flat 0.0 across all nine
    techniques (see asr_by_defense.png), so the only quantity that varies by
    technique is the detector's blind spot -- that is what this figure shows.
    """
    pt = data["per_technique"]
    per_tech = 8  # 4 resumes x 2 payload families
    # Sort so the technique with misses (the concealment dilemma) sits on top.
    techs = sorted(pt.keys(), key=lambda t: pt[t]["detector_missed"])
    missed = [pt[t]["detector_missed"] for t in techs]
    caught = [per_tech - m for m in missed]

    y = range(len(techs))
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.barh(list(y), caught, color=BLUE, label="Caught by detector", zorder=3)
    ax.barh(list(y), missed, left=caught, color=RED,
            label="Missed by detector", zorder=3)

    for i, m in enumerate(missed):
        if m:
            ax.text(per_tech + 0.12, i, f"{m} missed", va="center", ha="left",
                    fontsize=9.5, color=RED, fontweight="bold")

    ax.set_yticks(list(y))
    ax.set_yticklabels([t.replace("_", " ") for t in techs])
    ax.set_xlim(0, per_tech + 1.4)
    ax.set_xlabel("Injected documents (n=8 per technique)")
    ax.set_title("Detection catches every hidden payload — except the plain-visible one",
                 fontsize=13, fontweight="bold", pad=10)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    # Bars span the full width, so give the legend a surface box to stay legible.
    leg = ax.legend(loc="lower right", frameon=True, framealpha=1.0)
    leg.get_frame().set_facecolor(SURFACE)
    leg.get_frame().set_edgecolor("none")

    fig.tight_layout()
    out = FIGDIR / "per_technique.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_detection(data: dict) -> Path:
    """Confusion-matrix heatmap annotated with precision / recall / F1."""
    det = data["detection"]
    c = det["confusion"]
    m = det["reproduced"]
    matrix = [[c["tp"], c["fn"]],   # actual injected: predicted flag / miss
              [c["fp"], c["tn"]]]    # actual clean:    predicted flag / miss

    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    vmax = max(max(row) for row in matrix)
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)

    labels = [["TP", "FN"], ["FP", "TN"]]
    for i in range(2):
        for j in range(2):
            v = matrix[i][j]
            txt = INK if v < vmax * 0.6 else SURFACE
            ax.text(j, i, f"{labels[i][j]}\n{v}", ha="center", va="center",
                    fontsize=14, fontweight="bold", color=txt)

    ax.set_xticks([0, 1], ["flagged", "not flagged"])
    ax.set_yticks([0, 1], ["injected\n(n=72)", "clean\n(n=32)"])
    ax.set_xlabel("Detector prediction")
    ax.set_ylabel("Actual label")
    ax.set_title("Hidden-prompt detector confusion matrix",
                 fontsize=13, fontweight="bold", pad=10)
    for s in ax.spines.values():
        s.set_visible(False)

    caption = (f"precision {m['precision']:.2f}    "
               f"recall {m['recall']:.3f}    "
               f"F1 {m['f1']:.3f}")
    fig.text(0.5, 0.03, caption, ha="center", fontsize=11, color=MUTED)

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = FIGDIR / "detection.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    data = json.loads(RESULTS.read_text())
    FIGDIR.mkdir(parents=True, exist_ok=True)
    _style()
    for fn in (plot_asr_by_defense, plot_per_technique, plot_detection):
        print("wrote", fn(data).relative_to(ROOT))


if __name__ == "__main__":
    main()
