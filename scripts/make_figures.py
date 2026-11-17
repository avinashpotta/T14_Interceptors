#!/usr/bin/env python3
"""Report figures.

Palette and mark conventions follow the project's data-viz reference:
categorical slots assigned in fixed order (blue, orange, aqua...), never
cycled; one hue for magnitude; recessive grid and axes; text in ink colours
rather than series colours; direct labels instead of a number on every mark.
Light surface only - these go into a printed report.
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
MUTED     = "#8a8984"
GRID      = "#e5e4e0"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"      # categorical slots 1-3
S4, S8     = "#eda100", "#e34948"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 200,
})


def finish(fig, name, title, sub=None):
    """Place title/subtitle in INCHES from the top, not figure fractions.

    Fractional offsets collide on short figures: 0.05 of a 3.9in figure is
    14pt, which an 11.5pt bold title overruns. Working in inches keeps the
    spacing identical whatever the figure height.
    """
    import textwrap
    h = fig.get_size_inches()[1]
    sub_lines = textwrap.wrap(sub, 96) if sub else []
    reserve = 0.34 + 0.17 * len(sub_lines)          # inches of header space
    fig.tight_layout(rect=(0, 0, 1, 1 - reserve / h))
    fig.text(0.012, 1 - 0.10 / h, title, ha="left", va="top",
             fontsize=11.5, fontweight="bold", color=INK)
    for i, line in enumerate(sub_lines):
        fig.text(0.012, 1 - (0.33 + 0.16 * i) / h, line, ha="left", va="top",
                 fontsize=8.5, color=INK_2)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.25)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


# ---------------------------------------------------------------- fig 1
def fig_levels():
    """Magnitude, low->high, one measure -> horizontal bars, single hue,
    with the final system emphasised and the rest recessive."""
    rows = [
        ("Regex only",                 0.324, 0.27, 0.38, False),
        ("+ Western name list",        0.408, 0.34, 0.46, False),
        ("Presidio (off the shelf)",   0.528, 0.47, 0.58, False),
        ("+ Indian name list",         0.599, 0.54, 0.65, False),
        ("+ both lists",               0.616, 0.56, 0.67, False),
        ("Rule layer (no NER)",        0.641, 0.58, 0.70, False),
        ("Neural NER only",            0.677, 0.62, 0.72, False),
        ("Rule layer + NER",           0.725, 0.67, 0.77, True),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    y = np.arange(len(rows))[::-1]
    for i, (label, f2, lo, hi, hero) in enumerate(rows):
        c = S1 if hero else "#c7d7ee"
        ax.barh(y[i], f2, height=0.56, color=c,
                edgecolor=SURFACE, linewidth=2)
        ax.plot([lo, hi], [y[i]] * 2, color=INK_2 if hero else MUTED,
                lw=1.6, solid_capstyle="butt", zorder=3)
        ax.text(hi + 0.012, y[i], f"{f2:.3f}", va="center", fontsize=8.5,
                color=INK if hero else INK_2,
                fontweight="bold" if hero else "normal")
    ax.set_yticks(y, [r[0] for r in rows], fontsize=9,
                  color=INK)
    ax.set_xlim(0, 0.88)
    ax.set_xlabel("F2  (recall weighted 4x precision)")
    ax.xaxis.grid(True); ax.yaxis.grid(False)
    finish(fig, "fig1_levels", "Detection performance by approach",
           "Held-out test split, 92 dialogues / 268 PII spans. "
           "Bars show F2; lines show the 95% bootstrap interval over dialogues.")


# ---------------------------------------------------------------- fig 2
def fig_gazetteer():
    """Two distinct series -> categorical slots 1 and 2, both direct-labelled."""
    types = ["Given name", "Surname"]
    west  = [26, 7]
    india = [72, 55]
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    x = np.arange(len(types)); w = 0.33
    b1 = ax.bar(x - w/2 - 0.012, west,  w, color=S2, edgecolor=SURFACE, linewidth=2,
                label="Western name list")
    b2 = ax.bar(x + w/2 + 0.012, india, w, color=S1, edgecolor=SURFACE, linewidth=2,
                label="Indian name list")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width()/2, r.get_height() + 2,
                    f"{int(r.get_height())}%", ha="center", fontsize=9,
                    color=INK, fontweight="bold")
    ax.set_xticks(x, types, fontsize=9.5, color=INK)
    ax.set_ylim(0, 88); ax.set_ylabel("Recall (%)")
    ax.yaxis.grid(True); ax.xaxis.grid(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left",
              bbox_to_anchor=(0.0, 1.02), labelcolor=INK_2)
    finish(fig, "fig2_gazetteer", "A Western name list does not transfer",
           "End-to-end recall on Indian names, swapping only the gazetteer.")


# ---------------------------------------------------------------- fig 3
def fig_attribution():
    """Part-to-whole across four exclusive outcomes -> stacked horizontal bar."""
    types = ["PAN","AADHAAR","PHONE","LANDMARK","ADDRESS","EMAIL","SURNAME","GIVEN_NAME"]
    both     = [ 0, 35, 25, 21, 79, 81, 84, 82]
    lost     = [ 0, 15, 25,  0,  7, 19, 16, 10]
    rescued  = [81, 50, 46, 11,  4,  0,  0,  0]
    neither  = [19,  0,  4, 68, 11,  0,  0,  8]
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    y = np.arange(len(types))
    left = np.zeros(len(types))
    for vals, c, lab in ((both, S3, "found in both"),
                         (lost, S8, "lost to the ASR"),
                         (rescued, S1, "rescued by the ASR"),
                         (neither, "#d6d5d0", "found in neither")):
        ax.barh(y, vals, left=left, height=0.6, color=c,
                edgecolor=SURFACE, linewidth=2, label=lab)
        for i, v in enumerate(vals):
            if v >= 12:
                ax.text(left[i] + v/2, y[i], f"{v}", ha="center", va="center",
                        fontsize=8, color="#ffffff" if c != "#d6d5d0" else INK_2,
                        fontweight="bold")
        left = left + np.array(vals)
    ax.set_yticks(y, types, fontsize=9, color=INK)
    ax.set_xlim(0, 100); ax.set_xlabel("Share of PII items (%)")
    ax.xaxis.grid(True); ax.yaxis.grid(False)
    ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, -0.42), labelcolor=INK_2)
    finish(fig, "fig3_attribution", "Where detection failures come from",
           "The same detector run on the reference transcript and on Whisper's. "
           "'Rescued' = found only in Whisper's output.")


# ---------------------------------------------------------------- fig 4
def fig_latency():
    """Before -> after per item -> dumbbell, two shades of one hue."""
    types   = ["Given name","Surname","Landmark","Address","PAN","Phone","Aadhaar"]
    tts     = [0.49, 0.59, 1.35, 2.52, 2.31, 3.03, 3.91]
    human   = [0.49, 0.41, 1.42, 3.74, 4.98, 4.60, 5.80]
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    y = np.arange(len(types))
    for i in range(len(types)):
        ax.plot([tts[i], human[i]], [y[i]]*2, color="#c7d7ee", lw=3,
                solid_capstyle="round", zorder=1)
    ax.scatter(tts, y, s=58, color=MUTED, zorder=3, label="TTS corpus",
               edgecolor=SURFACE, linewidth=1.5)
    ax.scatter(human, y, s=58, color=S1, zorder=3, label="human readers",
               edgecolor=SURFACE, linewidth=1.5)
    for i in range(len(types)):
        if human[i] - tts[i] > 0.4:
            ax.text(human[i] + 0.16, y[i], f"+{human[i]-tts[i]:.1f}s",
                    va="center", fontsize=8, color=INK, fontweight="bold")
    ax.axvline(5.4, color=MUTED, lw=1, ls=(0,(4,3)), zorder=0)
    ax.axvline(8.0, color=S2, lw=1.2, ls=(0,(4,3)), zorder=0)
    # headroom above the top row so the threshold captions do not sit on a mark
    ax.set_ylim(-0.7, len(types) - 1 + 1.25)
    ax.text(5.4, len(types) - 1 + 1.18, " TTS p99  5.4s", fontsize=7.5,
            color=INK_2, va="top", ha="left")
    ax.text(8.0, len(types) - 1 + 1.18, " human p99  8.0s", fontsize=7.5,
            color=S2, va="top", ha="left", fontweight="bold")
    ax.set_yticks(y, types, fontsize=9, color=INK)
    ax.set_xlim(0, 9.9); ax.set_xlabel("Median spoken duration (seconds)")
    ax.xaxis.grid(True); ax.yaxis.grid(False)
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, -0.40), labelcolor=INK_2)
    finish(fig, "fig4_latency", "Synthetic speech understates the delay budget",
           "People slow down over identifiers and not over names. The streaming "
           "delay must exceed the longest PII utterance.")


# ---------------------------------------------------------------- fig 5
def fig_coverage():
    """Before -> after per item -> dumbbell. Detection vs actual redaction."""
    types = ["LANDMARK","ADDRESS","PHONE","EMAIL","PAN","GIVEN_NAME","SURNAME","AADHAAR"]
    found = [25, 39, 68, 75, 81, 81, 86, 85]
    cover = [ 0,  0, 50, 67, 38, 72, 64, 80]
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    y = np.arange(len(types))
    for i in range(len(types)):
        ax.plot([cover[i], found[i]], [y[i]]*2, color="#c7d7ee", lw=3,
                solid_capstyle="round", zorder=1)
    # dumbbell = one hue, two shades (not two hues): grey is the reference
    # point, blue is the quantity under scrutiny - same convention as fig 4,
    # so the two dumbbells read identically in the report.
    ax.scatter(found, y, s=58, color=MUTED, zorder=3, edgecolor=SURFACE, linewidth=1.5,
               label="detector found it")
    ax.scatter(cover, y, s=58, color=S1, zorder=3, edgecolor=SURFACE, linewidth=1.5,
               label="beep actually covered it")
    for i in range(len(types)):
        gap = found[i] - cover[i]
        if gap >= 15:
            ax.text(found[i] + 2, y[i], f"-{gap}", va="center", fontsize=8,
                    color=INK, fontweight="bold")
    ax.set_yticks(y, types, fontsize=9, color=INK)
    ax.set_xlim(0, 104); ax.set_xlabel("Share of PII spans (%)")
    ax.xaxis.grid(True); ax.yaxis.grid(False)
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, -0.38), labelcolor=INK_2)
    finish(fig, "fig5_coverage", "Detection success is not redaction success",
           "Before the span-expansion fix. A compressed ASR token under-represents "
           "the audio it came from.")


# ---------------------------------------------------------------- fig 6
def fig_streaming():
    """Magnitude on a grid -> heatmap, sequential (one hue, light->dark).

    A grouped bar would work but buries the point; the grid makes the diagonal
    visible at a glance, which IS the finding.
    """
    from matplotlib.colors import LinearSegmentedColormap
    buckets = ["0-1 s", "1-2 s", "2-3 s", "4 s +"]
    delays  = ["2 s", "4 s", "6 s", "9 s"]
    leak = np.array([[  5,   5,   5,   5],
                     [  0,   0,   0,   0],
                     [ 30,  20,  30,  20],
                     [100,  50,   0,   0]], dtype=float)
    cmap = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#2a78d6", "#143c6b"])
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    im = ax.imshow(leak, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(leak.shape[0]):
        for j in range(leak.shape[1]):
            v = leak[i, j]
            ax.text(j, i, f"{int(v)}%", ha="center", va="center", fontsize=9.5,
                    color="#ffffff" if v > 45 else INK,
                    fontweight="bold" if v > 45 else "normal")
    ax.set_xticks(range(4), delays, fontsize=9, color=INK)
    ax.set_yticks(range(4), buckets, fontsize=9, color=INK)
    ax.set_xlabel("delay budget"); ax.set_ylabel("spoken duration of the PII")
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label("leak rate", fontsize=8.5, color=INK_2)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=8, color=GRID)
    finish(fig, "fig6_streaming", "A span leaks unless the delay exceeds it",
           "Streaming redaction, 10 dialogues. Spans over 4 s leak entirely at a "
           "2 s budget and not at all by 6 s; short spans are unaffected.")


if __name__ == "__main__":
    fig_levels(); fig_gazetteer(); fig_attribution(); fig_latency()
    fig_coverage(); fig_streaming()
    print(f"\nfigures in {OUT}")
