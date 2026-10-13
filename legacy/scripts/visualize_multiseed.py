"""
Multi-Seed Results Visualization
FedGAN Failure Analysis - Anish Bharadwaj

Generates error-bar versions of Figures 2 and 3 using mean +/- std
across 3 seeds, replacing the single-seed point estimates.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

LOG_DIR    = "./logs"
OUTPUT_DIR = "./figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {
    "gan_iid"    : "#4C9BE8",
    "gan_noniid" : "#E8874C",
    "gan_noisy"  : "#E84C4C",
    "cgan_iid"   : "#4CE8A0",
    "cgan_noniid": "#B04CE8",
    "cgan_noisy" : "#E8D44C",
    "real"       : "#2ECC71",
    "bg"         : "#0F1117",
    "panel"      : "#1A1D27",
    "text"       : "#E8EAF0",
    "subtext"    : "#8B90A0",
    "grid"       : "#2A2D3A",
}

plt.rcParams.update({
    "figure.facecolor"  : COLORS["bg"],
    "axes.facecolor"    : COLORS["panel"],
    "axes.edgecolor"    : COLORS["grid"],
    "axes.labelcolor"   : COLORS["text"],
    "text.color"        : COLORS["text"],
    "xtick.color"       : COLORS["subtext"],
    "ytick.color"       : COLORS["subtext"],
    "grid.color"        : COLORS["grid"],
    "font.family"       : "monospace",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
})

EXP_COLORS = [COLORS["gan_iid"], COLORS["gan_noniid"], COLORS["gan_noisy"],
              COLORS["cgan_iid"], COLORS["cgan_noniid"], COLORS["cgan_noisy"]]


def load(name):
    with open(os.path.join(LOG_DIR, name)) as f:
        return json.load(f)


# ── Figure 2b: FID with error bars ────────────────────────────────────────────
def plot_fid_multiseed(fid_data):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["panel"])

    names  = list(fid_data.keys())
    means  = [fid_data[n]["mean"] for n in names]
    stds   = [fid_data[n]["std"] for n in names]
    short  = [n.replace("GAN — ", "GAN\n").replace("cGAN — ", "cGAN\n") for n in names]

    bars = ax.bar(short, means, yerr=stds, capsize=6, color=EXP_COLORS,
                  width=0.55, edgecolor=COLORS["bg"], linewidth=1.2,
                  error_kw=dict(ecolor=COLORS["text"], elinewidth=1.3, capthick=1.3))

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 3,
                f"{m:.1f}", ha="center", va="bottom", fontsize=9, color=COLORS["text"])

    baseline = means[0]
    ax.axhline(baseline, color=COLORS["gan_iid"], linestyle="--",
               linewidth=1, alpha=0.5, label=f"GAN IID Baseline ({baseline:.1f})")

    ax.set_title("Figure 2 — FID Score by Experiment (mean \u00b1 std, n=3 seeds)",
                 fontsize=12, color=COLORS["text"], pad=12)
    ax.set_ylabel("FID Score", fontsize=10)
    ax.set_ylim(0, max(means) * 1.2)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, framealpha=0.2)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig2_fid_comparison_multiseed.png")
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"Saved → {path}")


# ── Figure 3b: Silent Failure with error bars ─────────────────────────────────
def plot_silent_failure_multiseed(sf_data):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["panel"])

    ref_acc  = sf_data["Real Data (Reference)"]["mean"]
    exp_keys = [k for k in sf_data if k != "Real Data (Reference)"]
    means    = [sf_data[k]["mean"] for k in exp_keys]
    stds     = [sf_data[k]["std"] for k in exp_keys]
    short    = [k.replace("GAN — ", "GAN\n").replace("cGAN — ", "cGAN\n") for k in exp_keys]

    bars = ax.bar(short, means, yerr=stds, capsize=6, color=EXP_COLORS,
                  width=0.55, edgecolor=COLORS["bg"], linewidth=1.2,
                  error_kw=dict(ecolor=COLORS["text"], elinewidth=1.3, capthick=1.3))

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 1.5,
                f"{m:.1f}%", ha="center", va="bottom", fontsize=9, color=COLORS["text"])

    ax.axhline(ref_acc, color=COLORS["real"], linestyle="--",
               linewidth=1.5, label=f"Real data reference ({ref_acc:.1f}%)")

    # Shade overlapping cGAN region to visually flag statistical indistinguishability
    cgan_idx = [i for i, k in enumerate(exp_keys) if "cGAN" in k]
    if len(cgan_idx) == 3:
        lo = min(means[i] - stds[i] for i in cgan_idx)
        hi = max(means[i] + stds[i] for i in cgan_idx)
        ax.axhspan(lo, hi, xmin=0.5, xmax=1.0, color=COLORS["cgan_noniid"],
                  alpha=0.08, zorder=0)
        ax.text(4.5, hi + 2, "overlapping — not\nstatistically distinct",
               fontsize=7.5, color=COLORS["subtext"], ha="center", style="italic")

    ax.set_title("Figure 3 — Silent Failure: Downstream Accuracy (mean \u00b1 std, n=3 seeds)",
                 fontsize=12, color=COLORS["text"], pad=12)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8.5, framealpha=0.2, loc="upper left")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig3_silent_failure_multiseed.png")
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"Saved → {path}")


if __name__ == "__main__":
    fid_data = load("fid_multiseed.json")
    sf_data  = load("silent_failure_multiseed.json")

    plot_fid_multiseed(fid_data)
    plot_silent_failure_multiseed(sf_data)

    print("\nMulti-seed figures complete.")
