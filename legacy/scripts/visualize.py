"""
Results Visualization
FedGAN Failure Analysis - Anish Bharadwaj

Generates 4 figures for the paper:
1. Loss curves — all 6 experiments
2. FID comparison bar chart
3. Silent failure bar chart (accuracy gap)
4. Per-class accuracy heatmap
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

LOG_DIR    = "./logs"
OUTPUT_DIR = "./figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Palette
COLORS = {
    "gan_iid"       : "#4C9BE8",   # blue
    "gan_noniid"    : "#E8874C",   # orange
    "gan_noisy"     : "#E84C4C",   # red
    "cgan_iid"      : "#4CE8A0",   # green
    "cgan_noniid"   : "#B04CE8",   # purple
    "cgan_noisy"    : "#E8D44C",   # yellow
    "real"          : "#2ECC71",   # reference green
    "bg"            : "#0F1117",
    "panel"         : "#1A1D27",
    "text"          : "#E8EAF0",
    "subtext"       : "#8B90A0",
    "grid"          : "#2A2D3A",
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
    "grid.linewidth"    : 0.6,
    "font.family"       : "monospace",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
})

LABELS = {
    "iid"            : "GAN — IID",
    "non_iid"        : "GAN — Non-IID",
    "noisy_client"   : "GAN — Noisy Client",
    "cgan_iid"       : "cGAN — IID",
    "cgan_non_iid"   : "cGAN — Non-IID",
    "cgan_noisy_client": "cGAN — Noisy Client",
}
COLOR_LIST = list(COLORS.values())[:6]


# Load Data
def load_losses():
    files = {
        "iid"              : "losses_iid.json",
        "non_iid"          : "losses_non_iid.json",
        "noisy_client"     : "losses_noisy_client.json",
        "cgan_iid"         : "losses_cgan_iid.json",
        "cgan_non_iid"     : "losses_cgan_non_iid.json",
        "cgan_noisy_client": "losses_cgan_noisy_client.json",
    }
    data = {}
    for key, fname in files.items():
        path = os.path.join(LOG_DIR, fname)
        if os.path.exists(path):
            with open(path) as f:
                data[key] = json.load(f)
    return data

def load_fid():
    path = os.path.join(LOG_DIR, "fid_results_all.json")
    with open(path) as f:
        return json.load(f)

def load_silent_failure():
    path = os.path.join(LOG_DIR, "silent_failure_all.json")
    with open(path) as f:
        return json.load(f)


# Figure 1: Loss Curves 
def plot_loss_curves(loss_data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(COLORS["bg"])
    fig.suptitle("Figure 1 — Generator & Discriminator Loss Across Federated Rounds",
                 fontsize=13, color=COLORS["text"], y=1.01)

    plot_order = ["iid", "non_iid", "noisy_client",
                  "cgan_iid", "cgan_noisy_client", "cgan_non_iid"]
    color_map = {
        "iid"              : (COLORS["gan_iid"],    "-"),
        "non_iid"          : (COLORS["gan_noniid"], "-"),
        "noisy_client"     : (COLORS["gan_noisy"],  "-"),
        "cgan_iid"         : (COLORS["cgan_iid"],   "--"),
        "cgan_non_iid"     : (COLORS["cgan_noniid"],"--"),
        "cgan_noisy_client": (COLORS["cgan_noisy"], "--"),
    }

    for ax, loss_key, title in zip(axes, ["g_loss", "d_loss"],
                                    ["Generator Loss", "Discriminator Loss"]):
        ax.set_facecolor(COLORS["panel"])
        for key in plot_order:
            if key not in loss_data:
                continue
            label  = LABELS[key]
            color, style = color_map[key]
            rounds = loss_data[key]["round"]
            vals   = loss_data[key][loss_key]
            if key == "cgan_non_iid":
                ax.plot(rounds, vals, color=color, linestyle=style,
                        linewidth=3.2, label=label, alpha=1.0,
                        zorder=10, marker="o", markersize=3, markevery=5)
                if loss_key == "g_loss":
                    ax.annotate("cGAN Non-IID(diverging up)",
                                xy=(30, vals[-1]),
                                xytext=(22, vals[-1] + 0.5),
                                color=color, fontsize=7.5,
                                arrowprops=dict(arrowstyle="->",
                                                color=color, lw=1.2),
                                zorder=11)
            else:
                ax.plot(rounds, vals, color=color, linestyle=style,
                        linewidth=1.8, label=label, alpha=0.85, zorder=5)

        if loss_key == "d_loss":
            ax.axhline(0.693, color=COLORS["subtext"], linestyle=":",
                       linewidth=1.2, alpha=0.7, label="Equilibrium (0.693)")

        ax.set_title(title, color=COLORS["text"], fontsize=11, pad=10)
        ax.set_xlabel("Federated Round", fontsize=9)
        ax.set_ylabel("Loss", fontsize=9)
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=7.5, framealpha=0.2, loc="upper right")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig1_loss_curves.png")
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"Saved → {path}")


# Figure 2: FID Comparison
def plot_fid(fid_data):
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["panel"])

    names  = list(fid_data.keys())
    values = list(fid_data.values())
    colors = [COLORS["gan_iid"], COLORS["gan_noniid"], COLORS["gan_noisy"],
              COLORS["cgan_iid"], COLORS["cgan_noniid"], COLORS["cgan_noisy"]]
    short  = [n.replace("GAN — ", "GAN\n").replace("cGAN — ", "cGAN\n")
              for n in names]

    bars = ax.bar(short, values, color=colors, width=0.55,
                  edgecolor=COLORS["bg"], linewidth=1.2)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                f"{val:.1f}", ha="center", va="bottom",
                fontsize=9, color=COLORS["text"])

    baseline = values[0]
    ax.axhline(baseline, color=COLORS["gan_iid"], linestyle="--",
               linewidth=1, alpha=0.5, label=f"GAN IID Baseline ({baseline})")

    ax.set_title("Figure 2 — FID Score by Experiment  (lower = better)",
                 fontsize=12, color=COLORS["text"], pad=12)
    ax.set_ylabel("FID Score", fontsize=10)
    ax.set_ylim(0, max(values) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, framealpha=0.2)

    note = "⚠  cGAN Non-IID: singular covariance matrix detected — distribution collapsed"
    ax.annotate(note, xy=(0.5, -0.18), xycoords="axes fraction",
                ha="center", fontsize=8,
                color=COLORS["gan_noisy"], style="italic")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig2_fid_comparison.png")
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"Saved → {path}")


# Figure 3: Silent Failure 
def plot_silent_failure(sf_data):
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["panel"])

    ref_acc  = sf_data["Real Data (Reference)"]["overall"]
    exp_keys = [k for k in sf_data if k != "Real Data (Reference)"]
    accs     = [sf_data[k]["overall"] for k in exp_keys]
    gaps     = [ref_acc - a for a in accs]

    colors = [COLORS["gan_iid"], COLORS["gan_noniid"], COLORS["gan_noisy"],
              COLORS["cgan_iid"], COLORS["cgan_noniid"], COLORS["cgan_noisy"]]
    short  = [k.replace("GAN — ", "GAN\n").replace("cGAN — ", "cGAN\n")
              for k in exp_keys]

    x     = np.arange(len(exp_keys))
    width = 0.38

    b1 = ax.bar(x - width/2, accs, width, label="Synthetic-trained accuracy",
                color=colors, edgecolor=COLORS["bg"], linewidth=1)
    b2 = ax.bar(x + width/2, gaps, width, label="Silent failure gap",
                color=[c + "55" for c in colors],
                edgecolor=colors, linewidth=1.2, linestyle="--")

    ax.axhline(ref_acc, color=COLORS["real"], linestyle="--",
               linewidth=1.5, label=f"Real data reference ({ref_acc:.1f}%)")

    for bar, val in zip(b1, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=8, color=COLORS["text"])

    for bar, val in zip(b2, gaps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"-{val:.1f}%", ha="center", va="bottom",
                fontsize=8, color=COLORS["gan_noisy"])

    ax.set_title("Figure 3 — Silent Failure: Downstream Classifier Accuracy vs Gap",
                 fontsize=12, color=COLORS["text"], pad=12)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8.5, framealpha=0.2, loc="upper left")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig3_silent_failure.png")
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"Saved → {path}")


# Figure 4: Per-Class Heatmap
def plot_heatmap(sf_data):
    CLASS_NAMES = ["adipose", "background", "debris", "lymphocytes",
                   "mucus", "smooth muscle", "normal colon",
                   "cancer stroma", "adenocarcinoma"]

    col_labels = list(sf_data.keys())
    short_cols = ["Real\nData",
                  "GAN\nIID", "GAN\nNon-IID", "GAN\nNoisy",
                  "cGAN\nIID", "cGAN\nNon-IID", "cGAN\nNoisy"]

    matrix = np.array([
        [sf_data[col]["per_class"][i] for col in col_labels]
        for i in range(9)
    ])

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["panel"])

    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto",
                   vmin=0, vmax=100)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(short_cols, fontsize=9)
    ax.set_yticks(range(9))
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)

    for i in range(9):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            color = "black" if val > 55 else "white"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Accuracy (%)", color=COLORS["text"], fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=COLORS["subtext"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=COLORS["subtext"])

    ax.set_title(
        "Figure 4 — Per-Class Accuracy Heatmap  "
        "(green = high accuracy, red = failure)",
        fontsize=12, color=COLORS["text"], pad=12
    )

    # Highlight adenocarcinoma row — most clinically critical
    ax.add_patch(mpatches.FancyBboxPatch(
        (-0.5, 7.5), len(col_labels), 1,
        boxstyle="round,pad=0.05",
        linewidth=2, edgecolor=COLORS["gan_noisy"],
        facecolor="none", zorder=5
    ))
    ax.text(len(col_labels) - 0.1, 8, "← clinically critical",
            va="center", ha="left", fontsize=7.5,
            color=COLORS["gan_noisy"], style="italic")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig4_perclass_heatmap.png")
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"Saved → {path}")


# Main
if __name__ == "__main__":
    print("Generating paper figures...\n")

    loss_data = load_losses()
    fid_data  = load_fid()
    sf_data   = load_silent_failure()

    plot_loss_curves(loss_data)
    plot_fid(fid_data)
    plot_silent_failure(sf_data)
    plot_heatmap(sf_data)

    print(f"\nAll figures saved to ./{OUTPUT_DIR}/")
    print("Files: fig1_loss_curves.png, fig2_fid_comparison.png,")
    print("       fig3_silent_failure.png, fig4_perclass_heatmap.png")
