"""
Figures 2 and 3 (revised) — with centralized baselines
FedGAN Failure Analysis - Anish Bharadwaj

The previous versions of these figures predate the centralized control
runs. Finding #1 (federation carries a measurable cost even under ideal
IID conditions) is arguably the paper's strongest result and currently
appears in no figure at all. This adds it.

Figure 2 — FID, with centralized bars placed first and separated by a
divider, plus an annotated bracket showing the federated cost per
architecture.

Figure 3 — Downstream accuracy, with BOTH reference classifiers
(real-full and real-matched-10k) so the quantity-vs-quality
decomposition is visible, plus the centralized cGAN, which is the
correct comparator for the federated cGAN configurations.

Reads:
    logs/fid_multiseed.json
    logs/silent_failure_multiseed.json
    logs/centralized_results.json
    logs/reference_classifiers.json

Usage:
    python visualize_with_centralized.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

LOG_DIR = "./logs"
FIG_DIR = "./figures"
os.makedirs(FIG_DIR, exist_ok=True)

C = {
    "central"    : "#7A8296",   # grey — controls, visually set apart
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
    "figure.facecolor"  : C["bg"],
    "axes.facecolor"    : C["panel"],
    "axes.edgecolor"    : C["grid"],
    "axes.labelcolor"   : C["text"],
    "text.color"        : C["text"],
    "xtick.color"       : C["subtext"],
    "ytick.color"       : C["subtext"],
    "grid.color"        : C["grid"],
    "font.family"       : "monospace",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
})

ERRKW = dict(ecolor=C["text"], elinewidth=1.3, capthick=1.3)


def load(name):
    path = os.path.join(LOG_DIR, name)
    if not os.path.exists(path):
        print(f"MISSING: {path}")
        return None
    with open(path) as f:
        return json.load(f)


# ── Figure 2: FID with centralized baselines ─────────────────────────────────
def plot_fid(fid, central):
    labels = ["Centralized\nGAN", "Centralized\ncGAN",
              "GAN\nIID", "GAN\nNon-IID", "GAN\nNoisy",
              "cGAN\nIID", "cGAN\nNon-IID", "cGAN\nNoisy"]

    means = [central["Centralized GAN"]["fid_mean"],
             central["Centralized cGAN"]["fid_mean"],
             fid["GAN — IID Baseline"]["mean"],
             fid["GAN — Non-IID"]["mean"],
             fid["GAN — Noisy Client"]["mean"],
             fid["cGAN — IID Baseline"]["mean"],
             fid["cGAN — Non-IID"]["mean"],
             fid["cGAN — Noisy Client"]["mean"]]

    stds  = [central["Centralized GAN"]["fid_std"],
             central["Centralized cGAN"]["fid_std"],
             fid["GAN — IID Baseline"]["std"],
             fid["GAN — Non-IID"]["std"],
             fid["GAN — Noisy Client"]["std"],
             fid["cGAN — IID Baseline"]["std"],
             fid["cGAN — Non-IID"]["std"],
             fid["cGAN — Noisy Client"]["std"]]

    colors = [C["central"], C["central"],
              C["gan_iid"], C["gan_noniid"], C["gan_noisy"],
              C["cgan_iid"], C["cgan_noniid"], C["cgan_noisy"]]

    fig, ax = plt.subplots(figsize=(13, 5.8))
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["panel"])

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                  width=0.6, edgecolor=C["bg"], linewidth=1.2, error_kw=ERRKW)

    # Hatch the controls so they read as baselines, not results
    for i in (0, 1):
        bars[i].set_hatch("///")

    for b, m, s in zip(bars, means, stds):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + s + 3.5,
                f"{m:.1f}", ha="center", va="bottom",
                fontsize=8.5, color=C["text"])

    # Divider between centralized controls and federated runs
    ax.axvline(1.5, color=C["subtext"], linestyle="--", linewidth=1, alpha=0.55)
    ax.text(0.75, max(means) * 1.14, "centralized controls",
            fontsize=8, color=C["subtext"], ha="center", style="italic")
    ax.text(4.75, max(means) * 1.14, "federated",
            fontsize=8, color=C["subtext"], ha="center", style="italic")

    # Federated cost brackets, per architecture
    def bracket(i_central, i_fed, label, y):
        ax.annotate("", xy=(i_central, y), xytext=(i_fed, y),
                    arrowprops=dict(arrowstyle="<->", color=C["text"], lw=1.1))
        ax.text((i_central + i_fed) / 2, y + 2.5, label,
                ha="center", fontsize=7.5, color=C["text"])

    gan_cost  = means[2] - means[0]
    cgan_cost = means[5] - means[1]
    bracket(0, 2, f"GAN: +{gan_cost:.1f} FID "
                  f"(+{gan_cost/means[0]*100:.0f}%)", max(means) * 0.44)
    bracket(1, 5, f"cGAN: +{cgan_cost:.1f} FID "
                  f"(+{cgan_cost/means[1]*100:.0f}%)", max(means) * 0.30)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("FID Score", fontsize=10)
    ax.set_ylim(0, max(means) * 1.24)
    ax.set_title("Figure 2 — FID with centralized controls "
                 "(mean \u00b1 std, n=3 seeds; lower = better)",
                 fontsize=12, color=C["text"], pad=14)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig2_fid_with_centralized.png")
    plt.savefig(path, dpi=185, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    print(f"Saved → {path}")
    return gan_cost, cgan_cost


# ── Figure 3: Downstream accuracy with both references ───────────────────────
def plot_accuracy(sf, central, refs):
    labels = ["Real\nfull", "Real\nmatched 10k", "Centralized\ncGAN",
              "cGAN\nIID", "cGAN\nNon-IID", "cGAN\nNoisy",
              "GAN\nIID", "GAN\nNon-IID", "GAN\nNoisy"]

    means = [refs["real_full"]["mean"],
             refs["real_matched_10k"]["mean"],
             central["Centralized cGAN"]["acc_mean"],
             sf["cGAN — IID Baseline"]["mean"],
             sf["cGAN — Non-IID"]["mean"],
             sf["cGAN — Noisy Client"]["mean"],
             sf["GAN — IID Baseline"]["mean"],
             sf["GAN — Non-IID"]["mean"],
             sf["GAN — Noisy Client"]["mean"]]

    stds  = [refs["real_full"]["std"],
             refs["real_matched_10k"]["std"],
             central["Centralized cGAN"]["acc_std"],
             sf["cGAN — IID Baseline"]["std"],
             sf["cGAN — Non-IID"]["std"],
             sf["cGAN — Noisy Client"]["std"],
             sf["GAN — IID Baseline"]["std"],
             sf["GAN — Non-IID"]["std"],
             sf["GAN — Noisy Client"]["std"]]

    colors = [C["real"], C["real"], C["central"],
              C["cgan_iid"], C["cgan_noniid"], C["cgan_noisy"],
              C["gan_iid"], C["gan_noniid"], C["gan_noisy"]]

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["panel"])

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                  width=0.62, edgecolor=C["bg"], linewidth=1.2, error_kw=ERRKW)

    for i in (0, 1, 2):
        bars[i].set_hatch("///")

    for b, m, s in zip(bars, means, stds):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + s + 1.4,
                f"{m:.1f}%", ha="center", va="bottom",
                fontsize=8.5, color=C["text"])

    # Chance level for 9-class classification
    ax.axhline(100/9, color=C["subtext"], linestyle=":", linewidth=1.2,
               alpha=0.85)
    ax.text(6.0, 100/9 + 1.8, "chance (11.1%)", fontsize=7.5,
            color=C["subtext"], ha="center")

    # Reference lines
    ax.axhline(means[1], color=C["real"], linestyle="--", linewidth=1.2,
               alpha=0.6)
    ax.text(3.2, means[1] + 2.0,
            f"matched-size reference ({means[1]:.1f}%) — correct ceiling "
            f"for synthetic comparisons",
            fontsize=7.5, color=C["real"], ha="left")

    ax.axvline(1.5, color=C["subtext"], linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(5.5, color=C["subtext"], linestyle="--", linewidth=1, alpha=0.5)

    ax.text(0.5, 96, "references", fontsize=8, color=C["subtext"],
            ha="center", style="italic")
    ax.text(3.5, 96, "conditional (valid utility measurements)",
            fontsize=8, color=C["subtext"], ha="center", style="italic")
    ax.text(7.0, 96, "unconditional (labeling artifact)",
            fontsize=8, color=C["gan_noisy"], ha="center", style="italic")

    # Decomposition annotations
    quantity = means[0] - means[1]
    quality  = means[1] - means[3]
    fed_cost = means[2] - means[3]

    ax.annotate("", xy=(0, means[0]), xytext=(1, means[1]),
                arrowprops=dict(arrowstyle="<->", color=C["text"], lw=1.1))
    ax.text(0.5, (means[0] + means[1]) / 2 + 1.5,
            f"quantity\n{quantity:+.1f}pp", ha="center",
            fontsize=7, color=C["text"])

    ax.annotate("", xy=(2, means[2]), xytext=(3, means[3]),
                arrowprops=dict(arrowstyle="<->", color=C["text"], lw=1.1))
    ax.text(2.5, (means[2] + means[3]) / 2 + 1.5,
            f"federation\n{-fed_cost:+.1f}pp", ha="center",
            fontsize=7, color=C["text"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Accuracy on real test set (%)", fontsize=10)
    ax.set_ylim(0, 102)
    ax.set_title("Figure 3 — Downstream classifier accuracy with controls "
                 "(mean \u00b1 std, n=3 seeds)",
                 fontsize=12, color=C["text"], pad=14)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig3_accuracy_with_centralized.png")
    plt.savefig(path, dpi=185, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    print(f"Saved → {path}")
    return quantity, quality, fed_cost


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fid     = load("fid_multiseed.json")
    sf      = load("silent_failure_multiseed.json")
    central = load("centralized_results.json")
    refs    = load("reference_classifiers.json")

    if not all([fid, sf, central, refs]):
        print("\nOne or more log files missing — cannot build figures.")
        raise SystemExit(1)

    gan_cost, cgan_cost = plot_fid(fid, central)
    quantity, quality, fed_cost = plot_accuracy(sf, central, refs)

    print("\n" + "="*66)
    print(" NUMBERS FOR THE PAPER")
    print("="*66)
    print("\nFederated cost in FID (federated IID − centralized):")
    print(f"    GAN : +{gan_cost:.2f} FID")
    print(f"    cGAN: +{cgan_cost:.2f} FID")
    print("\nDownstream accuracy decomposition:")
    print(f"    data quantity effect (real-full − real-matched): "
          f"{quantity:+.2f} pp")
    print(f"    federation cost (centralized cGAN − federated cGAN IID): "
          f"{-fed_cost:+.2f} pp")
    print(f"    residual quality gap (real-matched − federated cGAN IID): "
          f"{-quality:+.2f} pp")
    print("\nNote: the matched-size reference, not real-full, is the correct")
    print("ceiling for synthetic comparisons — synthetic sets are 10k images.")
    print("="*66)
