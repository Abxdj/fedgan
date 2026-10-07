"""
Figure 7 — Pseudo-labeling: protocol artifact vs recoverable structure
FedGAN Failure Analysis - Anish Bharadwaj

Two panels:

  (a) Round-robin vs pseudo-labeled downstream accuracy for every
      unconditional generator. Round-robin bars sit at chance by
      construction; pseudo-labeled bars show what the generators
      actually carry. The centralized bar (4.6% -> 48.7%) is the
      cleanest demonstration that the original protocol measured
      itself rather than the generator.

  (b) Pseudo-label distribution entropy against pseudo-labeled
      accuracy. Entropy is an Inception-independent mode-collapse
      measure: it declines monotonically from centralized through
      federated IID, non-IID and noisy-client, and tracks downstream
      utility across all four.

Reads:
    logs/silent_failure_multiseed.json   (round-robin, federated)
    logs/centralized_results.json        (round-robin, centralized)
    logs/pseudo_label_results.json       (pseudo-labeled, all four)

Usage:
    python visualize_pseudo_label.py
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
    "roundrobin": "#7A8296",
    "pseudo"    : "#4C9BE8",
    "central"   : "#4CE8A0",
    "noisy"     : "#E8D44C",
    "warn"      : "#E84C4C",
    "bg"        : "#0F1117",
    "panel"     : "#1A1D27",
    "text"      : "#E8EAF0",
    "subtext"   : "#8B90A0",
    "grid"      : "#2A2D3A",
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

CHANCE = 100 / 9


def load(name):
    path = os.path.join(LOG_DIR, name)
    if not os.path.exists(path):
        print(f"MISSING: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def main():
    sf     = load("silent_failure_multiseed.json")
    cent   = load("centralized_results.json")
    pseudo = load("pseudo_label_results.json")
    if not all([sf, cent, pseudo]):
        raise SystemExit(1)

    pr = pseudo["results"]
    ceiling = pseudo["pseudo_labeler_accuracy_mean"]

    labels = ["Centralized\nGAN", "GAN\nIID", "GAN\nNon-IID", "GAN\nNoisy"]

    rr_mean = [cent["Centralized GAN"]["acc_mean"],
               sf["GAN — IID Baseline"]["mean"],
               sf["GAN — Non-IID"]["mean"],
               sf["GAN — Noisy Client"]["mean"]]
    rr_std  = [cent["Centralized GAN"]["acc_std"],
               sf["GAN — IID Baseline"]["std"],
               sf["GAN — Non-IID"]["std"],
               sf["GAN — Noisy Client"]["std"]]

    keys = ["Centralized GAN", "GAN — IID Baseline",
            "GAN — Non-IID", "GAN — Noisy Client"]
    pl_mean = [pr[k]["acc_mean"] for k in keys]
    pl_std  = [pr[k]["acc_std"] for k in keys]
    ent     = [pr[k]["label_entropy_mean"] for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.6),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    fig.patch.set_facecolor(C["bg"])

    # ── (a) Round-robin vs pseudo-labeled ─────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(C["panel"])
    x = np.arange(len(labels))
    w = 0.36

    b1 = ax.bar(x - w/2, rr_mean, w, yerr=rr_std, capsize=4,
                color=C["roundrobin"], hatch="///",
                edgecolor=C["bg"], linewidth=1.1,
                error_kw=dict(ecolor=C["text"], elinewidth=1.1, capthick=1.1),
                label="Round-robin labels (protocol artifact)")
    b2 = ax.bar(x + w/2, pl_mean, w, yerr=pl_std, capsize=4,
                color=C["pseudo"], edgecolor=C["bg"], linewidth=1.1,
                error_kw=dict(ecolor=C["text"], elinewidth=1.1, capthick=1.1),
                label="Pseudo-labels (recoverable structure)")

    # Highlight the noisy bar's unreliability
    b2[3].set_color(C["noisy"])
    b2[3].set_alpha(0.75)

    for b, m, s in zip(b1, rr_mean, rr_std):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + s + 1.2,
                f"{m:.1f}", ha="center", fontsize=8, color=C["subtext"])
    for b, m, s in zip(b2, pl_mean, pl_std):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + s + 1.2,
                f"{m:.1f}", ha="center", fontsize=8.5, color=C["text"])

    ax.axhline(CHANCE, color=C["subtext"], linestyle=":", linewidth=1.2)
    ax.text(-0.42, CHANCE + 1.5, f"chance ({CHANCE:.1f}%)",
            fontsize=7.5, color=C["subtext"], ha="left")

    ax.axhline(ceiling, color=C["central"], linestyle="--",
               linewidth=1.2, alpha=0.7)
    ax.text(-0.45, ceiling + 1.5,
            f"pseudo-labeler ceiling ({ceiling:.1f}%)",
            fontsize=7.5, color=C["central"], ha="left")

    # Federated cost bracket, pseudo-labeled scale
    fed_cost = pl_mean[0] - pl_mean[1]
    ax.annotate("", xy=(0 + w/2, pl_mean[0]), xytext=(1 + w/2, pl_mean[1]),
                arrowprops=dict(arrowstyle="<->", color=C["text"], lw=1.1))
    ax.text(0.5 + w/2, (pl_mean[0] + pl_mean[1]) / 2 + 2.5,
            f"federation\n{-fed_cost:.1f}pp", ha="center",
            fontsize=7.5, color=C["text"])

    ax.text(3.0, pl_mean[3] + pl_std[3] + 6,
            "high seed variance\n(6.3–46.5%)", fontsize=7,
            color=C["warn"], ha="center", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Accuracy on real test set (%)", fontsize=9.5)
    ax.set_ylim(0, 92)
    ax.set_title("(a) Labeling protocol determines the result\n"
                 "round-robin labels are random w.r.t. content by construction",
                 fontsize=10, color=C["text"], pad=8)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7.5, framealpha=0.2, loc="upper right")

    # ── (b) Entropy vs accuracy ───────────────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor(C["panel"])

    pt_colors = [C["central"], C["pseudo"], "#B04CE8", C["noisy"]]
    short = ["Centralized", "Fed IID", "Fed Non-IID", "Fed Noisy"]

    for xe, ya, sa, col, lab in zip(ent, pl_mean, pl_std, pt_colors, short):
        ax.errorbar(xe, ya, yerr=sa, fmt="o", markersize=10,
                    color=col, ecolor=col, elinewidth=1.4,
                    capsize=4, alpha=0.95)
        ax.annotate(lab, (xe, ya), textcoords="offset points",
                    xytext=(0, 14), ha="center", fontsize=8, color=col)

    # Trend line through the four points
    z = np.polyfit(ent, pl_mean, 1)
    xs = np.linspace(min(ent) - 0.04, max(ent) + 0.04, 50)
    ax.plot(xs, np.poly1d(z)(xs), color=C["subtext"],
            linestyle="--", linewidth=1.1, alpha=0.6)

    ax.axhline(CHANCE, color=C["subtext"], linestyle=":", linewidth=1.1)
    ax.text(min(ent) - 0.03, CHANCE + 1.5, f"chance ({CHANCE:.1f}%)",
            fontsize=7.5, color=C["subtext"])

    ax.set_xlabel("Pseudo-label distribution entropy\n"
                  "(1.0 = uniform across 9 classes, 0 = single class)",
                  fontsize=9)
    ax.set_ylabel("Pseudo-labeled accuracy (%)", fontsize=9.5)
    ax.set_title("(b) Mode collapse tracks downstream utility\n"
                 "entropy is Inception-independent",
                 fontsize=10, color=C["text"], pad=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 65)

    fig.suptitle("Figure 7 — Unconditional FedGAN: labeling protocol and "
                 "mode collapse (mean \u00b1 std, n=3 seeds)",
                 fontsize=11.5, color=C["text"], y=1.03)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig7_pseudo_label.png")
    plt.savefig(path, dpi=185, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    print(f"Saved → {path}")

    # ── Numbers ──
    print("\n" + "="*68)
    print(" NUMBERS FOR THE PAPER")
    print("="*68)
    print(f"\n  Pseudo-labeler ceiling: {ceiling:.2f}%   "
          f"Chance: {CHANCE:.2f}%\n")
    print(f"  {'Configuration':<18} {'Round-robin':>12} {'Pseudo':>16} "
          f"{'Entropy':>9}")
    print(f"  {'-'*58}")
    for k, lab, rm, rs, pm, ps, e in zip(keys, short, rr_mean, rr_std,
                                          pl_mean, pl_std, ent):
        print(f"  {lab:<18} {rm:>10.2f}% {pm:>9.2f}% +/-{ps:<4.2f} {e:>9.3f}")

    print(f"\n  Federated cost, unconditional (centralized − fed IID): "
          f"{-(pl_mean[0] - pl_mean[1]):+.2f} pp")
    print("  Compare: conditional architecture cost was -11.04 pp")
    print("  -> federation cost replicates independently in both architectures")

    r = np.corrcoef(ent, pl_mean)[0, 1]
    print(f"\n  Entropy vs accuracy correlation across 4 configs: r = {r:.3f}")
    print("  (n=4 points — descriptive only, not a statistical claim)")
    print("="*68)


if __name__ == "__main__":
    main()
