"""
Per-Client G-Loss Diagnostic — Visualization
FedGAN Failure Analysis - Anish Bharadwaj

Renders the two mechanism tests from perclient_gloss_diagnostic.py
(cGAN Non-IID, seed 42) as Figure 6:

  Left  — mean discriminator logit on generated images, split into
          classes present vs. absent in each client's local data.
          A persistent gap supports "clients reject classes they've
          never seen locally".
  Right — mean pairwise cosine similarity between clients' generator
          weight updates each round, overlaid with the aggregate
          G-loss. A falling/negative trend that coincides with rising
          G-loss supports "FedAvg averages contradictory gradients".
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
    "present" : "#4CE8A0",
    "absent"  : "#E84C4C",
    "cos_sim" : "#4C9BE8",
    "g_loss"  : "#E8D44C",
    "bg"      : "#0F1117",
    "panel"   : "#1A1D27",
    "text"    : "#E8EAF0",
    "subtext" : "#8B90A0",
    "grid"    : "#2A2D3A",
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


def load(name):
    with open(os.path.join(LOG_DIR, name)) as f:
        return json.load(f)


def per_round_present_absent(log):
    present_mask = {int(k): v for k, v in log["class_present_mask"].items()}
    rounds = log["round"]
    present_mean, present_std, absent_mean, absent_std = [], [], [], []

    for round_logits in log["d_logit_per_client_class"]:
        present_vals, absent_vals = [], []
        for cid_str, logits in round_logits.items():
            cid = int(cid_str)
            for c, logit in enumerate(logits):
                (present_vals if present_mask[cid][c] else absent_vals).append(logit)
        present_mean.append(np.mean(present_vals))
        present_std.append(np.std(present_vals))
        absent_mean.append(np.mean(absent_vals))
        absent_std.append(np.std(absent_vals))

    return (rounds, np.array(present_mean), np.array(present_std),
             np.array(absent_mean), np.array(absent_std))


def plot_diagnostic(log):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor(COLORS["bg"])

    # ── Panel 1: D logit on fake, present vs absent classes ──
    rounds, p_mean, p_std, a_mean, a_std = per_round_present_absent(log)
    ax1.set_facecolor(COLORS["panel"])
    ax1.plot(rounds, p_mean, color=COLORS["present"], linewidth=1.8,
              label="classes present locally")
    ax1.fill_between(rounds, p_mean - p_std, p_mean + p_std, color=COLORS["present"], alpha=0.15)
    ax1.plot(rounds, a_mean, color=COLORS["absent"], linewidth=1.8,
              label="classes absent locally")
    ax1.fill_between(rounds, a_mean - a_std, a_mean + a_std, color=COLORS["absent"], alpha=0.15)
    ax1.axhline(0, color=COLORS["subtext"], linestyle=":", linewidth=1, alpha=0.6)
    ax1.set_title("D logit on generated images\n(higher = D fooled, lower = D rejects)",
                   fontsize=10.5, color=COLORS["text"])
    ax1.set_xlabel("Federated round", fontsize=9.5)
    ax1.set_ylabel("Mean discriminator logit", fontsize=9.5)
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8.5, framealpha=0.2)

    # ── Panel 2: cosine similarity of client G updates vs G-loss ──
    ax2.set_facecolor(COLORS["panel"])
    cos_mean = log["cosine_sim_pairwise_mean"]
    ax2.plot(rounds, cos_mean, color=COLORS["cos_sim"], linewidth=1.8,
              label="mean pairwise cosine sim\n(client G updates)")
    ax2.axhline(0, color=COLORS["subtext"], linestyle=":", linewidth=1, alpha=0.6)
    ax2.set_ylabel("Cosine similarity", fontsize=9.5, color=COLORS["cos_sim"])
    ax2.set_xlabel("Federated round", fontsize=9.5)
    ax2.grid(alpha=0.25)

    ax2b = ax2.twinx()
    ax2b.plot(rounds, log["g_loss"], color=COLORS["g_loss"], linewidth=1.8,
               linestyle="--", label="aggregate G-loss")
    ax2b.set_ylabel("Aggregate G-loss", fontsize=9.5, color=COLORS["g_loss"])
    ax2b.spines["top"].set_visible(False)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, framealpha=0.2, loc="upper left")
    ax2.set_title("Client update conflict vs. G-loss divergence",
                   fontsize=10.5, color=COLORS["text"])

    fig.suptitle("Figure 6 — cGAN Non-IID Divergence Mechanism Diagnostic (seed 42)",
                  fontsize=12.5, color=COLORS["text"], y=1.02)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig6_perclient_diagnostic.png")
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"Saved -> {path}")


if __name__ == "__main__":
    log = load("perclient_gloss_diagnostic_seed42.json")
    plot_diagnostic(log)
