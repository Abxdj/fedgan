"""
Figure 6 (revised) — cGAN Divergence Mechanism, IID vs Non-IID
FedGAN Failure Analysis - Anish Bharadwaj

Replaces the single-condition fig6. The IID control is what makes the
mechanism claim causal rather than descriptive, so it must appear in
the figure.

Three panels:
  (a) G-loss trajectory, both conditions, all seeds — shows divergence
      is deterministic and specific to non-IID.
  (b) D logit on generated images. Non-IID split into locally-present
      vs locally-absent classes; IID shown as a single line (no absent
      classes exist). Shows progressive hardening and its absence
      under IID.
  (c) Client update alignment. Mean pairwise cosine similarity for both
      conditions, PLUS the minimum pairwise similarity, which is the
      sharper signal — it goes negative under non-IID and never
      under IID.

Reads logs/diagnostic_{iid,non_iid}_seed{42,123,2024}.json produced by
diagnostic_iid_control.py. Bands are min-max across seeds.

Usage:
    python visualize_diagnostic_control.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

LOG_DIR = "./logs"
FIG_DIR = "./figures"
SEEDS   = [42, 123, 2024]
os.makedirs(FIG_DIR, exist_ok=True)

C = {
    "iid"        : "#4CE8A0",   # green — the healthy control
    "noniid"     : "#B04CE8",   # purple — matches fig1/fig2 non-IID
    "present"    : "#4CE8A0",
    "absent"     : "#E84C4C",
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


# ── Load ──────────────────────────────────────────────────────────────────────
def load_all():
    logs = {"iid": [], "non_iid": []}
    for cond in logs:
        for seed in SEEDS:
            path = os.path.join(LOG_DIR, f"diagnostic_{cond}_seed{seed}.json")
            if not os.path.exists(path):
                print(f"MISSING: {path}")
                continue
            with open(path) as f:
                logs[cond].append(json.load(f))
    return logs


def band(series_list):
    """Returns (mean, min, max) per round across seeds."""
    arr = np.array(series_list)
    return arr.mean(axis=0), arr.min(axis=0), arr.max(axis=0)


def d_logit_series(log):
    """Per-round mean D logit, split by locally-present vs locally-absent.

    JSON serializes dict keys as strings, so class_present_mask comes back
    keyed "0".."4" rather than 0..4. Normalize to int keys on load.
    """
    mask = {int(k): v for k, v in log["class_present_mask"].items()}
    present_by_round, absent_by_round = [], []
    for round_logits in log["d_logit_per_client_class"]:
        p, a = [], []
        for cid_str, logits in round_logits.items():
            cid = int(cid_str)
            for c, val in enumerate(logits):
                (p if mask[cid][c] else a).append(val)
        present_by_round.append(np.mean(p))
        absent_by_round.append(np.mean(a) if a else np.nan)
    return np.array(present_by_round), np.array(absent_by_round)


# ── Plot ──────────────────────────────────────────────────────────────────────
def main():
    logs = load_all()
    if not logs["iid"] or not logs["non_iid"]:
        print("Missing logs — run diagnostic_iid_control.py for all seeds first.")
        return

    rounds = np.array(logs["non_iid"][0]["round"])
    n_seeds = len(logs["non_iid"])

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    fig.patch.set_facecolor(C["bg"])

    # ── (a) G-loss ────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(C["panel"])
    for cond, color, label in (("iid", C["iid"], "IID"),
                               ("non_iid", C["noniid"], "Non-IID")):
        m, lo, hi = band([lg["g_loss"] for lg in logs[cond]])
        ax.plot(rounds, m, color=color, linewidth=2.2, label=label)
        ax.fill_between(rounds, lo, hi, color=color, alpha=0.18)

    ax.set_title("(a) Generator loss\nIID converges, Non-IID diverges",
                 fontsize=10, color=C["text"], pad=8)
    ax.set_xlabel("Federated round", fontsize=9)
    ax.set_ylabel("Aggregate G-loss", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, framealpha=0.2)

    # ── (b) D logit ───────────────────────────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor(C["panel"])

    # IID: single line, no absent classes exist
    iid_present = [d_logit_series(lg)[0] for lg in logs["iid"]]
    m, lo, hi = band(iid_present)
    ax.plot(rounds, m, color=C["iid"], linewidth=2.0, linestyle=":",
            label="IID — all classes present")
    ax.fill_between(rounds, lo, hi, color=C["iid"], alpha=0.12)

    # Non-IID: split present vs absent
    non_present = [d_logit_series(lg)[0] for lg in logs["non_iid"]]
    non_absent  = [d_logit_series(lg)[1] for lg in logs["non_iid"]]

    m, lo, hi = band(non_present)
    ax.plot(rounds, m, color=C["present"], linewidth=2.0,
            label="Non-IID — classes present locally")
    ax.fill_between(rounds, lo, hi, color=C["present"], alpha=0.18)

    m_abs, lo, hi = band(non_absent)
    ax.plot(rounds, m_abs, color=C["absent"], linewidth=2.0,
            label="Non-IID — classes absent locally")
    ax.fill_between(rounds, lo, hi, color=C["absent"], alpha=0.18)

    # Annotate the widening gap
    m_pres, _, _ = band(non_present)
    early_gap = np.mean(m_pres[:5] - m_abs[:5])
    late_gap  = np.mean(m_pres[-5:] - m_abs[-5:])
    ax.annotate("", xy=(28, m_pres[-3]), xytext=(28, m_abs[-3]),
                arrowprops=dict(arrowstyle="<->", color=C["text"], lw=1.2))
    ax.text(26.5, (m_pres[-3] + m_abs[-3]) / 2,
            f"gap widens\n{early_gap:+.1f} → {late_gap:+.1f}",
            fontsize=7.5, color=C["text"], ha="right", va="center")

    ax.set_title("(b) Discriminator logit on generated images\n"
                 "lower = D rejects more confidently",
                 fontsize=10, color=C["text"], pad=8)
    ax.set_xlabel("Federated round", fontsize=9)
    ax.set_ylabel("Mean D logit", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, framealpha=0.2, loc="lower left")

    # ── (c) Client update alignment ───────────────────────────────────────────
    ax = axes[2]
    ax.set_facecolor(C["panel"])

    for cond, color, label in (("iid", C["iid"], "IID"),
                               ("non_iid", C["noniid"], "Non-IID")):
        m, lo, hi = band([lg["cosine_sim_pairwise_mean"] for lg in logs[cond]])
        ax.plot(rounds, m, color=color, linewidth=2.2, label=f"{label} — mean")
        ax.fill_between(rounds, lo, hi, color=color, alpha=0.18)

        m_min, lo_min, hi_min = band(
            [lg["cosine_sim_pairwise_min"] for lg in logs[cond]])
        ax.plot(rounds, m_min, color=color, linewidth=1.4, linestyle="--",
                alpha=0.75, label=f"{label} — min pair")

    ax.axhline(0, color=C["subtext"], linestyle=":", linewidth=1.1, alpha=0.8)
    ax.text(1.5, 0.004, "zero alignment", fontsize=7,
            color=C["subtext"], va="bottom")

    ax.set_title("(c) Client generator update alignment\n"
                 "Non-IID starts lower; min pair goes negative",
                 fontsize=10, color=C["text"], pad=8)
    ax.set_xlabel("Federated round", fontsize=9)
    ax.set_ylabel("Pairwise cosine similarity", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, framealpha=0.2, loc="upper right")

    fig.suptitle(
        f"Figure 6 — cGAN divergence mechanism: Non-IID vs IID control "
        f"(mean, min–max band over {n_seeds} seeds)",
        fontsize=11.5, color=C["text"], y=1.04)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig6_diagnostic_control.png")
    plt.savefig(path, dpi=190, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    print(f"Saved → {path}")

    # ── Numbers for the paper text ────────────────────────────────────────────
    print("\n" + "="*66)
    print(" NUMBERS FOR THE PAPER (mean across seeds)")
    print("="*66)

    print("\n(a) G-loss, round 1 → round 30:")
    for cond, label in (("iid", "IID"), ("non_iid", "Non-IID")):
        m, _, _ = band([lg["g_loss"] for lg in logs[cond]])
        print(f"    {label:<9} {m[0]:.3f} → {m[-1]:.3f}")

    print("\n(b) D logit gap (present − absent), Non-IID only:")
    print(f"    rounds  1–5 : {early_gap:+.3f}")
    print(f"    rounds 26–30: {late_gap:+.3f}   [widens]")
    m_iid, _, _ = band(iid_present)
    print(f"    IID present-class logit (no absent classes): "
          f"{m_iid.mean():+.3f}")

    print("\n(c) Cosine similarity:")
    for cond, label in (("iid", "IID"), ("non_iid", "Non-IID")):
        m, _, _ = band([lg["cosine_sim_pairwise_mean"] for lg in logs[cond]])
        mn, _, _ = band([lg["cosine_sim_pairwise_min"] for lg in logs[cond]])
        print(f"    {label:<9} mean r1: {m[0]:+.4f} → r30: {m[-1]:+.4f} "
              f"| lowest min-pair over run: {mn.min():+.4f}")
    print("\n    Key contrast is the STARTING level and the negative")
    print("    minimum, not the rate of decline.")
    print("="*66)


if __name__ == "__main__":
    main()
