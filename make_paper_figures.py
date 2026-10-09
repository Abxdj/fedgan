"""
Paper Figures — one script, one style, frozen numbers
FedGAN Failure Analysis - Anish Bharadwaj

Builds every results figure for the paper from the frozen evaluation and the
training/diagnostic logs. Nothing here trains or evaluates; it only reads.

Sources
  logs/final_results_v2.json            FID (pytorch-fid, all 7,180 test images)
                                         + downstream (val-selected, K=3)
  logs/losses_{tag}_seed{S}.json         per-round generator loss
  logs/diagnostic_{iid,non_iid}_seed{S}.json   mechanism diagnostic (Sync-G)

Outputs  figures/paper/   (PDF for LaTeX + 300-dpi PNG; legacy figures untouched)
  fig1_training_dynamics   G-loss per round, mean + min-max band over 3 seeds
  fig2_fid_vs_utility      FID vs downstream accuracy — the headline figure
  fig3_results_overview    FID | downstream accuracy, per configuration, with
                           the three per-seed values shown as dots
  fig4_perclass_heatmap    per-class test accuracy, conditional configurations
  fig5_mechanism           D-logit hardening + client-update alignment, IID vs non-IID
  fig6_labeling_artifact   unconditional: round-robin vs pseudo-labeled accuracy
  captions_draft.txt       caption drafts with the numbers filled in

One encoding for the whole paper (so a colour means the same thing everywhere):
  colour  = data condition   IID blue · non-IID orange · noisy client aqua ·
                             centralized baseline neutral grey
  shape / hatch = variant    non-IID per-seed partition = triangle / dashed;
                             Sync D&G aggregation = hollow marker / hatched bar
Palette: validated categorical slots 1-3 (all-pairs safe for scatter) plus a
neutral; aqua is below 3:1 contrast, so every series is also direct-labelled.

Usage:  python make_paper_figures.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

LOG_DIR = "./logs"
OUT_DIR = os.path.join("figures", "paper")
SEEDS   = [42, 123, 2024]
CHANCE  = 100 / 9
CLASS_NAMES = ["Adipose", "Background", "Debris", "Lymphocytes", "Mucus",
               "Smooth muscle", "Normal mucosa", "Stroma", "Adenocarcinoma"]
os.makedirs(OUT_DIR, exist_ok=True)

# ── Tokens ────────────────────────────────────────────────────────────────────
INK      = "#0b0b0b"   # primary text
INK2     = "#52514e"   # secondary text
MUTED    = "#898781"   # axis labels, ticks
GRID     = "#e1e0d9"   # hairline grid
AXIS     = "#c3c2b7"   # baseline / spines
SURFACE  = "#ffffff"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3
NEUTRAL  = "#898781"                                    # centralized baseline, 15k steps
NEUTRAL_LIGHT = "#bdbbb3"                               # centralized baseline, 3k steps
SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
            "#184f95", "#0d366b"]                       # sequential blue 100-700

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5,
    "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "grid.linestyle": "-", "axes.axisbelow": True,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "legend.frameon": False,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
COL1, COL2 = 3.5, 7.16      # IEEE single / double column widths (inches)

# ── Configuration catalogue: tag -> (short label, colour, variant) ────────────
#   variant: "" | "perseed" | "dg"
CGAN = [
    ("central_cgan",               "Centralized, 15k steps", NEUTRAL, "c15"),
    ("central_cgan_3k",            "Centralized, 3k steps",  NEUTRAL_LIGHT, "c3"),
    ("cgan_iid",                   "IID",                    BLUE,    ""),
    ("cgan_non_iid",               "Non-IID",                ORANGE,  ""),
    ("cgan_non_iid_perseed_syncG", "Non-IID, per-seed split", ORANGE, "perseed"),
    ("cgan_noisy_client",          "Noisy client",           AQUA,    ""),
    ("cgan_iid_syncDG",            "IID, Sync D&G",          BLUE,    "dg"),
    ("cgan_non_iid_syncDG",        "Non-IID, Sync D&G",      ORANGE,  "dg"),
]
GAN = [
    ("central_gan",                "Centralized, 15k steps", NEUTRAL, "c15"),
    ("central_gan_3k",             "Centralized, 3k steps",  NEUTRAL_LIGHT, "c3"),
    ("iid",                        "IID",                    BLUE,    ""),
    ("non_iid",                    "Non-IID",                ORANGE,  ""),
    ("gan_non_iid_perseed_syncG",  "Non-IID, per-seed split", ORANGE, "perseed"),
    ("noisy_client",               "Noisy client",           AQUA,    ""),
]


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=300,
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  saved figures/paper/{name}.pdf / .png")


def panel_letter(ax, letter, title=None):
    ax.set_title(f"({letter}) {title}" if title else f"({letter})", loc="left",
                 color=INK, fontsize=8, pad=4)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def marker_style(color, variant, size=6.5):
    """Filled circle by default; triangle = per-seed split; square = centralized;
    hollow (unfilled) = Sync D&G. Filled marks carry a thin surface ring."""
    m = {"": "o", "perseed": "^", "dg": "o", "c15": "s", "c3": "s"}[variant]
    if variant == "dg":
        return dict(marker=m, markersize=size, markerfacecolor="none",
                    markeredgecolor=color, markeredgewidth=1.5, linestyle="none")
    return dict(marker=m, markersize=size, markerfacecolor=color,
                markeredgecolor=SURFACE, markeredgewidth=0.8, linestyle="none")


# ── Fig 1: training dynamics ──────────────────────────────────────────────────
def fig1_training_dynamics():
    panels = [
        ("a", "Unconditional GAN, Sync-G", [
            ("iid", "IID", BLUE, "-"), ("non_iid", "Non-IID", ORANGE, "-"),
            ("gan_non_iid_perseed_syncG", "Per-seed", ORANGE, "--"),
            ("noisy_client", "Noisy", AQUA, "-")]),
        ("b", "Conditional GAN, Sync-G", [
            ("cgan_iid", "IID", BLUE, "-"), ("cgan_non_iid", "Non-IID", ORANGE, "-"),
            ("cgan_non_iid_perseed_syncG", "Per-seed", ORANGE, "--"),
            ("cgan_noisy_client", "Noisy", AQUA, "-")]),
        ("c", "Conditional GAN, Sync D&G", [
            ("cgan_iid_syncDG", "IID", BLUE, "-"),
            ("cgan_non_iid_syncDG", "Non-IID", ORANGE, "-")]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.25))
    for ax, (letter, title, series) in zip(axes, panels):
        ends = []
        for tag, label, color, ls in series:
            runs = []
            for s in SEEDS:
                p = os.path.join(LOG_DIR, f"losses_{tag}_seed{s}.json")
                if os.path.exists(p):
                    runs.append(load_json(p)["g_loss"])
            if not runs:
                print(f"  fig1: no logs for {tag}")
                continue
            arr = np.array(runs, dtype=float)
            r = np.arange(1, arr.shape[1] + 1)
            ax.fill_between(r, arr.min(0), arr.max(0), color=color, alpha=0.10, lw=0)
            ax.plot(r, arr.mean(0), color=color, lw=1.3, ls=ls)
            ends.append([arr.mean(0)[-1], label, color, ls])
        # direct end labels, nudged apart vertically
        ends.sort(key=lambda e: e[0])
        lo, hi = ax.get_ylim()
        gap = (hi - lo) * 0.07
        for i in range(1, len(ends)):
            if ends[i][0] - ends[i - 1][0] < gap:
                ends[i][0] = ends[i - 1][0] + gap
        for y, label, color, ls in ends:
            ax.text(31.0, y, label, va="center", ha="left", fontsize=6.5, color=INK2)
        ax.set_xlim(0.5, 30.5)
        ax.set_xticks([1, 10, 20, 30])
        ax.set_xlabel("Federated round")
        panel_letter(ax, letter, title)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Generator loss (mean of clients)")
    handles = [Line2D([], [], color=BLUE, lw=1.3, label="IID"),
               Line2D([], [], color=ORANGE, lw=1.3, label="Non-IID (frozen split)"),
               Line2D([], [], color=ORANGE, lw=1.3, ls="--", label="Non-IID (3 per-seed splits)"),
               Line2D([], [], color=AQUA, lw=1.3, label="Noisy client")]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(wspace=0.5)
    save(fig, "fig1_training_dynamics")


# ── Fig 2: FID vs downstream utility ─────────────────────────────────────────
LABEL_OFFSETS = {  # (dx, dy) in points, tuned for the frozen numbers
    "central_cgan": (7, 6), "central_cgan_3k": (-9, -8),
    "cgan_iid": (8, 5), "cgan_non_iid": (9, 8), "cgan_non_iid_perseed_syncG": (9, -6),
    "cgan_noisy_client": (14, -14), "cgan_iid_syncDG": (9, -9), "cgan_non_iid_syncDG": (-16, 14),
    "central_gan": (8, 5), "central_gan_3k": (8, 5), "iid": (9, -10), "non_iid": (9, 6),
    "gan_non_iid_perseed_syncG": (-16, 13), "noisy_client": (8, 6),
}


def place_label(ax, x, y, text, tag):
    dx, dy = LABEL_OFFSETS.get(tag, (7, 5))
    far = abs(dx) > 12 or abs(dy) > 12
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                fontsize=6.3, color=INK2, va="center", ha="right" if dx < 0 else "left",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=0, shrinkB=4)
                if far else None)


def fig2_fid_vs_utility(res):
    C = res["configs"]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.9))
    specs = [("a", "Conditional GAN", CGAN, "acc", "Downstream accuracy (%)"),
             ("b", "Unconditional GAN", GAN, "pseudo_acc",
              "Downstream accuracy, pseudo-labelled (%)")]
    rhos = {}
    for ax, (letter, title, cat, metric, ylab) in zip(axes, specs):
        xs, ys = [], []
        for tag, label, color, variant in cat:
            s = C.get(tag, {}).get("summary", {})
            f, a = s.get("fid", {}), s.get(metric, {})
            if not f.get("n") or not a.get("n"):
                continue
            ax.errorbar(f["mean"], a["mean"], xerr=f.get("std") or 0, yerr=a.get("std") or 0,
                        ecolor=color, elinewidth=0.6, capsize=0, zorder=2, alpha=0.55, fmt="none")
            ax.plot(f["mean"], a["mean"], zorder=3, **marker_style(color, variant))
            place_label(ax, f["mean"], a["mean"], label, tag)
            xs.append(f["mean"]); ys.append(a["mean"])
        if len(xs) >= 3:
            rhos[letter] = stats.spearmanr(xs, ys).correlation
            ax.text(0.98, 0.04, f"Spearman ρ = {rhos[letter]:+.2f}\n(FID ranking utility perfectly: −1)",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=6.3, color=INK2)
        if metric == "pseudo_acc":
            ax.axhline(CHANCE, color=MUTED, lw=0.7, ls=":")
            ax.text(0.01, CHANCE, " chance", transform=ax.get_yaxis_transform(),
                    fontsize=6.2, color=MUTED, va="bottom")
        ax.set_xlabel("FID (lower = closer to real images)")
        ax.set_ylabel(ylab)
        panel_letter(ax, letter, title)
        ax.margins(x=0.18, y=0.15)
    handles = [Line2D([], [], **marker_style(NEUTRAL, "c15"), label="Centralized, 15k steps"),
               Line2D([], [], **marker_style(NEUTRAL_LIGHT, "c3"), label="Centralized, 3k steps"),
               Line2D([], [], **marker_style(BLUE, ""), label="IID"),
               Line2D([], [], **marker_style(ORANGE, ""), label="Non-IID"),
               Line2D([], [], **marker_style(ORANGE, "perseed"), label="Non-IID, per-seed split"),
               Line2D([], [], **marker_style(AQUA, ""), label="Noisy client"),
               Line2D([], [], **marker_style(INK2, "dg"), label="Unfilled = Sync D&G")]
    fig.legend(handles=handles, loc="upper center", ncol=7, bbox_to_anchor=(0.5, -0.01),
               handletextpad=0.3, columnspacing=1.0)
    fig.subplots_adjust(wspace=0.3)
    save(fig, "fig2_fid_vs_utility")
    return rhos


# ── Fig 3: per-configuration overview ─────────────────────────────────────────
def _hbars(ax, cat, C, metric, value_fmt):
    labels = [c[1] for c in cat]
    y = np.arange(len(cat))[::-1]
    xmax = 0
    for yi, (tag, label, color, variant) in zip(y, cat):
        s = C.get(tag, {}).get("summary", {}).get(metric, {})
        if not s.get("n"):
            continue
        hatched = variant == "dg"
        ax.barh(yi, s["mean"], height=0.62, color=SURFACE if hatched else color,
                edgecolor=color, linewidth=0.9 if hatched else 0,
                hatch="//////" if hatched else None, zorder=2)
        sd = s.get("std") or 0
        ax.errorbar(s["mean"], yi, xerr=sd, ecolor=INK2, elinewidth=0.7, capsize=1.5,
                    capthick=0.7, fmt="none", zorder=3)
        vals = s.get("values", [])
        ax.plot(vals, [yi] * len(vals), "o", ms=2.6, mfc=INK, mec=SURFACE, mew=0.5,
                zorder=4, alpha=0.85)
        tip = max([s["mean"] + sd] + vals)
        ax.annotate(value_fmt(s["mean"]), (tip, yi), xytext=(4, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=6.2, color=INK2)
        xmax = max(xmax, tip)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, xmax * 1.16)
    return xmax


def fig3_results_overview(res):
    C, R = res["configs"], res["references"]
    fig, axes = plt.subplots(2, 2, figsize=(COL2, 4.2),
                             gridspec_kw={"height_ratios": [len(CGAN), len(GAN)]})
    _hbars(axes[0, 0], CGAN, C, "fid", lambda v: f"{v:.0f}")
    _hbars(axes[0, 1], CGAN, C, "acc", lambda v: f"{v:.1f}")
    _hbars(axes[1, 0], GAN, C, "fid", lambda v: f"{v:.0f}")
    _hbars(axes[1, 1], GAN, C, "pseudo_acc", lambda v: f"{v:.1f}")

    for ax in (axes[0, 1], axes[1, 1]):
        for key, ls in (("real_full", "-"), ("real_matched_10k", "--")):
            ref = R.get(key, {})
            if ref.get("mean") is not None:
                ax.axvline(ref["mean"], color=INK, lw=0.8, ls=ls, zorder=1)
        ax.set_xlim(0, max(ax.get_xlim()[1], R["real_full"]["mean"] * 1.1))
    axes[1, 1].axvline(CHANCE, color=MUTED, lw=0.8, ls=":", zorder=1)

    panel_letter(axes[0, 0], "a", "Conditional GAN — FID (lower = better)")
    panel_letter(axes[0, 1], "b", "Conditional GAN — downstream accuracy (%)")
    panel_letter(axes[1, 0], "c", "Unconditional GAN — FID")
    panel_letter(axes[1, 1], "d", "Unconditional GAN — pseudo-labelled accuracy (%)")
    for ax in axes[:, 1]:
        ax.set_yticklabels([])
    handles = [Patch(facecolor=NEUTRAL, label="Centralized, 15k"),
               Patch(facecolor=NEUTRAL_LIGHT, label="Centralized, 3k"),
               Patch(facecolor=BLUE, label="IID"), Patch(facecolor=ORANGE, label="Non-IID"),
               Patch(facecolor=AQUA, label="Noisy client"),
               Patch(facecolor=SURFACE, edgecolor=INK2, hatch="//////", label="Hatched = Sync D&G"),
               Line2D([], [], marker="o", ms=2.6, mfc=INK, mec=SURFACE, ls="none",
                      label="Individual seeds"),
               Line2D([], [], color=INK2, lw=0.7, label="± 1 s.d."),
               Line2D([], [], color=INK, lw=0.8, label="Real data, full"),
               Line2D([], [], color=INK, lw=0.8, ls="--", label="Real data, 10k"),
               Line2D([], [], color=MUTED, lw=0.8, ls=":", label="Chance")]
    fig.legend(handles=handles, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.4, columnspacing=0.9)
    fig.subplots_adjust(wspace=0.08, hspace=0.42)
    save(fig, "fig3_results_overview")


# ── Fig 4: per-class heatmap ──────────────────────────────────────────────────
def fig4_perclass_heatmap(res):
    C = res["configs"]
    rows, names = [], []
    if res.get("real_full_per_class_mean"):
        rows.append(res["real_full_per_class_mean"]); names.append("Real data (full)")
    for tag, label, _, _ in CGAN:
        pcs = C.get(tag, {}).get("per_class_mean_over_K", {})
        if pcs:
            rows.append(np.mean(list(pcs.values()), axis=0)); names.append(label)
    M = np.array(rows)
    cmap = LinearSegmentedColormap.from_list("seqblue", SEQ_RAMP)
    fig, ax = plt.subplots(figsize=(COL2, 2.9))
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.3,
                    color=SURFACE if v >= 55 else INK)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.grid(False)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axhline(0.5, color=SURFACE, lw=2.5)       # separate the real-data reference row
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cb.set_label("Test accuracy (%)", color=INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    save(fig, "fig4_perclass_heatmap")


# ── Fig 5: mechanism diagnostic ───────────────────────────────────────────────
def fig5_mechanism():
    logs = {c: [] for c in ("iid", "non_iid")}
    for c in logs:
        for s in SEEDS:
            p = os.path.join(LOG_DIR, f"diagnostic_{c}_seed{s}.json")
            if os.path.exists(p):
                logs[c].append(load_json(p))
    if not logs["iid"] or not logs["non_iid"]:
        print("  fig5: diagnostic logs missing — skipped")
        return None

    def split_logits(lg):
        mask = {int(k): v for k, v in lg["class_present_mask"].items()}
        pres, absn = [], []
        for rd in lg["d_logit_per_client_class"]:
            p, a = [], []
            for cid, vals in rd.items():
                for c, v in enumerate(vals):
                    (p if mask[int(cid)][c] else a).append(v)
            pres.append(np.mean(p)); absn.append(np.mean(a) if a else np.nan)
        return np.array(pres), np.array(absn)

    def tag_end(ax, y, text, dy=3):
        ax.annotate(text, (r[-1], y), xytext=(-2, dy), textcoords="offset points",
                    ha="right", va="bottom" if dy >= 0 else "top", fontsize=6.2, color=INK2,
                    zorder=5, bbox=dict(boxstyle="square,pad=0.15", fc=SURFACE, ec="none", alpha=0.85))

    def band(ax, r, arr, color, ls="-", lw=1.3):
        arr = np.array(arr, dtype=float)
        ax.fill_between(r, arr.min(0), arr.max(0), color=color, alpha=0.10, lw=0)
        ax.plot(r, arr.mean(0), color=color, lw=lw, ls=ls)
        return arr.mean(0)

    r = np.arange(1, len(logs["iid"][0]["g_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.25))

    ax = axes[0]
    e1 = band(ax, r, [lg["g_loss"] for lg in logs["iid"]], BLUE)
    e2 = band(ax, r, [lg["g_loss"] for lg in logs["non_iid"]], ORANGE)
    tag_end(ax, e1[-1], "IID")
    tag_end(ax, e2[-1], "Non-IID", dy=-3)
    ax.set_ylabel("Generator loss")
    panel_letter(ax, "a", "Generator loss")

    ax = axes[1]
    iid_p = band(ax, r, [split_logits(lg)[0] for lg in logs["iid"]], BLUE)
    p = band(ax, r, [split_logits(lg)[0] for lg in logs["non_iid"]], ORANGE)
    a = band(ax, r, [split_logits(lg)[1] for lg in logs["non_iid"]], ORANGE, ls="--")
    tag_end(ax, iid_p[-1], "IID")
    tag_end(ax, p[-1], "Non-IID, class held locally")
    tag_end(ax, a[-1], "Non-IID, class absent locally", dy=-3)
    ax.set_ylabel("Discriminator logit (generated)")
    panel_letter(ax, "b", "Rejection of generated samples")

    ax = axes[2]
    m1 = band(ax, r, [lg["cosine_sim_pairwise_mean"] for lg in logs["iid"]], BLUE)
    n1 = band(ax, r, [lg["cosine_sim_pairwise_min"] for lg in logs["iid"]], BLUE, ls="--", lw=1.0)
    m2 = band(ax, r, [lg["cosine_sim_pairwise_mean"] for lg in logs["non_iid"]], ORANGE)
    n2 = band(ax, r, [lg["cosine_sim_pairwise_min"] for lg in logs["non_iid"]], ORANGE, ls="--", lw=1.0)
    ax.axhline(0, color=INK2, lw=0.6)
    tag_end(ax, m1[-1], "IID, mean")
    tag_end(ax, n1[-1], "IID, least-aligned pair")
    tag_end(ax, m2[-1], "Non-IID, mean")
    tag_end(ax, n2[-1], "Non-IID, least-aligned pair", dy=-3)
    ax.set_ylabel("Cosine similarity")
    panel_letter(ax, "c", "Client-update alignment")

    for ax in axes:
        ax.set_xlim(0.5, 30.5)
        ax.set_xticks([1, 10, 20, 30])
        ax.set_xlabel("Federated round")
        ax.grid(axis="x", visible=False)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
    fig.legend(handles=[Line2D([], [], color=BLUE, lw=1.3, label="IID"),
                        Line2D([], [], color=ORANGE, lw=1.3, label="Non-IID (frozen split)"),
                        Line2D([], [], color=INK2, lw=1.0, ls="--",
                               label="(b) class absent locally · (c) least-aligned client pair")],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(wspace=0.38)
    save(fig, "fig5_mechanism")

    gap = p - a
    return {"gap_r1_5": float(np.mean(gap[:5])), "gap_r26_30": float(np.mean(gap[-5:])),
            "iid_mean_r1": float(m1[0]), "iid_mean_r30": float(m1[-1]),
            "noniid_mean_r1": float(m2[0]), "noniid_mean_r30": float(m2[-1]),
            "iid_min_of_seedavg_min": float(n1.min()), "noniid_min_of_seedavg_min": float(n2.min()),
            "iid_min_any_seed": float(min(min(lg["cosine_sim_pairwise_min"]) for lg in logs["iid"])),
            "noniid_min_any_seed": float(min(min(lg["cosine_sim_pairwise_min"]) for lg in logs["non_iid"]))}


# ── Fig 6: labelling-protocol artifact ────────────────────────────────────────
def fig6_labeling_artifact(res):
    C = res["configs"]
    fig, ax = plt.subplots(figsize=(COL1, 2.3))
    y = np.arange(len(GAN))[::-1]
    for yi, (tag, label, color, variant) in zip(y, GAN):
        s = C.get(tag, {}).get("summary", {})
        rr, pl = s.get("roundrobin_acc", {}), s.get("pseudo_acc", {})
        if rr.get("n"):
            ax.barh(yi + 0.19, rr["mean"], height=0.34, color=SURFACE, edgecolor=MUTED,
                    hatch="//////", lw=0.8, zorder=2)
        if pl.get("n"):
            ax.barh(yi - 0.19, pl["mean"], height=0.34, color=color, zorder=2)
            ax.errorbar(pl["mean"], yi - 0.19, xerr=pl.get("std") or 0, ecolor=INK2,
                        elinewidth=0.7, capsize=1.5, fmt="none", zorder=3)
    ax.axvline(CHANCE, color=INK2, lw=0.7, ls=":")
    ax.text(CHANCE + 0.6, len(GAN) - 0.45, "chance (11.1%)", fontsize=6, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([g[1] for g in GAN])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Downstream test accuracy (%)")
    ax.legend(handles=[Patch(facecolor=SURFACE, edgecolor=MUTED, hatch="//////",
                             label="Round-robin labels"),
                       Patch(facecolor=INK2, label="Pseudo-labels (colour = condition)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.2)
    save(fig, "fig6_labeling_artifact")


# ── Captions ──────────────────────────────────────────────────────────────────
def write_captions(res, rhos, mech):
    C, R = res["configs"], res["references"]

    def m(tag, metric):
        s = C.get(tag, {}).get("summary", {}).get(metric, {})
        if not s.get("n"):
            return "n/a"
        return f"{s['mean']:.1f} ± {s['std']:.1f}" if s.get("std") is not None else f"{s['mean']:.1f}"

    lines = [
        "CAPTION DRAFTS — rewrite in your own words; numbers are from final_results_v2.json",
        "Error bars / bands: ± 1 sample s.d. (ddof=1) or min–max over 3 generator seeds, as stated.",
        "",
        "Fig. 1  Generator loss per federated round (mean over clients; line = mean of 3 seeds,",
        "band = min–max). Under label skew the conditional GAN's loss rises and stalls in every",
        "run — frozen split, three per-seed splits, and Sync D&G — while every IID control falls;",
        "the unconditional GAN converges under the same skew. Loss values are not comparable",
        "across panels (different discriminator regimes).",
        "",
        "Fig. 2  FID against downstream accuracy for each configuration (mean ± 1 s.d., 3 seeds).",
        f"Conditional: Spearman ρ = {rhos.get('a', float('nan')):+.2f}; unconditional (pseudo-labelled): "
        f"ρ = {rhos.get('b', float('nan')):+.2f}. A metric that ranked utility would give ρ = −1.",
        "Unfilled markers: discriminator also synchronized (Sync D&G).",
        "",
        "Fig. 3  Per-configuration FID and downstream accuracy. Bars: mean of 3 generator seeds;",
        "dots: individual seeds; whiskers: ± 1 s.d. Downstream accuracy: classifier trained on",
        "10,000 synthetic images, best epoch selected on the PathMNIST validation split, averaged",
        "over 3 classifier seeds, tested on the 7,180 real test images. Vertical lines: classifiers",
        f"trained on real data, full ({R['real_full']['mean']:.1f}%) and 10,000-image "
        f"({R['real_matched_10k']['mean']:.1f}%) subsets.",
        "",
        "Fig. 4  Per-class test accuracy of classifiers trained on conditional-GAN synthetic data",
        "(mean over 3 generator seeds × 3 classifier seeds). Top row: classifier trained on real data.",
        "",
        "Fig. 5  Mechanism diagnostic, conditional GAN, Sync-G, frozen split, 3 seeds (line = mean,",
        "band = min–max). (b) Mean discriminator logit on generated images; under non-IID the gap",
        "between locally-held and locally-absent classes widens from "
        + (f"{mech['gap_r1_5']:+.2f} (rounds 1–5) to {mech['gap_r26_30']:+.2f} (rounds 26–30)." if mech else "n/a."),
        "(c) Cosine similarity of client generator updates; dashed = least-aligned client pair",
        "(seed-averaged). " + (f"Seed-averaged minimum: IID {mech['iid_min_of_seedavg_min']:+.3f}, "
                               f"non-IID {mech['noniid_min_of_seedavg_min']:+.3f}; single-seed minima "
                               f"{mech['iid_min_any_seed']:+.3f} and {mech['noniid_min_any_seed']:+.3f}."
                               if mech else ""),
        "",
        "Fig. 6  Unconditional generators: downstream accuracy under round-robin labels (random",
        "with respect to image content by construction, hence near chance) versus pseudo-labels",
        "from a classifier trained on real data. Centralized 15k: round-robin "
        f"{m('central_gan', 'roundrobin_acc')}%, pseudo-labelled {m('central_gan', 'pseudo_acc')}%.",
    ]
    with open(os.path.join(OUT_DIR, "captions_draft.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("  saved figures/paper/captions_draft.txt")


if __name__ == "__main__":
    src = os.path.join(LOG_DIR, "final_results_v2.json")
    if not os.path.exists(src):
        raise SystemExit(f"{src} not found — run evaluate_downstream_v2.py first.")
    res = load_json(src)
    print("Building paper figures from", src)
    fig1_training_dynamics()
    rhos = fig2_fid_vs_utility(res)
    fig3_results_overview(res)
    fig4_perclass_heatmap(res)
    mech = fig5_mechanism()
    fig6_labeling_artifact(res)
    write_captions(res, rhos, mech)
    print("Done.")
