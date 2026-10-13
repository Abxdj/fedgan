"""
Null Baseline for the Downstream Instrument + Welch Statistics
FedGAN Failure Analysis - Anish Bharadwaj

WHY THIS EXISTS
Round-robin labels for unconditional generators are random with respect to
image content, yet those classifiers scored up to 31.3 % on the real test set
(e.g. gan_non_iid_perseed_syncG: 24.9 / 31.3 / 10.1). The majority class
(Adipose) is only 18.6 % of the test set, so the extra accuracy comes from the
pipeline itself: picking the best of 15 epochs on 10,004 labelled real
validation images is a small label-guided search, even when the training
labels carry no information. Every downstream number in the paper went
through that same pipeline, so we need its output under NO signal.

WHAT IT DOES
  Part A — null distribution of the downstream instrument.
    For every checkpoint (14 configs x 3 generator seeds) the synthetic set is
    regenerated exactly as in evaluate_downstream_v2.py (seed s+1, 10,000
    images). Its labels are then replaced by a random permutation of a
    balanced label vector, so they are uniform over classes and independent of
    content. R classifiers are trained per checkpoint through the identical
    training function. Recorded per run:
      - test accuracy at the val-selected epoch  (what the paper reports)
      - test accuracy at the final epoch         (no selection)
      - test accuracy at epoch 1
      - the selected epoch
    Plus a generator-free null: R classifiers per seed on the 10,000-image
    real subset used for Real-matched-10k, with permuted labels.
  Part B — Welch t-tests for every derived difference in final_results_v2.json
    (descriptive; n = 3 per arm; for sorting claims, NOT for significance
    statements in the paper), and an exact-null p and Fisher-z 95 % interval
    for both Spearman correlations.

GUARANTEES
  - Reads checkpoints and final_results_v2.json; writes ONLY
    logs/null_baseline.json. Frozen results are not touched.
  - The classifier, data loading, generation and training functions are
    verbatim copies of evaluate_downstream_v2.py; the script refuses to run
    if any copy has drifted (source check at start-up).
  - train_classifier_trace() is train_classifier() plus a test-set evaluation
    after each epoch's validation step. Evaluation has no randomness and does
    not touch the optimiser, so the selected model is unchanged; --selfcheck
    verifies this numerically on one checkpoint before the main run.

SEEDS (disjoint from every seed used before)
  labels permutation : s + 30000 + r
  classifier         : s + 40000 + r
  real-subset null   : same N_MATCHED subset as v2 (RandomState(s)), same offsets

Usage:
    python evaluate_null_baseline.py                 # R=5, all 42 checkpoints
    python evaluate_null_baseline.py --reps 10       # tighter null
    python evaluate_null_baseline.py --stats-only    # Part B only, seconds
Runtime: R=5 is 210 synthetic + 15 real-subset classifier runs. Expect
roughly the time of evaluate_downstream_v2.py x 1.5 on the same GPU.
"""

import os
import sys
import json
import time
import inspect
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from medmnist import PathMNIST
from dcgan import Generator, LATENT_DIM as U_LATENT
from cgan import ConditionalGenerator, LATENT_DIM as C_LATENT, NUM_CLASSES

parser = argparse.ArgumentParser()
parser.add_argument("--reps", type=int, default=5, help="null classifiers per checkpoint")
parser.add_argument("--stats-only", action="store_true", help="Part B only (no GPU work)")
parser.add_argument("--selfcheck", action="store_true",
                    help="verify trace == original training on one checkpoint, then exit")
parser.add_argument("--configs", nargs="*", default=None, help="subset of config tags")
# smoke-test overrides — never use for the real run
parser.add_argument("--data-dir", default="./data")
parser.add_argument("--ckpt-dir", default="./checkpoints")
parser.add_argument("--log-dir", default="./logs")
parser.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)
args = parser.parse_args()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DATA_DIR, CKPT_DIR, LOG_DIR = args.data_dir, args.ckpt_dir, args.log_dir
SEEDS      = [42, 123, 2024]
BATCH      = 128
N_SYNTH    = 10000
N_MATCHED  = 10000
EPOCHS     = 15
LR         = 1e-3
if args.smoke:
    N_SYNTH, N_MATCHED, EPOCHS = 300, 300, 3
V2_SCRIPT  = "evaluate_downstream_v2.py"
IN_FILE    = os.path.join(LOG_DIR, "final_results_v2.json")
OUT_FILE   = os.path.join(LOG_DIR, "null_baseline.json")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PERM_OFF, CLF_OFF = 30000, 40000

CONFIGS = [
    ("central_gan",                "Centralized GAN (15k steps)",      False),
    ("central_gan_3k",             "Centralized GAN (3k steps)",       False),
    ("iid",                        "GAN IID (Sync-G)",                 False),
    ("non_iid",                    "GAN Non-IID (Sync-G, frozen)",     False),
    ("gan_non_iid_perseed_syncG",  "GAN Non-IID (Sync-G, per-seed)",   False),
    ("noisy_client",               "GAN Noisy (Sync-G)",               False),
    ("central_cgan",               "Centralized cGAN (15k steps)",     True),
    ("central_cgan_3k",            "Centralized cGAN (3k steps)",      True),
    ("cgan_iid",                   "cGAN IID (Sync-G)",                True),
    ("cgan_non_iid",               "cGAN Non-IID (Sync-G, frozen)",    True),
    ("cgan_non_iid_perseed_syncG", "cGAN Non-IID (Sync-G, per-seed)",  True),
    ("cgan_noisy_client",          "cGAN Noisy (Sync-G)",              True),
    ("cgan_iid_syncDG",            "cGAN IID (Sync D&G)",              True),
    ("cgan_non_iid_syncDG",        "cGAN Non-IID (Sync D&G, frozen)",  True),
]


# ═════════════ VERBATIM COPIES from evaluate_downstream_v2.py ═════════════════
def seed_all(s):
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)


def summ(v):
    v = [float(x) for x in v]
    return {"values": v, "n": len(v),
            "mean": float(np.mean(v)) if v else None,
            "std": float(np.std(v, ddof=1)) if len(v) > 1 else None}


def load_split(split):
    ds = PathMNIST(split=split, download=False, root=DATA_DIR)
    x = torch.from_numpy(np.asarray(ds.imgs))          # N,28,28,3 uint8
    if x.dim() == 3:                                     # safety: grayscale
        x = x.unsqueeze(-1).repeat(1, 1, 1, 3)
    x = x.permute(0, 3, 1, 2).contiguous()               # N,3,28,28 uint8
    y = torch.from_numpy(np.asarray(ds.labels).reshape(-1)).long()
    return x, y


def to_float(xb):
    return xb.float().div_(255.0) if xb.dtype == torch.uint8 else xb


class TissueClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((3, 3)))
        self.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(128 * 9, 256), nn.ReLU(True),
            nn.Dropout(0.3), nn.Linear(256, NUM_CLASSES))

    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))


@torch.no_grad()
def evaluate(model, X, Y):
    model.eval()
    preds = []
    for i in range(0, len(X), 512):
        preds.append(model(to_float(X[i:i + 512].to(DEVICE))).argmax(1).cpu())
    pred = torch.cat(preds)
    acc = (pred == Y).float().mean().item() * 100
    per_class = [((pred[Y == c] == c).float().mean().item() * 100) if (Y == c).any()
                 else 0.0 for c in range(NUM_CLASSES)]
    return acc, per_class, pred


def train_classifier(X, Y, seed, Xval, Yval):
    """Max EPOCHS epochs; returns the model restored to its best-validation epoch."""
    seed_all(seed)
    gen = torch.Generator().manual_seed(seed)
    model = TissueClassifier().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    sched = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
    crit = nn.CrossEntropyLoss()

    best_val, best_epoch, best_state = -1.0, 0, None
    n = len(X)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            if len(idx) < 2:          # BatchNorm cannot train on one sample
                continue
            xb = to_float(X[idx].to(DEVICE))
            yb = Y[idx].to(DEVICE)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        sched.step()

        val_acc, _, _ = evaluate(model, Xval, Yval)
        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_epoch, best_val


@torch.no_grad()
def generate(G, conditional, n, seed):
    """Images in [0,1] and round-robin labels (exact for cGAN)."""
    seed_all(seed)
    G.eval()
    imgs, lbls, done = [], [], 0
    while done < n:
        bs = min(BATCH, n - done)
        lab = torch.arange(done, done + bs) % NUM_CLASSES
        if conditional:
            out = G(torch.randn(bs, C_LATENT, device=DEVICE), lab.to(DEVICE))
        else:
            out = G(torch.randn(bs, U_LATENT, 1, 1, device=DEVICE))
        imgs.append(((out + 1) / 2).clamp(0, 1).cpu())
        lbls.append(lab.long())
        done += bs
    return torch.cat(imgs), torch.cat(lbls)
# ═══════════════════════════ end verbatim copies ══════════════════════════════

VERBATIM = [seed_all, summ, load_split, to_float, TissueClassifier, evaluate,
            train_classifier, generate]


def check_verbatim():
    """Refuse to run if any copied function differs from evaluate_downstream_v2.py."""
    if not os.path.exists(V2_SCRIPT):
        raise SystemExit(f"{V2_SCRIPT} not found next to this script.")
    with open(V2_SCRIPT) as f:
        src = f.read()
    drift = [fn.__name__ for fn in VERBATIM if inspect.getsource(fn) not in src]
    if drift:
        raise SystemExit(f"Copied code drifted from {V2_SCRIPT}: {drift}. Re-copy before running.")
    print(f"Verbatim check passed: {len(VERBATIM)} functions identical to {V2_SCRIPT}")


def train_classifier_trace(X, Y, seed, Xval, Yval, Xte, Yte):
    """train_classifier() + test accuracy logged after each epoch's val step."""
    seed_all(seed)
    gen = torch.Generator().manual_seed(seed)
    model = TissueClassifier().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    sched = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
    crit = nn.CrossEntropyLoss()

    best_val, best_epoch, best_state = -1.0, 0, None
    test_curve, val_curve = [], []
    n = len(X)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            if len(idx) < 2:
                continue
            xb = to_float(X[idx].to(DEVICE))
            yb = Y[idx].to(DEVICE)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        sched.step()

        val_acc, _, _ = evaluate(model, Xval, Yval)
        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        test_acc, _, _ = evaluate(model, Xte, Yte)          # logging only
        val_curve.append(val_acc)
        test_curve.append(test_acc)

    model.load_state_dict(best_state)
    sel_acc, _, sel_pred = evaluate(model, Xte, Yte)
    return {"selected_acc": sel_acc, "best_epoch": best_epoch, "best_val": best_val,
            "final_acc": test_curve[-1], "epoch1_acc": test_curve[0],
            "test_curve": test_curve, "val_curve": val_curve,
            "selected_pred_counts": np.bincount(sel_pred.numpy(), minlength=NUM_CLASSES).tolist()}


def null_labels(n, seed):
    """Balanced labels (same multiset as round-robin), randomly permuted."""
    lab = torch.arange(n) % NUM_CLASSES
    return lab[torch.randperm(n, generator=torch.Generator().manual_seed(seed))].long()


def pool_stats(v):
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return None
    return {"n": int(v.size), "mean": float(v.mean()),
            "std": float(v.std(ddof=1)) if v.size > 1 else None,
            "min": float(v.min()), "p05": float(np.percentile(v, 5)),
            "p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)),
            "max": float(v.max())}


# ── Part B: Welch + Spearman (no GPU) ─────────────────────────────────────────
def welch_block(v2):
    def values(tag, metric):
        if tag in v2.get("references", {}):
            return v2["references"][tag]["values"]
        return v2["configs"][tag]["summary"][metric]["values"]

    rows = []
    for d in v2["derived"]:
        if d.get("status") == "incomplete":
            continue
        a, b = np.asarray(values(d["a"], d["metric"])), np.asarray(values(d["b"], d["metric"]))
        res = stats.ttest_ind(a, b, equal_var=False)
        va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
        df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
        rows.append({"label": d["label"], "a": d["a"], "b": d["b"], "metric": d["metric"],
                     "difference": float(a.mean() - b.mean()),
                     "welch_t": float(res.statistic), "welch_df": float(df),
                     "welch_p": float(res.pvalue),
                     "one_std_intervals_overlap": d.get("one_std_intervals_overlap")})

    sp = {}
    for key, block in v2.get("spearman_fid_vs_acc", {}).items():
        if not block:
            continue
        tags = block["configs"]
        fid = [v2["configs"][t]["summary"]["fid"]["mean"] for t in tags]
        acc = [v2["configs"][t]["summary"]["acc"]["mean"] for t in tags]
        rho = stats.spearmanr(fid, acc).correlation
        perm = stats.permutation_test((acc,), lambda y: stats.spearmanr(fid, y).correlation,
                                      permutation_type="pairings", n_resamples=np.inf
                                      if len(tags) <= 8 else 20000)
        n = len(tags)
        z, se = np.arctanh(np.clip(rho, -0.999, 0.999)), 1.06 / np.sqrt(max(n - 3, 1))
        sp[key] = {"rho": float(rho), "n": n, "exact_null_p": float(perm.pvalue),
                   "fisher_z_95ci": [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))],
                   "note": "Fisher-z with Bonett-Wright SE 1.06/sqrt(n-3); rough at n<10"}
    return rows, sp


def print_welch(rows, sp):
    print("\n" + "=" * 100)
    print(" PART B — Welch t-tests on derived differences (n=3 per arm; descriptive, for sorting)")
    print("=" * 100)
    print(f"  {'Comparison':<52} {'diff':>8} {'t':>7} {'df':>5} {'p':>7}  1-sd flag")
    for r in rows:
        flag = {True: "overlap", False: "separated", None: "n/a"}[r["one_std_intervals_overlap"]]
        print(f"  {r['label']:<52} {r['difference']:+8.2f} {r['welch_t']:+7.2f} "
              f"{r['welch_df']:5.1f} {r['welch_p']:7.3f}  {flag}")
    for k, s in sp.items():
        print(f"\n  Spearman {k}: rho = {s['rho']:+.3f} (n={s['n']}), exact null p = "
              f"{s['exact_null_p']:.3f}, rough 95% CI [{s['fisher_z_95ci'][0]:+.2f}, "
              f"{s['fisher_z_95ci'][1]:+.2f}]")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    if not os.path.exists(IN_FILE):
        raise SystemExit(f"{IN_FILE} not found — run evaluate_downstream_v2.py first.")
    with open(IN_FILE) as f:
        v2 = json.load(f)

    welch_rows, spear = welch_block(v2)
    print_welch(welch_rows, spear)

    out = {"protocol": {
               "purpose": "output of the downstream instrument when training labels carry "
                          "no information about image content",
               "labels": "balanced label vector (i % 9) randomly permuted; seed s+30000+r",
               "classifier": "verbatim train_classifier from evaluate_downstream_v2.py "
                             "(+ per-epoch test logging); seed s+40000+r",
               "synthetic_sets": "regenerated exactly as in evaluate_downstream_v2.py "
                                 "(seed s+1, N_SYNTH images)",
               "real_null": "Real-matched-10k subset (RandomState(s)), labels permuted",
               "reps_per_checkpoint": args.reps, "seeds": SEEDS, "epochs": EPOCHS,
               "n_synth": N_SYNTH, "smoke": bool(args.smoke),
               "welch": "scipy ttest_ind(equal_var=False) on the per-generator-seed values "
                        "stored in final_results_v2.json; n=3 per arm; descriptive only"},
           "welch": welch_rows, "spearman": spear}

    if args.stats_only:
        with open(OUT_FILE, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved (stats only) -> {OUT_FILE}")
        return

    check_verbatim()
    Xtr, Ytr = load_split("train")
    Xva, Yva = load_split("val")
    Xte, Yte = load_split("test")
    print(f"\nDevice: {DEVICE} | R = {args.reps} null classifiers per checkpoint")
    print(f"Loaded train {len(Xtr)} | val {len(Xva)} | test {len(Xte)}")

    test_share = np.bincount(Yte.numpy(), minlength=NUM_CLASSES) / len(Yte) * 100
    val_share = np.bincount(Yva.numpy(), minlength=NUM_CLASSES) / len(Yva) * 100
    out["baselines"] = {"uniform_chance": 100.0 / NUM_CLASSES,
                        "test_class_share": test_share.tolist(),
                        "test_majority_class": int(test_share.argmax()),
                        "test_majority_share": float(test_share.max()),
                        "val_class_share": val_share.tolist()}
    print(f"Uniform chance {100 / NUM_CLASSES:.1f}% | test majority class "
          f"{int(test_share.argmax())} = {test_share.max():.1f}%")

    configs = [c for c in CONFIGS if args.configs is None or c[0] in args.configs]

    # ── Self-check: trace must reproduce train_classifier exactly ──
    if args.selfcheck:
        tag, _, cond = configs[0]
        s = SEEDS[0]
        G = (ConditionalGenerator() if cond else Generator()).to(DEVICE)
        G.load_state_dict(torch.load(os.path.join(CKPT_DIR, f"generator_{tag}_seed{s}_final.pt"),
                                     map_location=DEVICE))
        X, _ = generate(G, cond, N_SYNTH, seed=s + 1)
        Y = null_labels(len(X), s + PERM_OFF)
        m, ep, _ = train_classifier(X, Y, s + CLF_OFF, Xva, Yva)
        a_orig, _, _ = evaluate(m, Xte, Yte)
        tr = train_classifier_trace(X, Y, s + CLF_OFF, Xva, Yva, Xte, Yte)
        ok = abs(a_orig - tr["selected_acc"]) < 1e-9 and ep == tr["best_epoch"]
        print(f"\nSelf-check on {tag} seed {s}: original {a_orig:.4f}% @ epoch {ep} | "
              f"trace {tr['selected_acc']:.4f}% @ epoch {tr['best_epoch']} -> "
              f"{'IDENTICAL' if ok else 'MISMATCH'}")
        raise SystemExit(0 if ok else 1)

    # ── Generator-free null: real images, permuted labels ──
    print("\n== Real-matched-10k images, permuted labels ==")
    real_runs = []
    for s in SEEDS:
        idx = torch.from_numpy(np.random.RandomState(s).choice(len(Xtr), N_MATCHED, replace=False))
        Xr = Xtr[idx]
        for r in range(args.reps):
            tr = train_classifier_trace(Xr, null_labels(len(Xr), s + PERM_OFF + r),
                                        s + CLF_OFF + r, Xva, Yva, Xte, Yte)
            tr.update({"seed": s, "rep": r})
            real_runs.append(tr)
        sel = [x["selected_acc"] for x in real_runs if x["seed"] == s]
        print(f"  seed {s}: selected {[f'{a:.1f}' for a in sel]}")
    out["real_null"] = {"runs": real_runs,
                        "selected": pool_stats([x["selected_acc"] for x in real_runs]),
                        "final_epoch": pool_stats([x["final_acc"] for x in real_runs]),
                        "best_epoch": pool_stats([x["best_epoch"] for x in real_runs])}

    # ── Synthetic nulls ──
    out["configs"] = {}
    all_sel, all_fin = [], []
    for tag, name, cond in configs:
        print(f"\n== {name} ==")
        runs, missing = [], []
        for s in SEEDS:
            ckpt = os.path.join(CKPT_DIR, f"generator_{tag}_seed{s}_final.pt")
            if not os.path.exists(ckpt):
                missing.append(s)
                print(f"  seed {s}: MISSING {ckpt}")
                continue
            G = (ConditionalGenerator() if cond else Generator()).to(DEVICE)
            G.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            X, _ = generate(G, cond, N_SYNTH, seed=s + 1)
            for r in range(args.reps):
                tr = train_classifier_trace(X, null_labels(len(X), s + PERM_OFF + r),
                                            s + CLF_OFF + r, Xva, Yva, Xte, Yte)
                tr.update({"seed": s, "rep": r})
                runs.append(tr)
            sel = ["%.1f" % x["selected_acc"] for x in runs if x["seed"] == s]
            fin = ["%.1f" % x["final_acc"] for x in runs if x["seed"] == s]
            print(f"  seed {s}: selected {sel} | final-epoch {fin}")

        summ_v2 = v2["configs"][tag]["summary"]
        observed = summ_v2["acc"] if cond else summ_v2["pseudo_acc"]
        sel_stats = pool_stats([x["selected_acc"] for x in runs])
        obs_vals = observed.get("values", [])
        out["configs"][tag] = {
            "name": name, "conditional": cond, "missing_seeds": missing, "runs": runs,
            "null_selected": sel_stats,
            "null_final_epoch": pool_stats([x["final_acc"] for x in runs]),
            "null_epoch1": pool_stats([x["epoch1_acc"] for x in runs]),
            "null_best_epoch": pool_stats([x["best_epoch"] for x in runs]),
            "observed_metric": "acc" if cond else "pseudo_acc",
            "observed_seed_values": obs_vals,
            "observed_seeds_above_null_max": (sum(v > sel_stats["max"] for v in obs_vals)
                                              if sel_stats else None),
            "observed_seeds_above_null_p95": (sum(v > sel_stats["p95"] for v in obs_vals)
                                              if sel_stats else None),
        }
        all_sel += [x["selected_acc"] for x in runs]
        all_fin += [x["final_acc"] for x in runs]

    out["pooled_synthetic_null"] = {"selected": pool_stats(all_sel),
                                    "final_epoch": pool_stats(all_fin)}

    # ── Print ──
    print("\n" + "=" * 100)
    print(" PART A — downstream instrument under permuted (content-free) labels")
    print("=" * 100)
    b = out["baselines"]
    print(f"  Uniform chance {b['uniform_chance']:.1f}% | majority class {b['test_majority_share']:.1f}%")
    rn = out["real_null"]
    print(f"  Real images, permuted labels : selected {rn['selected']['mean']:.1f} ± "
          f"{rn['selected']['std']:.1f} (max {rn['selected']['max']:.1f}) | final-epoch "
          f"{rn['final_epoch']['mean']:.1f} ± {rn['final_epoch']['std']:.1f}")
    print(f"\n  {'Configuration':<34} {'null sel mean±sd':>17} {'p95':>6} {'max':>6} "
          f"{'final-ep':>9} {'observed (seed means)':>24} {'>p95':>5} {'>max':>5}")
    for tag, c in out["configs"].items():
        ns, nf = c["null_selected"], c["null_final_epoch"]
        if not ns:
            print(f"  {c['name']:<34} no runs")
            continue
        sd = f"{ns['std']:.1f}" if ns["std"] is not None else "—"
        obs = " / ".join(f"{v:.1f}" for v in c["observed_seed_values"])
        print(f"  {c['name']:<34} {ns['mean']:>9.1f} ± {sd:<5} {ns['p95']:>6.1f} {ns['max']:>6.1f} "
              f"{nf['mean']:>9.1f} {obs:>24} {c['observed_seeds_above_null_p95']:>3}/3 "
              f"{c['observed_seeds_above_null_max']:>3}/3")
    ps = out["pooled_synthetic_null"]["selected"]
    if ps:
        print(f"\n  Pooled synthetic null (n={ps['n']}): selected {ps['mean']:.1f} ± {ps['std']:.1f}, "
              f"p95 {ps['p95']:.1f}, max {ps['max']:.1f}")
    print("  Observed values are K=3 averages; null values are single classifiers, so "
          "'> p95' is a conservative test for signal.")

    with open(OUT_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {OUT_FILE}   (total {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
