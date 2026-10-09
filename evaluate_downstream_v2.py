"""
Downstream Evaluation v2 — validation-selected, multi-classifier
FedGAN Failure Analysis - Anish Bharadwaj

Fixes the last protocol problem left after evaluate_final.py: the
measuring instrument itself was noisy. The real-full reference classifier,
trained on IDENTICAL data with seeding and cuDNN determinism on, scored
76.13 / 71.04 / 83.50 % across three seeds — and seed 123 scored BELOW the
10k-image classifier. Fixed 15 epochs with no validation split lets each run
stop at an arbitrary point on a noisy trajectory.

Changes vs evaluate_final.py (downstream half only):
  1. Best-epoch selection on PathMNIST's official VALIDATION split
     (10,004 images). Test accuracy is reported at the epoch with the best
     validation accuracy. The test set is never used for selection.
  2. K classifier seeds per checkpoint (default K=3) on the SAME synthetic
     set. Per-checkpoint accuracy = mean over the K classifiers. This
     separates two variance sources:
       - instrument noise  : spread across classifier seeds, same data
       - generator variance: spread across the 3 generator seeds
     Reported mean ± std is across GENERATOR seeds (of the K-averaged
     values), so it measures what the paper is about.
  3. Data preloaded as uint8 tensors (identical pixels to ToTensor) so the
     extra classifier runs stay fast.

Unchanged:
  - FID: copied verbatim from logs/final_results.json (pytorch-fid, all
    7,180 test images). Not recomputed.
  - Synthetic sets: generated with the same seeds as evaluate_final.py
    (seed s+1, 10,000 images, round-robin labels — exact for cGAN).
  - Classifier architecture, optimiser, LR schedule, batch size, max epochs.
  - Pseudo-labeler = real-full classifier for generator seed s (k=0);
    downstream classifiers use different seeds (s+20000+k) — no shared init.
  - Sample std (ddof=1) throughout.

Round-robin labelling for unconditional generators is a demonstrated
protocol artifact, so it gets ONE classifier (k=0) — only to show chance.

Output: logs/final_results_v2.json   (final_results.json is NOT modified)

Usage:
    python evaluate_downstream_v2.py            # K=3
    python evaluate_downstream_v2.py --k 5      # tighter instrument estimate
Runtime: roughly 30-60 min on an RTX 3060 Laptop GPU at K=3
"""

import os
import json
import time
import copy
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
parser.add_argument("--k", type=int, default=3,
                    help="classifier seeds per checkpoint (instrument repeats)")
args = parser.parse_args()
K = args.k

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DATA_DIR, CKPT_DIR, LOG_DIR = "./data", "./checkpoints", "./logs"
SEEDS      = [42, 123, 2024]
BATCH      = 128
N_SYNTH    = 10000
N_MATCHED  = 10000
EPOCHS     = 15
LR         = 1e-3
IN_FILE    = os.path.join(LOG_DIR, "final_results.json")
OUT_FILE   = os.path.join(LOG_DIR, "final_results_v2.json")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# (checkpoint tag, display name, conditional) — identical to evaluate_final.py
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


def seed_all(s):
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)


def summ(v):
    v = [float(x) for x in v]
    return {"values": v, "n": len(v),
            "mean": float(np.mean(v)) if v else None,
            "std": float(np.std(v, ddof=1)) if len(v) > 1 else None}


# ── Data (uint8 CHW tensors; identical pixels to transforms.ToTensor) ────────
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


# ── Classifier (architecture identical to evaluate_final.py) ─────────────────
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


def label_entropy(labels):
    p = np.bincount(labels.numpy(), minlength=NUM_CLASSES).astype(float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log(p)).sum() / np.log(NUM_CLASSES))


# ── Generation (identical seeds and procedure to evaluate_final.py) ──────────
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


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    if not os.path.exists(IN_FILE):
        raise SystemExit(f"{IN_FILE} not found — run evaluate_final.py first.")
    with open(IN_FILE) as f:
        prev = json.load(f)
    print(f"Device: {DEVICE} | K = {K} classifier seeds per checkpoint\n")

    Xtr, Ytr = load_split("train")
    Xva, Yva = load_split("val")
    Xte, Yte = load_split("test")
    print(f"Loaded train {len(Xtr)} | val {len(Xva)} | test {len(Xte)}\n")

    # ── References ──
    print("== Reference classifiers (validation-selected) ==")
    ref = {"real_full": {}, "real_matched_10k": {}}
    ref_per_class, ref_epochs, labelers = [], {"real_full": [], "real_matched_10k": []}, {}
    for s in SEEDS:
        idx = np.random.RandomState(s).choice(len(Xtr), N_MATCHED, replace=False)
        idx = torch.from_numpy(idx)
        ref["real_full"][s], ref["real_matched_10k"][s] = [], []
        for k in range(K):
            cs = s + 1000 * k
            m, ep, _ = train_classifier(Xtr, Ytr, cs, Xva, Yva)
            acc, pc, _ = evaluate(m, Xte, Yte)
            ref["real_full"][s].append(acc)
            ref_epochs["real_full"].append(ep)
            ref_per_class.append(pc)
            if k == 0:
                labelers[s] = m
            m2, ep2, _ = train_classifier(Xtr[idx], Ytr[idx], cs, Xva, Yva)
            acc2, _, _ = evaluate(m2, Xte, Yte)
            ref["real_matched_10k"][s].append(acc2)
            ref_epochs["real_matched_10k"].append(ep2)
        print(f"  seed {s}: real-full {[f'{a:.2f}' for a in ref['real_full'][s]]} | "
              f"matched-10k {[f'{a:.2f}' for a in ref['real_matched_10k'][s]]}")

    def collapse(d):
        """dict seed -> list of K values  ->  (summary over seed means, instrument sd)"""
        means = [float(np.mean(d[s])) for s in SEEDS if d.get(s)]
        inst = [float(np.std(d[s], ddof=1)) for s in SEEDS if len(d.get(s, [])) > 1]
        return summ(means), (float(np.mean(inst)) if inst else None)

    REF, REF_INST = {}, {}
    for key in ("real_full", "real_matched_10k"):
        REF[key], REF_INST[key] = collapse(ref[key])
    all_full = [a for s in SEEDS for a in ref["real_full"][s]]

    # ── Synthetic ──
    results = {}
    for tag, name, cond in CONFIGS:
        print(f"\n== {name} ==")
        r = {"name": name, "conditional": cond, "missing_seeds": [],
             "acc": {}, "per_class": {}, "best_epochs": [],
             "roundrobin_acc": {}, "pseudo_acc": {},
             "pseudo_entropy": [], "pseudo_label_counts": []}
        for s in SEEDS:
            ckpt = os.path.join(CKPT_DIR, f"generator_{tag}_seed{s}_final.pt")
            if not os.path.exists(ckpt):
                r["missing_seeds"].append(s)
                print(f"  seed {s}: MISSING {ckpt}")
                continue
            G = (ConditionalGenerator() if cond else Generator()).to(DEVICE)
            G.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            simgs, slbls = generate(G, cond, N_SYNTH, seed=s + 1)

            if cond:
                accs, pcs = [], []
                for k in range(K):
                    m, ep, _ = train_classifier(simgs, slbls, s + 10000 + k, Xva, Yva)
                    a, pc, _ = evaluate(m, Xte, Yte)
                    accs.append(a); pcs.append(pc); r["best_epochs"].append(ep)
                r["acc"][s] = accs
                r["per_class"][s] = np.mean(pcs, axis=0).tolist()
                print(f"  seed {s}: acc {[f'{a:.2f}' for a in accs]} -> mean {np.mean(accs):.2f}%")
            else:
                m, ep, _ = train_classifier(simgs, slbls, s + 10000, Xva, Yva)
                rr, _, _ = evaluate(m, Xte, Yte)
                r["roundrobin_acc"][s] = [rr]
                _, _, pl = evaluate(labelers[s], simgs, slbls)   # labels = labeler predictions
                ent = label_entropy(pl)
                r["pseudo_entropy"].append(ent)
                r["pseudo_label_counts"].append(
                    np.bincount(pl.numpy(), minlength=NUM_CLASSES).tolist())
                accs = []
                for k in range(K):
                    m, ep, _ = train_classifier(simgs, pl, s + 20000 + k, Xva, Yva)
                    a, _, _ = evaluate(m, Xte, Yte)
                    accs.append(a); r["best_epochs"].append(ep)
                r["pseudo_acc"][s] = accs
                print(f"  seed {s}: round-robin {rr:.2f}% | pseudo {[f'{a:.2f}' for a in accs]}"
                      f" -> mean {np.mean(accs):.2f}% | entropy {ent:.3f}")
        results[tag] = r

    # ── Summaries: FID from final_results.json, downstream from this run ──
    S, INST = {}, {}
    for tag, _, cond in CONFIGS:
        r = results[tag]
        prev_sum = prev["configs"].get(tag, {}).get("summary", {})
        acc_s, acc_i = collapse(r["acc"])
        ps_s, ps_i = collapse(r["pseudo_acc"])
        rr_s, _ = collapse(r["roundrobin_acc"])
        S[tag] = {"fid": prev_sum.get("fid", summ([])),
                  "acc": acc_s, "pseudo_acc": ps_s, "roundrobin_acc": rr_s,
                  "pseudo_entropy": summ(r["pseudo_entropy"])}
        INST[tag] = acc_i if cond else ps_i
    for k in REF:
        S[k] = {"acc": REF[k]}

    def diff(label, a, b, metric):
        A, B = S[a][metric], S[b][metric]
        if not A["n"] or not B["n"]:
            return {"label": label, "status": "incomplete"}
        overlap = None
        if A["std"] is not None and B["std"] is not None:
            overlap = not (A["mean"] - A["std"] > B["mean"] + B["std"] or
                           B["mean"] - B["std"] > A["mean"] + A["std"])
        return {"label": label, "a": a, "b": b, "metric": metric,
                "a_mean": A["mean"], "b_mean": B["mean"],
                "difference": A["mean"] - B["mean"],
                "one_std_intervals_overlap": overlap}

    derived = [
        diff("Federation cost vs 15k, GAN FID", "iid", "central_gan", "fid"),
        diff("Federation cost vs 3k,  GAN FID", "iid", "central_gan_3k", "fid"),
        diff("Federation cost vs 15k, cGAN FID", "cgan_iid", "central_cgan", "fid"),
        diff("Federation cost vs 3k,  cGAN FID", "cgan_iid", "central_cgan_3k", "fid"),
        diff("Federation cost vs 15k, cGAN acc", "cgan_iid", "central_cgan", "acc"),
        diff("Federation cost vs 3k,  cGAN acc", "cgan_iid", "central_cgan_3k", "acc"),
        diff("Federation cost vs 15k, GAN pseudo acc", "iid", "central_gan", "pseudo_acc"),
        diff("Federation cost vs 3k,  GAN pseudo acc", "iid", "central_gan_3k", "pseudo_acc"),
        diff("Non-IID effect, Sync-G frozen, cGAN FID", "cgan_non_iid", "cgan_iid", "fid"),
        diff("Non-IID effect, Sync-G frozen, cGAN acc", "cgan_non_iid", "cgan_iid", "acc"),
        diff("Non-IID effect, Sync-G per-seed, cGAN FID", "cgan_non_iid_perseed_syncG", "cgan_iid", "fid"),
        diff("Non-IID effect, Sync-G per-seed, cGAN acc", "cgan_non_iid_perseed_syncG", "cgan_iid", "acc"),
        diff("Non-IID effect, Sync D&G, cGAN FID", "cgan_non_iid_syncDG", "cgan_iid_syncDG", "fid"),
        diff("Non-IID effect, Sync D&G, cGAN acc", "cgan_non_iid_syncDG", "cgan_iid_syncDG", "acc"),
        diff("Aggregation effect (D&G - G), IID, cGAN FID", "cgan_iid_syncDG", "cgan_iid", "fid"),
        diff("Aggregation effect (D&G - G), IID, cGAN acc", "cgan_iid_syncDG", "cgan_iid", "acc"),
        diff("Aggregation effect (D&G - G), Non-IID, cGAN FID", "cgan_non_iid_syncDG", "cgan_non_iid", "fid"),
        diff("Aggregation effect (D&G - G), Non-IID, cGAN acc", "cgan_non_iid_syncDG", "cgan_non_iid", "acc"),
        diff("Non-IID effect, GAN pseudo acc (frozen)", "non_iid", "iid", "pseudo_acc"),
        diff("Noisy-client effect, GAN FID", "noisy_client", "iid", "fid"),
        diff("Noisy-client effect, cGAN FID", "cgan_noisy_client", "cgan_iid", "fid"),
        diff("Noisy-client effect, cGAN acc", "cgan_noisy_client", "cgan_iid", "acc"),
        diff("Data-quantity effect (full - matched)", "real_full", "real_matched_10k", "acc"),
        diff("Quality gap (cGAN IID - matched real)", "cgan_iid", "real_matched_10k", "acc"),
        diff("Quality gap (centralized cGAN 15k - matched real)", "central_cgan", "real_matched_10k", "acc"),
    ]

    cond_tags = [t for t, _, c in CONFIGS if c and S[t]["fid"].get("n") and S[t]["acc"]["n"]]
    fed_tags = [t for t in cond_tags if not t.startswith("central")]
    def spear(tags):
        if len(tags) < 3:
            return None
        rho = stats.spearmanr([S[t]["fid"]["mean"] for t in tags],
                              [S[t]["acc"]["mean"] for t in tags]).correlation
        return {"rho": float(rho), "configs": tags}
    rho_all, rho_fed = spear(cond_tags), spear(fed_tags)

    # ── Print ──
    def fmt(x, pct=False):
        if not x or not x.get("n"):
            return "—"
        sd = f" ± {x['std']:.2f}" if x.get("std") is not None else ""
        return f"{x['mean']:.2f}{'%' if pct else ''}{sd}"

    def old(tag, key):
        return prev["configs"].get(tag, {}).get("summary", {}).get(key, {})

    print("\n" + "=" * 104)
    print(f" FINAL RESULTS v2  (mean ± sample std across 3 GENERATOR seeds; each value = mean of "
          f"K={K} val-selected classifiers)")
    print("=" * 104)
    print(f"  Real-full reference : {fmt(REF['real_full'], True)}   instrument sd "
          f"{REF_INST['real_full']:.2f}   (all {len(all_full)} runs: "
          f"{np.mean(all_full):.2f} ± {np.std(all_full, ddof=1):.2f})"
          f"   [v1: {fmt(prev['references']['real_full'], True)}]")
    print(f"  Real-matched-10k    : {fmt(REF['real_matched_10k'], True)}   instrument sd "
          f"{REF_INST['real_matched_10k']:.2f}"
          f"   [v1: {fmt(prev['references']['real_matched_10k'], True)}]")
    print(f"  Best-epoch range    : real-full {min(ref_epochs['real_full'])}-"
          f"{max(ref_epochs['real_full'])}, matched {min(ref_epochs['real_matched_10k'])}-"
          f"{max(ref_epochs['real_matched_10k'])} (of {EPOCHS})\n")

    print(f"  {'Configuration':<34} {'FID':>16} {'Downstream v2':>18} {'inst sd':>8} "
          f"{'[v1 downstream]':>18}")
    for tag, name, cond in CONFIGS:
        s = S[tag]
        if cond:
            new, v1 = fmt(s["acc"], True), fmt(old(tag, "acc"), True)
        else:
            new, v1 = "PL " + fmt(s["pseudo_acc"], True), "PL " + fmt(old(tag, "pseudo_acc"), True)
        inst = f"{INST[tag]:.2f}" if INST[tag] is not None else "—"
        print(f"  {name:<34} {fmt(s['fid']):>16} {new:>18} {inst:>8} {v1:>18}")
    print("  PL = pseudo-labeled (unconditional). Round-robin results are in the JSON;"
          " they remain a chance-level protocol artifact.")

    print("\n  Derived differences (a - b). 'overlap' = mean ± 1 sample std intervals overlap.")
    for d in derived:
        if d.get("status") == "incomplete":
            print(f"  {d['label']:<52} incomplete")
            continue
        ov = {True: "overlap", False: "separated", None: "n/a"}[d["one_std_intervals_overlap"]]
        print(f"  {d['label']:<52} {d['difference']:+8.2f}   [{ov}]")

    if rho_all:
        print(f"\n  Spearman(FID, downstream acc), {len(cond_tags)} conditional configs: "
              f"rho = {rho_all['rho']:+.3f}  (perfect FID ranking = -1)")
    if rho_fed:
        print(f"  Spearman(FID, downstream acc), {len(fed_tags)} federated conditional configs: "
              f"rho = {rho_fed['rho']:+.3f}")
    print("  Descriptive only — config means, small n, no significance claim.")

    out = {
        "protocol": {
            "fid": prev["protocol"]["fid"] + "  (copied from final_results.json)",
            "downstream": f"TissueClassifier, max {EPOCHS} epochs, best epoch selected on "
                          f"PathMNIST val split ({len(Xva)} images), Adam {LR}, StepLR(5,0.5), "
                          f"batch {BATCH}, {N_SYNTH} synthetic images; K={K} classifier seeds "
                          f"per checkpoint, averaged; std across generator seeds",
            "pseudo_label": "labeler = val-selected real-full classifier (seed s, k=0); "
                            "downstream seeds s+20000+k",
            "std": "sample standard deviation, ddof=1, across generator seeds",
            "seeds": SEEDS, "K": K, "cudnn_deterministic": True,
        },
        "references": {k: {**REF[k], "instrument_sd": REF_INST[k],
                           "raw": {str(s): ref[k][s] for s in SEEDS}} for k in REF},
        "references_best_epochs": ref_epochs,
        "real_full_per_class_mean": np.mean(ref_per_class, axis=0).tolist(),
        "configs": {tag: {**{k: v for k, v in results[tag].items()
                             if k not in ("acc", "pseudo_acc", "roundrobin_acc", "per_class")},
                          "acc_raw": {str(s): v for s, v in results[tag]["acc"].items()},
                          "pseudo_acc_raw": {str(s): v for s, v in results[tag]["pseudo_acc"].items()},
                          "roundrobin_acc_raw": {str(s): v for s, v in results[tag]["roundrobin_acc"].items()},
                          "per_class_mean_over_K": {str(s): v for s, v in results[tag]["per_class"].items()},
                          "instrument_sd": INST[tag],
                          "summary": S[tag]}
                    for tag in results},
        "derived": derived,
        "spearman_fid_vs_acc": {"all_conditional": rho_all, "federated_conditional": rho_fed},
        "training_divergence": prev.get("training_divergence", {}),
    }
    with open(OUT_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {OUT_FILE}   (total {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
