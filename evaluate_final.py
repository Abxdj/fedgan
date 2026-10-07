"""
Unified Final Evaluation — one reference, one protocol, every checkpoint
FedGAN Failure Analysis - Anish Bharadwaj

Replaces, for all reported numbers:
  fid_multiseed.py, silent_failure_multiseed.py, evaluate_centralized.py,
  evaluate_blocking.py, pseudo_label_eval.py, reference_classifiers.py

Fixes from REVIEW_REPORT.md:
  B1  One FID reference: ALL 7,180 test images, computed once, cached,
      shared by every checkpoint. No random subsets.
  B2  Standard FID: pytorch-fid's InceptionV3 (FID weights, 299 resize,
      [0,1] -> [-1,1] scaling). Values are comparable to published
      pytorch-fid numbers (still domain-shifted on 28x28 histology).
  N6  pytorch-fid's calculate_frechet_distance (finite check + eps offset).
  S1/S5  Every classifier trained through ONE seeded function; synthetic
      sets generated from a seeded RNG; cuDNN deterministic.
  S6  Pseudo-labeler seed s, downstream classifier seed s+20000 — no
      shared initialization.
  S8  All std are SAMPLE std (ddof=1).
  B3/C3  Pseudo-labeled evaluation also for the 3k centralized GAN, so the
      unconditional federation cost is bracketed too.
  S3  No automated verdict text. Divergence reported as numbers: early->late
      delta, second-half slope, and round-to-round noise scale.

Output: logs/final_results.json (every per-seed value + summaries + derived)

Requires: pip install pytorch-fid
Usage:    python evaluate_final.py
Runtime:  roughly 1 - 1.5 h on an RTX 3060 Laptop GPU
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import transforms
from medmnist import PathMNIST
from scipy import stats
try:
    from pytorch_fid.inception import InceptionV3
    from pytorch_fid.fid_score import calculate_frechet_distance
except ImportError:
    raise SystemExit("pytorch-fid not installed. Run: pip install pytorch-fid")
from dcgan import Generator, LATENT_DIM as U_LATENT
from cgan import ConditionalGenerator, LATENT_DIM as C_LATENT, NUM_CLASSES

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DATA_DIR, CKPT_DIR, LOG_DIR = "./data", "./checkpoints", "./logs"
SEEDS      = [42, 123, 2024]
BATCH      = 128
N_SYNTH    = 10000
N_MATCHED  = 10000
EPOCHS     = 15
LR         = 1e-3
REF_CACHE  = os.path.join(LOG_DIR, "fid_reference_stats_test_full.npz")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# (checkpoint tag, display name, conditional)
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

# Federated configs whose per-round training logs exist
FED_LOGS = ["iid", "non_iid", "gan_non_iid_perseed_syncG", "noisy_client",
            "cgan_iid", "cgan_non_iid", "cgan_non_iid_perseed_syncG",
            "cgan_noisy_client", "cgan_iid_syncDG", "cgan_non_iid_syncDG"]


def seed_all(s):
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)


def summ(v):
    v = [float(x) for x in v]
    return {"values": v, "n": len(v),
            "mean": float(np.mean(v)) if v else None,
            "std": float(np.std(v, ddof=1)) if len(v) > 1 else None}


# ── Classifier (single definition, single training function) ─────────────────
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


def train_classifier(dataset, seed):
    seed_all(seed)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed))
    model = TissueClassifier().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    sched = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
    crit = nn.CrossEntropyLoss()
    for _ in range(EPOCHS):
        model.train()
        for x, y in loader:
            opt.zero_grad()
            crit(model(x.to(DEVICE)), y.view(-1).long().to(DEVICE)).backward()
            opt.step()
        sched.step()
    return model


@torch.no_grad()
def evaluate(model, test_loader):
    model.eval()
    correct = total = 0
    pc_c, pc_t = np.zeros(NUM_CLASSES), np.zeros(NUM_CLASSES)
    for x, y in test_loader:
        y = y.view(-1).long()
        pred = model(x.to(DEVICE)).argmax(1).cpu()
        correct += (pred == y).sum().item()
        total += y.numel()
        for c in range(NUM_CLASSES):
            m = y == c
            pc_c[c] += (pred[m] == c).sum().item()
            pc_t[c] += m.sum().item()
    return correct / total * 100, (pc_c / np.maximum(pc_t, 1) * 100).tolist()


@torch.no_grad()
def predict(model, imgs):
    model.eval()
    preds = []
    for i in range(0, len(imgs), BATCH):
        preds.append(model(imgs[i:i + BATCH].to(DEVICE)).argmax(1).cpu())
    return torch.cat(preds)


def label_entropy(labels):
    p = np.bincount(labels.numpy(), minlength=NUM_CLASSES).astype(float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log(p)).sum() / np.log(NUM_CLASSES))


# ── Generation ────────────────────────────────────────────────────────────────
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


# ── FID ───────────────────────────────────────────────────────────────────────
class FIDExtractor:
    def __init__(self):
        self.model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(DEVICE).eval()

    @torch.no_grad()
    def stats(self, imgs01):
        feats = []
        for i in range(0, len(imgs01), BATCH):
            f = self.model(imgs01[i:i + BATCH].to(DEVICE))[0]
            feats.append(f.squeeze(-1).squeeze(-1).cpu().numpy())
        f = np.concatenate(feats).astype(np.float64)
        return f.mean(0), np.cov(f, rowvar=False)


def reference_stats(fidx, test_ds):
    if os.path.exists(REF_CACHE):
        d = np.load(REF_CACHE)
        print(f"Loaded cached FID reference ({int(d['n'])} images)")
        return d["mu"], d["sigma"], int(d["n"])
    imgs = torch.stack([test_ds[i][0] for i in range(len(test_ds))])  # [0,1]
    mu, sigma = fidx.stats(imgs)
    np.savez(REF_CACHE, mu=mu, sigma=sigma, n=len(imgs))
    print(f"Computed and cached FID reference from all {len(imgs)} test images")
    return mu, sigma, len(imgs)


# ── Training-log divergence statistics ────────────────────────────────────────
def divergence_stats(tag):
    out = []
    for s in SEEDS:
        p = os.path.join(LOG_DIR, f"losses_{tag}_seed{s}.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            g = np.array(json.load(f)["g_loss"], dtype=float)
        if len(g) < 30:
            continue
        rounds = np.arange(1, len(g) + 1)
        half = rounds > len(g) // 2
        out.append({
            "seed": s,
            "early_mean_r1_5": float(g[:5].mean()),
            "late_mean_r26_30": float(g[-5:].mean()),
            "delta": float(g[-5:].mean() - g[:5].mean()),
            "delta_from_r5": float(g[-5:].mean() - g[4]),
            "slope_r16_30_per_round": float(np.polyfit(rounds[half], g[half], 1)[0]),
            "roundtoround_noise_r16_30": float(np.std(np.diff(g[half]), ddof=1)),
        })
    return out


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()
    print(f"Device: {DEVICE}\n")
    to_tensor = transforms.ToTensor()
    train_ds = PathMNIST(split="train", transform=to_tensor, download=False, root=DATA_DIR)
    test_ds  = PathMNIST(split="test",  transform=to_tensor, download=False, root=DATA_DIR)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    fidx = FIDExtractor()
    ref_mu, ref_sigma, n_ref = reference_stats(fidx, test_ds)

    # References + pseudo-labelers (the real-full classifier IS the labeler)
    print("\n== Reference classifiers ==")
    labelers, ref = {}, {"real_full": [], "real_matched_10k": [],
                         "real_full_per_class": []}
    for s in SEEDS:
        m = train_classifier(train_ds, seed=s)
        acc, pc = evaluate(m, test_loader)
        labelers[s] = m
        ref["real_full"].append(acc)
        ref["real_full_per_class"].append(pc)
        idx = np.random.RandomState(s).choice(len(train_ds), N_MATCHED, replace=False)
        acc2, _ = evaluate(train_classifier(Subset(train_ds, idx.tolist()), seed=s),
                           test_loader)
        ref["real_matched_10k"].append(acc2)
        print(f"  seed {s}: real-full {acc:.2f}% | real-matched-10k {acc2:.2f}%")

    results = {}
    for tag, name, cond in CONFIGS:
        print(f"\n== {name} ==")
        r = {"name": name, "conditional": cond, "fid": [], "missing_seeds": [],
             "acc": [], "per_class": [],                  # conditional
             "roundrobin_acc": [], "pseudo_acc": [],      # unconditional
             "pseudo_entropy": [], "pseudo_label_counts": []}
        for s in SEEDS:
            ckpt = os.path.join(CKPT_DIR, f"generator_{tag}_seed{s}_final.pt")
            if not os.path.exists(ckpt):
                r["missing_seeds"].append(s)
                print(f"  seed {s}: MISSING {ckpt}")
                continue
            G = (ConditionalGenerator() if cond else Generator()).to(DEVICE)
            G.load_state_dict(torch.load(ckpt, map_location=DEVICE))

            imgs, _ = generate(G, cond, n_ref, seed=s)
            mu, sigma = fidx.stats(imgs)
            f = float(calculate_frechet_distance(mu, sigma, ref_mu, ref_sigma))
            r["fid"].append(f)
            line = f"  seed {s}: FID {f:.2f}"

            simgs, slbls = generate(G, cond, N_SYNTH, seed=s + 1)
            if cond:
                acc, pc = evaluate(train_classifier(TensorDataset(simgs, slbls),
                                                    seed=s + 10000), test_loader)
                r["acc"].append(acc)
                r["per_class"].append(pc)
                line += f" | acc {acc:.2f}%"
            else:
                rr, _ = evaluate(train_classifier(TensorDataset(simgs, slbls),
                                                  seed=s + 10000), test_loader)
                pl = predict(labelers[s], simgs)
                pacc, _ = evaluate(train_classifier(TensorDataset(simgs, pl),
                                                    seed=s + 20000), test_loader)
                r["roundrobin_acc"].append(rr)
                r["pseudo_acc"].append(pacc)
                r["pseudo_entropy"].append(label_entropy(pl))
                r["pseudo_label_counts"].append(
                    np.bincount(pl.numpy(), minlength=NUM_CLASSES).tolist())
                line += (f" | round-robin {rr:.2f}% | pseudo {pacc:.2f}% "
                         f"| entropy {label_entropy(pl):.3f}")
            print(line)
        results[tag] = r

    # ── Summaries ──
    S = {tag: {k: summ(results[tag][k]) for k in
               ("fid", "acc", "roundrobin_acc", "pseudo_acc", "pseudo_entropy")}
         for tag in results}
    REF = {k: summ(ref[k]) for k in ("real_full", "real_matched_10k")}

    def diff(label, a, b, metric):
        A = REF[a] if a in REF else S[a][metric]
        B = REF[b] if b in REF else S[b][metric]
        if not A["n"] or not B["n"]:
            return {"label": label, "status": "incomplete"}
        d = A["mean"] - B["mean"]
        overlap = None
        if A["std"] is not None and B["std"] is not None:
            overlap = not (A["mean"] - A["std"] > B["mean"] + B["std"] or
                           B["mean"] - B["std"] > A["mean"] + A["std"])
        return {"label": label, "a": a, "b": b, "metric": metric,
                "a_mean": A["mean"], "b_mean": B["mean"], "difference": d,
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
        diff("Noisy-client effect, GAN FID", "noisy_client", "iid", "fid"),
        diff("Noisy-client effect, cGAN FID", "cgan_noisy_client", "cgan_iid", "fid"),
        diff("Data-quantity effect (full - matched)", "real_full", "real_matched_10k", "acc"),
        diff("Quality gap (cGAN IID - matched real)", "cgan_iid", "real_matched_10k", "acc"),
    ]

    # FID-vs-utility rank agreement across conditional configs
    cond_tags = [t for t, _, c in CONFIGS if c and S[t]["fid"]["n"] and S[t]["acc"]["n"]]
    fid_m = [S[t]["fid"]["mean"] for t in cond_tags]
    acc_m = [S[t]["acc"]["mean"] for t in cond_tags]
    rho_all = stats.spearmanr(fid_m, acc_m) if len(cond_tags) >= 3 else None
    fed_tags = [t for t in cond_tags if not t.startswith("central")]
    rho_fed = (stats.spearmanr([S[t]["fid"]["mean"] for t in fed_tags],
                               [S[t]["acc"]["mean"] for t in fed_tags])
               if len(fed_tags) >= 3 else None)

    divergence = {tag: divergence_stats(tag) for tag in FED_LOGS}

    # ── Print ──
    def fmt(x, pct=False):
        if not x["n"]:
            return "—"
        sd = f" ± {x['std']:.2f}" if x["std"] is not None else ""
        return f"{x['mean']:.2f}{'%' if pct else ''}{sd}"

    print("\n" + "=" * 96)
    print(" FINAL RESULTS  (mean ± SAMPLE std, ddof=1, n=3; FID = pytorch-fid vs all "
          f"{n_ref} test images)")
    print("=" * 96)
    print(f"  Real-full reference      : {fmt(REF['real_full'], True)}")
    print(f"  Real-matched-10k         : {fmt(REF['real_matched_10k'], True)}\n")
    print(f"  {'Configuration':<34} {'FID':>16} {'Downstream':>16} "
          f"{'Pseudo-labeled':>16} {'Entropy':>12}")
    for tag, name, cond in CONFIGS:
        s = S[tag]
        down = fmt(s["acc"], True) if cond else f"RR {fmt(s['roundrobin_acc'], True)}"
        print(f"  {name:<34} {fmt(s['fid']):>16} {down:>16} "
              f"{('—' if cond else fmt(s['pseudo_acc'], True)):>16} "
              f"{('—' if cond else fmt(s['pseudo_entropy'])):>12}")

    print("\n  Derived differences (a - b). 'overlap' = mean ± 1 sample std intervals overlap.")
    for d in derived:
        if d.get("status") == "incomplete":
            print(f"  {d['label']:<52} incomplete")
            continue
        ov = {True: "overlap", False: "separated", None: "n/a"}[d["one_std_intervals_overlap"]]
        print(f"  {d['label']:<52} {d['difference']:+8.2f}   [{ov}]")

    if rho_all is not None:
        print(f"\n  Spearman(FID, downstream acc), all {len(cond_tags)} conditional configs: "
              f"rho = {rho_all.correlation:+.3f}  (FID predicting utility perfectly = -1)")
    if rho_fed is not None:
        print(f"  Spearman(FID, downstream acc), {len(fed_tags)} federated conditional configs: "
              f"rho = {rho_fed.correlation:+.3f}")
    print("  Descriptive only — config means, small n, no significance claim.")

    print("\n  Training-loss trajectories (per seed):  delta = mean(r26-30) - mean(r1-5); "
          "slope over r16-30; noise = sd of round-to-round change r16-30")
    for tag in FED_LOGS:
        for d in divergence[tag]:
            print(f"  {tag:<28} seed {d['seed']:<5} delta {d['delta']:+.3f} | "
                  f"from r5 {d['delta_from_r5']:+.3f} | slope {d['slope_r16_30_per_round']:+.4f}/rd "
                  f"| noise {d['roundtoround_noise_r16_30']:.3f}")

    out = {
        "protocol": {
            "fid": f"pytorch-fid InceptionV3 pool3, reference = all {n_ref} PathMNIST "
                   f"test images, {n_ref} generated images per checkpoint",
            "downstream": f"TissueClassifier, {EPOCHS} epochs, Adam {LR}, StepLR(5,0.5), "
                          f"batch {BATCH}, {N_SYNTH} synthetic images, seeded",
            "pseudo_label": "labeler = real-full classifier (seed s); downstream seed s+20000",
            "std": "sample standard deviation, ddof=1",
            "seeds": SEEDS,
            "cudnn_deterministic": True,
        },
        "references": {"real_full": REF["real_full"],
                       "real_matched_10k": REF["real_matched_10k"],
                       "real_full_per_class": ref["real_full_per_class"]},
        "configs": {tag: {**results[tag], "summary": S[tag]} for tag in results},
        "derived": derived,
        "spearman_fid_vs_acc": {
            "all_conditional": None if rho_all is None else
                {"rho": float(rho_all.correlation), "configs": cond_tags},
            "federated_conditional": None if rho_fed is None else
                {"rho": float(rho_fed.correlation), "configs": fed_tags}},
        "training_divergence": divergence,
    }
    with open(os.path.join(LOG_DIR, "final_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> logs/final_results.json   (total {(time.time() - t0) / 60:.1f} min)")
