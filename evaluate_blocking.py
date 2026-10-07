"""
Evaluation of the Blocking Experiments (C1, C2, C3)
FedGAN Failure Analysis - Anish Bharadwaj

Computes FID (5,000 samples) for every new checkpoint and downstream accuracy
(10,000 synthetic images, conditional generators only) under the SAME protocol
as the headline results, then compares each against its existing comparator
and prints a verdict per blocking item.

Unconditional generators get FID only: their round-robin downstream numbers
are a protocol artifact, and pseudo-labeled evaluation would need its own
labelers. Their divergence behaviour comes from the training logs.

Reads existing results from:
    logs/fid_multiseed.json, logs/silent_failure_multiseed.json,
    logs/centralized_results.json, data/non_iid_perseed_meta.json

Writes:
    logs/blocking_results.json

Usage:
    python evaluate_blocking.py
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, models
from medmnist import PathMNIST
from scipy import linalg
import warnings
warnings.filterwarnings("ignore")
from dcgan import Generator, LATENT_DIM as U_LATENT
from cgan import ConditionalGenerator, LATENT_DIM as C_LATENT, NUM_CLASSES

DATA_DIR, CKPT_DIR, LOG_DIR = "./data", "./checkpoints", "./logs"
SEEDS             = [42, 123, 2024]
N_FID             = 5000
N_SYNTHETIC       = 10000
BATCH_SIZE        = 128
CLASSIFIER_EPOCHS = 15
CLASSIFIER_LR     = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# (display name, checkpoint tag, conditional?, blocking item, federated?)
NEW = [
    ("cGAN IID — Sync D&G",              "cgan_iid_syncDG",             True,  "C1", True),
    ("cGAN Non-IID — Sync D&G",          "cgan_non_iid_syncDG",         True,  "C1", True),
    ("GAN Non-IID — per-seed partition", "gan_non_iid_perseed_syncG",   False, "C2", True),
    ("cGAN Non-IID — per-seed partition","cgan_non_iid_perseed_syncG",  True,  "C2", True),
    ("Centralized GAN — 3k steps",       "central_gan_3k",              False, "C3", False),
    ("Centralized cGAN — 3k steps",      "central_cgan_3k",             True,  "C3", False),
]

print(f"Device: {DEVICE}\n")


# ── Inception / FID ───────────────────────────────────────────────────────────
class InceptionFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        inc = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        self.features = nn.Sequential(
            inc.Conv2d_1a_3x3, inc.Conv2d_2a_3x3, inc.Conv2d_2b_3x3,
            nn.MaxPool2d(3, 2), inc.Conv2d_3b_1x1, inc.Conv2d_4a_3x3,
            nn.MaxPool2d(3, 2), inc.Mixed_5b, inc.Mixed_5c, inc.Mixed_5d,
            inc.Mixed_6a, inc.Mixed_6b, inc.Mixed_6c, inc.Mixed_6d, inc.Mixed_6e,
            inc.Mixed_7a, inc.Mixed_7b, inc.Mixed_7c, nn.AdaptiveAvgPool2d((1, 1)))
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = nn.functional.interpolate(x, size=(299, 299), mode="bilinear",
                                      align_corners=False)
        return self.features(x).view(x.size(0), -1)


@torch.no_grad()
def real_features(inception):
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(mean=[.5], std=[.5])])
    ds = PathMNIST(split="test", transform=tf, download=False, root=DATA_DIR)
    feats, total = [], 0
    for imgs, _ in DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True):
        feats.append(inception(((imgs + 1) / 2).clamp(0, 1).to(DEVICE)).cpu().numpy())
        total += imgs.size(0)
        if total >= N_FID:
            break
    return np.concatenate(feats)[:N_FID]


@torch.no_grad()
def generate(G, conditional, n):
    """Returns images in [-1,1] and round-robin labels (exact for cGAN)."""
    G.eval()
    imgs, lbls, done = [], [], 0
    while done < n:
        bs = min(BATCH_SIZE, n - done)
        lab = torch.tensor([i % NUM_CLASSES for i in range(done, done + bs)],
                           dtype=torch.long)
        if conditional:
            out = G(torch.randn(bs, C_LATENT, device=DEVICE), lab.to(DEVICE))
        else:
            out = G(torch.randn(bs, U_LATENT, 1, 1, device=DEVICE))
        imgs.append(out.cpu())
        lbls.append(lab)
        done += bs
    return torch.cat(imgs)[:n], torch.cat(lbls)[:n]


@torch.no_grad()
def fake_features(imgs, inception):
    feats = []
    for i in range(0, len(imgs), BATCH_SIZE):
        b = ((imgs[i:i + BATCH_SIZE] + 1) / 2).clamp(0, 1).to(DEVICE)
        feats.append(inception(b).cpu().numpy())
    return np.concatenate(feats)


def fid(real, fake):
    mu_r, mu_g = real.mean(0), fake.mean(0)
    s_r, s_g = np.cov(real, rowvar=False), np.cov(fake, rowvar=False)
    cov, _ = linalg.sqrtm(s_r.dot(s_g), disp=False)
    if np.iscomplexobj(cov):
        cov = cov.real
    d = mu_r - mu_g
    return float(d.dot(d) + np.trace(s_r + s_g - 2 * cov))


# ── Downstream classifier (identical protocol) ────────────────────────────────
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


def downstream_accuracy(imgs, labels, seed, test_loader):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    imgs = ((imgs + 1) / 2).clamp(0, 1)
    loader = DataLoader(TensorDataset(imgs, labels), batch_size=BATCH_SIZE, shuffle=True)
    model = TissueClassifier().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=CLASSIFIER_LR)
    sched = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
    crit = nn.CrossEntropyLoss()
    for _ in range(CLASSIFIER_EPOCHS):
        model.train()
        for x, y in loader:
            opt.zero_grad()
            crit(model(x.to(DEVICE)), y.to(DEVICE)).backward()
            opt.step()
        sched.step()
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x.to(DEVICE)).argmax(1).cpu()
            y = y.view(-1).long()
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total * 100


# ── Existing results ──────────────────────────────────────────────────────────
def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    print(f"  (missing {path})")
    return {}


def ms(vals):
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)


def fmt(m, s, pct=False):
    if m is None:
        return "—"
    return f"{m:.2f}{'%' if pct else ''} ± {s:.2f}"


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fid_ref  = load_json(os.path.join(LOG_DIR, "fid_multiseed.json"))
    acc_ref  = load_json(os.path.join(LOG_DIR, "silent_failure_multiseed.json"))
    cent_ref = load_json(os.path.join(LOG_DIR, "centralized_results.json"))
    meta     = load_json(os.path.join(DATA_DIR, "non_iid_perseed_meta.json"))

    print("Loading Inception-v3 and real features...")
    inception = InceptionFeatures().to(DEVICE)
    real = real_features(inception)
    test_ds = PathMNIST(split="test", transform=transforms.ToTensor(),
                        download=False, root=DATA_DIR)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    results = {}
    for name, tag, cond, item, federated in NEW:
        print(f"\n[{item}] {name}")
        fids, accs, early, late, div = [], [], [], [], []
        for seed in SEEDS:
            ckpt = os.path.join(CKPT_DIR, f"generator_{tag}_seed{seed}_final.pt")
            if not os.path.exists(ckpt):
                print(f"  seed {seed}: MISSING {ckpt}")
                continue
            G = (ConditionalGenerator() if cond else Generator()).to(DEVICE)
            G.load_state_dict(torch.load(ckpt, map_location=DEVICE))

            imgs, _ = generate(G, cond, N_FID)
            f = fid(real, fake_features(imgs, inception))
            fids.append(f)
            line = f"  seed {seed}: FID {f:.2f}"

            if cond:
                s_imgs, s_lbls = generate(G, cond, N_SYNTHETIC)
                a = downstream_accuracy(s_imgs, s_lbls, seed, test_loader)
                accs.append(a)
                line += f" | acc {a:.2f}%"

            if federated:
                lp = os.path.join(LOG_DIR, f"losses_{tag}_seed{seed}.json")
                if os.path.exists(lp):
                    with open(lp) as fh:
                        lg = json.load(fh)
                    early.append(lg["early_g"]); late.append(lg["late_g"])
                    div.append(lg["diverged"])
                    line += (f" | G-loss {lg['early_g']:.3f}->{lg['late_g']:.3f} "
                             f"{'DIVERGED' if lg['diverged'] else 'converged'}")
            print(line)

        results[tag] = {"name": name, "item": item,
                        "fid": ms(fids), "acc": ms(accs),
                        "fid_seeds": fids, "acc_seeds": accs,
                        "diverged_seeds": int(sum(div)), "n_seeds": len(fids),
                        "early_g": early, "late_g": late}

    # ── References ──
    def ref_fid(key):   # federated key in fid_multiseed.json
        r = fid_ref.get(key)
        return (r["mean"], r["std"]) if r else (None, None)

    def ref_acc(key):
        r = acc_ref.get(key)
        return (r["mean"], r["std"]) if r else (None, None)

    def ref_cent(key, metric):
        r = cent_ref.get(key)
        return (r[f"{metric}_mean"], r[f"{metric}_std"]) if r else (None, None)

    R = results
    print("\n" + "=" * 76)
    print(" C1 — AGGREGATION ABLATION (does divergence survive D synchronization?)")
    print("=" * 76)
    for cond_name, new_tag, ref_key in (
            ("IID",     "cgan_iid_syncDG",     "cGAN — IID Baseline"),
            ("Non-IID", "cgan_non_iid_syncDG", "cGAN — Non-IID")):
        r = R[new_tag]
        print(f"  cGAN {cond_name:<8} Sync-G  : FID {fmt(*ref_fid(ref_key))} | "
              f"acc {fmt(*ref_acc(ref_key), pct=True)}")
        print(f"  cGAN {cond_name:<8} Sync D&G: FID {fmt(*r['fid'])} | "
              f"acc {fmt(*r['acc'], pct=True)} | diverged "
              f"{r['diverged_seeds']}/{r['n_seeds']} seeds")
    d = R["cgan_non_iid_syncDG"]
    print()
    if d["n_seeds"] == 3 and d["diverged_seeds"] == 3:
        print("  -> Divergence SURVIVES D synchronization (3/3). Not a Sync-G")
        print("     artifact. Keep the claim; state both schemes were tested.")
        print("     Note the tension with Rasouli et al., whose convergence proof")
        print("     assumes both networks are synced — discuss it explicitly.")
    elif d["n_seeds"] == 3 and d["diverged_seeds"] == 0:
        print("  -> Divergence DISAPPEARS under Sync D&G (0/3). It is specific to")
        print("     generator-only aggregation. Reframe: the FedGAN convergence")
        print("     guarantee holds when its assumption (both networks synced)")
        print("     holds; dropping D sync — Fan & Liu's communication-saving")
        print("     option — produces divergence under label skew.")
    else:
        print("  -> MIXED or incomplete. Report per-seed; make no general claim.")
    print("  Also compare Sync D&G IID FID against Sync-G IID: if D&G is much")
    print("  better, part of the 'federation cost' is attributable to Sync-G.")

    print("\n" + "=" * 76)
    print(" C3 — BUDGET BRACKET (is the federation cost robust to budget choice?)")
    print("=" * 76)
    for arch, cent_key, new_tag, fed_key in (
            ("GAN",  "Centralized GAN",  "central_gan_3k",  "GAN — IID Baseline"),
            ("cGAN", "Centralized cGAN", "central_cgan_3k", "cGAN — IID Baseline")):
        c15 = ref_cent(cent_key, "fid")
        c3  = R[new_tag]["fid"]
        fed = ref_fid(fed_key)
        print(f"  {arch:<5} FID  centralized 15k {fmt(*c15)} | centralized 3k "
              f"{fmt(*c3)} | federated IID {fmt(*fed)}")
        if None not in (c15[0], c3[0], fed[0]):
            worst_cent = max(c15[0], c3[0])
            if fed[0] > worst_cent:
                print(f"        -> federated worse than BOTH budgets: cost robust "
                      f"(+{fed[0]-c15[0]:.1f} vs 15k, +{fed[0]-c3[0]:.1f} vs 3k)")
            else:
                print(f"        -> federated within/under the bracket: cost is "
                      f"BUDGET-DEPENDENT — report as a range, not a single number")
    c15a = ref_cent("Centralized cGAN", "acc")
    c3a  = R["central_cgan_3k"]["acc"]
    feda = ref_acc("cGAN — IID Baseline")
    print(f"  cGAN acc  centralized 15k {fmt(*c15a, pct=True)} | centralized 3k "
          f"{fmt(*c3a, pct=True)} | federated IID {fmt(*feda, pct=True)}")

    print("\n" + "=" * 76)
    print(" C2 — PARTITION ROBUSTNESS (does divergence replicate across draws?)")
    print("=" * 76)
    for seed in SEEDS:
        m = meta.get(str(seed))
        if m:
            print(f"  partition seed {seed}: {m['empty_cells']}/45 empty cells "
                  f"(original frozen partition: 10/45)")
    for arch, new_tag, ref_key in (
            ("GAN",  "gan_non_iid_perseed_syncG",  "GAN — Non-IID"),
            ("cGAN", "cgan_non_iid_perseed_syncG", "cGAN — Non-IID")):
        r = R[new_tag]
        print(f"  {arch:<5} frozen partition   FID {fmt(*ref_fid(ref_key))}")
        print(f"  {arch:<5} per-seed partitions FID {fmt(*r['fid'])} | diverged "
              f"{r['diverged_seeds']}/{r['n_seeds']}")
    c = R["cgan_non_iid_perseed_syncG"]
    print()
    if c["n_seeds"] == 3 and c["diverged_seeds"] == 3:
        print("  -> cGAN divergence replicates across 3 independent partitions.")
        print("     Drop the single-partition limitation; report partition count.")
    else:
        print("  -> Divergence does NOT replicate on every partition. Check empty-")
        print("     cell counts above: if non-diverging partitions have fewer empty")
        print("     cells, that SUPPORTS the absent-class mechanism — report it as")
        print("     such rather than as a failure to replicate.")
    print("  Note: per-seed std now mixes initialization AND partition variance,")
    print("  so expect it to be wider than the frozen-partition std.")

    out = os.path.join(LOG_DIR, "blocking_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out}")
