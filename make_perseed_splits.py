"""
Per-Seed Dirichlet Partitions — blocking fix C2
FedGAN Failure Analysis - Anish Bharadwaj

The headline non-IID runs all share ONE Dirichlet draw (data/non_iid_split.json),
so their 3 seeds vary initialization and shuffling but not the partition.
This script draws an independent Dirichlet(alpha=0.1) partition for each seed:

    data/non_iid_split_seed42.json
    data/non_iid_split_seed123.json
    data/non_iid_split_seed2024.json

Same algorithm as data_setup.py (per-class Dirichlet proportions over clients,
Hsu et al. 2019), using a local RNG seeded by the run seed.

Minimum-size guard: at alpha=0.1 a draw can leave a client with almost no data,
which breaks training (drop_last=True with batch 64 yields zero batches). Draws
with any client below MIN_CLIENT_SIZE samples are rejected and redrawn. This is
a mild conditioning of the partition distribution — disclose it in the paper:
"draws leaving any client with fewer than 500 samples were rejected and redrawn."

Also reports how many client-class cells are empty (a class entirely absent from
a client). The original frozen partition has 10/45. This number matters for
interpretation: the divergence mechanism runs through locally-absent classes, so
a draw with far fewer absent cells is a weaker stress test.

Usage:
    python make_perseed_splits.py
"""

import os
import json
import numpy as np
from medmnist import PathMNIST

DATA_DIR        = "./data"
NUM_CLIENTS     = 5
NUM_CLASSES     = 9
ALPHA           = 0.1
SEEDS           = [42, 123, 2024]
MIN_CLIENT_SIZE = 500
MAX_ATTEMPTS    = 200


def dirichlet_split(labels, rng):
    client_idx = {i: [] for i in range(NUM_CLIENTS)}
    for c in range(NUM_CLASSES):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        props  = rng.dirichlet(np.repeat(ALPHA, NUM_CLIENTS))
        counts = (props * len(idx)).astype(int)
        counts[-1] = len(idx) - counts[:-1].sum()
        start = 0
        for cid, n in enumerate(counts):
            client_idx[cid].extend(idx[start:start + n].tolist())
            start += n
    return client_idx


def describe(split, labels):
    counts = {cid: np.bincount(labels[idx], minlength=NUM_CLASSES).tolist()
              for cid, idx in split.items()}
    sizes  = {cid: len(idx) for cid, idx in split.items()}
    absent = sum(1 for c in counts.values() for x in c if x == 0)
    return counts, sizes, absent


if __name__ == "__main__":
    dataset = PathMNIST(split="train", download=False, root=DATA_DIR)
    labels  = np.asarray(dataset.labels).reshape(-1)
    n_total = len(labels)
    print(f"Loaded {n_total} training labels\n")

    # Reference: the original frozen partition
    orig_path = os.path.join(DATA_DIR, "non_iid_split.json")
    if os.path.exists(orig_path):
        with open(orig_path) as f:
            orig = {int(k): v for k, v in json.load(f).items()}
        _, _, orig_absent = describe(orig, labels)
        print(f"Original frozen partition: {orig_absent}/"
              f"{NUM_CLIENTS * NUM_CLASSES} client-class cells empty\n")

    meta = {}
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        for attempt in range(1, MAX_ATTEMPTS + 1):
            split = dirichlet_split(labels, rng)
            if min(len(v) for v in split.values()) >= MIN_CLIENT_SIZE:
                break
        else:
            raise SystemExit(f"Seed {seed}: no valid draw in {MAX_ATTEMPTS} attempts")

        # Sanity: every index assigned exactly once
        all_idx = [i for v in split.values() for i in v]
        assert len(all_idx) == n_total and len(set(all_idx)) == n_total, \
            f"Seed {seed}: partition does not cover the dataset exactly once"

        counts, sizes, absent = describe(split, labels)

        print("=" * 64)
        print(f" Seed {seed} — accepted on attempt {attempt}")
        print("=" * 64)
        for cid in range(NUM_CLIENTS):
            print(f"  Client {cid} ({sizes[cid]:6d}): {counts[cid]}")
        print(f"  Empty client-class cells: {absent}/{NUM_CLIENTS * NUM_CLASSES}\n")

        path = os.path.join(DATA_DIR, f"non_iid_split_seed{seed}.json")
        with open(path, "w") as f:
            json.dump({str(k): v for k, v in split.items()}, f)

        meta[str(seed)] = {"attempts": attempt, "empty_cells": absent,
                           "client_sizes": sizes, "class_counts": counts}

    with open(os.path.join(DATA_DIR, "non_iid_perseed_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved per-seed partitions and data/non_iid_perseed_meta.json")
    print("Disclose in the paper: draws with any client < "
          f"{MIN_CLIENT_SIZE} samples were rejected and redrawn.")
