"""
Runner for all blocking experiments — resumable, stops on failure.
FedGAN Failure Analysis - Anish Bharadwaj

Runs, in priority order:
  0. make_perseed_splits.py          (seconds)
  1. C1 Sync D&G ablation            cGAN non-IID x3, cGAN IID x3
  2. C3 centralized 3k-step bracket  x3 seeds (GAN + cGAN each)
  3. C2 per-seed partitions          cGAN x3, GAN x3
  4. evaluate_blocking.py            FID + downstream + verdicts

Unlike chaining with ';' in PowerShell, this STOPS at the first failure
instead of firing the remaining commands. Any step whose output already
exists is skipped, so after an interruption just run it again and it resumes.
Evaluation always re-runs at the end.

Usage:
    python run_blocking.py
"""

import os
import sys
import time
import subprocess

PY    = sys.executable
SEEDS = [42, 123, 2024]
CK    = "checkpoints"

steps = [("Per-seed partitions",
          [PY, "make_perseed_splits.py"],
          os.path.join("data", "non_iid_split_seed2024.json"))]

for split in ("non_iid", "iid"):
    for s in SEEDS:
        steps.append((f"C1  cGAN {split:<8} Sync D&G  seed {s}",
                      [PY, "ablation_train.py", "--arch", "cgan", "--split", split,
                       "--sync", "DG", "--seed", str(s)],
                      os.path.join(CK, f"generator_cgan_{split}_syncDG_seed{s}_final.pt")))

for s in SEEDS:
    steps.append((f"C3  centralized 3k        seed {s}",
                  [PY, "centralized_train.py", "--steps", "3000", "--seed", str(s)],
                  os.path.join(CK, f"generator_central_cgan_3k_seed{s}_final.pt")))

for arch in ("cgan", "gan"):
    for s in SEEDS:
        steps.append((f"C2  {arch:<4} per-seed partition seed {s}",
                      [PY, "ablation_train.py", "--arch", arch, "--split",
                       "non_iid_perseed", "--sync", "G", "--seed", str(s)],
                      os.path.join(CK, f"generator_{arch}_non_iid_perseed_syncG_seed{s}_final.pt")))

steps.append(("Evaluation", [PY, "evaluate_blocking.py"], None))


def hms(sec):
    return f"{int(sec // 3600)}h{int(sec % 3600 // 60):02d}m{int(sec % 60):02d}s"


if __name__ == "__main__":
    start = time.time()
    for i, (label, cmd, output) in enumerate(steps, 1):
        tag = f"[{i:2d}/{len(steps)}] {label}"
        if output and os.path.exists(output):
            print(f"{tag} — already done, skipping")
            continue
        print(f"\n{'#' * 70}\n{tag}\n{'#' * 70}", flush=True)
        t0 = time.time()
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"\nFAILED at step {i} ({label}), exit code {rc}.")
            print("Fix the error and re-run — completed steps will be skipped.")
            sys.exit(rc)
        print(f"{tag} — done in {hms(time.time() - t0)} "
              f"(elapsed {hms(time.time() - start)})", flush=True)

    print(f"\nAll blocking experiments complete in {hms(time.time() - start)}.")
