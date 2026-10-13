"""
Determinism check for the downstream instrument
FedGAN Failure Analysis - Anish Bharadwaj

evaluate_null_baseline.py --selfcheck reported a MISMATCH:
    train_classifier        30.10 % @ epoch 11
    train_classifier_trace  12.83 % @ epoch 7     (same data, same seed)
Two explanations, and they call for opposite actions:
  (a) GPU non-determinism. cudnn.deterministic=True does NOT make every op
      deterministic (AdaptiveAvgPool2d backward uses atomicAdd on CUDA). Tiny
      float differences can grow into a different trajectory, especially with
      content-free labels. Then the null results are valid draws, and the paper
      must not claim bit-reproducibility.
  (b) A real code difference between the two functions. Then the null results
      must be discarded.

This script separates them by running the ORIGINAL train_classifier twice with
the same seed. If two identical calls disagree, it is (a).

  Test 1  original x2, permuted labels   (the self-check case)
  Test 2  original x2, real signal       (cGAN IID seed 42, round-robin labels =
                                          true class labels) — how much does
                                          non-determinism move a reported number?
  Test 3  original vs trace, permuted labels, repeated

Run:
    python check_determinism.py --log logs/determinism_check.txt
                                                # tests 1-3, default mode; output also
                                                # written to the log as UTF-8
    python check_determinism.py                 # console only
    python check_determinism.py --strict        # same, with
        torch.use_deterministic_algorithms(True); may raise on ops without a
        deterministic CUDA kernel — the error message is itself the answer.
Writes nothing unless --log is given. Runtime: a few minutes.
Do not use shell redirection (>) on Windows: PowerShell/cp1252 can mangle or break it.
"""

import os
import sys

STRICT = "--strict" in sys.argv
LOG = None
if "--log" in sys.argv:
    i = sys.argv.index("--log")
    LOG = sys.argv[i + 1]
    del sys.argv[i:i + 2]


class _Tee:
    """Mirror stdout into a UTF-8 log file."""
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.out = sys.stdout
    def write(self, x):
        self.out.write(x); self.f.write(x); self.f.flush()
    def flush(self):
        self.out.flush(); self.f.flush()


if LOG:
    sys.stdout = _Tee(LOG)
if STRICT:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # before CUDA init

extra = [a for a in sys.argv[1:] if a != "--strict"]
sys.argv = [sys.argv[0]] + extra          # evaluate_null_baseline parses argv on import
import torch                               # noqa: E402
import evaluate_null_baseline as nb        # noqa: E402

if STRICT:
    torch.use_deterministic_algorithms(True)

TAG_NULL, TAG_SIG, S = "central_gan", "cgan_iid", 42


def load_gen(tag, cond):
    G = (nb.ConditionalGenerator() if cond else nb.Generator()).to(nb.DEVICE)
    G.load_state_dict(torch.load(os.path.join(nb.CKPT_DIR, f"generator_{tag}_seed{S}_final.pt"),
                                 map_location=nb.DEVICE))
    return G


def run_orig(X, Y, seed, Xva, Yva, Xte, Yte):
    m, ep, bv = nb.train_classifier(X, Y, seed, Xva, Yva)
    acc, _, _ = nb.evaluate(m, Xte, Yte)
    return acc, ep, bv


def main():
    nb.check_verbatim()
    print(f"Mode: {'STRICT (use_deterministic_algorithms)' if STRICT else 'default (as in v2)'}"
          f" | device {nb.DEVICE} | torch {torch.__version__}")
    Xva, Yva = nb.load_split("val")
    Xte, Yte = nb.load_split("test")

    Xn, _ = nb.generate(load_gen(TAG_NULL, False), False, nb.N_SYNTH, seed=S + 1)
    Yn = nb.null_labels(len(Xn), S + nb.PERM_OFF)
    Xs, Ys = nb.generate(load_gen(TAG_SIG, True), True, nb.N_SYNTH, seed=S + 1)
    seed = S + nb.CLF_OFF

    def report(name, a, b):
        same = abs(a[0] - b[0]) < 1e-9 and a[1] == b[1]
        print(f"  {name:<40} {a[0]:7.3f}% @ ep {a[1]:<2} | {b[0]:7.3f}% @ ep {b[1]:<2} -> "
              f"{'IDENTICAL' if same else f'DIFFERENT (diff {b[0] - a[0]:+.3f})'}")
        return same

    try:
        print("\nTest 1: original x2, permuted labels")
        r1 = [run_orig(Xn, Yn, seed, Xva, Yva, Xte, Yte) for _ in range(2)]
        t1 = report("original vs original", *r1)

        print("\nTest 2: original x3, real signal (cGAN IID seed 42)")
        r2 = [run_orig(Xs, Ys, S + 10000, Xva, Yva, Xte, Yte) for _ in range(3)]
        t2 = report("run 1 vs run 2", r2[0], r2[1]) & report("run 1 vs run 3", r2[0], r2[2])
        accs = [r[0] for r in r2]
        print(f"  spread across identical calls: {max(accs) - min(accs):.3f} pp "
              f"(v2 stored for this classifier: see acc_raw['42'][0])")

        print("\nTest 3: original vs trace, permuted labels, x2")
        for i in range(2):
            o = run_orig(Xn, Yn, seed, Xva, Yva, Xte, Yte)
            t = nb.train_classifier_trace(Xn, Yn, seed, Xva, Yva, Xte, Yte)
            report(f"original vs trace (#{i + 1})", o, (t["selected_acc"], t["best_epoch"]))
    except RuntimeError as e:
        print(f"\nRuntimeError: {e}")
        print("-> An op has no deterministic CUDA implementation; strict mode cannot run as-is.")
        return

    print("\nVERDICT")
    if not t1:
        print("  Identical calls to the ORIGINAL function disagree -> GPU non-determinism (a).")
        print("  The self-check mismatch is not a code difference; null results stand.")
    else:
        print("  Original is reproducible; if Test 3 differs, the trace function changes training (b).")
    if not t2:
        print("  Non-determinism also moves real-signal accuracy; it is part of the instrument noise"
              " already captured by K classifier seeds, but 'deterministic' must not be claimed.")


if __name__ == "__main__":
    main()
