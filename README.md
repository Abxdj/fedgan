# Federated GANs for medical image synthesis: failure modes and evaluation reliability

Code, logs and figures for an empirical study of federated GAN training on
PathMNIST (MedMNIST v2, colorectal histology, 9 classes, 28×28 RGB). A paper is
in preparation; this README describes the repository, not the findings in full.

**What is studied**
- An unconditional DCGAN and a conditional GAN trained across 5 simulated
  clients with FedAvg on the generator only, discriminators kept local
  ("Sync-G", Fan & Liu 2020). Averaging both networks ("Sync D&G", as in
  Rasouli et al. 2020) is run as an ablation.
- Client conditions: IID; label skew from Dirichlet(α = 0.1) partitions (one
  fixed partition plus an independent partition per seed); one client with
  Gaussian-noise-corrupted images.
- Centralized baselines at two budgets: 15,000 steps (the federation's total
  gradient steps) and 3,000 steps (the step count each federated model takes).
- Evaluation by FID and by downstream utility: test accuracy of a classifier
  trained on 10,000 synthetic images, measured against a meaningless-label
  baseline of the same pipeline.

All results are over 3 generator seeds (42, 123, 2024) and reported as
mean ± sample standard deviation.

## Repository layout

| path | contents |
|---|---|
| `dcgan.py`, `cgan.py` | architectures |
| `data_setup.py`, `make_perseed_splits.py` | download PathMNIST, build client partitions |
| `fedavg_train_multiseed.py`, `cgan_train_multiseed.py` | federated training, Sync-G: IID, non-IID (fixed partition), noisy client |
| `ablation_train.py` | federated training with `--sync G|DG` and `--split iid|non_iid|non_iid_perseed` |
| `centralized_train.py` | centralized baseline (used for the 3k-step arm) |
| `diagnostic_iid_control.py` | discriminator-logit and client-update-alignment diagnostic, IID vs non-IID |
| `evaluate_final.py` | FID for every checkpoint (pytorch-fid) |
| `evaluate_downstream_v2.py` | downstream accuracy, validation-selected, K = 3 classifiers per checkpoint |
| `evaluate_null_baseline.py` | the same pipeline with content-free labels; Welch and Spearman statistics |
| `check_determinism.py` | measures run-to-run GPU variation of the classifier |
| `make_paper_figures.py`, `sample_grid.py` | figures |
| `splits/` | the exact client partitions used (copies of the files `data_setup.py` / `make_perseed_splits.py` write to `data/`) |
| `logs/` | all results; `final_results_v2.json` and `null_baseline.json` are the authority files |
| `figures/paper/` | paper figures (PDF + PNG) and caption drafts |
| `legacy/` | retired scripts, logs and figures, kept for provenance — see `legacy/README.md`; do not cite |

## Setup

```
python -m venv venv
venv\Scripts\activate            # Windows;  source venv/bin/activate elsewhere
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Reproducing the results

Run everything from the repository root. Commands are PowerShell; on bash use
`for s in 42 123 2024; do ...; done`.

**1. Data and partitions**
```
python data_setup.py              # downloads PathMNIST to data/, writes iid_split.json, non_iid_split.json
python make_perseed_splits.py     # writes non_iid_split_seed{42,123,2024}.json
```
The partitions are seeded, but to use exactly the published ones, copy
`splits\*.json` into `data\` after this step (overwriting).

**2. Federated training (Sync-G)**
```
foreach ($s in 42,123,2024) {
  python fedavg_train_multiseed.py --seed $s     # GAN: IID, non-IID, noisy client
  python cgan_train_multiseed.py   --seed $s     # cGAN: same three conditions
  python ablation_train.py --arch gan  --split non_iid_perseed --sync G  --seed $s
  python ablation_train.py --arch cgan --split non_iid_perseed --sync G  --seed $s
  python ablation_train.py --arch cgan --split non_iid         --sync DG --seed $s
  python ablation_train.py --arch cgan --split iid             --sync DG --seed $s
}
```

**3. Centralized baselines**
```
foreach ($s in 42,123,2024) {
  python centralized_train.py --steps 3000 --seed $s
  $env:PYTHONPATH = "."; python legacy/scripts/centralized_train_v1_legacy.py --seed $s
}
```
The 15k-step checkpoints in the paper were produced by the legacy script, which
samples noise on the CPU; `centralized_train.py --steps 15000` runs the same
training math with GPU noise sampling and will not reproduce them exactly.

**4. Mechanism diagnostic**
```
foreach ($s in 42,123,2024) { python diagnostic_iid_control.py --seed $s }
```

**5. Evaluation and figures**
```
python evaluate_final.py                 # FID -> logs/final_results.json
python evaluate_downstream_v2.py         # downstream -> logs/final_results_v2.json
python evaluate_null_baseline.py         # null baseline + statistics -> logs/null_baseline.json
python check_determinism.py --log logs/determinism_check.txt
python make_paper_figures.py             # -> figures/paper/
python sample_grid.py
```

Approximate times on an RTX 3060 Laptop GPU: `evaluate_final.py` about 30 min,
`evaluate_downstream_v2.py` 30–60 min, `evaluate_null_baseline.py` about 30 min.

## Notes on reproducibility

- **Not bit-reproducible on GPU.** Every script is seeded and cuDNN runs in
  deterministic mode, but some CUDA kernels are not deterministic. Identical
  classifier runs on real-signal data differ by about 1 percentage point
  (`logs/determinism_check.txt`); K = 3 classifier repeats absorb this.
  Expect small differences from the logged numbers.
- **FID reference cache.** `evaluate_final.py` computes Inception statistics
  for all 7,180 test images once and caches them in `logs/` as an `.npz`
  (about 34 MB, gitignored). A fresh clone recomputes it on the first run.
- **FID sample size.** 7,180 generated against 7,180 real images, below the
  usual 10k–50k; values are biased upward but comparable across
  configurations in this repository, not to other papers.
- **Validation data.** Downstream accuracy selects the classifier epoch on the
  PathMNIST validation split (10,004 real labelled images).
  `logs/null_baseline.json` quantifies how much this selection alone produces
  when labels carry no information.
- **Checkpoints** are not in the repository (`*.pt` is gitignored).

## License

See `LICENSE`.
