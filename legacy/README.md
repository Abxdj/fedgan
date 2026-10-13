# legacy/ — retired code, results and figures

Nothing in here feeds the paper. Kept for provenance and for the limitations
section. **Do not cite any number from `legacy/logs/`.** Current results live in
`logs/final_results_v2.json` (downstream + FID) and `logs/null_baseline.json`.

Moved here on 2026-10-13 with `git mv`, so `git log --follow <file>` still
shows each file's history.

## Why each group was retired

| group | retired because | superseded by |
|---|---|---|
| June single-seed era: `fedavg_train.py`, `cgan_train.py`, `silent_failure.py`, `silent_failure_all.py`, `fid_all.py`, `visualize.py`, and their logs (`fid_results*.json`, `silent_failure_*.json`, `losses_{iid,non_iid,noisy_client,cgan_*}.json`) and figures (`fig1_loss_curves`, `fig2_fid_comparison`, `fig3_silent_failure`, `fig4_perclass_heatmap`) | n=1; contains the retracted 0.81 %, 49.82 %, 81.89 % | multi-seed training scripts in the root |
| Old multi-seed evaluation: `fid_multiseed.py`, `silent_failure_multiseed.py`, `evaluate_centralized.py`, `evaluate_blocking.py`, `pseudo_label_eval.py`, `reference_classifiers.py`, and `fid_multiseed.json`, `silent_failure_multiseed.json`, `centralized_results.json`, `reference_classifiers.json`, `pseudo_label_results.json`, `blocking_results.json` | random unseeded FID reference subset, non-standard Inception input, unseeded classifiers, population std | `evaluate_final.py` → `evaluate_downstream_v2.py` |
| Single-seed metrics: `per_class_fid.py`, `precision_recall.py`, `per_class_fid.json`, `precision_recall_all.json` | built from the no-seed June checkpoints; recall is degenerate (0.0000–0.0048) | per-class accuracy in `final_results_v2.json` (fig4) |
| Single-seed diagnostic: `perclient_gloss_diagnostic.py`, `perclient_gloss_diagnostic_seed42.json`, `visualize_gloss_diagnostic.py`, `fig6_perclient_diagnostic.png` | seed 42 only, no IID control | `diagnostic_iid_control.py` (3 seeds + control) |
| Pre-paper figure scripts: `visualize_multiseed.py`, `visualize_with_centralized.py`, `visualize_diagnostic_control.py`, `visualize_pseudo_label.py`, and their figures | read the retired JSONs above; `fig3_silent_failure_multiseed.png` plots the unstable 85.89 % reference | `make_paper_figures.py` → `figures/paper/` |
| `centralized_train_v1_legacy.py` | replaced by `centralized_train.py` (adds `--steps`) | `centralized_train.py`. **This is the script that produced the 15k centralized checkpoints** — same training math as the current one, but samples noise on CPU rather than GPU. Still runnable; see below |
| `run_blocking.py` | runner for the C1–C3 blocking experiments; its final step calls the retired `evaluate_blocking.py` | the explicit training loops in the top-level `README.md` (steps 2–3) |

## Running anything in here

These scripts expect to be run from the repository root (they use `./data`,
`./checkpoints`, `./logs`), and they import `dcgan` / `cgan` from the root.
From the root:

    PYTHONPATH=. python legacy/scripts/<script>.py

Don't. They write their outputs into the live `logs/` and `figures/`
directories, which would recreate the retired files next to the current ones.

**One exception: `centralized_train_v1_legacy.py`.** It is the only way to
reproduce the 15k centralized checkpoints, so run it exactly as step 3 of the
top-level `README.md` shows:

    $env:PYTHONPATH = "."; python legacy/scripts/centralized_train_v1_legacy.py --seed <S>

It writes only `checkpoints/generator_central_{gan,cgan}_seed{S}_final.pt` and
`logs/losses_central_{gan,cgan}_seed{S}.json`. Those are current files, not
retired ones.
