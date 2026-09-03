# Reproducibility experiments

A running, numbered log of the paper's results and how to reproduce each one.
Each entry says what's fully reproducible today, what isn't yet, and where
the code/outputs for it live.

## Experiment 1: LIMA scalability vs. Apriori / FP-Growth

**What it measures:** how LIMA's runtime scales against the Apriori and
FP-Growth baselines along two axes -- growing column count (dataset:
`miniboone`, `jannis`, `drive_diagnosis`, rows fixed) and growing sampling
budget / `1/approx` on the original, un-expanded dataset. Published as the
2-row grid figure over the three datasets.

**Origin:** `experiment_results/experiment_1_scalability/` (added directly
to the repo, not through this package) holds the original, pre-port script
(`benchmark.py`) plus its frozen output (`results.pickle`) and plotting
script (`squareplot.py`).
`benchmark.py` uses old flat imports (`from LIMA.LIMA import LIMA`, etc.)
that predate the `src/diada` package layout and no longer resolve.

**Status:**

| Piece | Reproducible today? |
|---|---|
| The published figure itself, from the frozen numbers | **Yes, exactly** -- `experiment_results/experiment_1_scalability/squareplot.py` run unmodified against `experiment_results/experiment_1_scalability/results.pickle` regenerates it (verified). |
| The "ours" / LIMA line, rerun from live code | **Yes** -- `src/diada/experiments/experiment_1_scalability/scalability_benchmark.py` (below). |
| The Apriori / FP-Growth lines, rerun from live code | **Not yet** -- `APRIORISampler`, `FPSampler`, `MonoScheduler` don't exist anywhere in the ported LIMA tree or in `LIMA_py`. `scalability_benchmark.py` auto-detects them (checks `diada.core.LIMA.LIMA.sampler.APRIORISampler`, `...sampler.FPSampler`, `...scheduler.MonoScheduler`) and will include them automatically, no code change needed, once those three files exist there. |

**Live rerun:** `python3 -m diada.experiments.experiment_1_scalability.scalability_benchmark`
(resumable -- safe to interrupt and rerun; checkpoints every timing point to
`results/experiment_1_scalability/scalability_raw.csv`). Writes
`results/experiment_1_scalability/results_live.pickle` (same schema as the
frozen `results.pickle`) and `scalability_live.png`. Full module docstring
in that file lists every deliberate difference from the original
`benchmark.py` (import paths, dead-code removal, dataset relabelling), plus
one now-reconciled discrepancy: the checked-in `benchmark.py`'s
`_colsSeries()` (10..60) and `APROX_INV_POINTS=40` didn't match what
actually produced `results.pickle`. Reverse-engineered
directly from the pickle's own x-values, the real generating constants were
`range(0, 100, 2)` for columns (50 points) and 20 log-spaced points (not 40)
for the sample-size sweep -- `squareplot.py`'s axis clipping (`COL_LIMITS =
(10, 100)`) and `xs[::2]` subsampling on the column series is what made the
published figure look like it only spanned ~10..96. `scalability_benchmark.py`
now uses these recovered values, so a live rerun lands on the same axes as
the published figure. See the docstring's point 5 for the full derivation.

**Known caveat -- runtime:** the sampling-budget ("row") series runs LIMA
directly on the full, un-expanded dataset (up to ~73k rows). A single point
at the *loosest* tested budget already took >150s on `miniboone`; the sweep
covers 40 such points per (dataset, algorithm), each averaged over 5
repeats, capped at 1800s per point before a series stops growing. Expect
the LIMA-only sweep across all 3 datasets to take a long time (likely
several hours) if run to completion; it was built resumable specifically
for that reason.

## Experiment 2: DIADA vs. Unsupervised Feature Selection (UFS) baselines

**What it measures:** whether DIADA drops injected noise columns better
than 16 generic UFS methods (variance score, Laplacian score, SPEC, MCFS,
NDFS, RSR, AGUFS, UDFS-like, correlation-uniqueness, mRMR, JMI, CMIM,
Inf-FS, cluster dispersion, PFA, feature-cluster-representative) plus an
optional Concrete Autoencoder (CAE, needs torch), across all 17
`benchmarks.yaml` datasets' `*_un_noise.csv` variants. Published as the
paper's noise-ratio comparison table.

**Origin:** `experiment_results/experiment_2_ufs/ufs_extended.ipynb` (added
directly to the repo, not through this package) plus its frozen output
`experiment_results/experiment_2_ufs/ufs_results_final.csv` (306 rows = 17
benchmarks x 18 methods). The notebook's checked-in state
does not reproduce its own committed CSV: it reads noise-injected input
from a stale absolute path (`C:/Projects/arclo/data/noise_cf1/...` --
`arclo` appears to be this project's name before it became DIADA, and
"cf1" the old name for what this package now calls "un"; the same files
already exist at `data/noise_un/` in this repo), its benchmark loop
is hardcoded to only 2 of the 17 datasets, its `cae` method is commented
out of the methods dict, and its final analysis cell reads a
`ufs_results_final_final.csv` that doesn't exist anywhere. Additionally,
the CSV's `"lima"` method rows (DIADA's own UN-cleaned output, scored the
same way as the UFS methods -- verified byte-for-byte against
`data/cleaned_un/bank_un_cleaned.csv`) aren't produced by any code in the
repo at all.

**Status:**

| Piece | Reproducible today? |
|---|---|
| All 16 UFS methods + CAE | **Yes** -- ported verbatim to `src/diada/experiments/experiment_2_ufs/ufs_baselines.py`, run by `src/diada/experiments/experiment_2_ufs/ufs_benchmark.py`. |
| DIADA's own comparison row | **Yes, and renamed.** Real code now (`invoke_diada` engine="jar" + `build_un_datasets`, same defaults as `generate_benchmarks.py`), labelled `"diada"` in the output and `"DIADA"` in the summary table -- not `"lima"`. LIMA is the internal algorithm name; DIADA is the system, and every results-facing label uses the latter (same override pattern `experiment_1_scalability/squareplot.py` already used for this). |
| The published table's exact numbers | **Not verified against the frozen CSV yet** -- the live runner reproduces the same methodology, but a full rerun (17 datasets x 18 methods) hasn't been executed to diff against `ufs_results_final.csv`. |

**Live rerun:** `python3 -m diada.experiments.experiment_2_ufs.ufs_benchmark`
(resumable -- checkpoints every (benchmark, method) row to
`results/experiment_2_ufs/ufs_raw.csv`; writes
`results/experiment_2_ufs/ufs_summary.csv` at the end, same
mean/noise-ratio/CI aggregation as the notebook's final cell). Smoke-tested
end-to-end on a tiny synthetic dataset (all 16 UFS methods + CAE + DIADA
ran, resumable-rerun idempotency verified, DIADA correctly kept 0/4 noise
columns vs. UFS methods' 2+/6). Full module docstring in
`ufs_benchmark.py` lists every deliberate difference from the notebook.

**Known, reproduced-not-fixed failures** (per explicit instruction --
recorded in the `error` column exactly like the frozen CSV, not silently
patched): `spec`/`mcfs`/`ndfs`/`rsr` do a dense N x N eigendecomposition (N
= dataset row count), which OOMs on the larger benchmarks (`jannis`,
`miniboone`, `bank`, `drive_diagnosis`, `covertype`, `default_payment`,
`nasa` in the frozen run) -- a real scalability ceiling, not a bug.
Separately, every W-consuming method failed specifically on `diamonds` in
the frozen run with `'NoneType' object has no attribute 'sum'` while every
non-W method succeeded on the same dataset -- root cause not identified.

**Not yet run at full scale.** 17 datasets x 18 methods; several UFS
methods took minutes per larger dataset in the frozen run (e.g. `spec` on
`pendigits`: 204s). Resumable/checkpointed specifically for this.

## Experiment 3: DIADA+ARDA across every benchmark x noise variant

**What it measures:** for every dataset in `benchmarks.yaml`, ARDA's full
1..N feature-count sweep on 5 variants -- raw (no noise, no cleaning, i.e.
the "clean baseline"), `_un_dirty`/`_un_clean` (univariate/UN noise,
uncleaned vs. DIADA-cleaned), `_mn_dirty`/`_mn_clean`
(multivariate-spurious-cluster/MN noise, uncleaned vs. DIADA-cleaned).
This is the paper's main results table.

**Origin:** unlike Experiments 1 & 2, there's no separate frozen
notebook/script pair here -- this is the package's own reproducibility
code (`src/diada/experiments/experiment_3_diada_arda/`:
`generate_benchmarks.py` for noise injection + DIADA cleaning,
`run_paper_experiments.py` for the ARDA sweep on top of it, both reusing
`noise.py`/`noise_benchmarks.py`/`spurious_cluster.py`/`synthetic_targets.py`).
The frozen reference is
`experiment_results/experiment_3_diada_arda/results_final_classification.csv`
/ `results_final_regression.csv` -- the paper's original numbers. (Moved
there from `results/` for consistency with Experiments 1 & 2's frozen
artifacts, once Experiment 4 -- below -- turned out to need the same
treatment for its own frozen CSVs, which used to sit in `results/` right
next to these.)

**Status:**

| Piece | Reproducible today? |
|---|---|
| Noise injection + DIADA cleaning artifacts (`data/noise_un/`, `cleaned_un/`, `noise_mn/`, `cleaned_mn/`, `*_synth_targets*`) | **Already generated, for all 17 benchmarks, via `data_original/`** -- copy `data_original/` into `data/` rather than regenerating (see README's "Getting the datasets" section for why regenerating won't reproduce it byte-for-byte, even though the noise-injection code is fully seeded). `generate_benchmarks.py`'s own standalone `main()` used to be hardcoded to just `"diamonds"` -- fixed, it now loops every `benchmarks.yaml` entry. A second, more serious bug found by smoke-testing `run_paper_experiments.py` against a byte-fresh `data_original/` copy of `data/`: `generate_for_benchmark()` writes to `data/cleaned_un_leftover/` etc. without ever creating those directories -- `generate_benchmarks.py`'s `main()` used to do that as a separate step, but `run_paper_experiments.py` calls `generate_for_benchmark()` directly and never did, so it crashed with `OSError: Cannot save file into a non-existent directory` on a clean `data/`. It only ever worked before because those directories already happened to exist from an earlier ad hoc `generate_benchmarks.py` run (the source of the stray `diamonds`/`students`-only entries in `data/cleaned_un_leftover/` etc. that don't exist in `data_original/` at all -- that output type was added post-reorg). Fixed: `generate_for_benchmark()` now creates its own output directories, so it's correct regardless of which entry point calls it. Naming note: `cf1`/`cf2` renamed to `un`/`mn` throughout (folders, files, function names, the `--mode` CLI flag) -- see `README.md`'s "UN vs MN" section. |
| The full ARDA sweep + `results/reproduced_*.csv` | **Smoke-tested, works.** `python -m diada.experiments.experiment_3_diada_arda.run_paper_experiments --datasets diamonds --max-features 2` ran end to end against a byte-fresh `data_original/`-copied `data/`: all 5 variants produced sane, consistent rows (`carat` ranked feature #1 every time, R^2 ~0.88 at 1 feature / ~0.94 at 2, matching what you'd expect for diamond price prediction), written to `results/reproduced_regression.csv`, frozen `results_final_*.csv` left untouched. One more real bug found and fixed in the process: `ARDA._log_result` (in `src/diada/arda/arda.py`) wrote its per-fit debug log to a bare cwd-relative `results.csv`/`results_regression.csv` -- the one remaining hardcoded-relative-path spot in the package, left a stray file at the repo root during the smoke test. Now routed through `paths.RESULTS_DIR` like everything else; this log is independent of and redundant with the `reproduced_*.csv` output (nothing else in the codebase reads it back). Not yet run at full scale (17 datasets, `max_features` up to 40). |
| Analysis/plots of the frozen numbers | **Yes** -- `experiment_results/experiment_3_diada_arda/results_analysis.ipynb` (sweep plots, noise-type breakdown, augmentation-time comparison). Split out of the old `results/results.ipynb`, which mixed this experiment's cells with Experiment 4's; the plotting code itself moved to `src/diada/experiments/results_plotting.py`, shared by both experiments' notebooks instead of duplicated (see Experiment 4 below). One real bug fixed in the split: the augmentation-time cell used to read `results_final_classification_2.csv` / `_regression_2.csv`, files that don't exist anywhere in this repo -- now reads the real `results_final_*.csv` files already loaded earlier in the same notebook. |

**Known discrepancy vs. the frozen file, found while cross-checking (not
fixed, informational):** the frozen `results_final_regression.csv`'s
`selected_features` column includes the target column itself alongside the
requested number of real features (e.g. `num_new_features=1` lists
`"carat,price"`, not just `"carat"`, for `diamonds_un_clean` -- `price` is
the target). The smoke-tested current code does not do this (verified:
its freshly-generated rows list only the real selected features, matching
what `candidate_cols = [c for c in dataset.columns if c != target_column]`
in `arda.py` should produce). Another confirmation that whatever produced
`results_final_*.csv` was an earlier version of this code, not just a
different random draw -- consistent with the `data_original/` vs. `data/`
finding above. Not treated as a bug to fix (excluding the target from
candidate features is clearly the correct behavior); noted here so a
column-by-column diff against the frozen file isn't mistaken for a
regression.

**Live rerun:** `python -m diada.experiments.experiment_3_diada_arda.run_paper_experiments`
(`--datasets <name> [...]` / `--max-features N` for a smaller slice). Not
resumable/checkpointed per-point the way Experiments 1 & 2 are -- it's
organized per-dataset (`generate_for_benchmark` + a full ARDA sweep per
variant), so use `--datasets` to split a full run into restartable chunks
if needed.

**Known caveat -- cost.** By far the most expensive of the three: every
dataset (17 by default) x 5 variants x up to 40 AutoGluon fits each (the
paper's original `max_features` setting). Data point from the smoke test
above: one small dataset (`diamonds`, 10 base columns) at `--max-features 2`
(5 variants x 2 feature counts = 10 fits total) took a few minutes
end-to-end -- RIFS feature-selection time per variant ranged ~90-490s on
its own (logged as `augmentation_time`), on top of the AutoGluon fits
themselves. Extrapolating to the full `max_features=40` x 17-dataset sweep,
expect this to run for a long time (very plausibly many hours) -- use
`--datasets` / `--max-features` to scope a run down.

## Experiment 4: synthetic-target ablation

**What it measures:** whether ARDA can be fooled into selecting features
that predict a synthetic decoy target rather than the real one. Two
synthetic proxy targets are built per benchmark (`synthetic_target_1` from
the first half of the real feature columns, `synthetic_target_2` from the
second half -- see `synthetic_targets.py`'s `build_synthetic_target`,
reused from Experiment 3's noise-generation code), then ARDA's feature
sweep is run against each synthetic target across three variants per
target: `_base` (no injected noise), `_noise` (noise injected, uncleaned),
`_clean` (noise injected, DIADA-cleaned) -- 6 variants total per benchmark.

**Origin:** built on the same infrastructure as Experiment 3
(`synthetic_targets.py`, `noise_benchmarks.py`'s
`generate_benchmark_noise_un_synth_targets`, both in
`src/diada/experiments/experiment_3_diada_arda/`) but is a **separate
ablation, not part of `run_paper_experiments.py`**'s 5-variant sweep. The
frozen reference is
`experiment_results/experiment_4_synthetic_targets/results_2targets_classification.csv`
/ `results_2targets_regression.csv` (moved there from `results/`, same
reasoning as Experiment 3 above).

**Status:**

| Piece | Reproducible today? |
|---|---|
| Noise injection + synthetic-target generation | **Yes, via existing Experiment 3 code** -- `generate_benchmark_noise_un_synth_targets` (writes `data/noise_un_synth_targets/`, `data/cleaned_un_synth_targets/`), already exercised by `generate_benchmarks.py`/`generate_for_benchmark()`. Nothing new needed here. |
| A dedicated ARDA sweep across the 6 synthetic-target variants, producing fresh `reproduced_2targets_*.csv` | **Not implemented.** No script builds the 6 variants (`_target_1_base/_noise/_clean`, `_target_2_base/_noise/_clean`) and runs ARDA's sweep over them the way `run_paper_experiments.py` does for Experiment 3's 5 variants. Deliberately not built yet -- flag if/when you want this scripted the same way (this is the "ask if you'd like that sweep scripted" the README used to mention). |
| Analysis/plots of the frozen numbers | **Yes** -- `experiment_results/experiment_4_synthetic_targets/results_analysis.ipynb`, split out of the old `results/results.ipynb` the same way as Experiment 3's, sharing `src/diada/experiments/results_plotting.py`'s plotting functions instead of duplicating them (the two notebooks' `plot_dataset` cells were near-identical, differing only in the `variants` config and dataset-ordering constants). |

**No live-rerun entry point exists for this experiment yet** -- unlike
Experiments 1-3, there's currently no `python -m
diada.experiments.experiment_4_synthetic_targets...` to run. Only the
frozen results + their analysis notebook are organized so far.
