# DIADA

**Status: experimental research code.** This repository accompanies an
academic paper and reflects the state of an active research project, not a
polished production tool. APIs, defaults, and output formats may still
change; there is no versioning/deprecation policy yet. It's shared for
reproducibility and for anyone who wants to build on the approach, not as a
maintained package with support guarantees.

DIADA scores how "sound" (statistically related) every pair of columns in a
table is, using that to decide which columns are relevant to a prediction
target and which look unrelated/noisy. ARDA (Automatic Relational Data
Augmentation) then takes a table of candidate columns and runs iterative
feature selection (RIFS) to pick the ones that actually help a model, using
AutoGluon to train and evaluate. Together: point DIADA + ARDA at a CSV and a
target column, and get back a cleaned dataset and a trained model.

**LIMA is the algorithm that actually computes DIADA's soundness scores.**
It's a lattice-search + statistical-sampling algorithm that tests, for
increasingly large groups of columns, whether adding one more column gives a
statistically significant ("sound") signal. `src/diada/core/LIMA/` is a
direct Python port of it (see "Two soundness-scoring engines" below) and is
the default engine `invoke_diada` uses; the original `DIADA-0.8.jar` (a Java
implementation of the same algorithm) is still bundled and usable as a
fallback/comparison engine.

## Prerequisites

- Python >= 3.9
- A Java runtime on your `PATH` (tested with OpenJDK 11) is **only** needed
  if you explicitly ask for the `engine="jar"` soundness-scoring engine (see
  below) -- the default `"lima"` engine is pure Python and needs nothing
  beyond the `pip install` below. Java runs the bundled
  `src/diada/core/DIADA-0.8.jar` via `java -jar`, which `pip install` can't
  set up for you; install a JRE/JDK yourself first if you want it (e.g.
  `apt install openjdk-11-jre-headless`, `brew install openjdk`).

## Install

```bash
git clone <this repo>
cd DIADA
pip install -e .
```

This installs the `diada` package (`pandas`, `numpy`, `scikit-learn`,
`scipy`, `networkx`, `matplotlib`, `pyyaml`, and
`autogluon.tabular[lightgbm,xgboost]`) and registers the `diada-clean`
console script. Everything is also runnable as `python -m diada.<module>`
if the console script isn't on your `PATH`.

## Getting the datasets

`data/` is **not committed to this repository** (it's in `.gitignore`) --
datasets are too large to keep in git history. All of it -- the raw
`base_datasets/` used in the paper plus the generated noisy/cleaned/output
artifacts -- is hosted instead at:

**https://mydisk.cs.upc.edu/s/ycJ4Any7Mz2RRLq**

Download and unzip it into the repo root so you end up with a `data/`
folder there (`data/base_datasets/`, `data/cleaned_cf1/`, etc., matching the
layout below) before running any of the reproducibility scripts. This isn't
needed just to run `diada-clean` on your own CSV, which doesn't touch
`data/` at all.

## Repository layout

```
DIADA/
├── pyproject.toml
├── benchmarks.yaml          per-dataset config (target column, task, metric) used
│                            by the reproducibility scripts below
├── data/                    NOT in git -- download from "Getting the datasets" above.
│                            base_datasets/ (raw, one per paper benchmark) + generated
│                            noisy/cleaned variants + DIADA soundness output
├── results/                 results_final_*.csv = the paper's original reference
│                            results; reproduced_*.csv = output of re-running the
│                            sweep yourself (see below) -- never overwrites the former
└── src/diada/
    ├── paths.py             central path resolution (repo-root-relative, not cwd-relative)
    ├── core/                 soundness scoring + soundness-graph -> dataset construction
    │   ├── diada_tool.py       invoke_diada(): runs LIMA (default) or the jar, returns
    │   │                       the soundness table -- see "Two soundness-scoring engines"
    │   ├── dataset_builder.py  build_cf1_datasets() / build_cf2_datasets(): threshold the
    │   │                       soundness graph and build the resulting cleaned dataset(s)
    │   ├── DIADA-0.8.jar       the original Java implementation (engine="jar")
    │   └── LIMA/               Python port of LIMA (engine="lima", the default) -- moved
    │                           here verbatim from the sibling LIMA_py repo, see its
    │                           __init__.py
    ├── arda/                 ARDA: RIFS feature selection + AutoGluon training/eval
    │   ├── arda.py             the ARDA class (see "Two ARDA entry points" below)
    │   └── run_benchmarks.py   reproducibility: sweep benchmarks.yaml through ARDA directly
    ├── experiments/          noise injection + the benchmark-generation sweep
    │   ├── generate_benchmarks.py     reproducibility: noise-inject + DIADA-clean every
    │   │                              benchmark, writing the artifacts under data/
    │   └── run_paper_experiments.py   reproducibility: the full paper experiment --
    │                                  chains noise injection -> DIADA cleaning -> ARDA
    │                                  across every dataset, producing results/reproduced_*.csv
    ├── pipeline.py            the "system": run DIADA + ARDA on any single CSV
    └── cli.py                 `diada-clean` command-line wrapper over pipeline.py
```

There are two separate ways to use this repository, covered next: running
the system on your own data, and reproducing the paper's experiments.

## Using the system on your own CSV

```bash
diada-clean --csv my_data.csv --target price --task regression
```

This runs DIADA cleaning in both CF1 and CF2 modes (see "CF1 vs CF2" below),
writes every resulting dataset to `./diada_pipeline_output/`, and (unless
`--no-arda` is passed) trains one ARDA model per dataset with the requested
number of features, printing a summary. From Python:

```python
from diada.pipeline import PipelineConfig, run_pipeline

config = PipelineConfig(csv_path="my_data.csv", target_column="price", task="regression")
result = run_pipeline(config)

result.dataset_paths   # {"cf1_kept": Path(...), "cf1_leftover": Path(...), "cf2_target_cluster": Path(...), ...}
result.datasets        # same, but as DataFrames instead of paths
result.arda_results    # {"cf1_kept": DataFrame(1 row), ...} -- one trained model's metrics per dataset
```

### CF1 vs CF2

DIADA's soundness scores are thresholded into a graph, and that graph is
used two ways:

- **CF1**: every column connected to *anything* above the threshold is kept
  in one `cf1_kept` dataset; everything else goes into `cf1_leftover`
  (nothing is silently discarded).
- **CF2**: the graph's connected components are treated as separate feature
  clusters. The component containing the target column becomes
  `cf2_target_cluster`; every other component becomes its own dataset
  (`cf2_cluster_2`, `cf2_cluster_3`, ...), and columns that never entered the
  graph at all become `cf2_leftover`. Every dataset (leftover and
  non-target clusters included) gets the target column appended if it isn't
  already present, so any of them can independently be handed to ARDA --
  useful as a negative control (does ARDA get fooled by a spurious cluster's
  own features?).

### Two soundness-scoring engines: LIMA vs. the jar

`invoke_diada(df, num_buckets, engine=...)` in `core/diada_tool.py` supports
two interchangeable-looking but **not numerically interchangeable** engines:

- `"lima"` (the default) -- the in-process Python port under
  `core/LIMA/`. Pure Python, no Java needed.
- `"jar"` -- the original `DIADA-0.8.jar`, invoked via `java -jar` (needs a
  Java runtime, see Prerequisites).

Both implement the same underlying soundness-testing algorithm, but:

- **LIMA is randomized and unseeded** (same as the upstream LIMA_py repo it
  was ported from) -- repeated calls on identical input return a different,
  though heavily overlapping, set of "sound" column pairs each time. **The
  jar was deterministic** in testing -- identical output across repeated
  runs.
- **Their soundness scores are on different scales.** LIMA reports z-scores
  from its own internal significance test, so its values already cluster
  above that test's significance bar; the jar computes a differently-scaled
  statistic over a much larger candidate set, most of which falls well
  below `DEFAULT_THRESHOLD=4` (the cutoff `dataset_builder.py` uses to build
  the relevance graph). That threshold ends up doing much more filtering
  work on the jar's output than on LIMA's. Don't compare raw soundness
  numbers between engines.
- **Measured on four base_datasets** (`num_buckets=10`, `threshold=4`; the
  jar was deterministic across repeated runs in every case, so this is jar
  vs. a single LIMA run):

  | dataset | rows x cols | jar time | lima time | jar pairs >threshold | lima pairs >threshold | Jaccard overlap | lima pairs also in jar's raw output |
  |---|---|---|---|---|---|---|---|
  | students | 4424 x 32 | 1.4s | 3.7s | 252 | 176 | ~0.45 | 98.9% |
  | pendigits | 10992 x 17 | 0.9s | 0.7s | 210 | 42 | 0.20 | 100% |
  | diamonds | 53940 x 10 | 1.8s | 0.5s | 30 | 22 | 0.73 | 100% |
  | mice_protein | 1080 x 81 | 20.4s | 12.3s | 2847 | 266 | 0.09 | 100% |

  Two consistent patterns: LIMA is *always* a near-total subset of the
  jar's raw (unfiltered) relations (>=98.9% in all four) -- it never
  contradicts the jar, it's just far more conservative about what it calls
  "sound" -- and that conservatism grows sharply with column count
  (81-column `mice_protein`: LIMA finds 266 vs. the jar's 2847; 10-column
  `diamonds`: the two are much closer, 22 vs. 30). Timing has no simple
  rule either way -- LIMA was faster on 3 of 4 datasets here (including the
  slowest case for both, `mice_protein`: 12.3s vs. 20.4s) but slower on
  `students`.

Practically: if you need bit-for-bit reproducible soundness scores (e.g.
regenerating a specific published number), use `engine="jar"`. For general
use, the default `"lima"` engine avoids the Java dependency, but expect it
to report meaningfully fewer "sound" relations than the jar, especially on
wide (many-column) tables -- if that matters for your dataset, check both
engines' output before trusting either one, or use `engine="jar"` for
continuity with previously-published results.

### Configuration reference (`diada-clean` flags / `PipelineConfig` fields)

| Flag | Field | Default | Meaning |
|---|---|---|---|
| `--task` | `task` | required | `classification` or `regression` |
| `--metric` | `metric` | `accuracy` | passed to ARDA/AutoGluon (accuracy/f1 for classification, root_mean_squared_error/mae for regression) |
| `--mode` | `mode` | `both` | `cf1`, `cf2`, or `both` -- which cleaning strategy to run |
| `--threshold` | `threshold` | `4` | soundness-score cutoff for the relevance graph; lower keeps more columns as "related" |
| `--num-buckets` | `num_buckets` | `10` | how DIADA discretizes numeric columns before scoring; `0` = all-string, `1` = typed/unbucketed, `>1` = bin into this many buckets |
| `--engine` | `engine` | `lima` | soundness-scoring engine, `lima` or `jar` -- see "Two soundness-scoring engines" above |
| `--num-features` | `num_features` | `10` | exact number of top-ranked features ARDA trains the model with (see below) |
| `--output-dir` | `output_dir` | `diada_pipeline_output` | where generated datasets + results are written |
| `--no-arda` | `run_arda` | on | skip model training, only produce cleaned CSVs |
| `--quiet` | `verbose` | on | suppress progress printing |

`rifs_k`, `rifs_eta`, `cv`, and `random_state` (ARDA/RIFS internals) are
available on `PipelineConfig` but not yet exposed as CLI flags.

**On `num_features`**: `diada-clean` trains exactly one model, using the
top-`num_features` ranked columns (capped automatically if a dataset has
fewer candidates than that). This is deliberately simple -- pick a number
that seems reasonable for your dataset and adjust it if the result isn't
good enough. It does *not* try to auto-select the "best" feature count for
you; if you want to see how performance changes across every feature count
from 1 upward (the sweep the paper's results are built from), that's a
separate, more expensive mode -- see `ARDA.fit_transform_full_table(...,
num_features=None, max_features=N)` in `arda/arda.py`, or the
reproducibility scripts below, which use exactly that sweep.

## Reproducing the paper's experiments

Three scripts, in increasing order of scope, all driven by `benchmarks.yaml`
(one entry per dataset in `data/base_datasets/`, giving its target column,
task, and metric):

- **`python -m diada.arda.run_benchmarks`** -- runs ARDA's feature-selection
  sweep directly on the raw base datasets (no noise, no DIADA cleaning). The
  "clean baseline" numbers.
- **`python -m diada.experiments.generate_benchmarks`** -- for each
  benchmark, injects noise (CF1 = univariate/"UN", CF2 =
  multivariate/spurious-cluster/"MN"), runs it through DIADA, and writes
  every noisy/cleaned/leftover variant to `data/`. Does **not** run ARDA;
  this only produces the CSV artifacts. Note: as shipped, its `main()` only
  processes `"diamonds"` (see the commented-out full loop in the script) --
  widen that yourself if you want the artifacts for every benchmark. This
  script (and `run_paper_experiments` below, which calls into it) is
  deliberately pinned to `engine="jar"` rather than the new default
  `"lima"` engine, specifically so it keeps producing the same soundness
  relations that `results/results_final_*.csv` was originally built from --
  see "Two soundness-scoring engines" above for why the two engines aren't
  interchangeable. Requires Java for this reason, even though the rest of
  the package (the `diada-clean` "system") doesn't.
- **`python -m diada.experiments.run_paper_experiments`** -- the full
  reproduction: for every dataset, generates the same noisy/cleaned variants
  as `generate_benchmarks` (reusing its code directly, so the two stay in
  sync) and additionally runs ARDA's full feature-count sweep on five
  variants of each -- raw, noisy-uncleaned (`_un_dirty`/`_mn_dirty`), and
  DIADA-cleaned (`_un_clean`/`_mn_clean`) -- matching the `benchmark` naming
  already used in `results/results_final_*.csv`. Results are written to
  `results/reproduced_regression.csv` / `results/reproduced_classification.csv`
  -- **the original `results_final_*.csv` files are never overwritten**, so
  you can always diff a fresh run against the paper's reference numbers.

  This is expensive: by default it's every dataset in `benchmarks.yaml` (17
  by default) x 5 variants x up to 40 AutoGluon fits each. Use
  `--datasets <name> [<name> ...]` and/or `--max-features N` to run a
  smaller slice, e.g.:

  ```bash
  python -m diada.experiments.run_paper_experiments --datasets diamonds --max-features 10
  ```

### Synthetic-target benchmarks

`experiments/synthetic_targets.py` and the `*_synth_targets` variants
(`generate_benchmark_noise_cf1_synth_targets`, written to
`data/noise_cf1_synth_targets/` / `data/cleaned_cf1_synth_targets/` by
`generate_benchmarks.py`) are used, but for a separate ablation from the main
results: they build synthetic proxy targets correlated with subsets of the
real features, to test whether ARDA can be fooled into selecting features
that predict a decoy rather than the real target. That experiment's results
live in `results/results_2targets_*.csv` (see `results/results.ipynb` for
the analysis) and are not part of `run_paper_experiments.py` above -- ask if
you'd like that sweep scripted the same way.

## Original ARDA (join-based) pipeline

`ARDA.fit_transform` / `fit_transform_from_connections` in `arda/arda.py`
reproduce the original ARDA paper's full pipeline end to end -- coreset
construction, join-graph discovery/execution across multiple tables, then
the same feature-selection stage described above. This project's own
experiments don't use this path (every benchmark here is already a single
assembled table), but it's kept for genuinely relational/multi-table data.
See the module docstring in `arda/arda.py` for details.
