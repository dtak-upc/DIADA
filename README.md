# DIADA

DIADA is the first system for **automatic data composition**: given a data lake withof scattered attributes, it organizes them into meaningful relations on its own, without being told up front what downstream task they'll be used for.

Data lakes accumulate attributes across many tables, and deciding which of them belong together is normally a manual, per-task effort. A naive join on shared keys just as easily combines genuinely related columns (e.g. a property's size and its market value) with spurious ones that only happen to
share a join key, and classical feature selection has to be redone from scratch for every new prediction target, discarding whatever wasn't relevant to it.

DIADA takes a different, **target-agnostic** approach: given a wide table of candidate attributes, it tests every group of columns for genuine statistical dependence. That is, whether knowing some of them actually changes what you'd expect
from the others, beyond what independence alone would predict. To do so, DIADA employs a principled statistical hypothesis test rather than an arbitrary similarity threshold. Columns found to be dependent are linked together into an association graph; the connected groups in that graph become the composed
relations, and columns unrelated to anything form their own group rather than being silently discarded. Because this composition doesn't reference any particular target, it only needs to be computed once per dataset and can then serve any number of downstream analytical tasks.

This repository also includes a modified version of ARDA (Automatic Relational Data Augmentation), which takes one of DIADA's composed relations plus a specific target column and runs iterative feature selection (RIFS) to pick out the columns that actually help a model, training and evaluating with AutoGluon. Put together, you can point this code at a CSV and a target column and get back a composed, cleaned dataset plus a trained model.

Please note that this is **an experimental research code** accompanying an academic [paper](https://arxiv.org/abs/2610.01646) and reflects the state of an active research project, not a polished tool. For a more production-oriented version of our work, check the
following [repository](https://github.com/dtak-upc/DIADA-demo).

## Prerequisites

- Python >= 3.9
- Optional: a Java runtime on your `PATH` (tested with OpenJDK 11), only to run the bundled original `src/diada/core/DIADA-0.8.jar` with `--engine jar` / `engine="jar"`. The default engine is pure Python and gives the same output (see "Soundness-scoring engine" below)

## Install

```bash
git clone <this repo>
cd DIADA
pip install -e .
```

This installs the `diada` package (`pandas`, `numpy`, `scikit-learn`, `scipy`, `networkx`, `matplotlib`, `pyyaml`, and `autogluon.tabular[lightgbm,xgboost]`) and registers the `diada-clean` console script. Everything is also runnable as `python -m diada.<module>` if the console script isn't on your `PATH`.

## Getting the datasets

If you want to reproduce the experiments you need to download the data first. All the data, that is, both the raw `base_datasets/` used in the paper plus the generated noisy/cleaned/output artifacts, exactly as they were when Experiments 2-4's published results were computed, is hosted instead at:

**https://mydisk.cs.upc.edu/s/oqKjXsMaHTYAMyk**

Download and unzip it into the repo root as **`data/`**, so you end up with `data/base_datasets/`, `data/cleaned_un/`, etc., matching the layout below (folders/files here use `un`/`mn` -> univariate-noise / multivariate-noise cleaning, see "UN vs MN" below).

**A note on reproducing exactly:** the noise-injection pipeline
(`experiments/experiment_3_diada_arda/noise.py`) is seeded (`random_state=0` by default, threaded through every RNG call), so it's deterministic *given the exact same code and library versions*. Practically: if you want to reproduce (or build on) the paper's exact published numbers for Experiments 2-4, use the downloaded `data/` as-is, don't regenerate over it. If you'd rather generate your own fresh noise/cleaned variants (e.g. to test a change to the noise-injection
logic itself), that's fine too, but expect the result to differ from the downloaded copy, and treat any results built on it as a new, independent run rather than a reproduction.

None of this is needed just to run `diada-clean` on your own CSV, which
doesn't touch `data/` at all.

## Repository layout

```
DIADA/
├── pyproject.toml
├── benchmarks.yaml          per-dataset config (target column, task, metric) used by the reproducibility scripts below
├── benchmark_2targets.yaml  the same benchmarks twice, once per synthetic target (Experiment 4)
├── data/                    Holds base_datasets/ (raw inputs) plus noise/cleaned/output artifacts, either the downloaded ones 
│                            original to the paper (for exact reproduction) or your own regenerated versions
├── experiment_results/      frozen results from the paper, one folder per experiment
└── src/diada/
    ├── paths.py             central path resolution
    ├── core/                 soundness scoring + soundness-graph into dataset construction
    │   ├── diada_tool.py       invoke_diada(): runs the system and returns the soundness table
    │   ├── dataset_builder.py  build_un_datasets() / build_mn_datasets(): threshold the
    │   │                       soundness graph and build the resulting cleaned dataset(s)
    │   ├── DIADA-0.8.jar       the original Java implementation (alternative engine)
    │   └── LIMA/               Python port of the jar's mining algorithm (default engine) + Experiment 1's baselines
    ├── arda/                 ARDA: RIFS feature selection + AutoGluon training/eval
    │   └── arda.py             the ARDA class 
    ├── experiments/          reproducibility experiments, one subpackage each
    │   ├── run_all.py                 run any/all experiments in one invocation
    │   ├── results_plotting.py        plotting/analysis helpers shared by Experiments 3 & 4's
    │   ├── experiment_1_scalability/  Mining algorithm vs. Apriori/FP-Growth (scalability_benchmark.py)
    │   ├── experiment_2_ufs/          DIADA vs. UFS baselines (ufs_baselines.py, ufs_benchmark.py)
    │   ├── experiment_3_diada_arda/   the full paper experiment -- noise injection ->
    │   │                              DIADA cleaning -> ARDA across every dataset x variant
    │   │                              (generate_benchmarks.py, run_paper_experiments.py,
    │   │                              noise.py, noise_benchmarks.py, spurious_cluster.py,
    │   │                              synthetic_targets.py), producing results/reproduced_*.csv
    │   └── experiment_4_synthetic_targets/  one composition, two synthetic targets
    │                                  (run_synthetic_targets.py)
    ├── pipeline.py            the "system": run DIADA + ARDA on any single CSV
    └── cli.py                 `diada-clean` command-line wrapper over pipeline.py
```

There are two separate ways to use this repository, covered next: running the system on your own data, and reproducing the paper's experiments.

## Using the system on your own CSV

```bash
diada-clean --csv my_data.csv --target price --task regression
```

This runs DIADA cleaning in both UN and MN modes (see "UN vs MN" below), writes every resulting dataset to `./diada_pipeline_output/`, and (unless `--no-arda` is passed) trains one ARDA model per dataset with the requested
number of features, printing a summary. From Python:

```python
from diada.pipeline import PipelineConfig, run_pipeline

config = PipelineConfig(csv_path="my_data.csv", target_column="price", task="regression")
result = run_pipeline(config)

result.dataset_paths   # {"un_kept": Path(...), "un_leftover": Path(...), "mn_target_cluster": Path(...), ...}
result.datasets        # same, but as DataFrames instead of paths
result.arda_results    # {"un_kept": DataFrame(1 row), ...} -- one trained model's metrics per dataset
```

### UN vs MN

DIADA's soundness scores are thresholded into a graph, and that graph is used two ways, named for what each one removes: **UN** (univariate noise) drops columns that don't correlate with anything else; **MN** (multivariate noise) additionally splits out sets of noisy, intra-correlated columns as their own partition(s). 

- **UN**: every column connected to *anything* above the threshold is kept
  in one `un_kept` dataset; everything else goes into `un_leftover`
  (nothing is silently discarded).
- **MN**: the graph's connected components are treated as separate feature
  clusters. The component containing the target column becomes
  `mn_target_cluster`; every other component becomes its own dataset
  (`mn_cluster_2`, `mn_cluster_3`, ...), and columns that never entered the
  graph at all become `mn_leftover`. Every dataset (leftover and
  non-target clusters included) gets the target column appended if it isn't
  already present, so any of them can independently be handed to ARDA.

### Soundness-scoring engine

Soundness is scored by `src/diada/core/LIMA/` by default (`--engine lima` / `PipelineConfig(engine="lima")` / `invoke_diada(..., engine="lima")`): a pure-Python port of the algorithm inside the original Java implementation, `DIADA-0.8.jar`, including its random number generator (Java's `SplittableRandom`, seed 42). It is deterministic and returns the jar's output exactly: every value of the output matrix, on the 4 old benchmarks and on the UN and MN versions of the 17 noise benchmarks (`explorations/jar_like_vs_jar.py`), in about a third of the jar's time. The jar stays available with `--engine jar` (needs Java). With Java's default heap it runs out of memory on the largest benchmarks (e.g. `mice_protein` MN).

With numeric (Integer/Double) columns, i.e. `num_buckets` 0 or 1, the jar skips a few patterns because of an integer overflow in its pattern hash. The port doesn't copy that bug, so a few values differ (the edges were identical on the datasets checked).

### Configuration reference (`diada-clean` flags / `PipelineConfig` fields)

| Flag | Field | Default | Meaning |
|---|---|---|---|
| `--task` | `task` | required | `classification` or `regression` |
| `--metric` | `metric` | `accuracy` | passed to ARDA/AutoGluon (accuracy/f1 for classification, root_mean_squared_error/mae for regression) |
| `--engine` | `engine` | `lima` | soundness-scoring engine: `lima` (pure Python) or `jar` (needs Java); same output, see above |
| `--mode` | `mode` | `both` | `un`, `mn`, or `both` -- which cleaning strategy to run |
| `--threshold` | `threshold` | `4` | soundness-score cutoff for the relevance graph; lower keeps more columns as "related" |
| `--num-buckets` | `num_buckets` | `10` | how DIADA discretizes numeric columns before scoring; `0` = all-string, `1` = typed/unbucketed, `>1` = bin into this many buckets |
| `--num-features` | `num_features` | `10` | exact number of top-ranked features ARDA trains the model with (see below) |
| `--output-dir` | `output_dir` | `diada_pipeline_output` | where generated datasets + results are written |
| `--no-arda` | `run_arda` | on | skip model training, only produce cleaned CSVs |
| `--quiet` | `verbose` | on | suppress progress printing |

`rifs_k`, `rifs_eta`, `cv`, and `random_state` (ARDA/RIFS internals) are
available on `PipelineConfig` but not yet exposed as CLI flags.

A dataset left with nothing but the target column (e.g. `un_leftover` when DIADA keeps every column) is still written to the output folder, but ARDA is skipped for it, since there are no features to select from. Each model is trained by AutoGluon with a 180-second time limit, so expect a few minutes per dataset.

**On `num_features`**: `diada-clean` trains exactly one model, using the top-`num_features` ranked columns (capped automatically if a dataset has fewer candidates than that). Thus, pick a number that seems reasonable for your dataset and adjust it if the result isn't good enough. It does *not* try to auto-select the "best" feature count for you; if you want to see how performance changes across every feature count from 1 upward (the sweep the paper's results are built from), that's a
separate, more expensive mode (see `ARDA.fit_transform_full_table(..., num_features=None, max_features=N)` in `arda/arda.py`, or the reproducibility scripts below, which use exactly that sweep.)

## Reproducing the paper's experiments

Four numbered experiments, each its own subpackage under
`src/diada/experiments/`; each module's docstring lists what it measures and every known deviation from the original scripts. `python -m diada.experiments.run_all` runs any/all of them in one invocation (`--experiments 1 2 3 4`, default: all four, each with its own default settings); each is independently runnable too, and all are resumable/checkpointed since each can take a long time on the full benchmark suite. Experiments 2-4 read `data/` (see "Getting the datasets") and use the default engine (same output as the jar); Experiment 1 benchmarks the Python LIMA itself against Apriori/FP-Growth baselines built on the same code, so it never calls the jar.

Every ARDA model fit (Experiments 3-4 and `diada-clean`) is also appended to a running log, `results/results.csv` (classification) / `results/results_regression.csv` (regression), in addition to each script's own output files.

- **`python -m diada.experiments.experiment_1_scalability.scalability_benchmark`**: Mining algorithm (LIMA) vs. Apriori/FP-Growth row/column scalability (Figure 4 in the paper). Add `--quick` for a few-minute smoke run (written to `results/experiment_1_scalability_quick/`), or select a subset with `--datasets`, `--algorithms`, `--series`, `--max-cols`, `--repeats` (see `--help`). Writes `scalability_raw.csv`, `results_live.pickle` (same schema as `experiment_results/experiment_1_scalability/results.pickle`) and a figure `scalability_live.png` to `results/experiment_1_scalability/`. Note: this sweep has not been rerun since LIMA was replaced by the port of the jar (the previous Python port was about 10x slower than the published LIMA timings).
- **`python -m diada.experiments.experiment_2_ufs.ufs_benchmark`**: DIADA vs. 16 Unsupervised Feature Selection baselines (+ optional CAE, needs `torch`) on how well each drops injected noise columns, read from `data/noise_un/`. `--datasets <name> ...` runs a subset, `--out-dir` changes the output folder (default `results/experiment_2_ufs/`: `ufs_raw.csv`, `ufs_summary.csv`). Some spectral methods (spec, mcfs, ndfs, rsr) build an N x N matrix over the rows and run out of memory on the larger datasets; those failures are recorded in the `error` column, as in the published run.
- **`python -m diada.experiments.experiment_3_diada_arda.generate_benchmarks`**: for each benchmark in `benchmarks.yaml` (one entry per dataset in `data/base_datasets/`, giving its target column, task, and metric), injects noise (UN = univariate, MN = multivariate/spurious-cluster), runs it through DIADA, and writes every noisy/cleaned/leftover variant to `data/`. Does **not** run ARDA; this only produces the CSV artifacts. **It overwrites the corresponding files in `data/`**, including the downloaded ones the published results were computed from. Noise injection is seeded (`random_state=0`), and in our checks it regenerated the downloaded noisy tables exactly, but back up `data/` first if you need the originals untouched.
- **`python -m diada.experiments.experiment_3_diada_arda.run_paper_experiments`**:the full reproduction: for every dataset, generates the same noisy/cleaned variants as `generate_benchmarks` and additionally runs ARDA's full  feature-count sweep on five variants of each: raw, noisy-uncleaned (`_un_dirty`/`_mn_dirty`), and DIADA-cleaned (`_un_clean`/`_mn_clean`), matching the `benchmark` naming already used in `experiment_results/experiment_3_diada_arda/results_final_*.csv` (the raw/no-suffix variant is the "clean baseline", that is, ARDA with no noise injection and no DIADA cleaning). Results are written to `results/reproduced_regression.csv` / `results/reproduced_classification.csv` (`--output-dir` to change). Like `generate_benchmarks`, it **writes the noisy/cleaned artifacts into `data/`**, overwriting existing files for the datasets it runs. **The original `results_final_*.csv` files are never overwritten**, so you can always diff a fresh run against the paper's reference numbers.

  This is expensive: by default it's every dataset in `benchmarks.yaml` (17 by default) x 5 variants x up to 40 AutoGluon fits each. Use `--datasets <name> [<name> ...]` and/or `--max-features N` to run a smaller slice, e.g.:

  ```bash
  python -m diada.experiments.experiment_3_diada_arda.run_paper_experiments --datasets diamonds --max-features 10
  ```

- **`python -m diada.experiments.experiment_4_synthetic_targets.run_synthetic_targets`**: one composition, any target (Section 6.4). For every dataset it injects UN noise plus two synthetic targets, runs DIADA once, and runs ARDA's sweep for each entry of `benchmark_2targets.yaml` (`<dataset>_synthetic_target_<k>`) on three variants: `_base` (no noise), `_noise` (pre-composition) and `_clean` (post-composition), matching the naming in `experiment_results/experiment_4_synthetic_targets/results_2targets_*.csv`. Results go to `results/experiment_4_synthetic_targets/reproduced_2targets_*.csv` and are compared against the frozen ones at the end (`comparison_reproduced.csv`). `--selection-only` skips model training and reproduces just the `selected_features` column (ARDA's ranking is computed before any model is trained), which takes hours instead of days; `--datasets`/`--max-features` run a slice; `--compare-only` re-runs the comparison; `--output-dir` changes the output folder. Unlike Experiment 3, it never writes into `data/`: its intermediate tables go to `<output-dir>/data/`. With `--selection-only`, a check run reproduced the published `selected_features` exactly on every classification dataset; on regression datasets a few near-tied features swap rank (at most 1-2 features at any feature count).


## Original ARDA (join-based) pipeline

`ARDA.fit_transform` / `fit_transform_from_connections` in `arda/arda.py`
reproduce the original ARDA paper's full pipeline end to end: coreset construction, join-graph discovery/execution across multiple tables, then the same feature-selection stage described above. This project's own experiments don't use this path (every benchmark here is already a single assembled table), but it's kept for genuinely relational/multi-table data.
See the module docstring in `arda/arda.py` for details.
