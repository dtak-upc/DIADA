# DIADA

DIADA is the first system for **automatic data composition**: given a data lake withof scattered attributes, it organizes them into meaningful relations on its own, without being told up front what downstream task they'll be used for.

Data lakes accumulate attributes across many tables, and deciding which of them belong together is normally a manual, per-task effort. A naive join on shared keys just as easily combines genuinely related columns (e.g. a property's size and its market value) with spurious ones that only happen to
share a join key, and classical feature selection has to be redone from scratch for every new prediction target, discarding whatever wasn't relevant to it.

DIADA takes a different, **target-agnostic** approach: given a wide table of candidate attributes, it tests every group of columns for genuine statistical dependence. That is, whether knowing some of them actually changes what you'd expect
from the others, beyond what independence alone would predict. To do so, DIADA employs a principled statistical hypothesis test rather than an arbitrary similarity threshold. Columns found to be dependent are linked together into an association graph; the connected groups in that graph become the composed
relations, and columns unrelated to anything form their own group rather than being silently discarded. Because this composition doesn't reference any particular target, it only needs to be computed once per dataset and can then serve any number of downstream analytical tasks.

This repository also includes a modified version of ARDA (Automatic Relational Data Augmentation), which takes one of DIADA's composed relations plus a specific target column and runs iterative feature selection (RIFS) to pick out the columns that actually help a model, training and evaluating with AutoGluon. Put together, you can point this code at a CSV and a target column and get back a composed, cleaned dataset plus a trained model.

Please note that this is **an experimental research code** accompanying an academic paper and reflects the state of an active research project, not a polished tool. For a more production-oriented version of our work, check the
following [repository](https://github.com/dtak-upc/DIADA-demo).

## Prerequisites

- Python >= 3.9
- A Java runtime on your `PATH` (tested with OpenJDK 11) **needed by default**
<!-- , since the default soundness-scoring engine is `engine="jar"` (see "Two soundness-scoring engines" below), which runs the bundled `src/diada/core/DIADA-0.8.jar` via `java -jar`. `pip install` can't set this up for you; install a JRE/JDK yourself first (e.g. `apt install openjdk-11-jre-headless`, `brew install openjdk`). If you'd rather avoid the Java dependency, pass `engine="lima"` (pure Python, no Java needed) explicitly -- it's a randomized approximation of the jar's output, not a numerically interchangeable substitute (same section). -->

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
├── data/                    Holds base_datasets/ (raw inputs) plus noise/cleaned/output artifacts, either the downloaded ones 
│                            original to the paper (for exact reproduction) or your own regenerated versions
├── experiment_results/      frozen results from the paper, one folder per experiment
└── src/diada/
    ├── paths.py             central path resolution
    ├── core/                 soundness scoring + soundness-graph into dataset construction
    │   ├── diada_tool.py       invoke_diada(): runs the system and returns the soundness table
    │   ├── dataset_builder.py  build_un_datasets() / build_mn_datasets(): threshold the
    │   │                       soundness graph and build the resulting cleaned dataset(s)
    │   ├── DIADA-0.8.jar       the original Java implementation
    ├── arda/                 ARDA: RIFS feature selection + AutoGluon training/eval
    │   └── arda.py             the ARDA class 
    ├── experiments/          reproducibility experiments, one subpackage each
    │   ├── run_all.py                 run any/all experiments in one invocation
    │   ├── results_plotting.py        plotting/analysis helpers shared by Experiments 3 & 4's
    │   ├── experiment_1_scalability/  Mining algorithm vs. Apriori/FP-Growth (scalability_benchmark.py)
    │   ├── experiment_2_ufs/          DIADA vs. UFS baselines (ufs_baselines.py, ufs_benchmark.py)
    │   └── experiment_3_diada_arda/   the full paper experiment -- noise injection ->
    │                                  DIADA cleaning -> ARDA across every dataset x variant
    │                                  (generate_benchmarks.py, run_paper_experiments.py,
    │                                  noise.py, noise_benchmarks.py, spurious_cluster.py,
    │                                  synthetic_targets.py), producing results/reproduced_*.csv
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

<!-- ### Two soundness-scoring engines: LIMA vs. the jar

`invoke_diada(df, num_buckets, engine=...)` in `core/diada_tool.py` supports
two interchangeable-looking but **not numerically interchangeable** engines:

- `"jar"` (the default) -- the original `DIADA-0.8.jar`, invoked via
  `java -jar` (needs a Java runtime, see Prerequisites). Deterministic.
- `"lima"` -- the in-process Python port under `core/LIMA/`. Pure Python,
  no Java needed, but a randomized approximation of the jar's output --
  opt-in only, see below.

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

Practically: the default `engine="jar"` gives bit-for-bit reproducible
soundness scores (useful if you ever need to regenerate a specific
published number) and is what every part of this package uses unless you
override it. If you'd rather avoid the Java dependency, pass
`engine="lima"` explicitly -- but expect it to report meaningfully fewer
"sound" relations than the jar (especially on wide, many-column tables),
and note it isn't seeded, so repeated `engine="lima"` runs on the very same
input can disagree with each other too. If either of those matters for your
dataset, check LIMA's output against the jar's before trusting it. -->

### Configuration reference (`diada-clean` flags / `PipelineConfig` fields)

| Flag | Field | Default | Meaning |
|---|---|---|---|
| `--task` | `task` | required | `classification` or `regression` |
| `--metric` | `metric` | `accuracy` | passed to ARDA/AutoGluon (accuracy/f1 for classification, root_mean_squared_error/mae for regression) |
| `--mode` | `mode` | `both` | `un`, `mn`, or `both` -- which cleaning strategy to run |
| `--threshold` | `threshold` | `4` | soundness-score cutoff for the relevance graph; lower keeps more columns as "related" |
| `--num-buckets` | `num_buckets` | `10` | how DIADA discretizes numeric columns before scoring; `0` = all-string, `1` = typed/unbucketed, `>1` = bin into this many buckets |
| `--num-features` | `num_features` | `10` | exact number of top-ranked features ARDA trains the model with (see below) |
| `--output-dir` | `output_dir` | `diada_pipeline_output` | where generated datasets + results are written |
| `--no-arda` | `run_arda` | on | skip model training, only produce cleaned CSVs |
| `--quiet` | `verbose` | on | suppress progress printing |

`rifs_k`, `rifs_eta`, `cv`, and `random_state` (ARDA/RIFS internals) are
available on `PipelineConfig` but not yet exposed as CLI flags.

**On `num_features`**: `diada-clean` trains exactly one model, using the top-`num_features` ranked columns (capped automatically if a dataset has fewer candidates than that). Thus, pick a number that seems reasonable for your dataset and adjust it if the result isn't good enough. It does *not* try to auto-select the "best" feature count for you; if you want to see how performance changes across every feature count from 1 upward (the sweep the paper's results are built from), that's a
separate, more expensive mode (see `ARDA.fit_transform_full_table(..., num_features=None, max_features=N)` in `arda/arda.py`, or the reproducibility scripts below, which use exactly that sweep.)

## Reproducing the paper's experiments

Three numbered experiments, each its own subpackage under
`src/diada/experiments/`. Check `EXPERIMENTS.md` for what each one measures, its reproducibility status, and known caveats. `python -m diada.experiments.run_all` runs any/all of them in one invocation (`--experiments 1 2 3`, default: all three); each is independently runnable too, and all are resumable/checkpointed since each can take a long time on the full benchmark suite.

- **`python -m diada.experiments.experiment_1_scalability.scalability_benchmark`**: Mining algorithm vs. Apriori/FP-Growth scalability.
- **`python -m diada.experiments.experiment_2_ufs.ufs_benchmark`**: DIADA vs. 16 Unsupervised Feature Selection baselines (+ optional CAE) on how well each drops injected noise columns.
- **`python -m diada.experiments.experiment_3_diada_arda.generate_benchmarks`**: for each benchmark in `benchmarks.yaml` (one entry per dataset in `data/base_datasets/`, giving its target column, task, and metric), injects noise (UN = univariate, MN = multivariate/spurious-cluster), runs it through DIADA, and writes every noisy/cleaned/leftover variant to `data/`. Does **not** run ARDA; this only produces the CSV artifacts. This overwrites whatever is in `data/` with a fresh,
independently-seeded run that will NOT match the downloaded `data/`'s original bytes, even for benchmarks it already covers.
- **`python -m diada.experiments.experiment_3_diada_arda.run_paper_experiments`**:the full reproduction: for every dataset, generates the same noisy/cleaned variants as `generate_benchmarks` and additionally runs ARDA's full  feature-count sweep on five variants of each: raw, noisy-uncleaned (`_un_dirty`/`_mn_dirty`), and DIADA-cleaned (`_un_clean`/`_mn_clean`), matching the `benchmark` naming already used in `experiment_results/experiment_3_diada_arda/results_final_*.csv` (the raw/no-suffix variant is the "clean baseline", that is, ARDA with no noise injection and no DIADA cleaning). Results are written to `results/reproduced_regression.csv` / `results/reproduced_classification.csv`. **The original `results_final_*.csv` files are never overwritten**, so you can always diff a fresh run against the paper's reference numbers.

  This is expensive: by default it's every dataset in `benchmarks.yaml` (17 by default) x 5 variants x up to 40 AutoGluon fits each. Use `--datasets <name> [<name> ...]` and/or `--max-features N` to run a smaller slice, e.g.:

  ```bash
  python -m diada.experiments.experiment_3_diada_arda.run_paper_experiments --datasets diamonds --max-features 10
  ```

<!-- ### Synthetic-target benchmarks (Experiment 4)

`experiments/experiment_3_diada_arda/synthetic_targets.py` and the `*_synth_targets` variants
(`generate_benchmark_noise_un_synth_targets`, written to
`data/noise_un_synth_targets/` / `data/cleaned_un_synth_targets/` by
`generate_benchmarks.py`) are used, but for a separate ablation from the main
results -- Experiment 4 (see `EXPERIMENTS.md`): they build synthetic proxy
targets correlated with subsets of the real features, to test whether ARDA
can be fooled into selecting features that predict a decoy rather than the
real target. That experiment's frozen results live in
`experiment_results/experiment_4_synthetic_targets/results_2targets_*.csv`
(see the same folder's `results_analysis.ipynb` for the analysis, sharing
plotting code with Experiment 3's via
`src/diada/experiments/results_plotting.py`) and are not part of
`run_paper_experiments.py` above -- there's no live-rerun script for this
one yet, ask if you'd like that sweep scripted the same way as Experiments
1-3. -->

## Original ARDA (join-based) pipeline

`ARDA.fit_transform` / `fit_transform_from_connections` in `arda/arda.py`
reproduce the original ARDA paper's full pipeline end to end: coreset construction, join-graph discovery/execution across multiple tables, then the same feature-selection stage described above. This project's own experiments don't use this path (every benchmark here is already a single assembled table), but it's kept for genuinely relational/multi-table data.
See the module docstring in `arda/arda.py` for details.
