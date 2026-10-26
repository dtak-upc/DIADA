# DIADA / LIMA research context (handover, 22 Oct 2026)

Written at the end of a long working session so that other sessions can continue without it.
Everything below was established in that session unless marked as an open question.

## 1. The project in one paragraph

DIADA composes datasets automatically: it scores how strongly every pair of columns is related
(with LIMA, or the original DIADA-0.8.jar), builds a graph (an edge where the score is above 4),
and groups columns into datasets for ARDA (feature augmentation / model training):
- UN cleaning: keep every column that has at least one edge (drops isolated noise);
- MN cleaning: keep the connected component that contains the target (drops unrelated clusters).
The research goal is to turn this into a **theory for organising a data lake by statistical
dependencies** (a statistical counterpart of Codd's normal forms), aimed at a top venue, with a
**scalability** angle (DIADA handled tables with hundreds of columns). The immediate technical goal
is detecting **transitive (mediated) dependencies**: in A-B-C where the A-C relation exists only
through B (e.g. two tables joined through a hub key such as country), DIADA should produce {A,B} and
{B,C}, not {A,B,C}.

## 2. Codebase map (C:\Projects\DIADA)

- `src/diada/core/LIMA/` Python port of LIMA. `LIMA/LIMA.py` (main loop: schedule -> sample ->
  update; `relatedness()` builds the column-pair matrix), `LIMA/lattice/` (Lattice, LIMALatticeState:
  Beta counters per node, soundness test, Apriori-style candidate edges), `LIMA/scheduler/`
  (LIMAScheduler: nodes adjacent to candidate edges with nodeGrad > approx=1e-6), `LIMA/sampler/`,
  `data/predicate/` (UnorderedPG: agree/differ; OrderedPG: <, >, = for numeric), `util/stats/betaDist.py`.
- `src/diada/core/diada_tool.py`: `_label_columns` (every column becomes categorical: numbers into 10
  buckets, equal-width or quantile if heavy-tailed), `invoke_diada(df, 10, engine="jar"|"lima")`.
- `src/diada/core/dataset_builder.py`: `build_soundness_graph` (threshold 4), UN/MN dataset builders.
- `src/diada/experiments/experiment_3_diada_arda/`: noise injection (UN: noise_* columns + fully
  reshuffled copies *_shuffled_*; MN: UN + a cluster of mutually correlated spurious_* columns),
  benchmarks in `benchmarks.yaml` (17 datasets), data in `data/noise_un`, `data/noise_mn`.
- `explorations/`: all scripts from this research (nothing there is part of DIADA itself).
- `results/`: outputs of those scripts (folders named after the experiment).

## 3. LIMA mechanics learned (needed to reason about it)

- Works on **pairs of rows** and per-column predicates "the two rows agree / differ". After DIADA's
  bucketing every column is categorical, so only agree/differ is used (order predicates unused).
- Nodes = patterns of predicates; counts per node are Beta distributions; an edge parent -> child
  (adding a predicate) is "sound" if every comparison against the parent-minus-one-column has |z|
  beyond 4 (originally one-sided z < -4; our V1 made it two-sided). z = m / sqrt(v), m = log-ratio of
  pattern probabilities (LIMA's statistic), variance overstated (z ~ 0.43 sd under no link).
- Sampling: pairs drawn at random, rounds of 2^10..2^17 pairs; a node is counted while
  nodeGrad = b/(a(a+b)^2) > 1e-6, i.e. until its own frequency is precise, NOT until its edge test is
  decided. Consequences found: weak real links stop being counted before z reaches 4 (bank
  duration-Class missing in seed 0, present in seed 1); patterns on the path to bigger explored
  patterns get counted ~1e6 times so tiny links become "sound" (Class-pdays); results depend on the
  seed and on what else LIMA explores. Bigger patterns never create new column pairs (Apriori rule),
  they only raise the score of existing ones.
- With enough pairs, almost any column pair becomes "sound" (z grows with sqrt(pairs) while the
  information is limited by rows). Pure-noise columns would pass at ~7-50 million pairs.
- **Fix found by the colleague (Albert Martin), ported on 22 Oct into DIADA (uncommitted)**: commit
  "Skew correction" in C:\Projects\LIMA_py (e393958): LIMA now keeps conditional edge distributions
  (child count given parent count) and tests soundness on them in log-odds, with an adjustment when an
  edge has little evidence; `relatedness()` uses every explored edge. Effect vs the jar on
  diamonds/bank/house_sales/bank_mn (3 seeds): edge overlap with the jar 0.28 -> 0.58-0.60
  (diamonds unchanged at 0.63), MN target group overlap 0.29 -> 0.81 (bank), 0.64 -> 0.93
  (house_sales); speed similar to the jar. Remaining differences: jar-only edges (diamonds grade links
  carat-clarity/color/cut, bank Class-personal_loan, age-marital, campaign-pdays, house_sales lat/long
  links) and Python-only (bank Class-pdays, month-pdays, housing_loan-contact).
  Files: `explorations/new_lima_vs_jar.py`, `results/new_lima_vs_jar/`. A full comparison on all 17
  noise benchmarks (does it change DIADA's UN/MN selections and downstream utility?) was running at
  handover: `explorations/jar_vs_fixed_lima_diada.py` + `jar_vs_fixed_lima_report.py`,
  `results/jar_vs_fixed_lima/`.
- Our V1-V3 explorations override the OLD `_isSound`; with the fix they need adapting.

## 4. Versions built (baselines)

- V0 = DIADA-0.8.jar (deterministic). V1 = Python LIMA with a two-sided test (old LIMA, pre-fix).
- V2 = V1 + splitting a triangle A-B, B-C, A-C into {A,B},{B,C} when B explains A-C.
  Test: pairs whose two rows have the same value of B (each value separately, combined with precision
  weights), LIMA's m and variance for the four agree/differ patterns, equivalence test: split only if
  |m| + 4 sd < 20% of the A-C link (unsure -> keep). Splits applied best-explained first, only while the
  triangle exists (near-duplicates end as a chain). Groups = largest sets connected through remaining
  edges with no split pair (B in both groups; overlapping groups); MN target dataset = union of groups
  containing the target. Code: `explorations/v2_mediation_split.py`.
- V3 = V2 + open paths A-B-C (no A-C edge): same test, bar = 20% of the weaker of A-B, B-C; keep if any
  middle column shows an interaction; separate A and C only if every middle shows none.
  Code: `explorations/v3_open_paths.py`, experiment `explorations/v3_experiment.py`, results `results/v0_v3/`.
- V2 inside LIMA (online pruning of patterns containing split pairs): same decisions, LIMA work -33..-90%
  when splits exist, but it exposed the sampling-allocation problem (lost 27 weak spurious edges in
  bank_mn). `explorations/v2_online.py`, `results/v2_online/`.
- Results (synthetic three-column benchmarks, `explorations/synthetic_triads.py`): V2 splits chain and
  common_cause 5/5, keeps direct_triangle, sum_collider, interaction; hub_join fails (A-C weak);
  V3 fixes hub_join 3/5 (the seeds with no A-C edge). Real benchmarks: V2/V3 make no doubtful splits but
  split very little (the 4-sd margin on LIMA's overstated variance is very conservative).
- Exact-count variant V2.exact (`explorations/exact_counts.py`): exact agreement counts over all pairs,
  row-level shuffles for chance, strength floor |phi| >= 0.01; synthetic all correct incl. hub_join 5/5;
  denser graphs; fast.

## 5. Methodology review (colleague's three points) and where we landed

1. **Proving equality (null = equality cannot be accepted).** Use an equivalence test on the *share of
   the A-C link left once B is known*, with a margin (e.g. B explains >= 80%). The margin is
   unavoidable ("transitive" must be defined approximately, like approximate FDs); make it meaningful,
   calibrate on known structures, or report the structure as a family over epsilon. Absolute differences
   fail for small probabilities; pure ratios fail for rare patterns; the principled "relative error" is
   frequency-weighted (KL / conditional mutual information). Classic tests and BIC fail (large n: nothing
   explained; BIC pushes many-valued columns to independence). Demo: `explorations/equivalence_demo.py`.
2. **Representation (open world).** Agree/differ dependence is sound (never invents dependence) but not
   complete: Cov(agree X, agree Y) = sum e^2 + 2 sum p(x)p(y)e (e = p(x,y)-p(x)p(y)); the second term can
   cancel the first (blind spot, ~2% of linked column pairs in our data, e.g. diamonds cut-clarity;
   dilution more common). It can make V2 split wrongly (constructed example). Fix keeping predicates:
   **HSIC with the same/different kernel** = T1 - 2 T2 + T3 using triples of rows = sum e^2, zero iff
   independent (at the bucket resolution); conditional version per value of B; rarity-weighted agreement
   gives phi^2 (~ 2 ln2 x CMI, pairs suffice); order/closeness kernels for ordered columns are complete
   too and much stronger (duration-Class z 116 -> 188, cut-clarity 23 -> 58). Agreement statistic
   without the correction = Mantel test (known flaw). Margins must be calibrated per kernel (ratios
   differ: plain HSIC is not on the CMI scale; rarity-weighted unstable with near-empty cells).
   Demos: `explorations/blind_spot_demo.py`. Implementation plan described (exact counts or sampling
   triples within B = v, jackknife over row blocks, equivalence decision).
3. **Combining independence statements into minimal sets.** Non-unique minimal sets arise exactly with
   (near-)deterministic relations / duplicates (intersection property fails): handle synonyms/FDs first
   (detectable with predicates), report alternatives. Redundancy does not transfer to bigger sets
   (colliders: sum_collider I(A;C)=0 but I(A;C|B)=1.36 bits): test removals in the current set (graphoid
   rules). Order dependence: use order-independent procedures (PC-stable) or stated tie-breaks.
   Proposed procedure: synonyms first -> PC-style levels with small conditioning sets (k=1 = V2/V3) ->
   grow-shrink Markov blanket for target datasets -> report non-uniqueness. Demo `explorations/point3_demo.py`.

Other established points:
- "Given B" must use pairs with the SAME value of B, per value: pairs differing on B mix kinds of pairs
  and show B's own effect (demos `explorations/b_differ_demo.py`); pooling B values causes Simpson
  reversals (bank pdays-month given housing_loan; `explorations/case4_*.py`, `simpson_examples.py`).
  Real-world reading of that case: the hidden common driver is the call date (campaign period), not
  housing_loan; a link no column explains may signal a missing hub (data-discovery angle).
- PC algorithm (causal-learn, level 1, G-test) on our benchmarks: as fast as V1/V2 (0.4-6 s), correct on
  all synthetic cases incl. hub_join and colliders, but near-complete graphs on real data (significance,
  not strength). `explorations/pc_baseline.py`, `results/pc_baseline/`. Interactive PC demo published as
  a private artifact "PC Algorithm Stepper" (https://claude.ai/artifact/8sgjqfg6gehLQg2NkdomZv), copy
  in `results/pc_demo/pc_stepper.html`.
- Literature to position against: Maimon (Kenig et al., SIGMOD 2020, approximate MVDs / acyclic
  schemes; MVD <-> conditional independence), PC / PC-stable, Markov-blanket methods (IAMB, HITON, MMPC,
  TIE*), HSIC (Gretton et al. 2005), distance covariance (Szekely 2007), Mantel test, causal data
  integration (Youngmann et al. PVLDB 2023), Suna (PVLDB 18), ARDA, Metam, Retrieve-Merge-Predict.
- Possible contributions identified: agree/differ predicates speak the language of FDs (exact and
  statistical dependence in one formalism; determinism handled), cardinality/type independence (keys,
  text; blocking within B for hubs), sampling cost independent of rows, context-specific patterns in
  LIMA's lattice (unused), scalability on wide tables (to be demonstrated vs PC level 1).

## 6. Open decisions (owner: the user)

- Which of the planned changes to implement: V1 stopping-rule fix (count until the edge test is decided,
  with a cap tied to rows), HSIC-based V2/V3 test with row-level intervals, kernel per column type,
  margin calibration / epsilon family, synonyms-first step, larger conditioning sets.
- Ground truth: bnlearn networks and semi-synthetic lakes (e.g. split bank into two tables joined on a
  key) were suggested; the user postponed this.

## 7. Working preferences of the user

- Go slowly; explain with concrete examples and numbers; avoid jargon; answer one point at a time.
- Keep DIADA's original code untouched except when the user explicitly asks (the LIMA fix was such a
  request); put experiments in `explorations/` and outputs in `results/`.
- Use the jar as V0.
- Never add Claude co-author / attribution lines to commits or PRs (the user removed them from GitHub).
