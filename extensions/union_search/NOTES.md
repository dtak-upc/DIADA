# Union search with dependencies: working notes (22 Oct 2026)

Discussion with the user: can DIADA judge what a unionable table T *adds* to a query table Q
(not only whether the union is correct)? Initial idea: compare the DIADA graph of Q with the graph of T.

## Literature checked
- DUST, Khatiwada, Shraga, Miller, EDBT 2026 (arXiv 2509.01012). Returns k tuples that are unionable
  and diverse with respect to Q. Holistic column alignment (RoBERTa column embeddings plus clustering),
  a fine-tuned RoBERTa tuple embedding (about 30 GPU-hours, trained on TUS pairs), pruning,
  hierarchical clustering into k*p clusters, re-ranking by minimum distance to Q. Per-query time:
  alignment 24-46 s, diversification 85 s on SANTOS. Does not model column relationships or value
  distributions, and does not evaluate downstream ML.
- Novel Table Search (ANTs), Kassaie and Miller, arXiv 2603.07235 (March 2026). Table-level novelty,
  with axioms (blatant duplicate, dilution). The score is syntactic: per-attribute (1-syn_sim)^b *
  sem_sim, using Jaccard for large domains and JSD over values for small ones. Scores each attribute
  separately, so there are no relationships between columns. One IMDb regression experiment. Benchmarks
  are TUS, SANTOS and Ugen-v2, with "diluted" copies added.
- SANTOS (Khatiwada et al., SIGMOD 2023) uses semantic column relationships to decide whether a union
  is *correct*. A statistical relationship that differs between tables can therefore also mean a wrong
  match, not novelty.

## Toy experiment (`toy_union.py`, output in `out/`)
Q has 3000 flats (zone, size, rooms, year, price). DIADA (jar) edges of Q: zone-price, size-price,
rooms-price, size-rooms. The candidates:

| T | what it is | graph diff, added / lost edges | union with source column S: direct S links |
|---|---|---|---|
| T1 | same generator | 0 / 0 | none |
| T7 | same generator, 300 rows | 0 / 2 | none |
| T2 | bigger flats, same price rule | 0 / 0 | size (rooms and price are explained by size) |
| T3 | price x1.5 per m2 | 0 / 0 | price (not explained) |
| T4 | price set by zone and year | 1 (year-price) / 2 | price, plus relations that change |
| T5 | each column shuffled | 0 / 4 | none, visible only conditionally (I(S;price)=0.001, I(S;price given size)=0.20) |
| T6 | rooms is really the floor | 0 / 2 | rooms, which has no relations left in T |

Findings:
1. Ranking by symmetric graph difference puts noise first (T5), then a wrong match (T6) and a small
   sample (T7), and misses T2 and T3.
2. Counting only the edges T *adds* is robust, because noise, small samples and wrong matches can only
   remove edges (soundness). It still misses coverage (T2) and level shifts (T3).
3. Union view: Q+T plus a source column S. Union value then becomes a DIADA question with S acting as
   a hub. The categories are redundant (S linked to nothing), coverage (S linked directly only to
   "input" columns, the rest explained), relation change (a direct S link that nothing explains), and
   suspicious (a direct link to a column that lost all its relations in T, or no structure in T).
4. Blind spot: a change in relations alone (T5) leaves no pairwise S edge, so DIADA's Apriori rule never
   explores {S, A, C}. A check on S together with a pair of columns is needed (HSIC per S value).
   Taking the maximum over single conditioning columns over-flags through colliders (T2 zone given
   price, T3 size given price). Proper conditioning sets are needed, i.e. a PC-style procedure on the
   union graph.

## Open questions for the user
- What is "utility"? Exploration (novel information) or a downstream target (ARDA-style)?
- Benchmark: the TUS, SANTOS and UGEN benchmarks label unionability, not utility. A semi-synthetic
  option is to split real tables into Q and T by selection, noise or wrong matches.

## Related work found for the novelty assessment (22 Oct)
- Statistical ingredients that already exist outside data management:
  - CD-NOD (Zhang, Huang et al., IJCAI 2017 / JMLR 2020, arXiv 1903.01672) adds a domain index as an
    extra variable and runs PC, i.e. the "source column S" trick. JCI (Mooij et al., JMLR 2020) is similar.
  - Feature shift localisation (Kulinski, Bagchi, Inouye, NeurIPS 2020, arXiv 2107.06929) finds which
    features shifted, using conditional distribution tests (our part A).
  - DCI (Wang, Squires et al., NeurIPS 2018, arXiv 1802.05631) estimates which edges changed between two
    datasets (our part B). It notes that running PC on each dataset and subtracting misses edges present
    in both with a different strength, which is the flaw of a plain graph difference.
- Table union search context:
  - "Something's Fishy in the Data Lake" (Boutaleb et al., TRL@ACL 2025, arXiv 2505.21329): table union
    search benchmarks are weak; over 90% of ground-truth pairs share at least 50% of exact column names.
  - Table-centric table union search (arXiv 2603.17298, 2026).
  - LakeMLB (2026 preprint): a machine-learning benchmark over data lakes with union scenarios.
- No paper found that uses statistical dependencies between columns to judge what a union adds.
