# Aldien: formulaic alpha discovery on the CSI 300

Final report, November 2026.

## Summary

Aldien searches a small formula language for stock-ranking signals on the
CSI 300, scores each formula by the absolute ICIR of its daily cross-sectional
IC with 20-day forward returns on the train years 2009–2018, keeps only the
formulas whose train IC survives a multiple-testing correction, and checks the
survivors once on later data. Three search methods were compared on five seeds
with the same evaluation budget: evolution, random search, and evolution with
an LLM proposal operator (local `qwen2.5:7b`), plus an ablation of that
operator without its factor-library retrieval on two seeds. The C++ evaluation
engine was timed against the Python reference. Each run's best survivor was
then checked once on a holdout year (2025) that nothing before had touched.

Findings:

1. **Evolution finds higher-scoring train formulas than random search at the
   same budget, by a modest margin.** Best train |ICIR| 0.286 ± 0.008 against
   0.269 ± 0.016; the mean of the top 10 is where the gap is clear, 0.269
   against 0.226 (Table 3).
2. **The pass count of the statistical filter mostly measures redundancy, not
   signal strength.** Evolution passed 331 formulas, random search 0, yet the
   single best random formula of a seed is as strong as evolution's. A
   population full of near-copies of its best formulas loosens the
   Benjamini–Hochberg threshold for every copy. Counted once per distinct
   signal, evolution found 40 signals over five seeds; random search still 0
   (Table 2).
3. **What passed on train held on test (330 of 331) but not on validation or
   holdout.** On the validation years 2019–2020 the top formulas of every
   method pointed the wrong way (directional ICIR −0.16 to −0.21), and on the
   holdout year 2025 only 1 of the 11 finalists kept its direction, with mean
   holdout ICIR −0.25 for evolution's finalists. The relationships found in
   2009–2018 recur in 2021–2024 and reverse in 2019–2020 and 2025: they are
   regime-dependent, not stable alphas.
4. **The LLM operator did not improve the search, and without retrieval it
   hurt.** With the factor library, evolution + LLM matches plain evolution
   (best 0.288 ± 0.014, 39 distinct signals). Without the library, on the same
   two seeds, the best score fell (0.282 → 0.269 and 0.287 → 0.212) and one
   seed passed nothing. The 7B model's answers were rejected 40% of the time,
   mostly as invalid JSON.
5. **The C++ engine is about 15 times faster than the Python reference and
   gives the same results**: bit-identical alpha values, same IC days, ICIR
   differences at the 1e-16 level, the same top 10 in a full search (Table 1).

## 1. Setup

**Data.** CSI 300 daily OHLCV, 1,213,602 stock-days from 2009-01-05 to
2026-01-09, with a date-aware universe (membership as of each day) and forward
returns at 1, 5, 20 and 30 days. Every row and every forward-return target
carries a split label; a return counts in a split only when its target date is
in the same split, so no return reaches across a boundary.

| Split | Years | Rows | Use |
|---|---|---|---|
| train | 2009–2018 | 703,279 | search fitness, statistical filter |
| validation | 2019–2020 | 145,781 | reported, never used for selection |
| test | 2021–2024 | 290,471 | one look per formula that passed the filter |
| holdout | 2025–Jan 2026 | 74,071 | one look per run's best survivor, at the end |

**Formula language.** Variables `OPEN HIGH LOW CLOSE VOLUME` and numeric
constants; `ABS LOG SIGN RANK` (cross-sectional rank); `ADD SUB MUL DIV`;
`TS_MEAN TS_STD TS_MIN TS_MAX` over an integer window of past days; depth at
most 8. A simplifier rewrites equivalent spellings into one canonical form so
that a formula is evaluated and counted once.

**Fitness.** For each day the IC is the Pearson correlation across stocks
between the formula's values and the 20-day forward return; ICIR is the mean
daily IC over its standard deviation. The fitness is |train ICIR|, and a
formula scores 0 when it has a value on fewer than half of the train rows or an
IC on fewer than half of the 2,411 train days that can have one. The sign of
the train mean IC fixes the formula's direction; every later number is
reported in that direction, so a positive validation, test or holdout ICIR
means the train relationship held.

**Methods and budget.** Seeds 17, 29, 41, 53, 67.

- *Evolution*: population 32, 10 generations, 4 elites, tournament selection,
  subtree mutation (rate 0.7) and crossover (0.3), starting from a fixed pool of
  16 domain-informed formulas (including Alpha#101) filled up with random
  formulas. Unique evaluations per seed: 255, 255, 262, 253, 264.
- *Random search*: the same starting pool, then random formulas until it has
  used exactly as many unique evaluations as evolution did on that seed.
- *Evolution + LLM*: as evolution, but with probability 0.3 a child is asked
  from `qwen2.5:7b` (Ollama, local) given the parent, the most similar factors
  from a 36-factor library and the run's recent failures; an answer that fails
  the validator is replaced by a random mutation. Same budget as evolution.
- *Evolution + LLM, no retrieval*: the same operator with the library removed
  from the prompt, on seeds 17 and 29.

**Statistical filter.** Every unique formula a run evaluated (not only the top
10) gets a t-test of its train daily IC in its train direction with the
Hansen–Hodrick long-run variance (20-day returns of neighbouring days
overlap), and Benjamini–Hochberg controls the false-discovery rate at
q = 0.05 over all formulas tried in that run. The passed formulas are also
grouped into distinct signals: in p-value order, a formula joins an earlier one
when their train daily IC series correlate above 0.9.

**One-look checks.** Formulas that pass the filter are evaluated once on the
test split; a ledger records every look and re-runs reuse it. The holdout check
takes, from each run, the formula that passed the filter, held its direction
on test and has the highest test ICIR (identical signals once), records the
selection in its own ledger, and refuses any different selection afterwards.

**Software.** Python 3.12, pandas 2.x, a pybind11 C++ engine that loads the
panel once and then receives only formulas; 334 tests; CI builds the engine
and runs the suite. The experiments were run with the C++ engine; the Python
reference gives the same results.

## 2. Table 1: evaluation speed, Python against C++

20 formulas (10 from the starting pool, 10 from the random generator), train
split, 20-day horizon; each time is the median of 5 runs after a warm-up.
Machine: Intel i5-13450HX, Windows 11, one core per engine.

| Measure | Python (pandas/NumPy) | C++ engine | Speed-up |
|---|---|---|---|
| Setup: load the panel into the evaluator (once) | 1.22 s | 1.11 s | — |
| Fitness of one formula, median | 971.4 ms | 65.7 ms | 14.8× |
| Fitness of one formula, mean | 1,284.6 ms | 85.6 ms | 15.0× |
| Fitness of one formula, fastest – slowest | 236.9 – 4,114.9 ms | 17.5 – 294.2 ms | 10.1× – 23.7× |
| All 20 formulas | 25.69 s | 1.71 s | 15.0× |
| Formulas per second | 0.8 | 11.7 | — |
| Alpha only (no IC), median | 566.0 ms | 41.4 ms | 13.7× |
| Full evolutionary search, seed 17 (255 formulas) | 367.75 s | 22.16 s | 16.6× |

Both engines gave the same IC days and observation counts for every formula;
the largest ICIR difference was 4.7e-16, and the two full searches found the
same top 10. The speed-up is flat across formula size (10× to 24× per formula)
because both engines are dominated by the per-row work, which the C++ engine
does without pandas' overhead; its rolling windows run in constant time per
row (compensated running sums, Welford variance, monotonic queues) and repeat
pandas 2.x's arithmetic step for step, which is what makes the alpha values
bit-identical. pandas 3 changed that arithmetic (differences of 1e-13 that
nested windows amplify to 1e-6), so the project pins pandas below 3.

The per-formula table is in `data/processed/benchmark/table1.md` after
`python -m python.evaluation.benchmark --with-search`.

## 3. Table 2: effect of the statistical filter

| Method | Seeds | Formulas tested | Passed filter | Pass rate | Distinct signals | Checked on test | Held on test | Held rate |
|---|---|---|---|---|---|---|---|---|
| Evolution | 5 | 1289 | 331 | 25.7% | 40 | 331 | 330 | 99.7% |
| Random | 5 | 1289 | 0 | 0.0% | 0 | 0 | 0 | — |
| Evolution + LLM | 5 | 1289 | 269 | 20.9% | 39 | 269 | 266 | 98.9% |
| Evolution + LLM, no retrieval | 2 | 510 | 102 | 20.0% | 13 | 102 | 102 | 100.0% |

Per seed (evolution / random / evolution + LLM):

| Seed | Best train \|ICIR\| | Passed filter | Distinct signals | BH cutoff (p) |
|---|---|---|---|---|
| 17 | 0.276 / 0.282 / 0.282 | 74 / 0 / 89 | 8 / 0 / 11 | 0.0112 / — / 0.0156 |
| 29 | 0.292 / 0.289 / 0.287 | 27 / 0 / 31 | 7 / 0 / 11 | 0.0053 / — / 0.0057 |
| 41 | 0.280 / 0.251 / 0.273 | 138 / 0 / 109 | 12 / 0 / 13 | 0.0249 / — / 0.0194 |
| 53 | 0.294 / 0.269 / 0.312 | 31 / 0 / 26 | 5 / 0 / 3 | 0.0055 / — / 0.0051 |
| 67 | 0.287 / 0.255 / 0.286 | 61 / 0 / 14 | 8 / 0 / 1 | 0.0112 / — / 0.0024 |

The headline contrast, 331 passed against 0, is not a contrast in signal
strength. On seeds 17 and 29 the best random formula scores as high on train
as the best evolved one, and its p-value is of the same order (on seed 41,
3.1e-4 against 3.2e-4). It fails because Benjamini–Hochberg over 255 formulas
admits the best one only if its p-value is below 0.05/255 ≈ 2e-4, and a lone
strong formula does not get there. An evolved population, by contrast, holds
dozens of near-copies of its best formulas; each copy with a small p-value
raises the threshold for the next, and the cutoffs above (0.005 to 0.025)
are 25 to 125 times the one a single formula faces. The distinct-signal count
corrects the headline (5 to 13 signals per evolved run, not 27 to 138
formulas) but still rests on the loosened threshold, so the filter's verdict
on a population should be read as "these few signals, found many times",
not as 331 discoveries.

That almost every passed formula held its direction on test (2021–2024) is
real information: the train relationships recur four to six years later. The
next two sections show that they do not recur in every period.

## 4. Table 3: search quality

Mean ± sd over seeds; top-k = 10. Validation ICIR is in each formula's train
direction. Diversity is the mean pairwise Jaccard distance between the
operator sets of the top-k formulas.

| Method | Seeds | Best train \|ICIR\| | Mean top-k train \|ICIR\| | Mean top-k validation ICIR (directional) | Direction agreement | Top-k diversity | Passed filter | Distinct signals | Held on test | Top-k from LLM | LLM acceptance |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Evolution | 5 | 0.286 ± 0.008 | 0.269 ± 0.005 | −0.210 ± 0.123 | 0.04 ± 0.09 | 0.343 ± 0.128 | 66.2 ± 44.8 | 8.0 ± 2.5 | 66.0 ± 44.4 | — | — |
| Random | 5 | 0.269 ± 0.016 | 0.226 ± 0.010 | −0.158 ± 0.066 | 0.20 ± 0.10 | 0.685 ± 0.040 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | — | — |
| Evolution + LLM | 5 | 0.288 ± 0.014 | 0.273 ± 0.015 | −0.171 ± 0.057 | 0.02 ± 0.04 | 0.384 ± 0.169 | 53.8 ± 42.3 | 7.8 ± 5.4 | 53.2 ± 42.3 | 2.4 ± 2.9 | 0.60 ± 0.16 |
| Evolution + LLM, no retrieval | 2 | 0.240 ± 0.040 | 0.223 ± 0.053 | −0.198 ± 0.054 | 0.00 ± 0.00 | 0.291 ± 0.205 | 51.0 ± 72.1 | 6.5 ± 9.2 | 51.0 ± 72.1 | 2.0 ± 1.4 | 0.84 ± 0.03 |

**Evolution against random search.** At the same budget, evolution's best
formula scores higher on 3 of 5 seeds and its top 10 scores higher on every
seed (0.269 against 0.226). Random search's top 10 is twice as diverse (0.69
against 0.34): evolution converges on a family of formulas, random search
does not. That convergence is also why evolution's populations pass the
filter in bulk (Section 3).

**Validation.** On 2019–2020 the top-10 formulas of every method had a
negative directional ICIR (evolution −0.21, random −0.16, LLM −0.17), and
almost none kept its train direction (agreement 0.04, 0.20, 0.02). The
validation years are a different regime from both the train years before
them and the test years after them. Validation was never used to select
formulas, so this is an observation, not a leak.

**The LLM operator.** With retrieval, evolution + LLM is level with evolution
on every column, within the seed-to-seed spread. 2.4 of the top 10 formulas
per seed came from the LLM, against an expected 3 at a 0.3 proposal rate, so
its proposals compete at about the rate of mutations. The model's answers
were accepted 60% of the time; the main rejection reason was invalid JSON
(168 of 565 calls over the five seeds), then formulas identical to the parent
and formulas that failed the validator.

The no-retrieval row averages two seeds against five, so the fair comparison
is on seeds 17 and 29 alone:

| Seeds 17 / 29 | Evolution | + LLM, retrieval | + LLM, no retrieval |
|---|---|---|---|
| Best train \|ICIR\| | 0.276 / 0.292 | 0.282 / 0.287 | 0.269 / 0.212 |
| Passed filter (distinct) | 74 (8) / 27 (7) | 89 (11) / 31 (11) | 102 (13) / 0 (0) |
| Top-k from LLM | — | 6 / 5 | 3 / 1 |
| LLM acceptance | — | 0.72 / 0.71 | 0.86 / 0.82 |

Without the library the model's answers are accepted more often, because its
proposals are simpler, but they are weaker: fewer reach the top 10, the best
score drops on both seeds, and on seed 29 the search ends up, like random
search, with nothing that survives the filter. Retrieval is what makes the
operator worth its cost; with it, on this problem and with a 7B model, it
breaks even with mutation. Two seeds are too few to size the gap.

## 5. Holdout check (2025-01-02 to 2025-12-10)

Selection `eee12bde68946297`: from each run, the formula that passed the
filter, held its direction on test, and has the highest test ICIR. Random
search had no eligible formula; the no-retrieval ablation had one (seed 29
passed nothing). The 11 finalists were evaluated on the holdout once.

| Method | Eligible | Checked | Held direction | Significant (p < 0.05) | Mean test ICIR | Mean holdout ICIR |
|---|---|---|---|---|---|---|
| Evolution | 293 | 5 | 0/5 | 0/5 | 0.351 | −0.247 |
| Random | 0 | 0 | 0/0 | 0/0 | — | — |
| Evolution + LLM | 234 | 5 | 1/5 | 0/5 | 0.330 | −0.162 |
| Evolution + LLM, no retrieval | 97 | 1 | 0/1 | 0/1 | 0.379 | −0.301 |

| Method | Formula | Seed | Test ICIR | Holdout ICIR | Held |
|---|---|---|---|---|---|
| Evolution | `LOG(MUL(ADD(ADD(LOG(SUB(CLOSE,TS_MIN(CLOSE,60))),RANK(0)),VOLUME),ADD(CLOSE,DIV(ADD(HIGH,HIGH),DIV(DIV(CLOSE,HIGH),OPEN)))))` | 41 | 0.435 | −0.472 | no |
| Evolution | `DIV(ADD(MUL(LOG(OPEN),TS_MAX(VOLUME,20)),VOLUME),TS_MEAN(2,60))` | 17 | 0.394 | −0.199 | no |
| Evolution | `MUL(LOG(MUL(CLOSE,MUL(2,CLOSE))),TS_STD(VOLUME,2))` | 29 | 0.339 | −0.155 | no |
| Evolution | `DIV(ADD(HIGH,MUL(MUL(ADD(CLOSE,HIGH),TS_MIN(SUB(LOW,VOLUME),5)),SUB(DIV(LOW,LOW),CLOSE))),HIGH)` | 67 | 0.332 | −0.345 | no |
| Evolution | `DIV(SUB(CLOSE,LOW),TS_MEAN(OPEN,60))` | 53 | 0.254 | −0.064 | no |
| Evolution + LLM | `ADD(2,MUL(ADD(LOG(ADD(LOW,TS_MIN(CLOSE,120))),RANK(DIV(MUL(LOW,RANK(-1)),MUL(LOW,TS_MEAN(VOLUME,60))))),TS_STD(VOLUME,30)))` | 17 | 0.434 | −0.166 | no |
| Evolution + LLM | `TS_MEAN(TS_STD(RANK(CLOSE),5),10)` | 29 | 0.394 | 0.013 | yes |
| Evolution + LLM | `RANK(MUL(ADD(ADD(ABS(SIGN(ADD(LOW,VOLUME))),RANK(LOW)),TS_STD(ADD(ADD(RANK(LOW),TS_STD(CLOSE,120)),VOLUME),40)),HIGH))` | 41 | 0.387 | −0.294 | no |
| Evolution + LLM | `MUL(ADD(HIGH,MUL(DIV(CLOSE,HIGH),TS_MIN(SUB(LOW,VOLUME),10))),CLOSE)` | 67 | 0.276 | −0.341 | no |
| Evolution + LLM | `TS_MEAN(TS_STD(TS_STD(MUL(ADD(MUL(OPEN,TS_MAX(LOW,40)),VOLUME),TS_MIN(TS_STD(CLOSE,252),10)),120),40),60)` | 53 | 0.161 | −0.023 | no |
| Evolution + LLM, no retrieval | `MUL(ADD(LOG(TS_MAX(OPEN,60)),TS_STD(CLOSE,60)),VOLUME)` | 17 | 0.379 | −0.301 | no |

The holdout has 228 days with a 20-day return inside it, about eleven
non-overlapping periods, so no single formula's t-test has power; the
direction count is the result. Ten of the eleven finalists reversed, and the
one that held (`TS_MEAN(TS_STD(RANK(CLOSE),5),10)`, a smoothed rank
volatility) has a holdout ICIR of 0.01, which is noise. Formulas with test
ICIR of 0.33 to 0.44, among the strongest in the whole experiment, have
holdout ICIR of −0.2 to −0.5.

Most finalists are a raw `VOLUME` or a volume window multiplied or added into
a price term; because `VOLUME` is orders of magnitude larger than any price,
such a formula ranks stocks mostly by traded volume, that is, by size and
activity. A size and activity premium that holds in 2009–2018 and 2021–2024
and reverses in 2019–2020 and 2025 fits a regime story, not a persistent
inefficiency. The pipeline did what it was built to do: the train-only
filter, the single test look and the single holdout look show that the
signals are not stable, instead of letting the search report them as
discoveries.

## 6. What was left out

- *Distributed workers (RabbitMQ or ZeroMQ).* The whole experiment runs on
  one laptop: a 255-formula search takes 15 to 25 s with the C++ engine, and
  in LLM runs the model sets the pace. The engine already has the shape a
  worker needs (data loaded once, formulas in, fitness out).
- *A persistent factor library fed by discovered formulas.* Every run starts
  from the same pool so that methods and seeds stay comparable.
- *Portfolios, transaction costs, backtests.* The project measures predictive
  IC; a trading strategy is a separate study, and given Section 5 there is
  nothing to trade.
- *Engine-independent tie-breaking.* Formulas with equal fitness to 1e-16 may
  be ordered differently by the two engines, which can change a run after
  many generations. The experiments were run entirely with the C++ engine, so
  this does not affect the results.

## 7. Limitations

- Five seeds, one market, one horizon, one fitness. The seed-to-seed spread
  (e.g. 27 to 138 passed formulas) is large next to the differences between
  methods, so "evolution beats random by 0.02 ICIR" is a tendency, not an
  estimate.
- The LLM is a 7B model running locally; a stronger model, a lower invalid-JSON
  rate or a prompt tuned to this DSL could change Section 4. The ablation has
  two seeds.
- The filter corrects for the number of formulas tried within a run but not
  for the overlap between runs, methods or the project's earlier development
  runs. The holdout check is the only number here that is free of every
  selection step.
- The holdout is one year, about eleven independent 20-day periods.

## 8. Reproduction

```bash
pip install -r requirements.txt
# build the C++ engine: see README
powershell -ExecutionPolicy Bypass -File scripts\run_experiment.ps1 -Engine cpp
powershell -ExecutionPolicy Bypass -File scripts\run_experiment.ps1 -Engine cpp -LlmRate 0.3 -LlmBackend ollama -AblationSeeds 17,29
python -m python.evaluation.benchmark --with-search
python -m python.stats.holdout --dry-run
python -m python.stats.holdout --engine cpp
```

The LLM runs replay from `data/processed/llm/cache.jsonl` when it is present,
so they reproduce without a model; without the cache they need Ollama with
`qwen2.5:7b` and will differ, since the model is not deterministic. Outputs:
`data/processed/experiment/tables.md` (Tables 2 and 3),
`comparison_summary.html` and `.json`, `holdout.md`,
`data/processed/benchmark/table1.md`.
