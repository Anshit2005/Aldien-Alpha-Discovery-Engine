# Aldien — Alpha Discovery Engine

Research prototype for formulaic alpha discovery. The deterministic evaluation
pipeline and a single-machine, non-LLM evolutionary baseline are implemented;
LLM-guided search, distributed execution, and statistical validation remain
planned work.

## Overview

Aldien is a research-oriented alpha discovery system designed to automatically generate, evaluate, evolve, and statistically validate quantitative trading factors.

The planned system combines:

- LLM-guided symbolic alpha generation
- Evolutionary search
- A message-queue-based distributed execution architecture
- A high-performance C++ alpha evaluation engine
- Statistical validation and robustness testing

The objective is to bridge symbolic hypothesis generation with scalable and rigorous quantitative validation.

## Current implementation status

### Implemented

- CSI 300 date-aware universe and prepared research data with train,
  validation, test, and holdout splits.
- Symbolic alpha DSL with tokenizer, parser, AST, and expression validator.
- Python reference evaluator and fitness calculation using daily IC and ICIR.
- C++ evaluator and fitness engine with a Python bridge.
- Split-aware fitness: search can use train signals and excludes targets that
  cross the selected split boundary.
- Single-machine evolutionary search without an LLM in `python/search/`, with
  random expression generation, tournament selection, elitism, subtree
  mutation/crossover, expression deduplication, and cached train fitness.
- Random-search baseline using the same AST generator, DSL constraints, train
  fitness evaluator, and validation procedure as evolutionary search.
- Paired comparison runner that matches random-search evaluation budget to the
  unique evaluations used by evolution for each seed.
- Generation progress output and a JSON report containing the top candidates,
  validation results, run configuration, dataset fingerprint, and evaluation
  counts.

The evolutionary search starts from a fixed, code-defined pool of 16
domain-informed expressions and fills the population with random DSL
expressions. The original eight formulas are preserved exactly. The prior
eight-seed pool remains selectable with `--seed-pool legacy`. The expanded pool
includes published Alpha#101 in its original formula, plus price normalization,
relative volatility, volume activity and interaction, and price-range signals.
The search does not load or update a persistent factor library between runs.
The output JSON stores the top reported candidates, not every evaluated
expression.

### Still outstanding

- Persistent, versioned alpha library and promotion rules for adding discovered
  formulas to future seed pools.
- LLM candidate generation or LLM-guided mutation/crossover.
- RabbitMQ task queue and distributed C++ workers (currently on hold).
- Statistical significance and multiple-testing controls, portfolio construction,
  transaction costs, and final backtesting.
- Execution and analysis of the planned repeated-seed comparison; the runner is
  implemented, but the full experiment has not yet been run.

The current evolutionary objective is absolute train ICIR. Validation is used
only to report the top train-ranked candidates and does not affect evolution.
Test and holdout are not used by the search.

Run a baseline experiment from the repository root:

```bash
.venv/bin/python -m python.search.run_evolution \
  --dataset data/processed/csi300_research_dataset.parquet \
  --population-size 32 \
  --generations 10 \
  --seed 17 \
  --output data/processed/evolution_results.json
```

The JSON report contains the search configuration, per-generation summaries,
unique evaluation count, best train-ranked expressions, and validation fitness
for the top candidates. Validation results are not used to select or evolve
candidates. The command prints a summary after each generation and a progress
heartbeat every 60 seconds while evaluating candidates. Change that interval
with `--progress-interval-seconds`.

Run random-only search with a budget matched to the evolutionary run's recorded
`unique_train_evaluations` count. The default is 257, matching the earlier saved
baseline; set it explicitly when comparing against a different evolutionary run.
For a fair paired comparison, give random search the same expanded seed pool as
evolution before it fills the remaining budget with random ASTs:

```bash
.venv/bin/python -m python.search.run_random \
  --dataset data/processed/csi300_research_dataset.parquet \
  --evaluation-budget 257 \
  --seed 17 \
  --seed-pool expanded \
  --output data/processed/random_results.json
```

Omit `--seed-pool expanded` for a pure random-only search.

For paired multi-seed comparisons, random search automatically receives the
same unique-evaluation budget as the evolutionary run for each seed:

```bash
.venv/bin/python -m python.search.run_comparison \
  --dataset data/processed/csi300_research_dataset.parquet \
  --seeds 17 29 41 53 67 \
  --population-size 32 \
  --generations 10 \
  --output-dir data/processed/search_comparison
```

The same runner supports evolutionary ablations. For example, compare against
an evolutionary run with crossover disabled:

```bash
.venv/bin/python -m python.search.run_comparison \
  --dataset data/processed/csi300_research_dataset.parquet \
  --seeds 17 29 41 53 67 \
  --crossover-rate 0 \
  --mutation-rate 0.7 \
  --output-dir data/processed/ablation_no_crossover
```

Use `--mutation-rate 0` to disable mutation or `--no-elitism` to disable elite
copying. Keep all other settings fixed when comparing an ablation to the full
evolutionary method.

The comparison report includes top-K train and validation ICIR summaries,
train-validation degradation and direction agreement, runtime, evaluation
counts, and expression diversity. This runner is implemented but the five-seed
experiment has not yet been run. Keep test and holdout untouched until the
search settings are finalized.

Generate a self-contained HTML dashboard from a completed paired comparison:

```bash
.venv/bin/python -m python.search.report_comparison \
  --input data/processed/search_comparison/comparison_summary.json
```

The dashboard is written beside the input as `comparison_summary.html`. Set
`--output` to choose a different location.

## Architecture

```text
Market Data
     │
     ▼
Universe & Data Preparation
     │
     ▼
LLM-Guided Alpha Generation
     │
     ▼
Evolutionary Search
     │
     ▼
Message Queue
     │
     ▼
Distributed C++ Execution Engine
     │
     ▼
Factor Performance Evaluation
     │
     ▼
Statistical Validation
     │
     ▼
Surviving Alpha Factors
