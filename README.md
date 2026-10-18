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
- Generation progress output and a JSON report containing the top candidates,
  validation results, run configuration, dataset fingerprint, and evaluation
  counts.

The search currently starts from a fixed, code-defined seed list and fills the
population with random DSL expressions. It does not load or update a persistent
factor library between runs. The output JSON stores the top reported candidates,
not every evaluated expression.

### Still outstanding

- Persistent, versioned alpha library and promotion rules for adding discovered
  formulas to future seed pools.
- LLM candidate generation or LLM-guided mutation/crossover.
- RabbitMQ task queue and distributed C++ workers (currently on hold).
- Statistical significance and multiple-testing controls, portfolio construction,
  transaction costs, and final backtesting.
- Repeated-seed experiments and a budget-matched comparison of the non-LLM and
  future LLM-guided search methods.

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
