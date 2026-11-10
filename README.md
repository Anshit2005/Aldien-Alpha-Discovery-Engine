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
- JSON serialization of the AST (`python/dsl/json_io.py`) so formulas can be
  supplied as structured data, for example by an LLM, and checked by the same
  validator as text formulas.
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
- Further repeated-seed experiments and analysis; the comparison runner is
  implemented and writes reproducible per-seed results.

The current evolutionary objective is absolute train ICIR. Validation is used
only to report the top train-ranked candidates and does not affect evolution.
Test and holdout are not used by the search.

## Clone and run

The prepared CSI 300 research dataset used by the search is included at
`data/processed/csi300_research_dataset.parquet` (about 99 MiB). It contains
1,213,602 rows and the train, validation, test, and holdout labels. The raw
upstream files are not included; you do not need them for the search commands
below. The prepared data was derived from the public
[QuantaAlpha Qlib CSI300 dataset](https://huggingface.co/datasets/QuantaAlpha/qlib_csi300).

From the repository root, create a virtual environment and install the Python
dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1` instead of `source .venv/bin/activate`.

Run an evolutionary-search baseline:

```bash
python -m python.search.run_evolution \
  --dataset data/processed/csi300_research_dataset.parquet \
  --population-size 32 \
  --generations 10 \
  --seed 17 \
  --output data/processed/evolution_results.json
```

For a paired random-versus-evolution comparison across five seeds, run:

```bash
python -m python.search.run_comparison \
  --dataset data/processed/csi300_research_dataset.parquet \
  --seeds 17 29 41 53 67 \
  --population-size 32 \
  --generations 10 \
  --output-dir data/processed/search_comparison
```

The comparison can take a long time because each formula is evaluated over the
training data. It prints progress while running. After it completes, generate
the HTML report with:

```bash
python -m python.search.report_comparison \
  --input data/processed/search_comparison/comparison_summary.json
```

### Optional: build the C++ engine and run the full test suite

The search baselines use the Python evaluator and do not need a C++ compiler.
For the C++ engine and integration tests, install CMake 3.16 or newer and a
C++17 compiler, then build the executable and Python extension:

```bash
cmake -S engine -B engine/build -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')"
cmake --build engine/build --parallel
ctest --test-dir engine/build --output-on-failure
```

Generate the small C++ integration-test fixtures from the included research
dataset, then run the tests:

```bash
python -m python.data.export_cpp_test_data
python -m python.data.export_cpp_fitness_data
pytest
```

The C++ build artifacts and generated CSV fixtures remain local and are ignored
by Git.

Without the module, `pytest` still runs everything else: the C++ tests skip
themselves, as do the tests whose fixtures have not been generated.
GitHub Actions (`.github/workflows/ci.yml`) builds the engine on Ubuntu and
runs the C++ tests and the whole suite on every push to `main`, and also runs
the suite once without the module.

#### Windows

Either MSVC (Visual Studio Build Tools) or MSYS2's UCRT64 GCC works. With
UCRT64, which needs no Visual Studio, install [MSYS2](https://www.msys2.org)
and its compiler (`pacman -S mingw-w64-ucrt-x86_64-gcc` in the MSYS2 UCRT64
shell), plus Ninja and pybind11 in your Python (`python -m pip install ninja pybind11`).
Then, in PowerShell from the repository root:

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"
$pybind = python -c "import pybind11; print(pybind11.get_cmake_dir())"
cmake -S engine -B engine/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ "-Dpybind11_DIR=$pybind"
cmake --build engine/build
ctest --test-dir engine/build --output-on-failure
python -c "from python.cpp_engine import alpha_engine_cpp; print(alpha_engine_cpp.__doc__)"
```

`engine/CMakeLists.txt` links the GCC runtime statically for MinGW builds, so
the module loads in the standard Windows Python without the MSYS2 DLLs on
`PATH`.

#### Searching with the C++ engine

Once the module is built, `--engine cpp` makes the searches compute fitness in
C++ (`python.evaluation.cpp_fitness.CppFitnessEvaluator`). The panel is copied
into C++ once and each formula then runs there, daily IC included:

```bash
python -m python.search.run_evolution --engine cpp --seed 17
python -m python.run_pipeline --engine cpp
```

`run_random`, `run_comparison` and `run_pipeline` take the same flag, and the
launchers take `-Engine cpp` (PowerShell) or `ENGINE=cpp` (bash). The default
is the Python engine. The C++ rolling windows repeat pandas' running-sum steps
(Kahan-compensated sums for `TS_MEAN`, Welford updates for `TS_STD`, and a
monotonic queue for `TS_MIN`/`TS_MAX`), so each window costs the same whatever
its length and the values match pandas bit for bit. On the full dataset, 60
formulas gave identical alpha values and identical daily IC on both engines;
ICIR differed by at most 7e-16 because the daily ICs are summed in a different
order. A seed-17 evolution found the same 255 formulas and the same top 10 on
both engines, in 15 s instead of 252 s. A search is still not guaranteed to
follow the same path on both engines: formulas with mathematically equal
scores, such as `X` and `DIV(X,2)`, are ordered by last-digit differences,
which can change which parents are selected. A saved run made with either
engine is reused by the other; each result file records the engine in its
`config`.

`python -m python.evaluation.benchmark` times both engines on the same 20
formulas and writes Table 1 (`data/processed/benchmark/table1.md`); add
`--with-search` to also time a full search on each. Run it on a quiet machine.

### JSON formula format

Besides the text form (`RANK(DIV(SUB(CLOSE,TS_MEAN(CLOSE,20)),TS_STD(CLOSE,20)))`),
a formula can be given as a JSON tree. Each node has a `node_type` of
`VARIABLE`, `CONSTANT`, or `OPERATOR`, a `value`, and for operators a
`children` list. The window of a time-series operator is its second child, an
integer `CONSTANT`:

```json
{"node_type": "OPERATOR", "value": "ADD", "children": [
  {"node_type": "OPERATOR", "value": "TS_MEAN", "children": [
    {"node_type": "VARIABLE", "value": "CLOSE"},
    {"node_type": "CONSTANT", "value": "10"}
  ]},
  {"node_type": "VARIABLE", "value": "VOLUME"}
]}
```

`python.dsl.from_json` builds the same typed AST that `parse` produces, and
rejects wrong arity, unknown operators, non-numeric constants, and non-integer
windows. Run `validate` on the result for the depth, window-range, and variable
checks, exactly as for a parsed text formula. `to_json`, `dumps`, and `loads`
go the other way. This format is intended as the output schema for the planned
LLM proposal operator.

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
counts, and expression diversity. The runner writes one result file per method
and seed, plus a comparison summary. Keep test and holdout untouched until the
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
