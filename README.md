# Aldien — Alpha Discovery Engine

Research prototype for formulaic alpha discovery on the CSI 300. Aldien
searches a small formula language with evolution, optionally guided by an LLM,
scores each formula on the train years, keeps only the formulas whose train IC
survives a multiple-testing correction, and checks the survivors once on
later data that the search never saw.

## Overview

A formula such as `RANK(DIV(SUB(CLOSE,TS_MEAN(CLOSE,20)),TS_STD(CLOSE,20)))`
gives every stock a value each day. Its daily IC is the cross-sectional
correlation between those values and the stocks' 20-day forward returns, and
its ICIR is the mean daily IC divided by its standard deviation. The search
looks for formulas with a large train ICIR; everything after the search asks
whether that ICIR was real or the result of trying many formulas.

```text
CSI 300 data ─► splits: train 2009–2018 | validation 2019–2020 | test 2021–2024 | holdout 2025–Jan 2026
                    │
                    ▼
Search (train only): evolution · random search (same budget) · evolution + LLM proposals
                    │  fitness = |train ICIR|, Python or C++ engine
                    ▼
Statistical filter (train only): Hansen–Hodrick t-test per formula,
                    │            Benjamini–Hochberg over every formula tried
                    ▼
Test split: survivors checked once (ledger) ─► report: Tables 2 and 3, HTML dashboard
                    │
                    ▼
Holdout: each run's best survivor checked once, at the very end
```

## What is implemented

- **Data.** A date-aware CSI 300 universe and a prepared panel with forward
  returns at 1, 5, 20 and 30 days. Each row and each forward-return target
  carries its split, and a return only counts in a split when its target date
  is in the same split, so no return reaches across a split boundary.
- **Formula language (DSL).** Variables `OPEN HIGH LOW CLOSE VOLUME`, numeric
  constants, `ABS LOG SIGN RANK`, `ADD SUB MUL DIV`, and
  `TS_MEAN TS_STD TS_MIN TS_MAX` with an integer window; depth at most 8.
  Tokenizer, parser, typed AST, validator, a JSON form for LLM output, and a
  simplifier that rewrites equivalent spellings (`ABS(HIGH)` and `HIGH`) into
  one canonical form so that they are evaluated and ranked once.
- **Evaluation.** A Python reference evaluator (pandas) and a C++ engine
  (pybind11) that loads the panel once and then receives only formulas. On
  the full dataset (pandas 2.2), 60 formulas gave bit-identical alpha values and daily IC
  on both; the C++ engine is about 15 times faster (Table 1,
  `python -m python.evaluation.benchmark`).
- **Fitness.** The absolute train ICIR at a 20-day horizon. A formula scores 0
  when it covers under half of the train rows or has an IC on fewer than half
  of the train days with returns (1,206 of 2,411), so a formula that is
  undefined most of the time cannot win on a few lucky days.
- **Search.** Evolution with tournament selection, elitism, subtree mutation
  and crossover, starting from a fixed pool of 16 domain-informed formulas
  (including Alpha#101) filled up with random formulas. Random search gets
  exactly as many unique evaluations as evolution used, seed by seed.
- **LLM proposal operator.** With probability `--llm-rate`, a child is
  proposed by an LLM instead of by mutation. The prompt holds the parent, the
  most similar factors from a 36-factor library (Jaccard overlap of operators
  and variables), and recent failures. The answer must pass the same
  validator as any other formula; when it does not, the search falls back to
  random mutation. Answers are cached by request, so re-running a seed sends
  nothing and reproduces the run. Backends: Ollama (local, default
  `qwen2.5:7b`), Groq, OpenRouter, Together, any OpenAI-compatible endpoint,
  and a mock for tests. An ablation runs the same operator without the library.
- **Statistical filter.** Every unique formula a run evaluated (not just the
  top 10) is tested on its train daily IC in its train direction. The 20-day
  returns of neighbouring days overlap, so the t-test uses the Hansen–Hodrick
  long-run variance; Benjamini–Hochberg then controls the false-discovery
  rate at 5% over all formulas tried. Because a population holds many
  near-copies of its best formulas, and each copy with a small p-value
  loosens the threshold for the next, the passed formulas are also grouped
  into distinct signals (train daily IC correlation above 0.9), and Table 2
  reports both counts.
- **One look at test and holdout.** Formulas that pass the filter are
  evaluated once on the test split. A ledger records every look, and running
  again reuses the recorded result, so changing a setting can never become a
  second, better look. The holdout check works the same way and comes last
  (see [Holdout check](#holdout-check)).
- **Pipeline and report.** `python -m python.run_pipeline` (or
  `scripts/run_experiment.ps1` / `.sh`) runs every method and seed, the filter
  and the test check, and writes the HTML dashboard and `tables.md`
  (Tables 2 and 3). Interrupted runs resume.
- **Tests and CI.** The pytest suite, the C++ tests (ctest), and a GitHub
  Actions workflow that builds the C++ engine on Ubuntu and runs both.

Validation results are reported but never used to select or evolve formulas;
the test and holdout splits are used only by the one-look checks.

## What was left out, and why

- **Distributed workers (RabbitMQ or ZeroMQ).** The whole experiment fits on
  one laptop: with the C++ engine a 255-formula search takes about 15 s, and
  in LLM runs the model, not the evaluation, sets the pace. A queue would add
  moving parts without changing any result. The C++ engine already has the
  shape a worker needs (data loaded once, then only formulas in and fitness
  out), so workers could be added later without touching the search.
- **A persistent factor library fed by discovered formulas.** Every run starts
  from the same fixed pool, so runs and methods stay comparable.
- **Portfolios, transaction costs and backtests.** The project measures the
  predictive IC of formulas; turning them into a trading strategy is a
  separate study.

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

### Run the whole experiment

The launcher runs evolution and random search for seeds 17, 29, 41, 53 and
67, the statistical filter and the test-split check, and then writes the
report, all into `data/processed/experiment` with a log file. Build the C++
engine first (below) and add `-Engine cpp`, or leave it out to use the Python
engine (about 15 times slower, same results):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_experiment.ps1 -Engine cpp
```

To add LLM-guided evolution as a third method, and the no-library ablation on
two seeds, run into the same folder; finished runs are reused. With Ollama
running and `qwen2.5:7b` pulled:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_experiment.ps1 -Engine cpp -LlmRate 0.3 -LlmBackend ollama -AblationSeeds 17,29
```

On Linux or macOS use `scripts/run_experiment.sh` with `ENGINE=cpp`,
`LLM_RATE`, `LLM_BACKEND` and `ABLATION_SEEDS`. The same steps without the
launcher are `python -m python.run_pipeline` with the same options. Hosted
backends read their API key from the environment (`GROQ_API_KEY`,
`OPENROUTER_API_KEY`, `TOGETHER_API_KEY`), for example from `.env`.

The results:

- `comparison_summary.html`: the dashboard (Tables 2 and 3, charts, LLM
  counts, the formulas found);
- `tables.md`: Table 2 (statistical filter) and Table 3 (factor quality);
- for each method and seed, `<method>_seed_<seed>.json` (the search result),
  `_daily_ic.npz` (train daily IC of every formula tried) and
  `_filtered.json` (filter and test results).

### Holdout check

Once every run above is final, check the finalists on the holdout split
(January 2025 to January 2026), once:

```bash
python -m python.stats.holdout --dry-run
python -m python.stats.holdout --engine cpp
```

The finalists are chosen from train and test results only: from each run
(method and seed), the formula that passed the filter, held its train
direction on test, and has the highest test ICIR. `--dry-run` writes and
prints that selection without touching the holdout. The check writes
`holdout.md` and `holdout_results.json` beside the summary and records the
look in `data/processed/holdout_ledger.jsonl`. Running it again shows the
recorded results; checking a different selection is refused, because a second
look at the holdout would no longer be a clean test. The holdout has about
230 IC days, roughly eleven independent 20-day periods, so how many formulas
keep their direction says more than any single p-value.

### Table 1: engine speed

```bash
python -m python.evaluation.benchmark --with-search
```

times the Python and C++ engines on the same 20 formulas and on a full
search, checks that they agree, and writes
`data/processed/benchmark/table1.md`. Run it with nothing else busy.

### Single searches

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
its length and the values match pandas 2.x bit for bit (pandas 3 changed its
rolling arithmetic; it differs by up to about 1e-10). On the full dataset, 60
formulas gave identical alpha values and identical daily IC on both engines;
ICIR differed by at most 7e-16 because the daily ICs are summed in a different
order. A seed-17 evolution found the same 255 formulas and the same top 10 on
both engines, in 15 s instead of 252 s. A search is still not guaranteed to
follow the same path on both engines: formulas with mathematically equal
scores, such as `X` and `DIV(X,2)`, are ordered by last-digit differences,
which can change which parents are selected. A saved run made with either
engine is reused by the other; each result file records the engine in its
`config`.

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
go the other way. The LLM proposal operator asks the model for this format and
checks its answer exactly this way.

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
