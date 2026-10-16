# Aldien — Distributed Alpha Discovery Engine

> **Project status:** Research/engineering prototype in active development  
> **Primary goal:** Build a reproducible, distributed system for discovering, executing, evaluating, and statistically validating symbolic quantitative trading factors ("alphas").

---

## 1. Project Overview

Aldien is a research-oriented **automated alpha/factor discovery engine** for quantitative finance.

The central idea is to combine:

1. **LLM-guided evolutionary search** to generate and evolve symbolic trading formulas.
2. A constrained **symbolic alpha DSL** so generated formulas are machine-readable, validatable, and executable.
3. A custom **C++ execution engine** for fast deterministic evaluation of formulas over historical market data.
4. A **distributed execution architecture** so many candidate formulas can eventually be evaluated in parallel.
5. A fitness/evaluation layer based on financial metrics such as **Information Coefficient (IC)** and **ICIR**.
6. A statistical validation stage intended to reduce false discoveries, redundancy, and overfitting.

The intended high-level workflow is:

```text
Historical Market Data
        |
        v
Date-aware Universe
        |
        v
Research Dataset + Time Splits
        |
        v
Initial Factor Pool
        |
        v
LLM-Guided Evolutionary Search
        |
        v
Candidate Symbolic Alpha
        |
        v
DSL Parsing + Validation
        |
        v
Distributed Task Queue
        |
        v
C++ Alpha Execution Workers
        |
        v
Fitness Evaluation
        |
        v
Evolutionary Selection
        |
        v
Statistical Validation
        |
        v
Validated Alpha Library
```

The project is deliberately being built **bottom-up**. Deterministic infrastructure is implemented before introducing the LLM into the core search loop.

---

# 2. Motivation

Formulaic alpha discovery can be expressed as the problem of finding mathematical expressions over market data that produce predictive signals for future returns.

For example:

```text
RANK(
    DIV(
        SUB(CLOSE, TS_MEAN(CLOSE, 20)),
        TS_STD(CLOSE, 20)
    )
)
```

This represents a standardized deviation of the current close from its recent moving average, followed by cross-sectional ranking.

Aldien is intended to automate the process of proposing, testing, evolving, and validating expressions like this.

The system must solve several problems:

### 2.1 Search space explosion

The number of possible symbolic expressions grows extremely quickly with:

- number of variables
- number of operators
- nesting depth
- time-series window choices
- constants
- combinations of existing factors

Brute-force enumeration is not practical.

### 2.2 Computational cost

Every candidate factor must be evaluated over historical data.

If thousands or millions of candidates are explored, repeatedly interpreting expressions in Python becomes expensive.

Therefore, the project includes a custom C++ execution engine.

### 2.3 Invalid expressions

An LLM or evolutionary algorithm can generate syntactically invalid or semantically invalid expressions.

Aldien therefore has:

```text
Tokenizer
   ->
Parser
   ->
AST
   ->
Validator
   ->
Executable expression
```

### 2.4 Statistical overfitting

A factor can appear successful simply because it was tested many times.

Therefore, the architecture separates:

- training/search
- validation
- testing
- holdout evaluation
- future statistical validation

### 2.5 Distributed execution

The eventual search process should not depend on a single machine evaluating every candidate.

The intended architecture distributes candidate evaluation across workers.

---

# 3. Design Philosophy

The project follows several important principles.

## 3.1 Deterministic first, LLM later

The LLM should not be introduced until the following are reliable:

- dataset
- expression language
- parser
- validator
- evaluator
- C++ evaluator
- fitness calculation
- Python/C++ equivalence

Otherwise, debugging becomes impossible because a failure could originate from either the model or the infrastructure.

## 3.2 Python as orchestration/reference layer

Python is currently responsible for:

- dataset preparation
- research splits
- DSL parsing
- validation
- reference evaluation
- fitness calculation
- C++ interface
- future evolutionary/LLM orchestration

Python is also the reference implementation against which C++ behavior can be tested.

## 3.3 C++ as execution layer

C++ is intended to become the high-performance execution engine.

The C++ engine currently supports:

- AST representation
- market data representation
- expression evaluation
- time-series operations
- cross-sectional ranking
- fitness calculation

## 3.4 Separate signal split from target availability

A subtle but important design rule is that a row's research split represents the split of the **signal observation**, while its future return can cross a temporal boundary.

For example, a training observation near the end of the training period may have a future return whose target date lies in validation.

For strict split-aware fitness, such a target must not be used as training fitness.

The dataset therefore stores both:

```text
split
target_split_1d
target_split_5d
target_split_20d
target_split_30d
```

---

# 4. Repository Structure

The intended repository structure is approximately:

```text
Aldien-Alpha Discovery Engine/
│
├── data/
│   ├── raw/
│   │   └── qlib_csi300/
│   │       ├── daily_pv.h5
│   │       └── instruments/
│   │           ├── all.txt
│   │           ├── csi300.txt
│   │           ├── csi500.txt
│   │           ├── csi800.txt
│   │           ├── csi1000.txt
│   │           └── csiall.txt
│   │
│   └── processed/
│       ├── csi300_universe.parquet
│       ├── csi300_market_data_clean.parquet
│       ├── csi300_returns.parquet
│       ├── csi300_research_dataset.parquet
│       ├── csi300_research_summary.txt
│       ├── cpp_fitness_data.csv
│       └── cpp_split_fitness_data.csv
│
├── python/
│   ├── data/
│   │   ├── inspect_dataset.py
│   │   ├── inspect_universe.py
│   │   └── ...
│   │
│   ├── dsl/
│   │   ├── __init__.py
│   │   ├── ast.py
│   │   ├── tokenizer.py
│   │   ├── parser.py
│   │   └── validator.py
│   │
│   ├── evaluation/
│   │   ├── evaluator.py
│   │   ├── fitness.py
│   │   └── cpp_ast_bridge.py
│   │
│   └── cpp_engine/
│       └── __init__.py
│
├── engine/
│   ├── CMakeLists.txt
│   ├── include/
│   │   ├── Data.h
│   │   ├── AST.h
│   │   ├── Evaluator.h
│   │   ├── CSVLoader.h
│   │   └── Fitness.h
│   │
│   ├── src/
│   │   ├── AST.cpp
│   │   ├── Evaluator.cpp
│   │   ├── CSVLoader.cpp
│   │   ├── Fitness.cpp
│   │   ├── main.cpp
│   │   └── bindings.cpp
│   │
│   ├── tests/
│   │   └── test_evaluator.cpp
│   │
│   └── build/
│
├── tests/
│   ├── dsl/
│   ├── evaluation/
│   └── integration/
│
├── pyproject.toml
├── README.md
└── .gitignore
```

The exact file list may change as development continues.

---

# 5. Dataset

## 5.1 Source

The current market dataset is based on the Qlib CSI300 data.

Raw files include:

```text
data/raw/qlib_csi300/daily_pv.h5
data/raw/qlib_csi300/instruments/csi300.txt
```

The raw HDF5 market table contains approximately:

```text
Rows:        14,215,449
Columns:     6
Columns:     open, close, high, low, volume, factor
Dates:       4,138 trading days
Instruments: 5,982
```

The CSI300 membership file is date-aware and contains constituent intervals.

---

# 6. Universe Construction

The project does NOT simply filter the market data by a static list of CSI300 instruments.

Instead, membership is treated as date-dependent.

The processed universe currently contains:

```text
Trading days:          4,138
Rows:                  1,241,394
Minimum constituents:  298
Maximum constituents:  300
Mean constituents:     300
Median constituents:   300
```

There are only 298 constituents on:

```text
2009-12-29
2009-12-30
2009-12-31
```

This was investigated and is associated with a membership transition rather than being treated as a generic data-processing error.

---

# 7. Clean Market Dataset

The cleaned CSI300 market dataset is:

```text
data/processed/csi300_market_data_clean.parquet
```

Current characteristics:

```text
Rows:              1,214,465
Unique instruments: 823
Trading dates:     4,138
Duplicate pairs:   0
Missing OHLCV:     0
Rows outside       universe: 0
Invalid values:    0
```

The raw filtered dataset contained rows where all OHLCV values were missing simultaneously.

The decision was:

> Do not impute missing market observations. Remove incomplete observations from the research dataset.

This is important because artificial imputation of market prices could create synthetic signals.

---

# 8. Returns Dataset

The processed return dataset is:

```text
data/processed/csi300_returns.parquet
```

Current 1-day return statistics:

```text
Valid 1D returns:  1,209,145
Missing 1D:        5,320

Mean:              0.0004078363
Std:               0.02534599
Minimum:           -0.200179
Median:            0
Maximum:           0.2792472
```

There are 4,497 rows following a data gap.

No absolute return greater than 100% was observed.

---

# 9. Research Dataset

The main research dataset is:

```text
data/processed/csi300_research_dataset.parquet
```

It currently contains:

```text
Rows:    1,213,602
Columns: 30
```

Schema:

```text
datetime
instrument
open
high
low
close
volume
factor
trading_day_id
split

future_close_1d
target_date_1d
target_split_1d
forward_return_1d
log_forward_return_1d

future_close_5d
target_date_5d
target_split_5d
forward_return_5d
log_forward_return_5d

future_close_20d
target_date_20d
target_split_20d
forward_return_20d
log_forward_return_20d

future_close_30d
target_date_30d
target_split_30d
forward_return_30d
log_forward_return_30d
```

---

# 10. Research Splits

The current signal-side split counts are:

```text
train:       703,279
validation:  145,781
test:        290,471
holdout:      74,071
```

The system supports four splits:

```text
train
validation
test
holdout
```

The intended role is:

### Train

Used for factor discovery and evolutionary search.

### Validation

Used for selecting/promoting candidates during development without directly training on them.

### Test

Used for final experimental evaluation.

### Holdout

Kept isolated until the final evaluation stage.

---

# 11. Forward Returns

Available forward-return coverage:

```text
Horizon       Valid rows
--------------------------------
1 day         1,207,410
5 days        1,196,561
20 days       1,164,393
30 days       1,144,514
```

Coverage:

```text
1D     99.49%
5D     98.60%
20D    95.95%
30D    94.31%
```

The project currently supports these four fitness horizons:

```text
1
5
20
30
```

---

# 12. Target Split Semantics

For every forward-return horizon there is a corresponding target split:

```text
target_split_1d
target_split_5d
target_split_20d
target_split_30d
```

Example:

```text
signal split = train
target split = validation
```

means the signal was observed in the training period but its future target lies in validation.

For strict train fitness, this target should not contribute to the training score.

Therefore:

```text
usable target
=
forward return exists
AND
target split == signal split
```

This distinction is essential to prevent temporal leakage.

---

# 13. Symbolic Alpha DSL

The project has its own symbolic expression language.

Current variables:

```text
OPEN
HIGH
LOW
CLOSE
VOLUME
```

Current binary operators:

```text
ADD
SUB
MUL
DIV
```

Current time-series operators:

```text
TS_MEAN
TS_STD
TS_MIN
TS_MAX
```

Current unary/cross-sectional operators:

```text
ABS
LOG
SIGN
RANK
```

Example:

```text
RANK(
    DIV(
        SUB(CLOSE,TS_MEAN(CLOSE,20)),
        TS_STD(CLOSE,20)
    )
)
```

---

# 14. DSL Pipeline

The DSL follows:

```text
String
  |
  v
Tokenizer
  |
  v
Parser
  |
  v
AST
  |
  v
Validator
  |
  v
Evaluator
```

The Python AST contains node types such as:

```text
ASTNode
Variable
Number
UnaryOp
BinaryOp
TimeSeriesOp
```

The validator enforces constraints including:

```text
Maximum depth: 8
Maximum window: 252
Finite numeric constants
Allowed variables only
Allowed operators only
Valid node types
```

---

# 15. Python Reference Evaluator

Implemented:

```text
python/evaluation/evaluator.py
```

The evaluator supports:

- variables
- constants
- ABS
- LOG
- SIGN
- RANK
- ADD
- SUB
- MUL
- DIV
- TS_MEAN
- TS_STD
- TS_MIN
- TS_MAX

Output format:

```text
datetime
instrument
alpha
```

Time-series operations currently operate using per-instrument rolling windows.

Cross-sectional `RANK` operates by date.

The current rolling implementation uses row-count windows per instrument.

Gap-aware semantics are a future consideration.

---

# 16. Python Fitness Engine

Implemented:

```text
python/evaluation/fitness.py
```

The fitness engine computes daily cross-sectional Pearson IC between:

```text
alpha
```

and:

```text
future return
```

For each trading day:

```text
IC_t = corr(alpha_t, return_t)
```

The aggregate statistics include:

```text
Mean IC
IC standard deviation
ICIR
Number of IC days
Number of valid observations
Coverage
```

ICIR is calculated conceptually as:

```text
ICIR = mean(IC) / std(IC)
```

If IC standard deviation is effectively zero, ICIR is reported as NaN rather than producing an invalid infinite statistic.

---

# 17. Split-Aware Fitness

The Python fitness API currently looks conceptually like:

```python
evaluator.evaluate(
    ast,
    horizon=20,
    split="train",
)
```

Supported splits:

```text
train
validation
test
holdout
```

Supported horizons:

```text
1
5
20
30
```

The Python implementation:

1. selects rows belonging to the requested signal split
2. evaluates the alpha on that split
3. joins the appropriate future-return column
4. excludes target returns whose target split does not match the requested split
5. computes daily IC
6. aggregates the results

This is the current reference behavior that the C++ interface should match.

---

# 18. C++ Execution Engine

The C++ engine lives under:

```text
engine/
```

Its main components are:

```text
Data.h
AST.h
Evaluator.h
CSVLoader.h
Fitness.h
```

and:

```text
AST.cpp
Evaluator.cpp
CSVLoader.cpp
Fitness.cpp
bindings.cpp
main.cpp
```

---

# 19. C++ Data Representation

The C++ market data representation stores:

```text
datetime
instrument
open
high
low
close
volume
```

and forward returns when required by fitness evaluation.

The data must be sorted appropriately for the evaluator.

The current implementation assumes instrument/date ordering for time-series operations.

---

# 20. C++ AST

The C++ AST supports node categories:

```text
VARIABLE
NUMBER
UNARY_OP
BINARY_OP
TIME_SERIES_OP
```

Variables include:

```text
OPEN
HIGH
LOW
CLOSE
VOLUME
```

Operators mirror the Python DSL.

Factory methods allow construction of C++ AST nodes.

---

# 21. C++ Evaluator

The C++ evaluator implements:

### Variables

```text
OPEN
HIGH
LOW
CLOSE
VOLUME
```

### Constants

Numeric constants.

### Unary operations

```text
ABS
LOG
SIGN
RANK
```

### Binary operations

```text
ADD
SUB
MUL
DIV
```

### Time-series operations

```text
TS_MEAN
TS_STD
TS_MIN
TS_MAX
```

Division by zero/non-finite results is handled as invalid/NaN rather than allowing invalid numerical values to propagate as normal values.

`TS_STD` uses sample standard deviation (`ddof=1`) to match the Python/pandas rolling implementation.

---

# 22. Cross-Sectional Ranking

The C++ implementation of `RANK` groups observations by date.

It calculates average ranks for ties and converts them into a percentile-style rank.

This was implemented specifically so that C++ and Python evaluations are semantically aligned.

---

# 23. C++ Fitness Engine

The C++ fitness engine is implemented in:

```text
engine/include/Fitness.h
engine/src/Fitness.cpp
```

It computes:

```text
Mean IC
IC Std
ICIR
Number of IC days
Valid observations
Coverage
```

The C++ fitness engine itself is intentionally relatively generic.

Split filtering is currently handled by the Python wrapper before data is passed to C++.

This keeps research-specific dataset/split policy in Python while leaving the C++ engine focused on deterministic numerical execution.

---

# 24. Python ↔ C++ Integration

The project uses **pybind11**.

The binding module is:

```text
alpha_engine_cpp
```

The Python wrapper is:

```text
python/evaluation/cpp_engine.py
```

and the AST bridge is:

```text
python/evaluation/cpp_ast_bridge.py
```

The intended pipeline is:

```text
Python expression
       |
       v
Python parser
       |
       v
Python AST
       |
       v
AST bridge
       |
       v
C++ AST
       |
       v
C++ evaluator
```

The C++ engine can also receive an already-created C++ AST directly.

---

# 25. Standalone C++ Executable

There is also a standalone executable:

```text
engine/build/alpha_engine
```

It is primarily used for C++ fitness execution/debugging.

It currently loads:

```text
data/processed/cpp_fitness_data.csv
```

using a relative path from the executable/build location.

Therefore, when manually running it, use:

```bash
cd engine/build
./alpha_engine
```

rather than launching it from an arbitrary working directory.

A previous successful run produced:

```text
Rows: 7304

Mean IC: -0.0520248771269
IC Std: 0.327282677141
ICIR: -0.158960069569
Number of IC days: 481
Valid observations: 6696
Coverage: 0.916757940854
```

These values were used to verify that the standalone C++ fitness implementation itself was functioning.

---

# 26. Test Suite

The project has extensive tests for the deterministic layers.

The DSL currently has:

```text
42 tests passing
```

The Python fitness layer currently has:

```text
15 tests passing
```

Important fitness tests include:

- supported horizons
- invalid horizons
- train split
- validation split
- test split
- holdout split
- invalid split
- split isolation
- cross-split target exclusion
- missing target split
- missing return
- NaN alpha
- result serialization
- missing required columns

---

# 27. Python/C++ Equivalence Testing

The project has integration tests intended to establish that the Python reference evaluator and C++ evaluator produce the same values.

Tests have been performed for expressions such as:

```text
RANK(
    SUB(
        CLOSE,
        TS_MEAN(CLOSE,2)
    )
)
```

and on real data:

```text
RANK(
    DIV(
        SUB(CLOSE,TS_MEAN(CLOSE,20)),
        TS_STD(CLOSE,20)
    )
)
```

A previous real-data comparison involved:

```text
7304 rows
6962 valid values
342 NaN values
```

and matched between Python and C++.

C++/Python fitness equivalence was also previously established for a non-split fixture with:

```text
Mean IC:             -0.0520248771269
IC Std:               0.327282677141
ICIR:                -0.158960069569
IC days:              481
Valid observations:  6696
Coverage:             0.916757940854
```

---

# 28. Important Fixture Distinction

There are two different fitness fixtures.

## Legacy fixture

```text
data/processed/cpp_fitness_data.csv
```

This was created for the original Python/C++ equivalence work.

It contains:

```text
7304 rows
18 instruments
500 dates
```

and does NOT contain split metadata.

It contains:

```text
datetime
instrument
open
high
low
close
volume
forward_return_1d
forward_return_5d
forward_return_20d
forward_return_30d
```

## Split-aware fixture

```text
data/processed/cpp_split_fitness_data.csv
```

This was introduced for split-aware fitness tests.

It contains:

```text
datetime
instrument
open
high
low
close
volume
split

forward_return_1d
target_split_1d

forward_return_5d
target_split_5d

forward_return_20d
target_split_20d

forward_return_30d
target_split_30d
```

Do not assume these two fixtures are interchangeable.

---

# 29. Current Development State

The deterministic foundation is substantially implemented.

### Completed

```text
[✓] Dataset inspection
[✓] CSI300 membership processing
[✓] Date-aware universe
[✓] Market-data cleaning
[✓] Forward returns
[✓] Research splits
[✓] Target split metadata
[✓] Python DSL
[✓] AST
[✓] Tokenizer
[✓] Parser
[✓] Validator
[✓] Python evaluator
[✓] Python fitness
[✓] C++ AST
[✓] C++ evaluator
[✓] C++ fitness
[✓] pybind11 interface
[✓] Python → C++ AST bridge
[✓] Python/C++ numerical equivalence work
[✓] Basic integration tests
```

However, the repository was recently being debugged around compatibility between older tests and the newer split-aware API.

**Do not assume the working tree contains all of the above after a Git reset.**

The user may intentionally reset the repository to the last known commit before continuing development.

Always run:

```bash
git status
git log --oneline -5
pytest -v
```

after restoring the repository.

---

# 30. Known Technical Issues / Things to Verify

## 30.1 `CppAlphaEngine.evaluate()` API

There were tests that called:

```python
engine.evaluate(ast)
```

where `ast` was already a C++ `ASTNode`.

The wrapper had previously been changed to route everything through:

```python
evaluate_expression()
```

which expects a string.

That produced:

```text
TypeError:
object of type 'alpha_engine_cpp.ASTNode' has no len()
```

because the Python tokenizer attempted to tokenize a C++ AST object.

Correct API design should distinguish:

```text
evaluate("CLOSE")
```

from:

```text
evaluate(cpp_ast)
```

Do not make the parser receive an already-built C++ AST.

---

# 31. Important Semantic Issue to Resolve

There is an important subtlety in the split-aware C++ fitness wrapper.

The Python reference behavior is conceptually:

```text
1. Filter signal rows by split
2. Evaluate alpha on all signal rows
3. Mask target returns whose target_split != signal split
4. Calculate fitness
```

A tempting implementation is instead:

```text
1. Filter signal rows by split
2. Filter rows again by target_split == split
3. Evaluate alpha
4. Calculate fitness
```

These are not necessarily equivalent because time-series operators depend on historical rows.

For example:

```text
TS_MEAN(CLOSE,20)
```

should be calculated from the complete signal-side history available within the split, rather than artificially removing observations merely because their future target crosses a split boundary.

Therefore, before finalizing split-aware C++ fitness, verify that the C++ wrapper follows the same semantics as the Python reference.

---

# 32. Another Important Semantic Issue: Rolling Windows

Current Python/C++ rolling semantics are based primarily on **row-count windows per instrument**.

This assumes appropriately ordered observations.

Future work should consider whether windows should instead be explicitly defined over trading-day IDs.

For example:

```text
TS_MEAN(CLOSE,20)
```

should ideally mean:

> mean over the previous 20 trading observations for that instrument

with clear behavior around:

- missing observations
- trading halts
- data gaps
- constituent changes

This should be documented and tested before large-scale alpha search.

---

# 33. Planned Architecture

The complete intended architecture is:

```text
                         ┌─────────────────────┐
                         │ Historical Dataset  │
                         └──────────┬──────────┘
                                    │
                                    v
                         ┌─────────────────────┐
                         │ Research Dataset    │
                         │ + Time Splits       │
                         └──────────┬──────────┘
                                    │
                                    v
                         ┌─────────────────────┐
                         │ Initial Alpha Pool  │
                         └──────────┬──────────┘
                                    │
                                    v
                         ┌─────────────────────┐
                         │ LLM Evolutionary    │
                         │ Search              │
                         └──────────┬──────────┘
                                    │
                                    v
                         ┌─────────────────────┐
                         │ DSL Parser /        │
                         │ Validator           │
                         └──────────┬──────────┘
                                    │
                                    v
                         ┌─────────────────────┐
                         │ Task Queue           │
                         │ RabbitMQ             │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    v               v               v
              ┌──────────┐    ┌──────────┐    ┌──────────┐
              │ Worker 1 │    │ Worker 2 │    │ Worker N │
              │ C++      │    │ C++      │    │ C++      │
              └────┬─────┘    └────┬─────┘    └────┬─────┘
                   │               │               │
                   └───────────────┼───────────────┘
                                   v
                         ┌─────────────────────┐
                         │ Fitness Evaluation  │
                         └──────────┬──────────┘
                                    │
                                    v
                         ┌─────────────────────┐
                         │ Selection /         │
                         │ Evolution           │
                         └──────────┬──────────┘
                                    │
                                    v
                         ┌─────────────────────┐
                         │ Statistical         │
                         │ Validation          │
                         └──────────┬──────────┘
                                    │
                                    v
                         ┌─────────────────────┐
                         │ Validated Alpha     │
                         │ Library             │
                         └─────────────────────┘
```

---

# 34. Development Roadmap

The original planned roadmap is approximately 12 weeks.

## Phase 1 — Dataset and Research Infrastructure

### Week 1 — Dataset

Status:

```text
DONE
```

Tasks completed:

- obtain Qlib CSI300 data
- inspect HDF5
- inspect instruments
- process membership
- construct date-aware universe
- clean OHLCV
- compute returns
- construct forward targets
- create time splits

---

## Phase 2 — Symbolic Execution

### Week 2 — Factor DSL and AST

Status:

```text
MOSTLY DONE
```

Completed:

- tokenizer
- parser
- AST
- validator
- expression syntax
- unit tests

Remaining:

- potentially expand operator library
- improve semantic validation
- define canonicalization rules
- define expression complexity metrics

---

## Week 3 — C++ Execution Engine

Status:

```text
MOSTLY DONE
```

Completed:

- C++ AST
- data structures
- expression evaluation
- rolling operators
- ranking
- C++ tests

Remaining:

- performance benchmarks
- memory optimization
- larger-data execution
- robust error handling
- explicit rolling/gap semantics

---

## Week 4 — Fitness Engine

Status:

```text
MOSTLY DONE
```

Completed:

- daily IC
- IC mean
- IC standard deviation
- ICIR
- coverage
- observation counts
- Python implementation
- C++ implementation

Remaining:

- finalize split semantics
- add more portfolio-level metrics
- benchmark performance

---

## Week 5 — Python/C++ Interface

Status:

```text
MOSTLY DONE
```

Completed:

- pybind11
- AST conversion
- C++ evaluator exposure
- C++ fitness exposure
- Python/C++ equivalence tests

Remaining:

- clean public API
- resolve legacy/new fixture compatibility
- finalize `CppAlphaEngine.evaluate()` semantics
- improve error handling

---

## Week 6 — Message Queue

Status:

```text
NOT IMPLEMENTED YET
```

Planned technology:

```text
RabbitMQ
```

Expected task:

```text
Candidate Alpha
        |
        v
Task Queue
        |
        v
Worker
```

A task should contain enough information for a worker to reproduce the evaluation.

Potential task fields:

```json
{
  "task_id": "...",
  "expression": "...",
  "horizon": 20,
  "split": "train",
  "dataset_version": "..."
}
```

The exact schema should be designed before implementation.

---

## Week 7 — Distributed C++ Workers

Status:

```text
NOT IMPLEMENTED YET
```

Planned:

```text
RabbitMQ
    |
    +--> Worker 1
    +--> Worker 2
    +--> Worker 3
    +--> ...
```

Each worker should:

1. receive an alpha expression
2. validate/compile it
3. evaluate it using the C++ engine
4. calculate fitness
5. return the result

The worker should be stateless where practical.

---

## Week 8 — Evolutionary Search Without LLM

Status:

```text
NOT IMPLEMENTED YET
```

This should be implemented before integrating the LLM.

Basic evolutionary loop:

```text
Initial population
        |
        v
Evaluate
        |
        v
Select
        |
        v
Mutation / Crossover
        |
        v
New population
        |
        v
Repeat
```

This provides a baseline against which LLM-guided search can later be compared.

---

## Week 9 — LLM Evolutionary Operator

Status:

```text
NOT IMPLEMENTED YET
```

The LLM should be used to improve operations such as:

- mutation
- crossover
- explanation
- candidate generation
- candidate repair

The LLM should NOT directly execute arbitrary code.

It should output constrained symbolic expressions that are checked by:

```text
Parser
+
Validator
```

Potential LLM flow:

```text
Parent Factors
     +
Fitness Feedback
     +
Search Context
     |
     v
    LLM
     |
     v
Candidate Expression
     |
     v
Parser
     |
     v
Validator
```

Invalid expressions should be rejected or repaired.

---

## Week 10 — Closed-Loop Alpha Search

Status:

```text
NOT IMPLEMENTED YET
```

The full system should become:

```text
LLM
 ↓
Generate candidates
 ↓
Validate
 ↓
Queue
 ↓
Distributed C++ evaluation
 ↓
Fitness
 ↓
Feedback
 ↓
LLM / Evolutionary search
```

At this point the project becomes a complete automated discovery loop.

---

## Week 11 — Statistical Validation

Status:

```text
NOT IMPLEMENTED YET
```

This stage is particularly important because repeated factor mining creates multiple-testing problems.

Potential validation components:

- out-of-sample performance
- train/validation/test separation
- holdout evaluation
- IC stability
- ICIR stability
- factor correlation/redundancy
- turnover
- transaction-cost-aware performance
- bootstrap/permutation tests
- multiple-testing correction
- probability of backtest overfitting
- deflated Sharpe ratio or related methods where appropriate

Exact methodology should be selected and documented rather than adding metrics indiscriminately.

---

## Week 12 — Experiments and Ablations

Status:

```text
NOT IMPLEMENTED YET
```

Planned comparisons:

```text
Random search
vs
Evolutionary search
vs
LLM-guided evolutionary search
```

Potential ablations:

```text
Without LLM reasoning
Without crossover
Without mutation
Without diversity mechanism
Without validation filtering
Without statistical validation
```

Metrics should include both:

### Search efficiency

- number of candidates evaluated
- wall-clock time
- valid-expression rate
- unique-expression rate

### Factor quality

- mean IC
- ICIR
- out-of-sample IC
- stability
- redundancy

### Portfolio quality

Eventually:

- Sharpe
- turnover
- drawdown
- transaction-cost-adjusted return

---

# 35. Literature Context

The architecture was influenced by recent research on LLM-assisted formulaic alpha discovery.

Two relevant papers were studied:

1. **EvoAlpha: Evolutionary Alpha Factor Discovery with Large Language Models**
2. **EVOALPHA: An LLM-Enhanced Evolutionary Framework for Formulaic Alpha Mining**

Both use evolutionary-style factor discovery with LLM involvement.

An important project note is that the name **EvoAlpha** appears in both papers, with closely related terminology but different implementations/experimental setups.

Aldien should therefore clearly distinguish itself through its engineering architecture and implementation.

The intended differentiators include:

- explicit symbolic DSL
- custom C++ execution engine
- Python/C++ reference equivalence
- distributed task execution
- explicit dataset/version handling
- split-aware target semantics
- statistical validation layer
- modular LLM integration

Do not claim novelty simply because the system uses an LLM for alpha discovery. The novelty should be framed around the actual implemented architecture and experimental contribution.

---

# 36. What Has NOT Been Implemented

The following major components are still outstanding:

```text
[ ] RabbitMQ integration
[ ] Distributed task protocol
[ ] C++ worker process
[ ] Worker result protocol
[ ] Evolutionary population manager
[ ] Mutation operator
[ ] Crossover operator
[ ] Selection strategy
[ ] Diversity management
[ ] LLM candidate generation
[ ] LLM candidate mutation
[ ] LLM candidate crossover
[ ] LLM feedback loop
[ ] Candidate deduplication/canonicalization
[ ] Search experiment tracking
[ ] Alpha library
[ ] Statistical validation
[ ] Portfolio construction
[ ] Transaction cost modeling
[ ] Final backtesting
[ ] Large-scale performance benchmarking
[ ] Full ablation study
```

---

# 37. Recommended Next Step

Do NOT immediately start RabbitMQ or the LLM.

First restore the repository to a clean known Git state:

```bash
git status
git log --oneline -5
```

Then run:

```bash
pytest -v
```

The immediate objective should be:

> Establish one clean commit in which the deterministic Python DSL, Python evaluator, Python fitness, C++ evaluator, C++ fitness, and Python/C++ integration tests all pass.

Only after this baseline is reproducible should the distributed architecture be added.

Recommended sequence:

```text
1. Clean Git baseline
        ↓
2. All tests passing
        ↓
3. Finalize C++/Python API
        ↓
4. Benchmark single-node C++
        ↓
5. Define task schema
        ↓
6. Add RabbitMQ
        ↓
7. Add one worker
        ↓
8. Add multiple workers
        ↓
9. Add evolutionary search
        ↓
10. Add LLM
        ↓
11. Add statistical validation
        ↓
12. Run experiments
```

---

# 38. How a New Agent Should Continue

If another coding agent takes over this project, it should follow this procedure.

## Step 1 — Inspect Git

```bash
git status
git log --oneline -10
```

Determine whether the working tree is clean.

## Step 2 — Inspect project structure

```bash
find . -maxdepth 3 -type f | sort
```

Do not assume files exist solely because they are described in this README.

## Step 3 — Run tests

```bash
pytest -v
```

Record exactly which tests fail.

## Step 4 — Inspect current implementation

Start with:

```text
python/dsl/
python/evaluation/
engine/include/
engine/src/
tests/
```

## Step 5 — Do not rewrite working infrastructure unnecessarily

In particular:

- do not replace the Python evaluator
- do not replace the C++ evaluator
- do not redesign the DSL
- do not change dataset semantics

unless tests demonstrate an actual problem.

## Step 6 — Make one architectural change at a time

After every meaningful change:

```bash
pytest -v
```

Then commit when the baseline is stable.

---

# 39. Coding Conventions

Prefer:

- small modules
- deterministic functions
- explicit data schemas
- type annotations where useful
- unit tests for every operator
- integration tests for Python/C++
- no hidden global state
- reproducible dataset generation
- explicit configuration
- clear error messages

Avoid:

- silently imputing financial data
- silently dropping invalid alpha expressions
- hiding split leakage
- letting LLM output execute arbitrary code
- coupling the LLM directly to the C++ engine
- using the test set during factor search
- using holdout data for development decisions

---

# 40. Core Invariants

These should remain true as the system evolves.

### Data

```text
No duplicate (datetime, instrument)
```

### Universe

```text
Only date-valid CSI300 membership should be included.
```

### Targets

```text
Future returns must correspond to the requested horizon.
```

### Split-aware fitness

```text
A target crossing into another split must not contribute
to fitness for the current split.
```

### DSL

```text
Every executable alpha must be parseable and valid.
```

### C++

```text
C++ execution should match the Python reference implementation.
```

### Search

```text
The test and holdout sets must not influence candidate discovery.
```

### Validation

```text
A high in-sample score alone is not sufficient to call an alpha valid.
```

---

# 41. Example End-to-End Candidate

A candidate could be:

```text
RANK(
    DIV(
        SUB(CLOSE,TS_MEAN(CLOSE,20)),
        TS_STD(CLOSE,20)
    )
)
```

The intended lifecycle is:

```text
1. LLM/evolutionary system proposes expression
                ↓
2. Parser tokenizes it
                ↓
3. Parser constructs AST
                ↓
4. Validator checks it
                ↓
5. Candidate is submitted to execution queue
                ↓
6. C++ worker receives candidate
                ↓
7. C++ AST is evaluated over market data
                ↓
8. Forward returns are aligned
                ↓
9. Daily IC is calculated
                ↓
10. Mean IC / ICIR are calculated
                ↓
11. Result returned to search engine
                ↓
12. Candidate participates in selection
                ↓
13. Promising candidate evaluated out-of-sample
                ↓
14. Candidate eventually reaches statistical validation
```

---

# 42. Final Project Goal

The final system should allow a user to provide:

```text
dataset
search configuration
operator set
horizon
population size
number of generations
LLM configuration
worker count
```

and obtain a ranked collection of candidate alpha expressions with:

```text
expression
fitness
IC
ICIR
validation performance
test performance
holdout performance
redundancy
stability
statistical validation results
```

The ultimate objective is not merely to produce formulas with high backtest scores.

The objective is to build a **reproducible, distributed research system capable of discovering potentially useful symbolic factors while explicitly controlling computational cost, data leakage, redundancy, and statistical overfitting.**

---

## Quick Reference

### Main dataset

```text
data/processed/csi300_research_dataset.parquet
```

### Universe

```text
data/processed/csi300_universe.parquet
```

### Clean market data

```text
data/processed/csi300_market_data_clean.parquet
```

### Returns

```text
data/processed/csi300_returns.parquet
```

### Python DSL

```text
python/dsl/
```

### Python evaluator

```text
python/evaluation/evaluator.py
```

### Python fitness

```text
python/evaluation/fitness.py
```

### C++ engine

```text
engine/
```

### Python/C++ wrapper

```text
python/evaluation/cpp_engine.py
```

### AST bridge

```text
python/evaluation/cpp_ast_bridge.py
```

### Tests

```bash
pytest -v
```

### Build C++

```bash
cd engine
cmake -S . -B build
cmake --build build
```

### Run standalone C++ executable

```bash
cd engine/build
./alpha_engine
```

---

# 43. Current Priority

**Highest priority:**

> Restore and freeze a clean deterministic baseline before implementing distributed execution.

Once the baseline is stable, the next major engineering milestone is:

```text
RabbitMQ task queue
        +
single C++ worker
        +
end-to-end candidate → fitness pipeline
```

Only after that should evolutionary and LLM components be connected.
