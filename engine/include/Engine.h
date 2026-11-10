#pragma once

// Load-once evaluation engine.
//
// The market panel is handed over once, when the Engine is created, and kept
// in memory; after that only formulas are passed in. Rows must be sorted by
// instrument, then by day, and instruments and days are integer ids
// (0..n-1), so cross-sectional work (RANK, daily IC) uses precomputed row
// lists per day instead of grouping by date strings on every call.
//
// Semantics follow the Python reference (python/evaluation/fitness.py):
//   - The alpha is evaluated on the whole panel, so time-series windows see
//     the full history of each instrument.
//   - With a split, only rows of that split count, and a row's forward
//     return counts only if its target is in the same split.
//   - A day's IC is the Pearson correlation of alpha and forward return
//     across instruments; it is defined when at least two rows are valid and
//     both alpha and return vary that day (min < max), and it is clipped to
//     [-1, 1].
//   - Non-finite values (NaN, +-inf) are treated as missing by every
//     operator, as in the original C++ evaluator.

#include "AST.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

struct EngineFitness {
    int horizon = 0;
    double mean_ic = 0.0;
    double ic_std = 0.0;
    double icir = 0.0;
    std::size_t num_days = 0;          // days with a defined IC
    std::size_t num_observations = 0;  // rows with finite alpha and return
    double coverage = 0.0;             // num_observations / rows considered

    // One entry per day considered (every day, or the days of the split),
    // in day order; NaN where the IC is undefined.
    std::vector<std::int32_t> days;
    std::vector<double> daily_ic;
};

class Engine {
public:
    static constexpr int NUM_HORIZONS = 4;
    static constexpr int HORIZONS[NUM_HORIZONS] = {1, 5, 20, 30};

    // `split` and each `target_split[h]` hold one code per row (-1 = none);
    // they may be empty when the data has no splits. `returns` and
    // `target_split` are ordered like HORIZONS.
    Engine(
        std::vector<std::int32_t> instrument,
        std::vector<std::int32_t> day,
        std::vector<double> open,
        std::vector<double> high,
        std::vector<double> low,
        std::vector<double> close,
        std::vector<double> volume,
        std::vector<std::vector<double>> returns,
        std::vector<std::int8_t> split,
        std::vector<std::vector<std::int8_t>> target_split
    );

    std::vector<double> evaluate(const std::shared_ptr<ASTNode>& ast) const;

    // split < 0: every row; otherwise the split code.
    EngineFitness fitness(
        const std::shared_ptr<ASTNode>& ast, int horizon, int split
    ) const;

    std::size_t size() const { return open_.size(); }
    std::size_t num_days() const { return day_offsets_.size() - 1; }
    std::size_t num_instruments() const { return block_starts_.size() - 1; }

private:
    std::vector<std::int32_t> instrument_;
    std::vector<std::int32_t> day_;
    std::vector<double> open_, high_, low_, close_, volume_;
    std::vector<std::vector<double>> returns_;
    std::vector<std::int8_t> split_;
    std::vector<std::vector<std::int8_t>> target_split_;

    // Rows [block_starts_[k], block_starts_[k+1]) belong to one instrument.
    std::vector<std::size_t> block_starts_;
    // Rows of day d: day_rows_[day_offsets_[d] .. day_offsets_[d+1]), in
    // row order (that is, instrument order).
    std::vector<std::size_t> day_offsets_;
    std::vector<std::size_t> day_rows_;

    std::vector<double> evaluateNode(const ASTNode& node) const;
    std::vector<double> rank(const std::vector<double>& input) const;
    std::vector<double> timeSeries(
        const std::vector<double>& input, TimeSeriesOperator op, int window
    ) const;
    static int horizonIndex(int horizon);
};
