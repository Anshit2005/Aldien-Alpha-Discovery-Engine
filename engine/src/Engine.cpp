#include "Engine.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

inline bool isFiniteValue(double x) { return std::isfinite(x); }

template <typename T>
void requireSize(const std::vector<T>& values, std::size_t n, const char* name) {
    if (values.size() != n) {
        throw std::invalid_argument(
            std::string(name) + " has " + std::to_string(values.size()) +
            " rows, expected " + std::to_string(n)
        );
    }
}

}  // namespace

constexpr int Engine::HORIZONS[Engine::NUM_HORIZONS];

Engine::Engine(
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
)
    : instrument_(std::move(instrument)),
      day_(std::move(day)),
      open_(std::move(open)),
      high_(std::move(high)),
      low_(std::move(low)),
      close_(std::move(close)),
      volume_(std::move(volume)),
      returns_(std::move(returns)),
      split_(std::move(split)),
      target_split_(std::move(target_split)) {
    const std::size_t n = open_.size();
    requireSize(instrument_, n, "instrument");
    requireSize(day_, n, "day");
    requireSize(high_, n, "high");
    requireSize(low_, n, "low");
    requireSize(close_, n, "close");
    requireSize(volume_, n, "volume");
    if (returns_.size() != NUM_HORIZONS) {
        throw std::invalid_argument("returns must hold one array per horizon (1, 5, 20, 30)");
    }
    for (const auto& r : returns_) requireSize(r, n, "forward return");
    if (!split_.empty()) {
        requireSize(split_, n, "split");
        if (target_split_.size() != NUM_HORIZONS) {
            throw std::invalid_argument("target_split must hold one array per horizon");
        }
        for (const auto& t : target_split_) requireSize(t, n, "target split");
    }

    // Instrument blocks; rows must be sorted by instrument, then day.
    std::int32_t max_day = -1;
    block_starts_.push_back(0);
    for (std::size_t i = 0; i < n; ++i) {
        if (day_[i] < 0 || instrument_[i] < 0) {
            throw std::invalid_argument("instrument and day ids must be non-negative");
        }
        max_day = std::max(max_day, day_[i]);
        if (i == 0) continue;
        if (instrument_[i] != instrument_[i - 1]) {
            if (instrument_[i] < instrument_[i - 1]) {
                throw std::invalid_argument("rows must be sorted by instrument, then day");
            }
            block_starts_.push_back(i);
        } else if (day_[i] <= day_[i - 1]) {
            throw std::invalid_argument(
                "rows must be sorted by instrument, then day, with one row per instrument and day"
            );
        }
    }
    if (n > 0) block_starts_.push_back(n);

    // Rows per day (counting sort keeps row order within a day).
    const std::size_t days = static_cast<std::size_t>(max_day + 1);
    day_offsets_.assign(days + 1, 0);
    for (std::size_t i = 0; i < n; ++i) ++day_offsets_[static_cast<std::size_t>(day_[i]) + 1];
    for (std::size_t d = 0; d < days; ++d) day_offsets_[d + 1] += day_offsets_[d];
    day_rows_.resize(n);
    std::vector<std::size_t> next(day_offsets_.begin(), day_offsets_.end() - 1);
    for (std::size_t i = 0; i < n; ++i) day_rows_[next[static_cast<std::size_t>(day_[i])]++] = i;
}

int Engine::horizonIndex(int horizon) {
    for (int h = 0; h < NUM_HORIZONS; ++h) {
        if (HORIZONS[h] == horizon) return h;
    }
    throw std::invalid_argument("Unsupported horizon: " + std::to_string(horizon));
}

std::vector<double> Engine::evaluate(const std::shared_ptr<ASTNode>& ast) const {
    if (!ast) throw std::invalid_argument("AST is null");
    return evaluateNode(*ast);
}

std::vector<double> Engine::evaluateNode(const ASTNode& node) const {
    const std::size_t n = size();
    switch (node.type) {
        case NodeType::VARIABLE:
            switch (node.variable) {
                case VariableType::OPEN: return open_;
                case VariableType::HIGH: return high_;
                case VariableType::LOW: return low_;
                case VariableType::CLOSE: return close_;
                case VariableType::VOLUME: return volume_;
            }
            throw std::runtime_error("Unknown variable");

        case NodeType::NUMBER:
            return std::vector<double>(n, node.value);

        case NodeType::UNARY_OP: {
            if (!node.operand) throw std::invalid_argument("Unary operator without operand");
            std::vector<double> x = evaluateNode(*node.operand);
            if (node.unary_op == UnaryOperator::RANK) return rank(x);
            for (double& v : x) {
                if (!isFiniteValue(v)) {
                    v = kNaN;
                    continue;
                }
                switch (node.unary_op) {
                    case UnaryOperator::ABS: v = std::abs(v); break;
                    case UnaryOperator::LOG: v = v > 0.0 ? std::log(v) : kNaN; break;
                    case UnaryOperator::SIGN: v = v > 0.0 ? 1.0 : (v < 0.0 ? -1.0 : 0.0); break;
                    default: throw std::runtime_error("Unknown unary operator");
                }
            }
            return x;
        }

        case NodeType::BINARY_OP: {
            if (!node.left || !node.right) throw std::invalid_argument("Binary operator without operands");
            std::vector<double> a = evaluateNode(*node.left);
            const std::vector<double> b = evaluateNode(*node.right);
            for (std::size_t i = 0; i < n; ++i) {
                if (!isFiniteValue(a[i]) || !isFiniteValue(b[i])) {
                    a[i] = kNaN;
                    continue;
                }
                switch (node.binary_op) {
                    case BinaryOperator::ADD: a[i] += b[i]; break;
                    case BinaryOperator::SUB: a[i] -= b[i]; break;
                    case BinaryOperator::MUL: a[i] *= b[i]; break;
                    case BinaryOperator::DIV: a[i] = b[i] != 0.0 ? a[i] / b[i] : kNaN; break;
                }
            }
            return a;
        }

        case NodeType::TIME_SERIES_OP: {
            if (!node.operand) throw std::invalid_argument("Time-series operator without operand");
            if (node.window <= 0) throw std::invalid_argument("Window must be positive");
            return timeSeries(evaluateNode(*node.operand), node.ts_op, node.window);
        }
    }
    throw std::runtime_error("Unknown AST node type");
}

// Cross-sectional percentile rank per day, average rank for ties
// (pandas rank(method="average", pct=True)); non-finite values stay NaN.
std::vector<double> Engine::rank(const std::vector<double>& input) const {
    std::vector<double> result(input.size(), kNaN);
    std::vector<std::pair<double, std::size_t>> values;
    for (std::size_t d = 0; d + 1 < day_offsets_.size(); ++d) {
        values.clear();
        for (std::size_t k = day_offsets_[d]; k < day_offsets_[d + 1]; ++k) {
            const std::size_t row = day_rows_[k];
            if (isFiniteValue(input[row])) values.emplace_back(input[row], row);
        }
        std::sort(values.begin(), values.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        const std::size_t m = values.size();
        for (std::size_t i = 0; i < m;) {
            std::size_t j = i;
            while (j + 1 < m && values[j + 1].first == values[i].first) ++j;
            const double percentile =
                (static_cast<double>(i + 1) + static_cast<double>(j + 1)) / 2.0 /
                static_cast<double>(m);
            for (std::size_t k = i; k <= j; ++k) result[values[k].second] = percentile;
            i = j + 1;
        }
    }
    return result;
}

// Rolling window over each instrument's rows; a value needs `window`
// consecutive finite inputs (pandas rolling(window, min_periods=window)).
std::vector<double> Engine::timeSeries(
    const std::vector<double>& input, TimeSeriesOperator op, int window
) const {
    std::vector<double> result(input.size(), kNaN);
    const std::size_t w = static_cast<std::size_t>(window);
    for (std::size_t b = 0; b + 1 < block_starts_.size(); ++b) {
        const std::size_t start = block_starts_[b], end = block_starts_[b + 1];
        std::size_t run = 0;   // consecutive finite values ending at i
        std::size_t same = 0;  // consecutive equal values ending at i
        for (std::size_t i = start; i < end; ++i) {
            if (!isFiniteValue(input[i])) {
                run = same = 0;
                continue;
            }
            same = (run > 0 && input[i] == input[i - 1]) ? same + 1 : 1;
            ++run;
            if (run < w) continue;
            const std::size_t first = i + 1 - w;
            // A window of identical values: pandas returns that value as the
            // mean and exactly 0 as the standard deviation. Summing would leave
            // rounding noise (~1e-16) that differs between stocks and would
            // make a constant formula look like it varies across stocks.
            if (same >= w && (op == TimeSeriesOperator::TS_MEAN || op == TimeSeriesOperator::TS_STD)) {
                result[i] = op == TimeSeriesOperator::TS_MEAN ? input[i] : (w >= 2 ? 0.0 : kNaN);
                continue;
            }
            switch (op) {
                case TimeSeriesOperator::TS_MEAN: {
                    double sum = 0.0;
                    for (std::size_t j = first; j <= i; ++j) sum += input[j];
                    result[i] = sum / static_cast<double>(w);
                    break;
                }
                case TimeSeriesOperator::TS_STD: {
                    if (w < 2) break;
                    double sum = 0.0;
                    for (std::size_t j = first; j <= i; ++j) sum += input[j];
                    const double mean = sum / static_cast<double>(w);
                    double squares = 0.0;
                    for (std::size_t j = first; j <= i; ++j) {
                        const double diff = input[j] - mean;
                        squares += diff * diff;
                    }
                    result[i] = std::sqrt(squares / static_cast<double>(w - 1));
                    break;
                }
                case TimeSeriesOperator::TS_MIN: {
                    double m = input[first];
                    for (std::size_t j = first + 1; j <= i; ++j) m = std::min(m, input[j]);
                    result[i] = m;
                    break;
                }
                case TimeSeriesOperator::TS_MAX: {
                    double m = input[first];
                    for (std::size_t j = first + 1; j <= i; ++j) m = std::max(m, input[j]);
                    result[i] = m;
                    break;
                }
            }
        }
    }
    return result;
}

EngineFitness Engine::fitness(
    const std::shared_ptr<ASTNode>& ast, int horizon, int split
) const {
    const int h = horizonIndex(horizon);
    if (split >= 0 && split_.empty()) {
        throw std::invalid_argument("This engine was created without split columns");
    }
    const std::vector<double> alpha = evaluate(ast);
    const std::vector<double>& returns = returns_[static_cast<std::size_t>(h)];
    const std::vector<std::int8_t>* target =
        split >= 0 ? &target_split_[static_cast<std::size_t>(h)] : nullptr;
    const auto code = static_cast<std::int8_t>(split);

    EngineFitness out;
    out.horizon = horizon;
    std::size_t rows_considered = 0;
    std::vector<double> x, y;

    for (std::size_t d = 0; d + 1 < day_offsets_.size(); ++d) {
        x.clear();
        y.clear();
        bool day_in_split = split < 0;
        for (std::size_t k = day_offsets_[d]; k < day_offsets_[d + 1]; ++k) {
            const std::size_t row = day_rows_[k];
            if (split >= 0) {
                if (split_[row] != code) continue;
                day_in_split = true;
            }
            ++rows_considered;
            const double a = alpha[row];
            const double r = (target == nullptr || (*target)[row] == code) ? returns[row] : kNaN;
            if (isFiniteValue(a) && isFiniteValue(r)) {
                x.push_back(a);
                y.push_back(r);
            }
        }
        if (!day_in_split) continue;
        out.num_observations += x.size();
        out.days.push_back(static_cast<std::int32_t>(d));

        double ic = kNaN;
        const std::size_t m = x.size();
        if (m >= 2) {
            const auto [xmin, xmax] = std::minmax_element(x.begin(), x.end());
            const auto [ymin, ymax] = std::minmax_element(y.begin(), y.end());
            if (*xmin < *xmax && *ymin < *ymax) {
                double sx = 0.0, sy = 0.0;
                for (std::size_t i = 0; i < m; ++i) {
                    sx += x[i];
                    sy += y[i];
                }
                const double mx = sx / static_cast<double>(m);
                const double my = sy / static_cast<double>(m);
                double sxy = 0.0, sxx = 0.0, syy = 0.0;
                for (std::size_t i = 0; i < m; ++i) {
                    const double dx = x[i] - mx, dy = y[i] - my;
                    sxy += dx * dy;
                    sxx += dx * dx;
                    syy += dy * dy;
                }
                const double value = sxy / std::sqrt(sxx * syy);
                if (isFiniteValue(value)) ic = std::clamp(value, -1.0, 1.0);
            }
        }
        out.daily_ic.push_back(ic);
    }

    double sum = 0.0;
    for (double ic : out.daily_ic) {
        if (isFiniteValue(ic)) {
            sum += ic;
            ++out.num_days;
        }
    }
    if (out.num_days == 0) {
        out.mean_ic = out.ic_std = out.icir = kNaN;
        out.num_observations = 0;
        out.coverage = 0.0;
        return out;
    }
    out.mean_ic = sum / static_cast<double>(out.num_days);
    if (out.num_days >= 2) {
        double squares = 0.0;
        for (double ic : out.daily_ic) {
            if (isFiniteValue(ic)) squares += (ic - out.mean_ic) * (ic - out.mean_ic);
        }
        out.ic_std = std::sqrt(squares / static_cast<double>(out.num_days - 1));
    } else {
        out.ic_std = kNaN;
    }
    out.icir = (isFiniteValue(out.ic_std) && out.ic_std > 0.0) ? out.mean_ic / out.ic_std : kNaN;
    out.coverage = rows_considered > 0
        ? static_cast<double>(out.num_observations) / static_cast<double>(rows_considered)
        : 0.0;
    return out;
}
