#include "Evaluator.h"

#include <limits>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

AlphaEvaluator::AlphaEvaluator(const MarketData& data)
    : data(data) {}


std::vector<double>
AlphaEvaluator::evaluate(
    const std::shared_ptr<ASTNode>& ast
) {
    return evaluateNode(ast);
}


std::vector<double>
AlphaEvaluator::evaluateNode(
    const std::shared_ptr<ASTNode>& node
) {

    if (!node) {
        throw std::runtime_error("Null AST node");
    }


    // =========================================================
    // VARIABLE
    // =========================================================

    if (node->type == NodeType::VARIABLE) {

        switch (node->variable) {

            case VariableType::OPEN:
                return data.open;

            case VariableType::HIGH:
                return data.high;

            case VariableType::LOW:
                return data.low;

            case VariableType::CLOSE:
                return data.close;

            case VariableType::VOLUME:
                return data.volume;
        }
    }


    // =========================================================
    // NUMBER
    // =========================================================

    if (node->type == NodeType::NUMBER) {

        return std::vector<double>(
            data.size(),
            node->value
        );
    }


    // =========================================================
    // BINARY OPERATIONS
    // =========================================================

    if (node->type == NodeType::BINARY_OP) {

        auto left = evaluateNode(node->left);
        auto right = evaluateNode(node->right);

        if (left.size() != right.size()) {
            throw std::runtime_error(
                "Binary operands have different sizes"
            );
        }

        std::vector<double> result(data.size());

        for (size_t i = 0; i < data.size(); ++i) {

            switch (node->binary_op) {

                case BinaryOperator::ADD:
                    result[i] = left[i] + right[i];
                    break;

                case BinaryOperator::SUB:
                    result[i] = left[i] - right[i];
                    break;

                case BinaryOperator::MUL:
                    result[i] = left[i] * right[i];
                    break;

                case BinaryOperator::DIV:

                    if (right[i] == 0.0) {
                        result[i] = NAN;
                    }
                    else {
                        result[i] = left[i] / right[i];
                    }

                    break;
            }
        }

        return result;
    }


    // =========================================================
    // UNARY OPERATIONS
    // =========================================================

    if (node->type == NodeType::UNARY_OP) {

        auto input = evaluateNode(node->operand);

        std::vector<double> result(data.size());

        for (size_t i = 0; i < data.size(); ++i) {

            switch (node->unary_op) {

                case UnaryOperator::ABS:
                    result[i] = std::abs(input[i]);
                    break;

                case UnaryOperator::LOG:

                    if (input[i] <= 0.0) {
                        result[i] = NAN;
                    }
                    else {
                        result[i] = std::log(input[i]);
                    }

                    break;

                case UnaryOperator::SIGN:

                    if (input[i] > 0.0)
                        result[i] = 1.0;

                    else if (input[i] < 0.0)
                        result[i] = -1.0;

                    else
                        result[i] = 0.0;

                    break;

                case UnaryOperator::RANK: {
                    std::vector<double> result(input.size(), NAN);

                    // Process each date independently.
                    size_t start = 0;

                    while (start < input.size()) {
                        size_t end = start + 1;

                        // Find all rows belonging to the same date.
                        while (end < input.size() &&
                            data.datetime[end] == data.datetime[start]) {
                            ++end;
                        }

                        // Collect finite values for this date.
                        std::vector<std::pair<double, size_t>> values;

                        for (size_t i = start; i < end; ++i) {
                            if (std::isfinite(input[i])) {
                                values.push_back({input[i], i});
                            }
                        }

                        // Sort by value.
                        std::sort(
                            values.begin(),
                            values.end(),
                            [](const auto& a, const auto& b) {
                                return a.first < b.first;
                            }
                        );

                        size_t n = values.size();

                        // Average-rank handling for ties.
                        size_t i = 0;

                        while (i < n) {
                            size_t j = i;

                            while (j + 1 < n &&
                                values[j + 1].first == values[i].first) {
                                ++j;
                            }

                            // 1-based average rank.
                            double average_rank =
                                (static_cast<double>(i + 1) +
                                static_cast<double>(j + 1)) / 2.0;

                            // Match pandas rank(pct=True).
                            double percentile = average_rank / static_cast<double>(n);

                            for (size_t k = i; k <= j; ++k) {
                                result[values[k].second] = percentile;
                            }

                            i = j + 1;
                        }

                        start = end;
                    }

                    return result;
                }
            }
        }

        return result;
    }


    // =========================================================
    // TIME-SERIES OPERATIONS
    // =========================================================

    if (node->type == NodeType::TIME_SERIES_OP) {

        auto input = evaluateNode(node->operand);

        if (input.size() != data.size()) {
            throw std::runtime_error(
                "Time-series operand has incorrect size"
            );
        }

        const int window = node->window;

        if (window <= 0) {
            throw std::runtime_error(
                "Time-series window must be positive"
            );
        }

        std::vector<double> result(
            data.size(),
            NAN
        );


        // -----------------------------------------------------
        // Process each position
        // -----------------------------------------------------

        for (size_t i = 0; i < data.size(); ++i) {

            // Not enough observations yet.
            //
            // Example:
            // window = 3
            //
            // i = 0 → invalid
            // i = 1 → invalid
            // i = 2 → valid

            if (i + 1 < static_cast<size_t>(window)) {
                continue;
            }

            const size_t start =
                i + 1 - static_cast<size_t>(window);


            // -------------------------------------------------
            // Check whether entire window is valid
            // -------------------------------------------------

            bool valid = true;

            for (size_t j = start; j <= i; ++j) {

                if (!std::isfinite(input[j])) {
                    valid = false;
                    break;
                }
            }

            if (!valid) {
                continue;
            }


            // -------------------------------------------------
            // TS_MEAN
            // -------------------------------------------------

            if (node->ts_op ==
                TimeSeriesOperator::TS_MEAN) {

                double sum = 0.0;

                for (size_t j = start; j <= i; ++j) {
                    sum += input[j];
                }

                result[i] =
                    sum / static_cast<double>(window);
            }


            // -------------------------------------------------
            // TS_STD
            // -------------------------------------------------

            else if (node->ts_op ==
                     TimeSeriesOperator::TS_STD) {

                double sum = 0.0;

                for (size_t j = start; j <= i; ++j) {
                    sum += input[j];
                }

                double mean =
                    sum / static_cast<double>(window);


                // Pandas rolling().std() uses sample
                // standard deviation (ddof=1).

                if (window == 1) {
                    result[i] = NAN;
                    continue;
                }

                double squared_sum = 0.0;

                for (size_t j = start; j <= i; ++j) {

                    double diff =
                        input[j] - mean;

                    squared_sum += diff * diff;
                }

                result[i] =
                    std::sqrt(
                        squared_sum /
                        static_cast<double>(window - 1)
                    );
            }


            // -------------------------------------------------
            // TS_MIN
            // -------------------------------------------------

            else if (node->ts_op ==
                     TimeSeriesOperator::TS_MIN) {

                double minimum =
                    input[start];

                for (size_t j = start + 1; j <= i; ++j) {
                    minimum =
                        std::min(minimum, input[j]);
                }

                result[i] = minimum;
            }


            // -------------------------------------------------
            // TS_MAX
            // -------------------------------------------------

            else if (node->ts_op ==
                     TimeSeriesOperator::TS_MAX) {

                double maximum =
                    input[start];

                for (size_t j = start + 1; j <= i; ++j) {
                    maximum =
                        std::max(maximum, input[j]);
                }

                result[i] = maximum;
            }


            else {

                throw std::runtime_error(
                    "Unsupported time-series operator"
                );
            }
        }

        return result;
    }


    // =========================================================
    // UNSUPPORTED NODE
    // =========================================================

    throw std::runtime_error(
        "Unsupported AST node"
    );
}