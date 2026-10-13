#include "Evaluator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

namespace {

double NaN() {
    return std::numeric_limits<double>::quiet_NaN();
}

bool isFinite(double x) {
    return std::isfinite(x);
}

} // namespace


AlphaEvaluator::AlphaEvaluator(const MarketData& data)
    : data(data) {}


std::vector<double>
AlphaEvaluator::evaluate(
    const std::shared_ptr<ASTNode>& ast
) {
    if (!ast) {
        throw std::runtime_error("AST is null");
    }

    return evaluateNode(ast);
}


std::vector<double>
AlphaEvaluator::evaluateNode(
    const std::shared_ptr<ASTNode>& node
) {

    if (!node) {
        throw std::runtime_error("AST node is null");
    }


    // ==================================================
    // VARIABLE
    // ==================================================

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

        throw std::runtime_error("Unknown variable");
    }


    // ==================================================
    // NUMBER
    // ==================================================

    if (node->type == NodeType::NUMBER) {

        return std::vector<double>(
            data.size(),
            node->value
        );
    }


    // ==================================================
    // UNARY OPERATORS
    // ==================================================

    if (node->type == NodeType::UNARY_OP) {

        auto input = evaluateNode(node->operand);

        std::vector<double> result(
            input.size(),
            NaN()
        );


        // ----------------------------------------------
        // ABS
        // ----------------------------------------------

        if (node->unary_op == UnaryOperator::ABS) {

            for (size_t i = 0; i < input.size(); ++i) {

                if (isFinite(input[i])) {
                    result[i] = std::abs(input[i]);
                }
            }
        }


        // ----------------------------------------------
        // LOG
        // ----------------------------------------------

        else if (node->unary_op == UnaryOperator::LOG) {

            for (size_t i = 0; i < input.size(); ++i) {

                if (
                    isFinite(input[i]) &&
                    input[i] > 0.0
                ) {
                    result[i] = std::log(input[i]);
                }
            }
        }


        // ----------------------------------------------
        // SIGN
        // ----------------------------------------------

        else if (node->unary_op == UnaryOperator::SIGN) {

            for (size_t i = 0; i < input.size(); ++i) {

                if (!isFinite(input[i])) {
                    continue;
                }

                if (input[i] > 0.0) {
                    result[i] = 1.0;
                }
                else if (input[i] < 0.0) {
                    result[i] = -1.0;
                }
                else {
                    result[i] = 0.0;
                }
            }
        }


        // ----------------------------------------------
        // RANK
        //
        // Cross-sectional:
        // group by DATE, rank across instruments.
        // ----------------------------------------------

        else if (node->unary_op == UnaryOperator::RANK) {

            std::map<
                std::string,
                std::vector<size_t>
            > dateGroups;


            // Group rows by date.
            for (size_t i = 0; i < input.size(); ++i) {

                if (isFinite(input[i])) {

                    dateGroups[
                        data.datetime[i]
                    ].push_back(i);
                }
            }


            // Process each date independently.
            for (const auto& [date, indices] : dateGroups) {

                std::vector<
                    std::pair<double, size_t>
                > values;


                for (size_t index : indices) {

                    values.push_back({
                        input[index],
                        index
                    });
                }


                // Sort ascending.
                std::sort(
                    values.begin(),
                    values.end(),
                    [](const auto& a, const auto& b) {
                        return a.first < b.first;
                    }
                );


                const size_t n = values.size();

                size_t i = 0;

                while (i < n) {

                    size_t j = i;

                    // Find tied values.
                    while (
                        j + 1 < n &&
                        values[j + 1].first ==
                            values[i].first
                    ) {
                        ++j;
                    }


                    // Average rank for ties.
                    double averageRank =
                        (
                            static_cast<double>(i + 1) +
                            static_cast<double>(j + 1)
                        ) / 2.0;


                    // pandas rank(pct=True)
                    double percentile =
                        averageRank /
                        static_cast<double>(n);


                    for (size_t k = i; k <= j; ++k) {

                        result[
                            values[k].second
                        ] = percentile;
                    }


                    i = j + 1;
                }
            }
        }


        else {
            throw std::runtime_error(
                "Unknown unary operator"
            );
        }


        return result;
    }


    // ==================================================
    // BINARY OPERATORS
    // ==================================================

    if (node->type == NodeType::BINARY_OP) {

        auto left =
            evaluateNode(node->left);

        auto right =
            evaluateNode(node->right);


        if (left.size() != right.size()) {

            throw std::runtime_error(
                "Binary operand size mismatch"
            );
        }


        std::vector<double> result(
            left.size(),
            NaN()
        );


        for (size_t i = 0; i < left.size(); ++i) {

            if (
                !isFinite(left[i]) ||
                !isFinite(right[i])
            ) {
                continue;
            }


            switch (node->binary_op) {

                case BinaryOperator::ADD:

                    result[i] =
                        left[i] + right[i];

                    break;


                case BinaryOperator::SUB:

                    result[i] =
                        left[i] - right[i];

                    break;


                case BinaryOperator::MUL:

                    result[i] =
                        left[i] * right[i];

                    break;


                case BinaryOperator::DIV:

                    if (right[i] != 0.0) {

                        result[i] =
                            left[i] / right[i];
                    }

                    break;
            }
        }


        return result;
    }


    // ==================================================
    // TIME-SERIES OPERATORS
    //
    // Group by INSTRUMENT, then move through TIME.
    // ==================================================

    if (node->type == NodeType::TIME_SERIES_OP) {

        auto input =
            evaluateNode(node->operand);


        std::vector<double> result(
            input.size(),
            NaN()
        );


        const int window =
            node->window;


        if (window <= 0) {

            throw std::runtime_error(
                "Window must be positive"
            );
        }


        // ------------------------------------------------
        // Find contiguous instrument blocks.
        //
        // Data must be sorted:
        // instrument -> datetime
        // ------------------------------------------------

        size_t start = 0;


        while (start < input.size()) {

            size_t end = start + 1;


            while (
                end < input.size() &&
                data.instrument[end] ==
                    data.instrument[start]
            ) {
                ++end;
            }


            // --------------------------------------------
            // Rolling calculation
            // --------------------------------------------

            for (size_t i = start; i < end; ++i) {

                if (
                    i - start + 1 <
                    static_cast<size_t>(window)
                ) {
                    continue;
                }


                size_t windowStart =
                    i + 1 -
                    static_cast<size_t>(window);


                std::vector<double> values;

                double sum = 0.0;

                double minValue =
                    std::numeric_limits<double>::infinity();

                double maxValue =
                    -std::numeric_limits<double>::infinity();


                for (
                    size_t j = windowStart;
                    j <= i;
                    ++j
                ) {

                    if (!isFinite(input[j])) {
                        continue;
                    }


                    values.push_back(input[j]);

                    sum += input[j];

                    minValue =
                        std::min(
                            minValue,
                            input[j]
                        );

                    maxValue =
                        std::max(
                            maxValue,
                            input[j]
                        );
                }


                // Require complete window.
                if (
                    values.size() !=
                    static_cast<size_t>(window)
                ) {
                    continue;
                }


                // ----------------------------------------
                // TS_MEAN
                // ----------------------------------------

                if (
                    node->ts_op ==
                    TimeSeriesOperator::TS_MEAN
                ) {

                    result[i] =
                        sum /
                        static_cast<double>(window);
                }


                // ----------------------------------------
                // TS_MIN
                // ----------------------------------------

                else if (
                    node->ts_op ==
                    TimeSeriesOperator::TS_MIN
                ) {

                    result[i] = minValue;
                }


                // ----------------------------------------
                // TS_MAX
                // ----------------------------------------

                else if (
                    node->ts_op ==
                    TimeSeriesOperator::TS_MAX
                ) {

                    result[i] = maxValue;
                }


                // ----------------------------------------
                // TS_STD
                //
                // Sample standard deviation.
                // Matches pandas rolling().std()
                // ----------------------------------------

                else if (
                    node->ts_op ==
                    TimeSeriesOperator::TS_STD
                ) {

                    double mean =
                        sum /
                        static_cast<double>(window);


                    double squaredSum = 0.0;


                    for (double value : values) {

                        double diff =
                            value - mean;

                        squaredSum +=
                            diff * diff;
                    }


                    result[i] =
                        std::sqrt(
                            squaredSum /
                            static_cast<double>(
                                window - 1
                            )
                        );
                }
            }


            start = end;
        }


        return result;
    }


    throw std::runtime_error(
        "Unknown AST node type"
    );
}