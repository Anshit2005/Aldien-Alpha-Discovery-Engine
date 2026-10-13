#include <iostream>
#include <iomanip>
#include <string>

#include "AST.h"
#include "Data.h"
#include "Evaluator.h"

int main() {

    // --------------------------------------------------
    // Same data that we will use in Python
    // --------------------------------------------------

    MarketData data;

    data.datetime = {
        "2020-01-01",
        "2020-01-02",

        "2020-01-01",
        "2020-01-02",

        "2020-01-01",
        "2020-01-02"
    };

    data.instrument = {
        "A",
        "A",

        "B",
        "B",

        "C",
        "C"
    };

    data.close = {
        105,
        108,

        120,
        118,

        110,
        115
    };

    data.open = {
        100,
        102,

        110,
        112,

        105,
        108
    };

    data.high = {
        110,
        112,

        125,
        122,

        115,
        118
    };

    data.low = {
        95,
        98,

        108,
        110,

        100,
        105
    };

    data.volume = {
        1000,
        1100,

        2000,
        2100,

        1500,
        1600
    };


    AlphaEvaluator evaluator(data);


    // --------------------------------------------------
    // Expression:
    //
    // RANK(
    //     SUB(
    //         CLOSE,
    //         TS_MEAN(CLOSE, 2)
    //     )
    // )
    //
    // --------------------------------------------------

    auto close =
        ASTNode::makeVariable(
            VariableType::CLOSE
        );

    auto rollingMean =
        ASTNode::makeTimeSeries(
            TimeSeriesOperator::TS_MEAN,
            close,
            2
        );

    auto difference =
        ASTNode::makeBinary(
            BinaryOperator::SUB,
            close,
            rollingMean
        );

    auto alpha =
        ASTNode::makeUnary(
            UnaryOperator::RANK,
            difference
        );


    auto result =
        evaluator.evaluate(alpha);


    // --------------------------------------------------
    // Output
    // --------------------------------------------------

    std::cout << std::fixed
              << std::setprecision(10);

    for (size_t i = 0; i < result.size(); ++i) {

        std::cout
            << data.datetime[i]
            << ","
            << data.instrument[i]
            << ","
            << result[i]
            << "\n";
    }

    return 0;
}