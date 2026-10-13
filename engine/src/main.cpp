#include <iostream>
#include <iomanip>

#include "AST.h"
#include "Data.h"
#include "Evaluator.h"

int main() {

    MarketData data;

    // --------------------------------------------------
    // Two dates, three stocks per date
    // --------------------------------------------------

    data.datetime = {
        "2020-01-01",
        "2020-01-01",
        "2020-01-01",

        "2020-01-02",
        "2020-01-02",
        "2020-01-02"
    };

    data.instrument = {
        "A",
        "B",
        "C",

        "A",
        "B",
        "C"
    };

    data.close = {
        100,
        120,
        100,

        102,
        118,
        115
    };

    AlphaEvaluator evaluator(data);

    // CLOSE
    auto close =
        ASTNode::makeVariable(VariableType::CLOSE);

    // RANK(CLOSE)
    auto rank =
        ASTNode::makeUnary(
            UnaryOperator::RANK,
            close
        );

    auto result = evaluator.evaluate(rank);

    std::cout << std::fixed
              << std::setprecision(4);

    std::cout << "RANK(CLOSE):\n";

    for (size_t i = 0; i < result.size(); ++i) {

        std::cout
            << data.datetime[i]
            << " "
            << data.instrument[i]
            << " "
            << result[i]
            << "\n";
    }

    return 0;
}