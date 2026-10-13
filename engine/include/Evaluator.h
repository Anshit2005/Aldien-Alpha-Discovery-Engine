#pragma once

#include "AST.h"
#include "Data.h"

#include <vector>

class AlphaEvaluator {

public:

    explicit AlphaEvaluator(const MarketData& data);

    std::vector<double> evaluate(
        const std::shared_ptr<ASTNode>& ast
    );

private:

    const MarketData& data;

    std::vector<double> evaluateNode(
        const std::shared_ptr<ASTNode>& node
    );
};