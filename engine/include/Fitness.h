#pragma once

#include "AST.h"
#include "Data.h"

#include <cstddef>
#include <memory>
#include <vector>

struct FitnessResult {
    double mean_ic;
    double ic_std;
    double icir;

    std::size_t num_days;
    std::size_t num_observations;

    double coverage;
};

class FitnessEvaluator {
public:

    explicit FitnessEvaluator(const MarketData& data);

    FitnessResult evaluate(
        const std::shared_ptr<ASTNode>& ast,
        const std::vector<double>& forward_returns
    );

private:

    const MarketData& data;

    double calculatePearsonIC(
        const std::vector<double>& alpha,
        const std::vector<double>& returns,
        const std::vector<std::size_t>& indices
    ) const;
};