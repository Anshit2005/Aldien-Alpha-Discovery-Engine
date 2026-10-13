#include "Fitness.h"
#include "Evaluator.h"

#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {

double NaN() {
    return std::numeric_limits<double>::quiet_NaN();
}

bool isFinite(double x) {
    return std::isfinite(x);
}

double mean(
    const std::vector<double>& values
) {
    if (values.empty()) {
        return NaN();
    }

    double sum = 0.0;

    for (double value : values) {
        sum += value;
    }

    return sum / static_cast<double>(values.size());
}

double sampleStd(
    const std::vector<double>& values,
    double meanValue
) {
    if (values.size() < 2) {
        return NaN();
    }

    double sumSquared = 0.0;

    for (double value : values) {
        const double diff = value - meanValue;
        sumSquared += diff * diff;
    }

    return std::sqrt(
        sumSquared /
        static_cast<double>(values.size() - 1)
    );
}

} // namespace


FitnessEvaluator::FitnessEvaluator(
    const MarketData& data
)
    : data(data)
{
}


double FitnessEvaluator::calculatePearsonIC(
    const std::vector<double>& alpha,
    const std::vector<double>& returns,
    const std::vector<std::size_t>& indices
) const {

    std::vector<double> validAlpha;
    std::vector<double> validReturns;

    for (std::size_t index : indices) {

        double a = alpha[index];
        double r = returns[index];

        if (!isFinite(a) || !isFinite(r)) {
            continue;
        }

        validAlpha.push_back(a);
        validReturns.push_back(r);
    }

    // Need at least two observations
    if (validAlpha.size() < 2) {
        return NaN();
    }

    const double alphaMean =
        mean(validAlpha);

    const double returnMean =
        mean(validReturns);

    double numerator = 0.0;
    double alphaVariance = 0.0;
    double returnVariance = 0.0;

    for (std::size_t i = 0;
         i < validAlpha.size();
         ++i) {

        const double alphaDiff =
            validAlpha[i] - alphaMean;

        const double returnDiff =
            validReturns[i] - returnMean;

        numerator +=
            alphaDiff * returnDiff;

        alphaVariance +=
            alphaDiff * alphaDiff;

        returnVariance +=
            returnDiff * returnDiff;
    }

    if (alphaVariance <= 0.0 ||
        returnVariance <= 0.0) {

        return NaN();
    }

    return numerator /
           std::sqrt(
               alphaVariance *
               returnVariance
           );
}


FitnessResult FitnessEvaluator::evaluate(
    const std::shared_ptr<ASTNode>& ast,
    const std::vector<double>& forward_returns
) {

    if (!ast) {
        throw std::invalid_argument(
            "AST cannot be null"
        );
    }

    if (forward_returns.size() != data.size()) {
        throw std::invalid_argument(
            "Forward return vector size "
            "does not match market data size"
        );
    }

    // ------------------------------------------------------------
    // Evaluate alpha
    // ------------------------------------------------------------

    AlphaEvaluator alphaEvaluator(data);

    std::vector<double> alpha =
        alphaEvaluator.evaluate(ast);

    // ------------------------------------------------------------
    // Group rows by date
    // ------------------------------------------------------------

    std::unordered_map<
        std::string,
        std::vector<std::size_t>
    > dateGroups;

    for (std::size_t i = 0;
         i < data.size();
         ++i) {

        dateGroups[data.datetime[i]]
            .push_back(i);
    }

    // ------------------------------------------------------------
    // Calculate daily IC
    // ------------------------------------------------------------

    std::vector<double> dailyIC;

    std::size_t totalValidObservations = 0;

    for (const auto& [date, indices] :
         dateGroups) {

        std::vector<double> validAlpha;
        std::vector<double> validReturns;

        for (std::size_t index : indices) {

            if (!isFinite(alpha[index]) ||
                !isFinite(forward_returns[index])) {

                continue;
            }

            validAlpha.push_back(alpha[index]);
            validReturns.push_back(
                forward_returns[index]
            );

            ++totalValidObservations;
        }

        if (validAlpha.size() < 2) {
            continue;
        }

        const double ic =
            calculatePearsonIC(
                alpha,
                forward_returns,
                indices
            );

        if (isFinite(ic)) {
            dailyIC.push_back(ic);
        }
    }

    // ------------------------------------------------------------
    // No valid IC
    // ------------------------------------------------------------

    if (dailyIC.empty()) {

        return FitnessResult{
            NaN(),
            NaN(),
            NaN(),
            0,
            0,
            0.0
        };
    }

    // ------------------------------------------------------------
    // Mean IC
    // ------------------------------------------------------------

    const double meanIC =
        mean(dailyIC);

    // ------------------------------------------------------------
    // IC standard deviation
    // ------------------------------------------------------------

    const double icStd =
        sampleStd(
            dailyIC,
            meanIC
        );

    // ------------------------------------------------------------
    // ICIR
    // ------------------------------------------------------------

    double icir = NaN();

    if (isFinite(icStd) &&
        icStd > 1e-12) {

        icir =
            meanIC / icStd;
    }

    // ------------------------------------------------------------
    // Coverage
    // ------------------------------------------------------------

    const double coverage =
        data.size() > 0
        ? static_cast<double>(
              totalValidObservations
          ) /
          static_cast<double>(
              data.size()
          )
        : 0.0;

    return FitnessResult{
        meanIC,
        icStd,
        icir,
        dailyIC.size(),
        totalValidObservations,
        coverage
    };
}