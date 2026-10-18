#include "AST.h"
#include "CSVLoader.h"
#include "Evaluator.h"
#include "Fitness.h"

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>


int main() {

    try {

        std::filesystem::path inputPath =
            "data/processed/cpp_fitness_data.csv";
        if (!std::filesystem::exists(inputPath)) {
            inputPath = "../../data/processed/cpp_fitness_data.csv";
        }
        const std::string inputFile = inputPath.string();

        std::cout
            << "Loading: "
            << inputFile
            << "\n";

        MarketData data =
            CSVLoader::load(inputFile);

        std::cout
            << "Rows: "
            << data.size()
            << "\n";

        // ------------------------------------------------------------
        // Alpha:
        //
        // RANK(
        //     DIV(
        //         SUB(CLOSE, TS_MEAN(CLOSE,20)),
        //         TS_STD(CLOSE,20)
        //     )
        // )
        // ------------------------------------------------------------

        auto close =
            ASTNode::makeVariable(
                VariableType::CLOSE
            );

        auto mean20 =
            ASTNode::makeTimeSeries(
                TimeSeriesOperator::TS_MEAN,
                close,
                20
            );

        auto std20 =
            ASTNode::makeTimeSeries(
                TimeSeriesOperator::TS_STD,
                close,
                20
            );

        auto deviation =
            ASTNode::makeBinary(
                BinaryOperator::SUB,
                close,
                mean20
            );

        auto standardized =
            ASTNode::makeBinary(
                BinaryOperator::DIV,
                deviation,
                std20
            );

        auto alpha =
            ASTNode::makeUnary(
                UnaryOperator::RANK,
                standardized
            );

        // ------------------------------------------------------------
        // Fitness using 20-day forward return
        // ------------------------------------------------------------

        FitnessEvaluator fitnessEvaluator(data);

        FitnessResult fitness =
            fitnessEvaluator.evaluate(
                alpha,
                data.forward_return_20d
            );

        // ------------------------------------------------------------
        // Print results
        // ------------------------------------------------------------

        std::cout << "\n";
        std::cout
            << "===== FITNESS RESULT =====\n";

        std::cout
            << std::setprecision(12);

        std::cout
            << "Mean IC: "
            << fitness.mean_ic
            << "\n";

        std::cout
            << "IC Std: "
            << fitness.ic_std
            << "\n";

        std::cout
            << "ICIR: "
            << fitness.icir
            << "\n";

        std::cout
            << "Number of IC days: "
            << fitness.num_days
            << "\n";

        std::cout
            << "Valid observations: "
            << fitness.num_observations
            << "\n";

        std::cout
            << "Coverage: "
            << fitness.coverage
            << "\n";

        return 0;

    } catch (
        const std::exception& e
    ) {

        std::cerr
            << "ERROR: "
            << e.what()
            << "\n";

        return 1;
    }
}
