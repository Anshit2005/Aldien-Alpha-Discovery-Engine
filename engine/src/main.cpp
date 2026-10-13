#include "AST.h"
#include "CSVLoader.h"
#include "Evaluator.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

int main() {

    try {

        const std::string inputFile =
            "../../data/processed/cpp_test_data.csv";

        const std::string outputFile =
            "../../data/processed/cpp_alpha_output.csv";

        std::cout << "Loading: "
                  << inputFile << "\n";

        MarketData data =
            CSVLoader::load(inputFile);

        std::cout << "Rows: "
                  << data.size() << "\n";

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
        // Evaluate
        // ------------------------------------------------------------

        AlphaEvaluator evaluator(data);

        std::vector<double> result =
            evaluator.evaluate(alpha);

        // ------------------------------------------------------------
        // Write output
        // ------------------------------------------------------------

        std::ofstream output(outputFile);

        if (!output.is_open()) {
            throw std::runtime_error(
                "Could not create output file: " +
                outputFile
            );
        }

        output
            << "datetime,instrument,alpha\n";

        output << std::setprecision(17);

        for (size_t i = 0;
             i < result.size();
             ++i) {

            output
                << data.datetime[i]
                << ","
                << data.instrument[i]
                << ",";

            if (std::isfinite(result[i])) {
                output << result[i];
            } else {
                output << "nan";
            }

            output << "\n";
        }

        output.close();

        // ------------------------------------------------------------
        // Statistics
        // ------------------------------------------------------------

        size_t valid = 0;

        for (double value : result) {
            if (std::isfinite(value)) {
                ++valid;
            }
        }

        std::cout
            << "Evaluation complete.\n";

        std::cout
            << "Output size: "
            << result.size()
            << "\n";

        std::cout
            << "Valid alpha values: "
            << valid
            << "\n";

        std::cout
            << "Invalid/NaN values: "
            << result.size() - valid
            << "\n";

        std::cout
            << "Saved output: "
            << outputFile
            << "\n";

        return 0;

    } catch (const std::exception& e) {

        std::cerr
            << "ERROR: "
            << e.what()
            << "\n";

        return 1;
    }
}