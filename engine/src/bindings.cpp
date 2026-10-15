#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "AST.h"
#include "Evaluator.h"
#include "Fitness.h"
#include "Data.h"

#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace py = pybind11;


// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------

static std::shared_ptr<ASTNode> make_variable(const std::string& name)
{
    if (name == "OPEN")
        return ASTNode::makeVariable(VariableType::OPEN);

    if (name == "HIGH")
        return ASTNode::makeVariable(VariableType::HIGH);

    if (name == "LOW")
        return ASTNode::makeVariable(VariableType::LOW);

    if (name == "CLOSE")
        return ASTNode::makeVariable(VariableType::CLOSE);

    if (name == "VOLUME")
        return ASTNode::makeVariable(VariableType::VOLUME);

    throw std::runtime_error("Unknown variable: " + name);
}


// ------------------------------------------------------------
// Python-facing data structure
// ------------------------------------------------------------

struct PyMarketData
{
    std::vector<std::string> datetime;
    std::vector<std::string> instrument;

    std::vector<double> open;
    std::vector<double> high;
    std::vector<double> low;
    std::vector<double> close;
    std::vector<double> volume;

    std::vector<double> forward_return_1d;
    std::vector<double> forward_return_5d;
    std::vector<double> forward_return_20d;
    std::vector<double> forward_return_30d;

    MarketData to_market_data() const
    {
        MarketData data;

        data.datetime = datetime;
        data.instrument = instrument;

        data.open = open;
        data.high = high;
        data.low = low;
        data.close = close;
        data.volume = volume;

        data.forward_return_1d = forward_return_1d;
        data.forward_return_5d = forward_return_5d;
        data.forward_return_20d = forward_return_20d;
        data.forward_return_30d = forward_return_30d;

        return data;
    }
};


// ------------------------------------------------------------
// AST factory
// ------------------------------------------------------------

std::shared_ptr<ASTNode> create_variable(
    const std::string& variable)
{
    return make_variable(variable);
}


std::shared_ptr<ASTNode> create_number(double value)
{
    return ASTNode::makeNumber(value);
}


std::shared_ptr<ASTNode> create_unary(
    const std::string& op,
    std::shared_ptr<ASTNode> operand)
{
    if (op == "ABS")
        return ASTNode::makeUnary(
            UnaryOperator::ABS,
            operand
        );

    if (op == "LOG")
        return ASTNode::makeUnary(
            UnaryOperator::LOG,
            operand
        );

    if (op == "SIGN")
        return ASTNode::makeUnary(
            UnaryOperator::SIGN,
            operand
        );

    if (op == "RANK")
        return ASTNode::makeUnary(
            UnaryOperator::RANK,
            operand
        );

    throw std::runtime_error("Unknown unary operator: " + op);
}


std::shared_ptr<ASTNode> create_binary(
    const std::string& op,
    std::shared_ptr<ASTNode> left,
    std::shared_ptr<ASTNode> right)
{
    if (op == "ADD")
        return ASTNode::makeBinary(
            BinaryOperator::ADD,
            left,
            right
        );

    if (op == "SUB")
        return ASTNode::makeBinary(
            BinaryOperator::SUB,
            left,
            right
        );

    if (op == "MUL")
        return ASTNode::makeBinary(
            BinaryOperator::MUL,
            left,
            right
        );

    if (op == "DIV")
        return ASTNode::makeBinary(
            BinaryOperator::DIV,
            left,
            right
        );

    throw std::runtime_error("Unknown binary operator: " + op);
}


std::shared_ptr<ASTNode> create_time_series(
    const std::string& op,
    std::shared_ptr<ASTNode> operand,
    int window)
{
    if (op == "TS_MEAN")
        return ASTNode::makeTimeSeries(
            TimeSeriesOperator::TS_MEAN,
            operand,
            window
        );

    if (op == "TS_STD")
        return ASTNode::makeTimeSeries(
            TimeSeriesOperator::TS_STD,
            operand,
            window
        );

    if (op == "TS_MIN")
        return ASTNode::makeTimeSeries(
            TimeSeriesOperator::TS_MIN,
            operand,
            window
        );

    if (op == "TS_MAX")
        return ASTNode::makeTimeSeries(
            TimeSeriesOperator::TS_MAX,
            operand,
            window
        );

    throw std::runtime_error(
        "Unknown time-series operator: " + op
    );
}


// ------------------------------------------------------------
// Module
// ------------------------------------------------------------

PYBIND11_MODULE(alpha_engine_cpp, m)
{
    m.doc() =
        "C++ execution engine for symbolic alpha evaluation";


    // --------------------------------------------------------
    // AST
    // --------------------------------------------------------

    py::class_<ASTNode,
               std::shared_ptr<ASTNode>>(m, "ASTNode")
        .def_static(
            "variable",
            &create_variable
        )
        .def_static(
            "number",
            &create_number
        )
        .def_static(
            "unary",
            &create_unary
        )
        .def_static(
            "binary",
            &create_binary
        )
        .def_static(
            "time_series",
            &create_time_series
        );


    // --------------------------------------------------------
    // Market data
    // --------------------------------------------------------

    py::class_<PyMarketData>(m, "MarketData")
        .def(py::init<>())

        .def_readwrite(
            "datetime",
            &PyMarketData::datetime
        )

        .def_readwrite(
            "instrument",
            &PyMarketData::instrument
        )

        .def_readwrite(
            "open",
            &PyMarketData::open
        )

        .def_readwrite(
            "high",
            &PyMarketData::high
        )

        .def_readwrite(
            "low",
            &PyMarketData::low
        )

        .def_readwrite(
            "close",
            &PyMarketData::close
        )

        .def_readwrite(
            "volume",
            &PyMarketData::volume
        )

        .def_readwrite(
            "forward_return_1d",
            &PyMarketData::forward_return_1d
        )

        .def_readwrite(
            "forward_return_5d",
            &PyMarketData::forward_return_5d
        )

        .def_readwrite(
            "forward_return_20d",
            &PyMarketData::forward_return_20d
        )

        .def_readwrite(
            "forward_return_30d",
            &PyMarketData::forward_return_30d
        );


    // --------------------------------------------------------
    // Evaluator
    // --------------------------------------------------------

    m.def(
        "evaluate",
        [](const PyMarketData& py_data,
           std::shared_ptr<ASTNode> ast)
        {
            MarketData data =
                py_data.to_market_data();

            AlphaEvaluator evaluator(data);

            return evaluator.evaluate(ast);
        },
        "Evaluate an alpha expression using C++"
    );


    // --------------------------------------------------------
    // Fitness
    // --------------------------------------------------------

    py::class_<FitnessResult>(
        m,
        "FitnessResult"
    )
        .def_readonly(
            "mean_ic",
            &FitnessResult::mean_ic
        )

        .def_readonly(
            "ic_std",
            &FitnessResult::ic_std
        )

        .def_readonly(
            "icir",
            &FitnessResult::icir
        )

        .def_readonly(
            "num_days",
            &FitnessResult::num_days
        )

        .def_readonly(
            "num_observations",
            &FitnessResult::num_observations
        )

        .def_readonly(
            "coverage",
            &FitnessResult::coverage
        );


    m.def(
        "evaluate_fitness",
        [](const PyMarketData& py_data,
           std::shared_ptr<ASTNode> ast,
           int horizon)
        {
            MarketData data =
                py_data.to_market_data();

            FitnessEvaluator evaluator(data);

            const std::vector<double>* returns = nullptr;

            switch (horizon)
            {
                case 1:
                    returns = &data.forward_return_1d;
                    break;

                case 5:
                    returns = &data.forward_return_5d;
                    break;

                case 20:
                    returns = &data.forward_return_20d;
                    break;

                case 30:
                    returns = &data.forward_return_30d;
                    break;

                default:
                    throw std::runtime_error(
                        "Unsupported horizon"
                    );
            }

            return evaluator.evaluate(
                ast,
                *returns
            );
        },
        "Evaluate alpha fitness using C++"
    );
}