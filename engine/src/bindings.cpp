#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "AST.h"
#include "Evaluator.h"
#include "Fitness.h"
#include "Data.h"
#include "Engine.h"

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
// numpy <-> std::vector
// ------------------------------------------------------------

template <typename T>
static std::vector<T> to_vector(
    const py::array_t<T, py::array::c_style | py::array::forcecast>& array)
{
    if (array.ndim() != 1)
        throw std::invalid_argument("expected a one-dimensional array");
    const T* data = array.data();
    return std::vector<T>(data, data + array.shape(0));
}

template <typename T>
static py::array_t<T> to_array(const std::vector<T>& values)
{
    py::array_t<T> array(static_cast<py::ssize_t>(values.size()));
    std::copy(values.begin(), values.end(), array.mutable_data());
    return array;
}

using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Int32Array = py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>;
using Int8Array = py::array_t<std::int8_t, py::array::c_style | py::array::forcecast>;


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

    // --------------------------------------------------------
    // Load-once engine
    // --------------------------------------------------------

    py::class_<EngineFitness>(m, "EngineFitness")
        .def_readonly("horizon", &EngineFitness::horizon)
        .def_readonly("mean_ic", &EngineFitness::mean_ic)
        .def_readonly("ic_std", &EngineFitness::ic_std)
        .def_readonly("icir", &EngineFitness::icir)
        .def_readonly("num_days", &EngineFitness::num_days)
        .def_readonly("num_observations", &EngineFitness::num_observations)
        .def_readonly("coverage", &EngineFitness::coverage)
        .def_property_readonly("days", [](const EngineFitness& f) { return to_array(f.days); })
        .def_property_readonly("daily_ic", [](const EngineFitness& f) { return to_array(f.daily_ic); });

    py::class_<Engine>(m, "Engine")
        .def(
            py::init([](const Int32Array& instrument,
                        const Int32Array& day,
                        const DoubleArray& open,
                        const DoubleArray& high,
                        const DoubleArray& low,
                        const DoubleArray& close,
                        const DoubleArray& volume,
                        const std::vector<DoubleArray>& returns,
                        const Int8Array& split,
                        const std::vector<Int8Array>& target_split) {
                std::vector<std::vector<double>> returns_v;
                for (const auto& r : returns) returns_v.push_back(to_vector<double>(r));
                std::vector<std::vector<std::int8_t>> target_v;
                for (const auto& t : target_split) target_v.push_back(to_vector<std::int8_t>(t));
                return Engine(
                    to_vector<std::int32_t>(instrument), to_vector<std::int32_t>(day),
                    to_vector<double>(open), to_vector<double>(high), to_vector<double>(low),
                    to_vector<double>(close), to_vector<double>(volume),
                    std::move(returns_v), to_vector<std::int8_t>(split), std::move(target_v));
            }),
            py::arg("instrument"), py::arg("day"),
            py::arg("open"), py::arg("high"), py::arg("low"), py::arg("close"), py::arg("volume"),
            py::arg("returns"), py::arg("split"), py::arg("target_split"),
            "Copy the panel into C++ once. Rows sorted by instrument, then day; "
            "returns and target_split ordered by horizon (1, 5, 20, 30); "
            "split codes -1 for none (pass empty arrays when there are no splits).")
        .def("evaluate",
             [](const Engine& engine, std::shared_ptr<ASTNode> ast) {
                 std::vector<double> alpha;
                 {
                     py::gil_scoped_release release;
                     alpha = engine.evaluate(ast);
                 }
                 return to_array(alpha);
             },
             py::arg("ast"), "Alpha values for every row")
        .def("fitness",
             [](const Engine& engine, std::shared_ptr<ASTNode> ast, int horizon, int split) {
                 py::gil_scoped_release release;
                 return engine.fitness(ast, horizon, split);
             },
             py::arg("ast"), py::arg("horizon") = 20, py::arg("split") = -1,
             "Fitness and daily IC; split -1 means every row")
        .def_property_readonly("size", &Engine::size)
        .def_property_readonly("num_days", &Engine::num_days)
        .def_property_readonly("num_instruments", &Engine::num_instruments);
}
