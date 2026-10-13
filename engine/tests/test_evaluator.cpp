#include <iostream>
#include <vector>
#include <cmath>
#include <string>

#include "AST.h"
#include "Data.h"
#include "Evaluator.h"


// --------------------------------------------------
// Simple assertion helpers
// --------------------------------------------------

void assertEqual(double actual, double expected, double tolerance = 1e-6) {

    if (std::isnan(expected)) {
        if (!std::isnan(actual)) {
            std::cerr
                << "Expected NaN but got "
                << actual << "\n";
            std::exit(1);
        }
        return;
    }

    if (std::abs(actual - expected) > tolerance) {

        std::cerr
            << "Expected "
            << expected
            << " but got "
            << actual
            << "\n";

        std::exit(1);
    }
}


void assertVectorEqual(
    const std::vector<double>& actual,
    const std::vector<double>& expected,
    double tolerance = 1e-5
) {

    if (actual.size() != expected.size()) {
        std::cerr << "Vector size mismatch\n";
        std::exit(1);
    }

    for (size_t i = 0; i < actual.size(); ++i) {
        assertEqual(actual[i], expected[i], tolerance);
    }
}


// --------------------------------------------------
// Test 1: Variable
// --------------------------------------------------

void testVariable() {

    MarketData data;

    data.close = {
        10,
        20,
        30
    };

    AlphaEvaluator evaluator(data);

    auto close =
        ASTNode::makeVariable(VariableType::CLOSE);

    auto result =
        evaluator.evaluate(close);

    assertVectorEqual(
        result,
        {10, 20, 30}
    );

    std::cout << "PASS: Variable\n";
}


// --------------------------------------------------
// Test 2: Arithmetic
// --------------------------------------------------

void testArithmetic() {

    MarketData data;

    data.open = {
        10,
        20,
        30
    };

    data.close = {
        15,
        25,
        20
    };

    AlphaEvaluator evaluator(data);

    auto open =
        ASTNode::makeVariable(VariableType::OPEN);

    auto close =
        ASTNode::makeVariable(VariableType::CLOSE);

    auto expression =
        ASTNode::makeBinary(
            BinaryOperator::SUB,
            close,
            open
        );

    auto result =
        evaluator.evaluate(expression);

    assertVectorEqual(
        result,
        {5, 5, -10}
    );

    std::cout << "PASS: Arithmetic\n";
}


// --------------------------------------------------
// Test 3: Division by zero
// --------------------------------------------------

void testDivisionByZero() {

    MarketData data;

    data.close = {
        10,
        20,
        30
    };

    data.volume = {
        2,
        0,
        5
    };

    AlphaEvaluator evaluator(data);

    auto close =
        ASTNode::makeVariable(VariableType::CLOSE);

    auto volume =
        ASTNode::makeVariable(VariableType::VOLUME);

    auto expression =
        ASTNode::makeBinary(
            BinaryOperator::DIV,
            close,
            volume
        );

    auto result =
        evaluator.evaluate(expression);

    assertEqual(result[0], 5.0);
    assertEqual(result[1], NAN);
    assertEqual(result[2], 6.0);

    std::cout << "PASS: Division by zero\n";
}


// --------------------------------------------------
// Test 4: TS_MEAN
// --------------------------------------------------

void testTimeSeriesMean() {

    MarketData data;

    data.close = {
        10,
        12,
        15,
        13,
        16
    };

    AlphaEvaluator evaluator(data);

    auto close =
        ASTNode::makeVariable(VariableType::CLOSE);

    auto mean =
        ASTNode::makeTimeSeries(
            TimeSeriesOperator::TS_MEAN,
            close,
            3
        );

    auto result =
        evaluator.evaluate(mean);

    assertVectorEqual(
        result,
        {
            NAN,
            NAN,
            12.3333333333,
            13.3333333333,
            14.6666666667
        }
    );

    std::cout << "PASS: TS_MEAN\n";
}


// --------------------------------------------------
// Test 5: TS_MIN / TS_MAX
// --------------------------------------------------

void testTimeSeriesMinMax() {

    MarketData data;

    data.close = {
        10,
        12,
        15,
        13,
        16
    };

    AlphaEvaluator evaluator(data);

    auto close =
        ASTNode::makeVariable(VariableType::CLOSE);

    auto minNode =
        ASTNode::makeTimeSeries(
            TimeSeriesOperator::TS_MIN,
            close,
            3
        );

    auto maxNode =
        ASTNode::makeTimeSeries(
            TimeSeriesOperator::TS_MAX,
            close,
            3
        );

    auto minResult =
        evaluator.evaluate(minNode);

    auto maxResult =
        evaluator.evaluate(maxNode);

    assertVectorEqual(
        minResult,
        {
            NAN,
            NAN,
            10,
            12,
            13
        }
    );

    assertVectorEqual(
        maxResult,
        {
            NAN,
            NAN,
            15,
            15,
            16
        }
    );

    std::cout << "PASS: TS_MIN / TS_MAX\n";
}


// --------------------------------------------------
// Test 6: TS_STD
// --------------------------------------------------

void testTimeSeriesStd() {

    MarketData data;

    data.close = {
        10,
        12,
        15,
        13,
        16
    };

    AlphaEvaluator evaluator(data);

    auto close =
        ASTNode::makeVariable(VariableType::CLOSE);

    auto stdNode =
        ASTNode::makeTimeSeries(
            TimeSeriesOperator::TS_STD,
            close,
            3
        );

    auto result =
        evaluator.evaluate(stdNode);

    assertVectorEqual(
        result,
        {
            NAN,
            NAN,
            2.5166114784,
            1.5275252317,
            1.5275252317
        }
    );

    std::cout << "PASS: TS_STD\n";
}


// --------------------------------------------------
// Test 7: RANK
// --------------------------------------------------

void testRank() {

    MarketData data;

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
        110,

        102,
        118,
        115
    };

    AlphaEvaluator evaluator(data);

    auto close =
        ASTNode::makeVariable(VariableType::CLOSE);

    auto rank =
        ASTNode::makeUnary(
            UnaryOperator::RANK,
            close
        );

    auto result =
        evaluator.evaluate(rank);

    assertVectorEqual(
        result,
        {
            1.0 / 3.0,
            1.0,
            2.0 / 3.0,

            1.0 / 3.0,
            1.0,
            2.0 / 3.0
        }
    );

    std::cout << "PASS: RANK\n";
}


// --------------------------------------------------
// Test 8: RANK with ties
// --------------------------------------------------

void testRankTies() {

    MarketData data;

    data.datetime = {
        "2020-01-01",
        "2020-01-01",
        "2020-01-01"
    };

    data.instrument = {
        "A",
        "B",
        "C"
    };

    data.close = {
        100,
        120,
        100
    };

    AlphaEvaluator evaluator(data);

    auto close =
        ASTNode::makeVariable(VariableType::CLOSE);

    auto rank =
        ASTNode::makeUnary(
            UnaryOperator::RANK,
            close
        );

    auto result =
        evaluator.evaluate(rank);

    assertVectorEqual(
        result,
        {
            0.5,
            1.0,
            0.5
        }
    );

    std::cout << "PASS: RANK ties\n";
}


// --------------------------------------------------
// Test 9: Nested expression
// --------------------------------------------------

void testNestedExpression() {

    MarketData data;

    data.open = {
        10,
        20,
        30
    };

    data.close = {
        15,
        25,
        35
    };

    AlphaEvaluator evaluator(data);

    auto open =
        ASTNode::makeVariable(VariableType::OPEN);

    auto close =
        ASTNode::makeVariable(VariableType::CLOSE);

    // ABS(CLOSE - OPEN)

    auto difference =
        ASTNode::makeBinary(
            BinaryOperator::SUB,
            close,
            open
        );

    auto expression =
        ASTNode::makeUnary(
            UnaryOperator::ABS,
            difference
        );

    auto result =
        evaluator.evaluate(expression);

    assertVectorEqual(
        result,
        {
            5,
            5,
            5
        }
    );

    std::cout << "PASS: Nested expression\n";
}


// --------------------------------------------------
// Main
// --------------------------------------------------

int main() {

    testVariable();
    testArithmetic();
    testDivisionByZero();

    testTimeSeriesMean();
    testTimeSeriesMinMax();
    testTimeSeriesStd();

    testRank();
    testRankTies();

    testNestedExpression();

    std::cout << "\n";
    std::cout << "ALL TESTS PASSED\n";

    return 0;
}