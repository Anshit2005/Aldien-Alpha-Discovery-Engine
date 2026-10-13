#pragma once

#include <memory>
#include <string>

enum class NodeType {
    VARIABLE,
    NUMBER,
    UNARY_OP,
    BINARY_OP,
    TIME_SERIES_OP
};

enum class VariableType {
    OPEN,
    HIGH,
    LOW,
    CLOSE,
    VOLUME
};

enum class UnaryOperator {
    ABS,
    LOG,
    SIGN,
    RANK
};

enum class BinaryOperator {
    ADD,
    SUB,
    MUL,
    DIV
};

enum class TimeSeriesOperator {
    TS_MEAN,
    TS_STD,
    TS_MIN,
    TS_MAX
};

struct ASTNode {
    NodeType type;

    // VARIABLE
    VariableType variable;

    // NUMBER
    double value;

    // Operators
    UnaryOperator unary_op;
    BinaryOperator binary_op;
    TimeSeriesOperator ts_op;

    std::shared_ptr<ASTNode> left;
    std::shared_ptr<ASTNode> right;
    std::shared_ptr<ASTNode> operand;

    // Time-series window
    int window;

    static std::shared_ptr<ASTNode> makeVariable(VariableType variable);

    static std::shared_ptr<ASTNode> makeNumber(double value);

    static std::shared_ptr<ASTNode> makeUnary(
        UnaryOperator op,
        std::shared_ptr<ASTNode> operand
    );

    static std::shared_ptr<ASTNode> makeBinary(
        BinaryOperator op,
        std::shared_ptr<ASTNode> left,
        std::shared_ptr<ASTNode> right
    );

    static std::shared_ptr<ASTNode> makeTimeSeries(
        TimeSeriesOperator op,
        std::shared_ptr<ASTNode> operand,
        int window
    );
};