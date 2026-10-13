#include "AST.h"

std::shared_ptr<ASTNode>
ASTNode::makeVariable(VariableType variable) {

    auto node = std::make_shared<ASTNode>();

    node->type = NodeType::VARIABLE;
    node->variable = variable;

    return node;
}


std::shared_ptr<ASTNode>
ASTNode::makeNumber(double value) {

    auto node = std::make_shared<ASTNode>();

    node->type = NodeType::NUMBER;
    node->value = value;

    return node;
}


std::shared_ptr<ASTNode>
ASTNode::makeUnary(
    UnaryOperator op,
    std::shared_ptr<ASTNode> operand
) {

    auto node = std::make_shared<ASTNode>();

    node->type = NodeType::UNARY_OP;
    node->unary_op = op;
    node->operand = operand;

    return node;
}


std::shared_ptr<ASTNode>
ASTNode::makeBinary(
    BinaryOperator op,
    std::shared_ptr<ASTNode> left,
    std::shared_ptr<ASTNode> right
) {

    auto node = std::make_shared<ASTNode>();

    node->type = NodeType::BINARY_OP;
    node->binary_op = op;
    node->left = left;
    node->right = right;

    return node;
}


std::shared_ptr<ASTNode>
ASTNode::makeTimeSeries(
    TimeSeriesOperator op,
    std::shared_ptr<ASTNode> operand,
    int window
) {

    auto node = std::make_shared<ASTNode>();

    node->type = NodeType::TIME_SERIES_OP;
    node->ts_op = op;
    node->operand = operand;
    node->window = window;

    return node;
}