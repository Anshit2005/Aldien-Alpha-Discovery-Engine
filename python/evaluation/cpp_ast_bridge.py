from python.dsl.ast import (
    ASTNode,
    Variable,
    Number,
    UnaryOp,
    BinaryOp,
    TimeSeriesOp,
)

from python.cpp_engine import alpha_engine_cpp


class CppASTConversionError(Exception):
    """Raised when a Python AST cannot be converted to a C++ AST."""

    pass


VARIABLES = {
    "OPEN",
    "HIGH",
    "LOW",
    "CLOSE",
    "VOLUME",
}

UNARY_OPERATORS = {
    "ABS",
    "LOG",
    "SIGN",
    "RANK",
}

BINARY_OPERATORS = {
    "ADD",
    "SUB",
    "MUL",
    "DIV",
}

TIME_SERIES_OPERATORS = {
    "TS_MEAN",
    "TS_STD",
    "TS_MIN",
    "TS_MAX",
}


def python_ast_to_cpp(node: ASTNode):
    """
    Convert our Python DSL AST into the corresponding C++ AST.

    Python:
        RANK(DIV(CLOSE, TS_MEAN(CLOSE, 20)))

    becomes a C++ AST that can be evaluated by
    alpha_engine_cpp.
    """

    if isinstance(node, Variable):
        if node.name not in VARIABLES:
            raise CppASTConversionError(
                f"Unsupported variable: {node.name}"
            )

        return alpha_engine_cpp.ASTNode.variable(
            node.name
        )

    if isinstance(node, Number):
        return alpha_engine_cpp.ASTNode.number(
            float(node.value)
        )

    if isinstance(node, UnaryOp):
        if node.operator not in UNARY_OPERATORS:
            raise CppASTConversionError(
                f"Unsupported unary operator: "
                f"{node.operator}"
            )

        operand = python_ast_to_cpp(
            node.operand
        )

        return alpha_engine_cpp.ASTNode.unary(
            node.operator,
            operand
        )

    if isinstance(node, BinaryOp):
        if node.operator not in BINARY_OPERATORS:
            raise CppASTConversionError(
                f"Unsupported binary operator: "
                f"{node.operator}"
            )

        left = python_ast_to_cpp(
            node.left
        )

        right = python_ast_to_cpp(
            node.right
        )

        return alpha_engine_cpp.ASTNode.binary(
            node.operator,
            left,
            right
        )

    if isinstance(node, TimeSeriesOp):
        if node.operator not in TIME_SERIES_OPERATORS:
            raise CppASTConversionError(
                f"Unsupported time-series operator: "
                f"{node.operator}"
            )

        operand = python_ast_to_cpp(
            node.operand
        )

        return alpha_engine_cpp.ASTNode.time_series(
            node.operator,
            operand,
            int(node.window)
        )

    raise CppASTConversionError(
        f"Unsupported Python AST node: "
        f"{type(node).__name__}"
    )