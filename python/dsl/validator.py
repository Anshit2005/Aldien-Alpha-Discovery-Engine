from math import isfinite

from .ast import ASTNode, Variable, Number, UnaryOp, BinaryOp, TimeSeriesOp


ALLOWED_VARIABLES = {
    "OPEN",
    "HIGH",
    "LOW",
    "CLOSE",
    "VOLUME",
}

ALLOWED_BINARY_OPERATORS = {
    "ADD",
    "SUB",
    "MUL",
    "DIV",
}

ALLOWED_UNARY_OPERATORS = {
    "ABS",
    "LOG",
    "SIGN",
    "RANK",
}

ALLOWED_TIME_SERIES_OPERATORS = {
    "TS_MEAN",
    "TS_STD",
    "TS_MIN",
    "TS_MAX",
}

MAX_DEPTH = 8
MAX_WINDOW = 252


class ValidationError(ValueError):
    """Raised when an AST violates DSL research constraints."""


def validate(ast: ASTNode) -> None:
    """
    Validate an AST.

    Raises:
        ValidationError: if the AST violates any DSL constraint.
    """
    _validate_node(ast, depth=1)


def _validate_node(node: ASTNode, depth: int) -> None:

    # ---------------------------------------------------------
    # Depth constraint
    # ---------------------------------------------------------

    if depth > MAX_DEPTH:
        raise ValidationError(
            f"Maximum AST depth exceeded: {depth} > {MAX_DEPTH}"
        )

    # ---------------------------------------------------------
    # Variable
    # ---------------------------------------------------------

    if isinstance(node, Variable):
        if node.name not in ALLOWED_VARIABLES:
            raise ValidationError(
                f"Unknown variable: {node.name}"
            )
        return

    # ---------------------------------------------------------
    # Numeric constant
    # ---------------------------------------------------------

    if isinstance(node, Number):
        if not isfinite(node.value):
            raise ValidationError(
                f"Invalid numeric constant: {node.value}"
            )
        return

    # ---------------------------------------------------------
    # Unary operator
    # ---------------------------------------------------------

    if isinstance(node, UnaryOp):

        if node.operator not in ALLOWED_UNARY_OPERATORS:
            raise ValidationError(
                f"Unknown unary operator: {node.operator}"
            )

        _validate_node(node.operand, depth + 1)
        return

    # ---------------------------------------------------------
    # Binary operator
    # ---------------------------------------------------------

    if isinstance(node, BinaryOp):

        if node.operator not in ALLOWED_BINARY_OPERATORS:
            raise ValidationError(
                f"Unknown binary operator: {node.operator}"
            )

        _validate_node(node.left, depth + 1)
        _validate_node(node.right, depth + 1)

        return

    # ---------------------------------------------------------
    # Time-series operator
    # ---------------------------------------------------------

    if isinstance(node, TimeSeriesOp):

        if node.operator not in ALLOWED_TIME_SERIES_OPERATORS:
            raise ValidationError(
                f"Unknown time-series operator: {node.operator}"
            )

        if node.window <= 0:
            raise ValidationError(
                f"Window must be positive: {node.window}"
            )

        if node.window > MAX_WINDOW:
            raise ValidationError(
                f"Window exceeds maximum of {MAX_WINDOW}: {node.window}"
            )

        _validate_node(node.operand, depth + 1)

        return

    # ---------------------------------------------------------
    # Unknown AST node
    # ---------------------------------------------------------

    raise ValidationError(
        f"Unsupported AST node type: {type(node).__name__}"
    )