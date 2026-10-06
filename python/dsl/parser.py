from .ast import (
    AST,
    Variable,
    Number,
    UnaryOp,
    BinaryOp,
    TimeSeriesOp,
)

from .tokenizer import tokenize, Token


# ============================================================
# DSL Definition
# ============================================================

VARIABLES = {
    "OPEN",
    "HIGH",
    "LOW",
    "CLOSE",
    "VOLUME",
}

BINARY_OPERATORS = {
    "ADD",
    "SUB",
    "MUL",
    "DIV",
}

UNARY_OPERATORS = {
    "ABS",
    "LOG",
    "SIGN",
    "RANK",
}

TIME_SERIES_OPERATORS = {
    "TS_MEAN",
    "TS_STD",
    "TS_MIN",
    "TS_MAX",
}


class Parser:
    def __init__(self, expression: str):
        self.expression = expression
        self.tokens = tokenize(expression)
        self.position = 0

    # --------------------------------------------------------
    # Token helpers
    # --------------------------------------------------------

    def current(self) -> Token | None:
        if self.position >= len(self.tokens):
            return None

        return self.tokens[self.position]

    def consume(self, token_type: str) -> Token:
        token = self.current()

        if token is None:
            raise ValueError(
                f"Expected {token_type}, but reached end of expression"
            )

        if token.type != token_type:
            raise ValueError(
                f"Expected {token_type} at position {token.position}, "
                f"got {token.type} ({token.value!r})"
            )

        self.position += 1

        return token

    # --------------------------------------------------------
    # Entry point
    # --------------------------------------------------------

    def parse(self) -> AST:

        if not self.tokens:
            raise ValueError("Expression is empty")

        node = self.parse_expression()

        if self.current() is not None:
            token = self.current()

            raise ValueError(
                f"Unexpected token {token.value!r} "
                f"at position {token.position}"
            )

        return node

    # --------------------------------------------------------
    # Expression
    # --------------------------------------------------------

    def parse_expression(self) -> AST:

        token = self.current()

        if token is None:
            raise ValueError("Unexpected end of expression")

        # Variable or operator
        if token.type == "IDENTIFIER":

            identifier = token.value.upper()

            self.position += 1

            # ------------------------------------------------
            # Variable
            # ------------------------------------------------

            if identifier in VARIABLES:

                # Variables must not have arguments.
                if self.current() is not None:
                    if self.current().type == "LPAREN":
                        raise ValueError(
                            f"Variable {identifier} cannot have arguments"
                        )

                return Variable(identifier)

            # ------------------------------------------------
            # Operator
            # ------------------------------------------------

            if identifier in (
                BINARY_OPERATORS
                | UNARY_OPERATORS
                | TIME_SERIES_OPERATORS
            ):

                return self.parse_operator(identifier)

            raise ValueError(
                f"Unknown identifier: {identifier}"
            )

        # ----------------------------------------------------
        # Number
        # ----------------------------------------------------

        if token.type == "NUMBER":

            self.position += 1

            return Number(float(token.value))

        raise ValueError(
            f"Unexpected token {token.value!r} "
            f"at position {token.position}"
        )

    # --------------------------------------------------------
    # Operator
    # --------------------------------------------------------

    def parse_operator(self, operator: str) -> AST:

        self.consume("LPAREN")

        # ----------------------------------------------------
        # Binary
        # ----------------------------------------------------

        if operator in BINARY_OPERATORS:

            left = self.parse_expression()

            self.consume("COMMA")

            right = self.parse_expression()

            self.consume("RPAREN")

            return BinaryOp(
                operator=operator,
                left=left,
                right=right,
            )

        # ----------------------------------------------------
        # Unary
        # ----------------------------------------------------

        if operator in UNARY_OPERATORS:

            operand = self.parse_expression()

            self.consume("RPAREN")

            return UnaryOp(
                operator=operator,
                operand=operand,
            )

        # ----------------------------------------------------
        # Time Series
        # ----------------------------------------------------

        if operator in TIME_SERIES_OPERATORS:

            operand = self.parse_expression()

            self.consume("COMMA")

            window_token = self.consume("NUMBER")

            window = float(window_token.value)

            if not window.is_integer():
                raise ValueError(
                    f"Window must be an integer, got {window}"
                )

            window = int(window)

            if window <= 0:
                raise ValueError(
                    f"Window must be positive, got {window}"
                )

            self.consume("RPAREN")

            return TimeSeriesOp(
                operator=operator,
                operand=operand,
                window=window,
            )

        raise ValueError(
            f"Unsupported operator: {operator}"
        )


def parse(expression: str) -> AST:
    """Parse a DSL expression into an AST."""
    return Parser(expression).parse()