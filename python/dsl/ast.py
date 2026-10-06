from dataclasses import dataclass
from typing import Union


# ============================================================
# Base Node
# ============================================================

class ASTNode:
    """Base class for every node in the alpha AST."""

    def to_string(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.to_string()


# ============================================================
# Leaf Nodes
# ============================================================

@dataclass(frozen=True)
class Variable(ASTNode):
    name: str

    def to_string(self) -> str:
        return self.name


@dataclass(frozen=True)
class Number(ASTNode):
    value: float

    def to_string(self) -> str:
        # Keep integers clean: 20 instead of 20.0
        if self.value.is_integer():
            return str(int(self.value))

        return str(self.value)


# ============================================================
# Unary Operation
# ============================================================

@dataclass(frozen=True)
class UnaryOp(ASTNode):
    operator: str
    operand: ASTNode

    def to_string(self) -> str:
        return f"{self.operator}({self.operand.to_string()})"


# ============================================================
# Binary Operation
# ============================================================

@dataclass(frozen=True)
class BinaryOp(ASTNode):
    operator: str
    left: ASTNode
    right: ASTNode

    def to_string(self) -> str:
        return (
            f"{self.operator}("
            f"{self.left.to_string()},"
            f"{self.right.to_string()}"
            f")"
        )


# ============================================================
# Time-Series Operation
# ============================================================

@dataclass(frozen=True)
class TimeSeriesOp(ASTNode):
    operator: str
    operand: ASTNode
    window: int

    def to_string(self) -> str:
        return (
            f"{self.operator}("
            f"{self.operand.to_string()},"
            f"{self.window}"
            f")"
        )


# ============================================================
# Type Alias
# ============================================================

AST = Union[
    Variable,
    Number,
    UnaryOp,
    BinaryOp,
    TimeSeriesOp,
]