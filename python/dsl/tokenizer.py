from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Token:
    type: str
    value: str
    position: int


TOKEN_SPEC = [
    ("NUMBER", r"(?:\d+(?:\.\d*)?|\.\d+)"),
    ("IDENTIFIER", r"[A-Za-z_][A-Za-z0-9_]*"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("COMMA", r","),
    ("WHITESPACE", r"\s+"),
]


TOKEN_REGEX = re.compile(
    "|".join(
        f"(?P<{name}>{pattern})"
        for name, pattern in TOKEN_SPEC
    )
)


def tokenize(expression: str) -> list[Token]:
    tokens = []
    position = 0

    while position < len(expression):

        match = TOKEN_REGEX.match(expression, position)

        if not match:
            raise ValueError(
                f"Unexpected character at position {position}: "
                f"{expression[position]!r}"
            )

        token_type = match.lastgroup
        value = match.group()

        if token_type != "WHITESPACE":
            tokens.append(
                Token(
                    type=token_type,
                    value=value,
                    position=position,
                )
            )

        position = match.end()

    return tokens