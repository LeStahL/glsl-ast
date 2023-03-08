from typing import Any
from glsl_ast.Expression import Expression
from glsl_ast.TypeSpecifier import TypeSpecifier

class UnaryExpression(Expression):
    def __init__(self,
        operator: Any,
        operand: Expression,
    ) -> None:
        super().__init__()

        self.operator = operator
        self.operand = operand

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'UnaryExpression @\n{}\n{}'.format(
            self.operator.toString(depth + 1),
            self.operand.toString(depth + 1),
        )
