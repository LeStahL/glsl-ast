from glsl_ast.UnaryPrefixOperator import UnaryPrefixOperator
from glsl_ast.Expression import Expression
from glsl_ast.TypeSpecifier import TypeSpecifier

class UnaryPrefixExpression(Expression):
    def __init__(self,
        operator: UnaryPrefixOperator,
        operand: Expression,
    ) -> None:
        super().__init__()

        self.operator = operator
        self.operand = operand

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'Unary Prefix Expression @ {}{}\n'.format(
            self.operator.value,
            ('\n' + self.operand.toString(depth + 1)) if 'toString' in dir(self.operand) else str(self.operand),
        )

    def resultType(self) -> TypeSpecifier:
        return self.operand.resultType()
    
    def dimension(self) -> int:
        return self.operand.dimension()
