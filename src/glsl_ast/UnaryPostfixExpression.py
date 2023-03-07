from glsl_ast.UnaryPostfixOperator import UnaryPostfixOperator
from glsl_ast.Expression import Expression

class UnaryPostfixExpression(Expression):
    def __init__(self,
        appendix: UnaryPostfixOperator,
        operand: Expression,
    ) -> None:
        super().__init__()

        self.appendix = appendix
        self.operand = operand

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'Unary Postfix Expression @ {}{}'.format(
            ('\n' + self.operand.toString(depth + 1)) if 'toString' in dir(self.operand) else '{}\n'.format(str(self.operand)),
            self.appendix.toString(depth + 1),
        )
