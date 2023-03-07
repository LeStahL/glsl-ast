from glsl_ast.UnaryPostfixOperator import UnaryPostfixOperator
from glsl_ast.Appendix import Appendix

class UnaryPostfixAppendix(Appendix):
    def __init__(self,
        operator: UnaryPostfixOperator,
    ) -> None:
        super().__init__(operator, None)

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'Unary Postfix Appendix @ {}\n'.format(
            self.operator.value,
        )
