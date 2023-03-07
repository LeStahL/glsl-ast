from glsl_ast.Expression import Expression

class ParenthesisExpression(Expression):
    def __init__(self,
        child: Expression,
    ) -> None:
        super().__init__()

        self.child = child

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'Parenthesis @\n{}'.format(
            self.child.toString(depth + 1),
        )
