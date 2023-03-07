from glsl_ast.Expression import Expression

# TODO: split in expression and appendix
# TODO: implement ternary expression
class TernaryExpression(Expression):
    def __init__(self,
        lhs: Expression,
        cv: Expression,
        rhs: Expression,           
    ) -> None:
        super().__init__()

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'TernaryExpression @\n{}\n{}\n{}\n'.format(
            self.lhs.toString(depth + 1),
            self.cv.toString(depth + 1),
            self.rhs.toString(depth + 1),
        )
