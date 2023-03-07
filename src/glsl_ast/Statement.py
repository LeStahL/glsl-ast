from glsl_ast.Expression import Expression

class Statement:
    def __init__(self,
        expression: Expression,
    ) -> None:
        self.expression = expression

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'Statement @{}'.format(("\n" + self.expression.toString(depth + 1)) if 'toString' in dir(self.expression) else ' {}\n'.format(str(self.expression)))
