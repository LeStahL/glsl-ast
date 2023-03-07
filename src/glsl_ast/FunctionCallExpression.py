from glsl_ast.Expression import Expression

class FunctionCallExpression(Expression):
    def __init__(self,
        name: str,
        arguments: Expression,             
    ) -> None:
        super().__init__()

        self.name = name
        self.arguments = arguments

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'Function Call @ {}{}'.format(
            self.name,
            ','.join(map(
                lambda argument: ('\n' + argument.toString(depth + 1)) if 'toString' in dir(argument) else ' {}'.format(str(argument)),
                self.arguments,
            )),
        )
