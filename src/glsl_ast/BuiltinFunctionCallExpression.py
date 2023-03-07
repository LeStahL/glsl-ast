from glsl_ast.BuiltinFunctionId import BuiltinFunctionId
from glsl_ast.Expression import Expression

class BuiltinFunctionCallExpression(Expression):
    def __init__(self,
        name: BuiltinFunctionId,
        arguments: Expression,             
    ) -> None:
        super().__init__()

        self.name = name
        self.arguments = arguments

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'Builtin Function Call @ {}\n{}'.format(
            self.name.value,
            self.arguments.toString(depth + 1),
        )
