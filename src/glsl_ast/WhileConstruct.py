from glsl_ast.Expression import Expression
from glsl_ast.Block import Block

class WhileConstruct:
    def __init__(self,
        condition: Expression,
        operation: Block,
    ) -> None:
        self.condition = condition
        self.operation = operation

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'While Construct @\n{}{}'.format(
            self.condition.toString(depth + 1),
            self.operation.toString(depth + 1),
        )
