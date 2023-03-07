from glsl_ast.Expression import Expression
from glsl_ast.Block import Block

from typing import Optional

class IfCase:
    def __init__(self,
        condition: Optional[Expression],
        operation: Block,
    ) -> None:
        self.condition = condition
        self.operation = operation

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'If Case @{}{}'.format(
            ('\n' + self.condition.toString(depth + 1)) if 'toString' in dir(self.condition) else ' {}'.format(str(self.condition)),
            self.operation.toString(depth + 1) if 'toString' in dir(self.operation) else str(self.operation),
        )
