from glsl_ast.Statement import Statement
from glsl_ast.Expression import Expression
from glsl_ast.Block import Block

from typing import Optional

class ForConstruct:
    def __init__(self,
        initializer: Optional[Statement],
        bounds: Optional[Statement],
        updater: Optional[Expression],
        body: Optional[Block],
    ) -> None:
        self.initializer = initializer
        self.updater = updater
        self.bounds = bounds
        self.body = body

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'ForConstruct @\n{}{}{}{}'.format(
            self.initializer.toString(depth + 1),
            self.bounds.toString(depth + 1),
            self.updater.toString(depth + 1),
            self.body.toString(depth + 1) if self.body is not None else '',
        )
