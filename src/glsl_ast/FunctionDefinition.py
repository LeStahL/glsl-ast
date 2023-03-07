from glsl_ast.QualifiedName import QualifiedName
from glsl_ast.Block import Block

from typing import Optional, Iterable

class FunctionDefinition:
    def __init__(self,
        name: QualifiedName,
        body: Block,
        arguments: Optional[Iterable[QualifiedName]],
    ) -> None:
        self.name = name
        self.arguments = arguments if arguments is not None else []
        self.body = body

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'FunctionDefinition @\n{}{}'.format(
            ''.join(map(
                lambda argument: argument.toString(depth + 1) if 'toString' in dir(argument) else str(argument),
                self.arguments,
            )),
            self.name.toString(depth + 1),
        ) + ' ' * (depth + 1) + 'body:\n{}'.format(
            self.body.toString(depth + 2) if 'toString' in dir(self.body) else str(self.body),
        )
