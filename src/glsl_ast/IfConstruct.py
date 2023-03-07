from glsl_ast.IfCase import IfCase

from typing import Iterable

class IfConstruct:
    def __init__(self,
        ifCase: IfCase,
        elseIfCases: Iterable[IfCase],
        elseCase: IfCase,
    ) -> None:
        self.ifCase = ifCase
        self.elseIfCases = elseIfCases
        self.elseCase = elseCase

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'If Construct @\n{}{}{}'.format(
            self.ifCase.toString(depth + 1),
            ''.join(map(
                lambda case: case.toString(depth + 1),
                self.elseIfCases,
            )),
            self.elseCase.toString(depth + 1) if self.elseCase is not None else '',
        )
