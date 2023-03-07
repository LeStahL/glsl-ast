from glsl_ast.Expression import Expression
from glsl_ast.Appendix import Appendix

from typing import Iterable

class BinaryExpression(Expression):
    def __init__(self,
        lhs: Expression,
        appendices: Iterable[Appendix],
    ) -> None:
        self.lhs = lhs
        self.appendices = appendices

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'BinaryExpression @{}{}'.format(
            ('\n' + self.lhs.toString(depth + 1)) if 'toString' in dir(self.lhs) else ' {}\n'.format(str(self.lhs)),
            ''.join(map(
                lambda appendix: appendix.toString(depth + 1) if 'toString' in dir(appendix) else str(appendix),
                self.appendices,
            )),
        )
