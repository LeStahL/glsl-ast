from glsl_ast.Expression import Expression
from glsl_ast.Appendix import Appendix
from glsl_ast.TypeSpecifier import TypeSpecifier

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

    def resultType(self) -> TypeSpecifier:
        lhsDimension = self.lhs.dimension()
        result = self.lhs.resultType()

        for appendix in self.appendices:
            appendixDimension = appendix.rhs.dimension()
            if appendixDimension != lhsDimension:
                # This is only allowed if one of lhs or appendix dimension is one.
                if lhsDimension == 1:
                    result = appendix.rhs.resultType()
                elif appendixDimension == 1:
                    continue
                else:
                    raise ValueError("Dimension mismatch in binary operation. Don't know what to do.")
                
        return result
