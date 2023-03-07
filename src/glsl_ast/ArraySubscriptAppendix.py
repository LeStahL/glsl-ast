from glsl_ast.Expression import Expression
from glsl_ast.Appendix import Appendix

class ArraySubscriptAppendix(Appendix):
    def __init__(self,
        subscript: Expression,             
    ) -> None:
        super().__init__('arraySubscript', subscript)

    def toString(self, depth: int) -> str:
        return super().toString(depth)
