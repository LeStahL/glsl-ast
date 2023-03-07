from glsl_ast.Expression import Expression

from typing import Any

class Appendix:
    def __init__(self,
        operator: Any,
        rhs: Expression,             
    ) -> None:
        self.operator = operator
        self.rhs = rhs

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'Appendix @ {}{}'.format(
            self.operator,
            ('\n' + self.rhs.toString(depth + 1)) if 'toString' in dir(self.rhs) else ' {}\n'.format(str(self.rhs)),
        )
