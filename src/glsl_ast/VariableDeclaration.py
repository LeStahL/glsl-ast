from glsl_ast.TypeQualifier import TypeQualifier
from glsl_ast.TypeSpecifier import TypeSpecifier
from glsl_ast.Statement import Statement

from typing import Optional

class VariableDeclaration:
    def __init__(self,
        qualifier: Optional[TypeQualifier],
        specifier: TypeSpecifier,
        statement: Optional[Statement],
    ) -> None:
        self.qualifier = qualifier
        self.specifier = specifier
        self.statement = statement

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'Variable Declaration @ {} {}\n{}'.format(
            self.qualifier.value if self.qualifier is not None else '',
            self.specifier.value if self.specifier is not None else '',
            self.statement.toString(depth + 1),
        )
