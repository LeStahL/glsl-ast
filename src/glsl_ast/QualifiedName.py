from typing import Optional
from glsl_ast.TypeSpecifier import TypeSpecifier
from glsl_ast.TypeQualifier import TypeQualifier

class QualifiedName:
    def __init__(self,
        name: Optional[str],
        specifier: TypeSpecifier,
        qualifier: Optional[TypeQualifier] = None,
    ) -> None:
        self.name = name
        self.qualifier = qualifier
        self.specifier = specifier

    def toGLSL(self) -> str:
        return '{}{} {}'.format(
            (self.qualifier.value + ' ') if self.qualifier is not None else '',
            self.specifier.value,
            self.name,
        )
    
    def toString(self, depth: int) -> str:
        return ' ' * depth + 'QualifiedName @ {} ({}, {})\n'.format(
            self.name if self.name is not None else 'Anonymous',
            self.qualifier if self.qualifier is not None else 'Unqualified',
            self.specifier if self.specifier is not None else 'Unspecified',
        )
