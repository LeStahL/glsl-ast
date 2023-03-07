from glsl_ast.TypeSpecifier import TypeSpecifier
from glsl_ast.QualifiedName import QualifiedName

from typing import Iterable

class FunctionPrototype:
    def __init__(self,
        returnType: TypeSpecifier,
        name: str,
        arguments: Iterable[QualifiedName] = None,
    ) -> None:
        self.returnType = returnType
        self.name = name
        self.arguments = arguments if arguments != None else []

    def toGLSL(self) -> str:
        return '{} {}({})'.format(
            self.returnType.value,
            self.name,
            ','.join(map(
                lambda argument: argument.toGLSL(),
                self.arguments,
            )) if self.arguments != [] else '',
        )

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'FunctionPrototype: {} {}\n{}'.format(
            self.returnType.value,
            self.name,
            (' ' * (depth - 1)).join(map(
                lambda argument: argument.toString(depth + 1) if 'toString' in dir(argument) else '{}\n'.format(str(argument)),
                self.arguments,
            )) if self.arguments != [] else 'void',
        )
