from glsl_ast.VersionDirective import VersionDirective

from typing import Iterable, Any

class ShaderFile:
    def __init__(self,
        version: VersionDirective,
        body: Iterable[Any] = None,
    ) -> None:
        self.version = version
        self.body = body if body is not None else []
        if type(self.body) is not Iterable:
            self.body = [self.body]

    def toGLSL(self) -> str:
        return '{}{}'.format(
            self.version.toGLSL(),
            ''.join(map(
                lambda entry: entry.toGLSL(),
                self.body,
            )),
        )

    def toString(self,
        depth: int = 0,
    ) -> str:
        return ' ' * depth + "ShaderFile:\n" + self.version.toString(depth + 1) + ''.join(map(
            lambda child: child.toString(depth + 1) if 'toString' in dir(child) else str(child),
            self.body,
        ))
