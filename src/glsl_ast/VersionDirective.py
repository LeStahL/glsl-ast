
class VersionDirective:
    def __init__(self,
        version: int,
        es: bool = False,
    ) -> None:
        self.version = version
        self.es = es

    def toGLSL(self) -> str:
        return '#version {}{}\n'.format(
            self.version,
            ' es' if self.es else '',
        )

    def toString(self,
        depth: int = 1,
    ) -> str:
        return ' ' * depth + 'VersionDirective: {}{}\n'.format(
            self.version,
            ' es' if self.es else '',
        )

    def __eq__(self, __o: object) -> bool:
        if type(__o) is not VersionDirective:
            return False
        
        return __o.es == self.es and __o.version == self.version
