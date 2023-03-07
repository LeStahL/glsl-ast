from typing import Optional, Iterable

class MacroDefinition:
    def __init__(self,
        name: str,
        argumentList: Optional[Iterable[str]] = None,
        body: Optional[str] = None,
    ) -> None:
        self.name = name
        self.body = body
        self.argumentList = argumentList

    def toGLSL(self) -> str:
        return "#define {}{}\n".format(
            self.name,
            (' ' + self.body) if self.body is not None else '',
        )

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'MacroDefinition: {}{}{}\n'.format(
            self.name,
            '({})'.format(','.join(self.argumentList)) if self.argumentList is not None else '',
            ' @ {}'.format(self.body[:50].encode('unicode_escape')) if self.body is not None else '',
        )
