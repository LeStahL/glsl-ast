from typing import Iterable, Any

class Block:
    def __init__(self,
        entries: Iterable[Any] = None, # These can be Block or Statement
    ) -> None:
        self.entries = entries if entries is not None else []

    def toGLSL(self) -> str:
        return '{{{}}}'.format(

        )
    
    def toString(self, depth: int) -> str:
        return ' ' * depth + 'Block @\n' + ''.join(map(
            lambda entry: entry.toString(depth + 1) if 'toString' in dir(entry) else str(entry),
            self.entries,
        ))
