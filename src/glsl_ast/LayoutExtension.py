
class LayoutExtension:
    def __init__(self,
        location: int,         
    ) -> None:
        self.location = location

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'LayoutExtension @ {}\n'.format(self.location)
