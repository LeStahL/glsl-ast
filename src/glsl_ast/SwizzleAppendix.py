from glsl_ast.Appendix import Appendix

class SwizzleAppendix(Appendix):
    def __init__(self,
        swizzle: str,
    ) -> None:
        super().__init__('swizzle', swizzle)

        self.swizzle = swizzle

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'Swizzle @ {}\n'.format(self.swizzle)
