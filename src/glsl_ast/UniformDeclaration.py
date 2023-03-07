from glsl_ast.LayoutExtension import LayoutExtension
from glsl_ast.VariableDeclaration import VariableDeclaration

class UniformDeclaration:
    def __init__(self,
        layoutExtension: LayoutExtension,
        variableDeclaration: VariableDeclaration,
    ) -> None:
        self.layoutExtension = layoutExtension
        self.variableDeclaration = variableDeclaration

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'UniformDeclaration @\n{}{}'.format(
            self.layoutExtension.toString(depth + 1) if self.layoutExtension is not None else '',
            self.variableDeclaration.toString(depth + 1),
        )

    # def toGLSL(self) -> str:
    #     return "{}uniform {} {};".format(
    #         'layout(location={})'.format(self.location) if self.location is not None else '',
    #         self.type,
    #         ','.join(self.names),
    #     )

    # def toString(self, depth: int):
    #     return ' ' * depth + 'UniformDeclaration: {} {}\n'.format(
    #         self.type,
    #         ','.join(self.names),
    #     )
