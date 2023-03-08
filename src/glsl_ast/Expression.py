from glsl_ast.TypeSpecifier import TypeSpecifier

# Inherit for the specific expressions
class Expression:
    def __init__(self) -> None:
        pass

    def resultType(self) -> TypeSpecifier:
        raise NotImplementedError("Expression.resultType was called.")

    def dimension(self) -> int:
        raise NotImplementedError("Expression.dimension was called.")
