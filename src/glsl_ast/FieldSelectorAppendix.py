from glsl_ast.Appendix import Appendix

class FieldSelectorAppendix(Appendix):
    def __init__(self,
        fieldName: str,
    ) -> None:
        super().__init__('fieldSelector', fieldName)

        self.fieldName = fieldName

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'Field Selector @ {}\n'.format(self.fieldName)
