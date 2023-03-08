from glsl_ast.VersionDirective import VersionDirective
from glsl_ast.ASTVisitor import ASTVisitor
from glsl_ast import grammar

from unittest import TestCase, main
from parsimonious import ParseError

class TestShaderFile(TestCase):
    EmptyShaderFile = """
#version 450
"""
    ShaderFileWithMain = """
#version 450 es

void main() {

}
"""
    NontrivialBodyShaderFile = """
#version 450

float x;
vec2 y = 5;

void main() {

}
"""

    ASTVisitorInstance = ASTVisitor()

    def test_EmptyShaderFile(self):
        shaderFile = TestShaderFile.ASTVisitorInstance.visit(grammar.parse(TestShaderFile.EmptyShaderFile))
        self.assertEqual(shaderFile.version, VersionDirective(450))
        self.assertListEqual(shaderFile.body, [])

    # TODO: Test two other cases.

if __name__ == '__main__':
    main()
