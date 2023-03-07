from sys import path
from os.path import join, dirname
path.append(join(dirname(__file__), '../../src'))

from glsl_ast.VersionDirective import VersionDirective
from glsl_ast.ASTVisitor import ASTVisitor
from glsl_ast.VariableDeclaration import VariableDeclaration
from glsl_ast.TypeSpecifier import TypeSpecifier
from glsl_ast import grammar

from unittest import TestCase, main

class TestArraySubscriptAppendix(TestCase):
    GlobalDeclaration = """
#version 450

float data[4] = float[4](
    0,
    0,
    0,
    0
);
"""

    LocalDeclaration = """
#version 450

void main() {
    float data[4] = float[4](
        0,
        0,
        0,
        0
    );

    data[0] = 25.;
    data[1] = 50.;
    data[2] = 33.;
    data [3] = 21.;
}
"""

    ASTVisitorInstance = ASTVisitor()

    def test_GlobalDeclaration(self):
        shaderFile = TestArraySubscriptAppendix.ASTVisitorInstance.visit(grammar.parse(TestArraySubscriptAppendix.GlobalDeclaration))

        self.assertEqual(shaderFile.version, VersionDirective(450))
        
        self.assertEqual(len(shaderFile.body), 1)
        self.assertTrue(type(shaderFile.body[0]) is VariableDeclaration)
        
        self.assertIsNone(shaderFile.body[0].qualifier)
        self.assertIsNotNone(shaderFile.body[0].specifier)
        self.assertIsNotNone(shaderFile.body[0].statement)
        self.assertEqual(shaderFile.body[0].specifier, TypeSpecifier.Float)
    
    
    def test_LocalDeclaration(self):
        pass

if __name__ == '__main__':
    main()
