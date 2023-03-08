from glsl_ast.VersionDirective import VersionDirective
from glsl_ast.ASTVisitor import ASTVisitor
from glsl_ast import grammar

from unittest import TestCase, main
from parsimonious import ParseError

class TestArraySubscriptAppendix(TestCase):
    PlainVersionDirective = """
#version 450

void main() {

}
"""
    ESVersionDirective = """
#version 450 es

void main() {

}
"""
    NoVersionDirective = """
void main() {

}    
"""

    ASTVisitorInstance = ASTVisitor()

    def test_PlainVersionDirective(self):
        shaderFile = TestArraySubscriptAppendix.ASTVisitorInstance.visit(grammar.parse(TestArraySubscriptAppendix.PlainVersionDirective))
        self.assertEqual(shaderFile.version, VersionDirective(450))
    
    def test_ESVersionDirective(self):
        shaderFile = TestArraySubscriptAppendix.ASTVisitorInstance.visit(grammar.parse(TestArraySubscriptAppendix.ESVersionDirective))
        self.assertEqual(shaderFile.version, VersionDirective(450, True))

    def test_NoVersionDirective(self):
        with self.assertRaises(ParseError):
            TestArraySubscriptAppendix.ASTVisitorInstance.visit(grammar.parse(TestArraySubscriptAppendix.NoVersionDirective))

if __name__ == '__main__':
    main()
