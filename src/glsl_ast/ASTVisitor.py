from parsimonious import NodeVisitor

from glsl_ast.Appendix import Appendix
from glsl_ast.BinaryExpression import BinaryExpression
from glsl_ast.LogicalOperator import LogicalOperator
from glsl_ast.BitWiseOperator import BitWiseOperator
from glsl_ast.BuiltinVariable import BuiltinVariable
from glsl_ast.BuiltinFunctionId import BuiltinFunctionId
from glsl_ast.SequentialOperator import SequentialOperator
from glsl_ast.AssignmentOperator import AssignmentOperator
from glsl_ast.UnaryPrefixOperator import UnaryPrefixOperator
from glsl_ast.UnaryPostfixOperator import UnaryPostfixOperator
from glsl_ast.ArraySubscriptAppendix import ArraySubscriptAppendix
from glsl_ast.SwizzleAppendix import SwizzleAppendix
from glsl_ast.FieldSelectorAppendix import FieldSelectorAppendix
from glsl_ast.UnaryPrefixExpression import UnaryPrefixExpression
from glsl_ast.UnaryPostfixAppendix import UnaryPostfixAppendix
from glsl_ast.UnaryPostfixExpression import UnaryPostfixExpression
from glsl_ast.VersionDirective import VersionDirective
from glsl_ast.Statement import Statement
from glsl_ast.TypeQualifier import TypeQualifier
from glsl_ast.TypeSpecifier import TypeSpecifier
from glsl_ast.VariableDeclaration import VariableDeclaration
from glsl_ast.LayoutExtension import LayoutExtension
from glsl_ast.UniformDeclaration import UniformDeclaration
from glsl_ast.Block import Block
from glsl_ast.ForConstruct import ForConstruct
from glsl_ast.IfCase import IfCase
from glsl_ast.IfConstruct import IfConstruct
from glsl_ast.WhileConstruct import WhileConstruct
from glsl_ast.MacroDefinition import MacroDefinition
from glsl_ast.QualifiedName import QualifiedName
from glsl_ast.FunctionPrototype import FunctionPrototype
from glsl_ast.FunctionDefinition import FunctionDefinition
from glsl_ast.ShaderFile import ShaderFile
from glsl_ast.FunctionCallExpression import FunctionCallExpression
from glsl_ast.ReturnStatement import ReturnStatement

from typing import Iterable

class ASTVisitor(NodeVisitor):
    def __init__(self) -> None:
        super().__init__()

    StringLikeRules = [
        'macroBody',
        'equals',
        'star',
        'slash',
        'percent',
        'moreThan',
        'lessThan',
        'bang',
        'pipe',
        'caret',
        'questionMark',
        'ampersand',
        'tilde',
        'minus',
        'plus',
        'identifier',
        'es',
        'comma',
        'void',
        'uniform',
        'const',
        'in',
        'out',
        'swizzle',
        'typeQualifier',
        'typeSpecifier',
    ]

    BinaryReductionBasedRules = [
        'multiplicativeAppendix',
        'additiveAppendix',
        'bitWiseShiftAppendix',
        'relationalAppendix',
        'equalityAppendix',
        'bitWiseAndAppendix',
        'bitWiseExclusiveOrAppendix',
        'bitWiseInclusiveOrAppendix',
        'logicalAndAppendix',
        'logicalExclusiveOrAppendix',
        'logicalInclusiveOrAppendix',
        'assignmentAppendix',
        'sequentialAppendix',
    ]

    BinaryRootBasedRules = [
        'multiplicativeExpressionRoot',
        'additiveExpressionRoot',
        'bitWiseShiftExpressionRoot',
        'relationalExpressionRoot',
        'equalityExpressionRoot',
        'bitWiseAndExpressionRoot',
        'bitWiseExclusiveOrExpressionRoot',
        'bitWiseInclusiveOrExpressionRoot',
        'logicalAndExpressionRoot',
        'logicalExclusiveOrExpressionRoot',
        'logicalInclusiveOrExpressionRoot',
        'assignmentExpressionRoot',
        'sequentialExpressionRoot',
        # 'ternarySelectionExpressionRoot',
    ]

    LogicalOperatorBasedRules = [
        'logicalInclusiveOrOperator',
        'logicalExclusiveOrOperator',
        'logicalAndOperator',
        'equalityOperator',
        'relationalOperator',
    ]

    BitWiseOperatorBasedRules = [
        'bitWiseInclusiveOrOperator',
        'bitWiseExclusiveOrOperator',
        'bitWiseAndOperator',
        'bitWiseShiftOperator',
    ]

    JoinChildrenBasedRules = [
        'plusPlus',
        'minusMinus',
        'equals',
        'plusEquals',
        'minusEquals',
        'starEquals',
        'slashEquals',
        'percentEquals',
        'lessThanLessThanEquals',
        'moreThanMoreThanEquals',
        'ampersandEquals',
        'caretEquals',
        'pipeEquals',
        'logicalExclusiveOrOperator',
        'logicalInclusiveOrOperator',
        'logicalAndOperator',
        'equalsEquals',
        'bangEquals',
        'lessThanEquals',
        'moreThanEquals',
        'lessThanLessThan',
        'moreThanMoreThan',
    ]

    @staticmethod
    def Existing(children: Iterable) -> list:
        return list(filter(
            lambda child: child is not None,
            children,
        ))

    def generic_visit(self, node, visited_children):
        children = ASTVisitor.Existing(visited_children)
        if node.expr_name in ASTVisitor.StringLikeRules:
            return str(node.text)
        if node.expr_name in ASTVisitor.BinaryReductionBasedRules:
            return Appendix(children[0], children[1])
        if node.expr_name in ASTVisitor.BinaryRootBasedRules:
            if len(children) == 0:
                return None
            elif len(children) == 1:
                return children[0]
            if type(children[1]) is not list:
                children[1] = [children[1]]
            return BinaryExpression(children[0], children[1])
        if node.expr_name in ASTVisitor.LogicalOperatorBasedRules:
            return LogicalOperator(node.text)
        if node.expr_name in ASTVisitor.BitWiseOperatorBasedRules:
            return BitWiseOperator(node.text)
        if node.expr_name in ASTVisitor.JoinChildrenBasedRules:
            assert False
            return ''.join(children)
        if len(children) == 1:
            return children[0]
        if len(children) == 0:
            return None
        return children
    
    def visit_intConstant(self, node, visited_children):
        return int(node.text)
    
    def visit_floatConstant(self, node, visited_children):
        return float(node.text)
    
    def visit_builtinVariableId(self, node, visited_children):
        return BuiltinVariable(node.text)
    
    def visit_builtinFunctionId(self, node, visited_children):
        return BuiltinFunctionId(node.text)
    
    def visit_sequentialOperator(self, node, visited_children):
        return SequentialOperator(node.text)
    
    def visit_assignmentOperator(self, node, visited_children):
        return AssignmentOperator(ASTVisitor.Existing(visited_children)[0])

    def visit_bitWiseShiftOperator(self, node, visited_children):
        return BitWiseOperator(ASTVisitor.Existing(visited_children)[0])
    
    def visit_unaryPrefixOperator(self, node, visited_children):
        return UnaryPrefixOperator(ASTVisitor.Existing(visited_children)[0])
    
    def visit_unaryPostfixOperator(self, node, visited_children):
        return UnaryPostfixOperator(ASTVisitor.Existing(visited_children)[0])

    def visit_arraySubscriptAppendix(self, node, visited_children):
        return ArraySubscriptAppendix(ASTVisitor.Existing(visited_children)[0])

    def visit_swizzleAppendix(self, node, visited_children):
        return SwizzleAppendix(ASTVisitor.Existing(visited_children)[0])
    
    def visit_fieldSelectorAppendix(self, node, visited_children):
        return FieldSelectorAppendix(ASTVisitor.Existing(visited_children)[0])

    def visit_unaryPrefixExpression(self, node, visited_children):
        children = ASTVisitor.Existing(visited_children)
        
        return UnaryPrefixExpression(children[0], children[1])

    def visit_primaryAppendix(self, node, visited_children):
        children = ASTVisitor.Existing(visited_children)
        if type(children[0]) is UnaryPostfixOperator:
            return UnaryPostfixAppendix(children[0])
        return children[0]

    def visit_primaryExpressionRoot(self, node, visited_children):
        children = ASTVisitor.Existing(visited_children)
        if len(children) == 1:
            return children[0]
        elif len(children) > 1:
            if type(children[1]) is list:
                first = children[0]
                for subchild in children[1]:
                    first = UnaryPostfixExpression(subchild, first)
                return first

        return BinaryExpression(children[0], [children[1]])
    
    def visit_versionDirective(self, node, visited_children):
        children = ASTVisitor.Existing(visited_children)
        
        return VersionDirective(children[0], len(children) == 2)

    def visit_statement(self, node, visited_children):
        children = ASTVisitor.Existing(visited_children)

        if len(children) == 1:
            return Statement(children[0])
        return Statement(None)
    
    def visit_typeQualifier(self, node, visited_children):
        return TypeQualifier(node.text)
    
    def visit_typeSpecifier(self, node, visited_children):
        return TypeSpecifier(node.text)

    def visit_variableDeclaration(self, node, visited_children):
        children = ASTVisitor.Existing(visited_children)

        qualifier = None
        specifier = None
        statement = None

        for child in children:
            if type(child) is TypeQualifier:
                qualifier = child
            elif type(child) is TypeSpecifier:
                specifier = child
            else:
                statement = child

        return VariableDeclaration(qualifier, specifier, statement)
    
    def visit_uniformDeclaration(self, node, visited_children):
        children = ASTVisitor.Existing(visited_children)

        layoutExtension = None
        variableDeclaration = None

        for child in children:
            if type(child) is LayoutExtension:
                layoutExtension = child
            else:
                variableDeclaration = child

        return UniformDeclaration(layoutExtension, variableDeclaration)

    def visit_layoutExtension(self, node, visited_children):
        return LayoutExtension(ASTVisitor.Existing(visited_children)[0])

    def visit_block(self, node, visited_children):
        children = ASTVisitor.Existing(visited_children)
        
        result = []
        for child in children:
            if type(child) is list:
                result.append(Block(child))
            else:
                result.append(child)

        return Block(result)

    def visit_forConstruct(self, node, visited_children):
        children = ASTVisitor.Existing(visited_children)
        return ForConstruct(children[0], children[1], children[2], children[3] if len(children) == 4 else None)

    def visit_ifCase(self, node, visited_children):
        children = ASTVisitor.Existing(visited_children)
        return IfCase(children[0], children[1])
    
    def visit_elseIfCase(self, node, visited_children):
        children = ASTVisitor.Existing(visited_children)
        return IfCase(children[0], children[1])
    
    def visit_elseCase(self, node, visited_children):
        return IfCase(None, ASTVisitor.Existing(visited_children)[0])

    def visit_ifConstruct(self, node, visited_children):
        children = ASTVisitor.Existing(visited_children)
        if len(children) == 1:
            return IfConstruct(children[0], [], None)
        elif len(children) == 2:
            return IfConstruct(children[0], [], children[1])
        return IfConstruct(children[0], children[1] if type(children[1]) is list else [children[1]], children[2])

    def visit_whileConstruct(self, node, visited_children):
        children = ASTVisitor.Existing(visited_children)
        return WhileConstruct(children[0], children[1])

    def visit_defineDirective(self, node, visited_children):
        children = ASTVisitor.Existing(visited_children)
        
        arguments = None
        body = ""
        for child in children[1:]:
            if type(child) is list:
                arguments = child
            else:
                body = child
        
        return MacroDefinition(children[0], arguments, body)

    def visit_macroArgumentListAppendix(self, node, visited_children):
        return ASTVisitor.Existing(visited_children)[1]

    def visit_macroArgument(self, node, visited_children):
        children = ASTVisitor.Existing(visited_children)
        return children[0]

    # TODO: try and remove the code duplication below.
    def visit_macroArgumentList(self, node, visited_children):
        children = ASTVisitor.Existing(visited_children)
        
        if len(children) == 0:
            return None
        elif len(children) == 1:
            return children[0]
        return [children[0]] + children[1]
    
    def visit_functionPrototypeParameterListAppendix(self, node, visited_children):
        children = ASTVisitor.Existing(visited_children)
        
        qualifier = None
        specifier = None
        identifier = ''

        for child in children:
            if type(child) is TypeQualifier:
                qualifier = child
            elif type(child) is TypeSpecifier:
                specifier = child
            elif type(child) is str:
                identifier = child
        
        return QualifiedName(identifier, specifier, qualifier)
    
    def visit_functionPrototypeParameterList(self, node, visited_children):
        children = ASTVisitor.Existing(visited_children)

        if len(children) == 0:
            return None
        elif len(children) == 1:
            return children[0]
        return [children[0]] + children[1]

    def visit_functionPrototypeParameterListEntry(self, node, visited_children):
        children = ASTVisitor.Existing(visited_children)
        
        qualifier = None
        specifier = None
        identifier = ''

        for child in children:
            if type(child) is TypeQualifier:
                qualifier = child
            elif type(child) is TypeSpecifier:
                specifier = child
            elif type(child) is str:
                identifier = child
        
        return QualifiedName(identifier, specifier, qualifier) 

    def visit_functionPrototype(self, node, visited_children):
        children = ASTVisitor.Existing(visited_children)
        return FunctionPrototype(children[0], children[1], children[2])
    
    def visit_functionParameterListAppendix(self, node, visited_children):
        children = ASTVisitor.Existing(visited_children)
        
        qualifier = None
        specifier = None
        identifier = ''

        for child in children:
            if type(child) is TypeQualifier:
                qualifier = child
            elif type(child) is TypeSpecifier:
                specifier = child
            elif type(child) is str:
                identifier = child
        
        return QualifiedName(identifier, specifier, qualifier)
    
    def visit_functionParameterList(self, node, visited_children):
        children = ASTVisitor.Existing(visited_children)

        if len(children) == 0:
            return None
        elif len(children) == 1:
            return children[0]
        return [children[0]] + children[1]

    def visit_functionParameterListEntry(self, node, visited_children):
        children = ASTVisitor.Existing(visited_children)
        
        qualifier = None
        specifier = None
        identifier = ''

        for child in children:
            if type(child) is TypeQualifier:
                qualifier = child
            elif type(child) is TypeSpecifier:
                specifier = child
            elif type(child) is str:
                identifier = child
        
        return QualifiedName(identifier, specifier, qualifier) 

    def visit_functionDefinition(self, node, visited_children):
        children = ASTVisitor.Existing(visited_children)

        specifier = None
        name = ""
        args = None
        body = None

        for child in children:
            if type(child) is str:
                name = child
            elif type(child) is list:
                args = child
            elif type(child) is TypeSpecifier:
                specifier = child
            else:
                body = child

        return FunctionDefinition(QualifiedName(name, specifier), body, args)
    
    def visit_validFile(self, node, visited_children):
        children = ASTVisitor.Existing(visited_children)
        return ShaderFile(children[0], children[1])
    
    def visit_functionCallExpression(self, node, visited_children):
        children = ASTVisitor.Existing(visited_children)
        return FunctionCallExpression(children[0], children[1] if type(children[1]) is list else [children[1]])

    def visit_returnStatement(self, node, visited_children):
        children = ASTVisitor.Existing(visited_children)
        return ReturnStatement(children[0])
