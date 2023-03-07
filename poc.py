from parsimonious.grammar import Grammar
from parsimonious import NodeVisitor
from enum import Enum
from json import dumps
from typing import Optional, Iterable, Any

class TypeQualifier(Enum):
    Const = 'const'
    In = 'in'
    Out = 'out'
    InOut = 'inout'
    Uniform = 'uniform'

class TypeSpecifier(Enum):
    Void = 'void'
    Float = 'float'
    Int = 'int'
    Ivec2 = 'ivec2'
    Ivec3 = 'ivec3'
    Ivec4 = 'ivec4'
    Uint = 'uint'
    Uvec2 = 'uvec2'
    Uvec3 = 'uvec3'
    Uvec4 = 'uvec4'
    Bool = 'bool'
    Vec2 = 'vec2'
    Vec3 = 'vec3'
    Vec4 = 'vec4'
    Bvec2 = 'bvec2'
    Bvec3 = 'bvec3'
    Bvec4 = 'bvec4'
    Mat2 = 'mat2'
    Mat3 = 'mat3'
    Mat4 = 'mat4'

class UnaryPrefixOperator(Enum):
    Increment = '++'
    Decrement = '--'
    Plus = '+'
    Minus = '-'
    BitWiseNot = '~'
    LogicalNot = '!'

class UnaryPostfixOperator(Enum):
    Increment = '++'
    Decrement = '--'

class SequentialOperator(Enum):
    Another = ','

class AssignmentOperator(Enum):
    Equals = '='
    PlusEquals = '+='
    MinusEquals = '-='
    StarEquals = '*='
    SlashEquals = '/='
    PercentEquals = '%='
    LeftShiftEquals = '<<='
    RightShiftEquals = '>>='
    AmpersandEquals = '&='
    CaretEquals = '^='
    PipeEquals = '|='

class TernaryOperator(Enum):
    QuestionMark = '?'
    Colon = ':'

class LogicalOperator(Enum):
    LogicalInclusiveOr = '||'
    LogicalExclusiveOr = '^^'
    LogicalAnd = '&&'
    Equality = '=='
    Inequality = '!='
    LessThanEqual = '<='
    LessThan = '<'
    MoreThanEqual = '>='
    MoreThan = '>'

class BitWiseOperator(Enum):
    InclusiveOr = '|'
    ExclusiveOr = '^'
    And = '&'
    LeftShift = '<<'
    RightShift = '>>'

class ArithmeticOperator(Enum):
    Plus = '+'
    Minus = '-'
    Star = '*'
    Slash = '/'
    Percent = '%'

class QualifiedName:
    def __init__(self,
        name: Optional[str],
        specifier: TypeSpecifier,
        qualifier: Optional[TypeQualifier] = None,
    ) -> None:
        self.name = name
        self.qualifier = qualifier
        self.specifier = specifier

    def toGLSL(self) -> str:
        return '{}{} {}'.format(
            (self.qualifier.value + ' ') if self.qualifier is not None else '',
            self.specifier.value,
            self.name,
        )
    
    def toString(self, depth: int) -> str:
        return ' ' * depth + 'QualifiedName @ {} ({}, {})\n'.format(
            self.name if self.name is not None else 'Anonymous',
            self.qualifier if self.qualifier is not None else 'Unqualified',
            self.specifier if self.specifier is not None else 'Unspecified',
        )

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

class FunctionDefinition:
    def __init__(self,
        name: QualifiedName,
        body: Block,
        arguments: Optional[Iterable[QualifiedName]],
    ) -> None:
        self.name = name
        self.arguments = arguments if arguments is not None else []
        self.body = body

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'FunctionDefinition @\n{}{}'.format(
            ''.join(map(
                lambda argument: argument.toString(depth + 1) if 'toString' in dir(argument) else str(argument),
                self.arguments,
            )),
            self.name.toString(depth + 1),
        ) + ' ' * (depth + 1) + 'body:\n{}'.format(
            self.body.toString(depth + 2) if 'toString' in dir(self.body) else str(self.body),
        )

class VersionDirective:
    def __init__(self,
        version: int,
        es: bool = False,
    ) -> None:
        self.version = version
        self.es = es

    def toGLSL(self) -> str:
        return '#version {}{}\n'.format(
            self.version,
            ' es' if self.es else '',
        )

    def toString(self,
        depth: int = 1,
    ) -> str:
        return ' ' * depth + 'VersionDirective: {}{}\n'.format(
            self.version,
            ' es' if self.es else '',
        )

class MacroDefinition:
    def __init__(self,
        name: str,
        argumentList: Optional[Iterable[str]] = None,
        body: Optional[str] = None,
    ) -> None:
        self.name = name
        self.body = body
        self.argumentList = argumentList

    def toGLSL(self) -> str:
        return "#define {}{}\n".format(
            self.name,
            (' ' + self.body) if self.body is not None else '',
        )

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'MacroDefinition: {}{}{}\n'.format(
            self.name,
            '({})'.format(','.join(self.argumentList)) if self.argumentList is not None else '',
            ' @ {}'.format(self.body[:50].encode('unicode_escape')) if self.body is not None else '',
        )

# Inherit for the specific expressions
class Expression:
    def __init__(self) -> None:
        pass

class Appendix:
    def __init__(self,
        operator: Any,
        rhs: Expression,             
    ) -> None:
        self.operator = operator
        self.rhs = rhs

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'Appendix @ {}{}'.format(
            self.operator,
            ('\n' + self.rhs.toString(depth + 1)) if 'toString' in dir(self.rhs) else ' {}\n'.format(str(self.rhs)),
        )

class UnaryExpression(Expression):
    def __init__(self,
        operator: Any,
        operand: Expression,
    ) -> None:
        super().__init__()

        self.operator = operator
        self.operand = operand

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'UnaryExpression @\n{}\n{}'.format(
            self.operator.toString(depth + 1),
            self.operand.toString(depth + 1),
        )

class BinaryExpression(Expression):
    def __init__(self,
        lhs: Expression,
        appendices: Iterable[Appendix],
    ) -> None:
        self.lhs = lhs
        self.appendices = appendices

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'BinaryExpression @{}{}'.format(
            ('\n' + self.lhs.toString(depth + 1)) if 'toString' in dir(self.lhs) else ' {}\n'.format(str(self.lhs)),
            ''.join(map(
                lambda appendix: appendix.toString(depth + 1) if 'toString' in dir(appendix) else str(appendix),
                self.appendices,
            )),
        )

# TODO: split in expression and appendix
# TODO: implement ternary expression
class TernaryExpression(Expression):
    def __init__(self,
        lhs: Expression,
        cv: Expression,
        rhs: Expression,           
    ) -> None:
        super().__init__()

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'TernaryExpression @\n{}\n{}\n{}\n'.format(
            self.lhs.toString(depth + 1),
            self.cv.toString(depth + 1),
            self.rhs.toString(depth + 1),
        )

class SwizzleAppendix(Appendix):
    def __init__(self,
        swizzle: str,
    ) -> None:
        super().__init__('swizzle', swizzle)

        self.swizzle = swizzle

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'Swizzle @ {}\n'.format(self.swizzle)

class FieldSelectorAppendix(Appendix):
    def __init__(self,
        fieldName: str,
    ) -> None:
        super().__init__('fieldSelector', fieldName)

        self.fieldName = fieldName

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'Field Selector @ {}\n'.format(self.fieldName)

class Statement:
    def __init__(self,
        expression: Expression,
    ) -> None:
        self.expression = expression

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'Statement @{}'.format(("\n" + self.expression.toString(depth + 1)) if 'toString' in dir(self.expression) else ' {}\n'.format(str(self.expression)))

class VariableDeclaration:
    def __init__(self,
        qualifier: TypeQualifier,
        specifier: TypeSpecifier,
        statement: Statement,
    ) -> None:
        self.qualifier = qualifier
        self.specifier = specifier
        self.statement = statement

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'Variable Declaration @ {} {}\n{}'.format(
            self.qualifier.value if self.qualifier is not None else '',
            self.specifier.value if self.specifier is not None else '',
            self.statement.toString(depth + 1),
        )

class LayoutExtension:
    def __init__(self,
        location: int,         
    ) -> None:
        self.location = location

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'LayoutExtension @ {}\n'.format(self.location)

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

class FunctionPrototype:
    def __init__(self,
        returnType: TypeSpecifier,
        name: str,
        arguments: Iterable[QualifiedName] = None,
    ) -> None:
        self.returnType = returnType
        self.name = name
        self.arguments = arguments if arguments != None else []

    def toGLSL(self) -> str:
        return '{} {}({})'.format(
            self.returnType.value,
            self.name,
            ','.join(map(
                lambda argument: argument.toGLSL(),
                self.arguments,
            )) if self.arguments != [] else '',
        )

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'FunctionPrototype: {} {}\n{}'.format(
            self.returnType.value,
            self.name,
            (' ' * (depth - 1)).join(map(
                lambda argument: argument.toString(depth + 1) if 'toString' in dir(argument) else '{}\n'.format(str(argument)),
                self.arguments,
            )) if self.arguments != [] else 'void',
        )

class ShaderFile:
    def __init__(self,
        version: VersionDirective,
        body: Iterable[Any] = None,
    ) -> None:
        self.version = version
        self.body = body if body is not None else []

    def toGLSL(self) -> str:
        return '{}{}'.format(
            self.version.toGLSL(),
            ''.join(map(
                lambda entry: entry.toGLSL(),
                self.body,
            )),
        )

    def toString(self,
        depth: int = 0,
    ) -> str:
        return ' ' * depth + "ShaderFile:\n" + self.version.toString(depth + 1) + ''.join(map(
            lambda child: child.toString(depth + 1) if 'toString' in dir(child) else str(child),
            self.body,
        ))

class ForConstruct:
    def __init__(self,
        initializer: Optional[Statement],
        bounds: Optional[Statement],
        updater: Optional[Expression],
        body: Optional[Block],
    ) -> None:
        self.initializer = initializer
        self.updater = updater
        self.bounds = bounds
        self.body = body

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'ForConstruct @\n{}{}{}{}'.format(
            self.initializer.toString(depth + 1),
            self.bounds.toString(depth + 1),
            self.updater.toString(depth + 1),
            self.body.toString(depth + 1) if self.body is not None else '',
        )

class IfCase:
    def __init__(self,
        condition: Optional[Expression],
        operation: Block,
    ) -> None:
        self.condition = condition
        self.operation = operation

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'If Case @{}{}'.format(
            ('\n' + self.condition.toString(depth + 1)) if 'toString' in dir(self.condition) else ' {}'.format(str(self.condition)),
            self.operation.toString(depth + 1) if 'toString' in dir(self.operation) else str(self.operation),
        )

class IfConstruct:
    def __init__(self,
        ifCase: IfCase,
        elseIfCases: Iterable[IfCase],
        elseCase: IfCase,
    ) -> None:
        self.ifCase = ifCase
        self.elseIfCases = elseIfCases
        self.elseCase = elseCase

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'If Construct @\n{}{}{}'.format(
            self.ifCase.toString(depth + 1),
            ''.join(map(
                lambda case: case.toString(depth + 1),
                self.elseIfCases,
            )),
            self.elseCase.toString(depth + 1) if self.elseCase is not None else '',
        )

class WhileConstruct:
    def __init__(self,
        condition: Expression,
        operation: Block,
    ) -> None:
        self.condition = condition
        self.operation = operation

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'While Construct @\n{}{}'.format(
            self.condition.toString(depth + 1),
            self.operation.toString(depth + 1),
        )

class UnaryPrefixExpression(Expression):
    def __init__(self,
        operator: UnaryPrefixOperator,
        operand: Expression,
    ) -> None:
        super().__init__()

        self.operator = operator
        self.operand = operand

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'Unary Prefix Expression @ {}{}\n'.format(
            self.operator.value,
            ('\n' + self.operand.toString(depth + 1)) if 'toString' in dir(self.operand) else str(self.operand),
        )
    
class UnaryPostfixExpression(Expression):
    def __init__(self,
        appendix: UnaryPostfixOperator,
        operand: Expression,
    ) -> None:
        super().__init__()

        self.appendix = appendix
        self.operand = operand

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'Unary Postfix Expression @ {}{}'.format(
            ('\n' + self.operand.toString(depth + 1)) if 'toString' in dir(self.operand) else '{}\n'.format(str(self.operand)),
            self.appendix.toString(depth + 1),
        )
    
class UnaryPostfixAppendix(Appendix):
    def __init__(self,
        operator: UnaryPostfixOperator,
    ) -> None:
        super().__init__(operator, None)

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'Unary Postfix Appendix @ {}\n'.format(
            self.operator.value,
        )

class ArraySubscriptAppendix(Appendix):
    def __init__(self,
        subscript: Expression,             
    ) -> None:
        super().__init__('arraySubscript', subscript)

    def toString(self, depth: int) -> str:
        return super().toString(depth)

class FunctionCallExpression(Expression):
    def __init__(self,
        name: str,
        arguments: Expression,             
    ) -> None:
        super().__init__()

        self.name = name
        self.arguments = arguments

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'Function Call @ {}{}'.format(
            self.name,
            ','.join(map(
                lambda argument: ('\n' + argument.toString(depth + 1)) if 'toString' in dir(argument) else ' {}'.format(str(argument)),
                self.arguments,
            )),
        )

class ParenthesisExpression(Expression):
    def __init__(self,
        child: Expression,
    ) -> None:
        super().__init__()

        self.child = child

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'Parenthesis @\n{}'.format(
            self.child.toString(depth + 1),
        )

class BuiltinFunctionId(Enum):
    radians = "radians"
    degrees = "degrees"
    sin = "sin"
    cos = "cos"
    tan = "tan"
    asin = "asin"
    acos = "acos"
    atan = "atan"
    sinh = "sinh"
    cosh = "cosh"
    tanh = "tanh"
    asinh = "asinh"
    acosh = "acosh"
    atanh = "atanh"
    pow = "pow"
    exp = "exp"
    log = "log"
    exp2 = "exp2"
    log2 = "log2"
    sqrt = "sqrt"
    inversesqrt = "inversesqrt"
    abs = "abs"
    sign = "sign"
    floor = "floor"
    trunc = "trunc"
    round = "round"
    roundEven = "roundEven"
    ceil = "ceil"
    fract = "fract"
    mod = "mod"
    min = "min"
    max = "max"
    modf = "modf"
    clamp = "clamp"
    mix = "mix"
    step = "step"
    smoothstep = "smoothstep"
    isnan = "isnan"
    isinf = "isinf"
    floatBitsToInt = "floatBitsToInt"
    intBitsToFloat = "intBitsToFloat"
    floatBitsToUint = "floatBitsToUint"
    uintBitsToFloat = "uintBitsToFloat"
    fma = "fma"
    frexp = "frexp"
    ldexp = "ldexp"
    packUnorm2x16 = "packUnorm2x16"
    packSnorm2x16 = "packSnorm2x16"
    packUnorm4x8 = "packUnorm4x8"
    packSnorm4x8 = "packSnorm4x8"
    unpackUnorm2x16 = "unpackUnorm2x16"
    unpackSnorm2x16 = "unpackSnorm2x16"
    unpackUnorm4x8 = "unpackUnorm4x8"
    unpackSnorm4x8 = "unpackSnorm4x8"
    packDouble2x32 = "packDouble2x32"
    unpackDouble2x32 = "unpackDouble2x32"
    packHalf2x16 = "packHalf2x16"
    unpackHalf2x16 = "unpackHalf2x16"
    length = "length"
    distance = "distance"
    dot = "dot"
    cross = "cross"
    normalize = "normalize"
    ftransform = "ftransform"
    faceforward = "faceforward"
    reflect = "reflect"
    refract = "refract"
    matrixCompMult = "matrixCompMult"
    outerProduct = "outerProduct"
    transpose = "transpose"
    determinant = "determinant"
    inverse = "inverse"
    lessThan = "lessThan"
    lessThanEqual = "lessThanEqual"
    greaterThan = "greaterThan"
    greaterThanEqual = "greaterThanEqual"
    equal = "equal"
    notEqual = "notEqual"
    any = "any"
    all = "all"
    _not = "not"
    uaddCarry = "uaddCarry"
    usubBorrow = "usubBorrow"
    umulExtended = "umulExtended"
    imulExtended = "imulExtended"
    bitfieldExtract = "bitfieldExtract"
    bitfieldInsert = "bitfieldInsert"
    bitfieldReverse = "bitfieldReverse"
    bitCount = "bitCount"
    findLSB = "findLSB"
    findMSB = "findMSB"
    textureSize = "textureSize"
    textureQueryLod = "textureQueryLod"
    textureQueryLevels = "textureQueryLevels"
    textureSamples = "textureSamples"
    texture = "texture"
    textureProj = "textureProj"
    textureLod = "textureLod"
    textureOffset = "textureOffset"
    texelFetch = "texelFetch"
    texelFetchOffset = "texelFetchOffset"
    textureProjOffset = "textureProjOffset"
    textureLodOffset = "textureLodOffset"
    textureProjLod = "textureProjLod"
    textureProjLodOffset = "textureProjLodOffset"
    textureGrad = "textureGrad"
    textureGradOffset = "textureGradOffset"
    textureProjGrad = "textureProjGrad"
    textureProjGradOffset = "textureProjGradOffset"
    textureGather = "textureGather"
    textureGatherOffset = "textureGatherOffset"
    textureGatherOffsets = "textureGatherOffsets"
    atomicCounterIncrement = "atomicCounterIncrement"
    atomicCounterDecrement = "atomicCounterDecrement"
    atomicCounter = "atomicCounter"
    atomicAdd = "atomicAdd"
    atomicMin = "atomicMin"
    atomicMax = "atomicMax"
    atomicAnd = "atomicAnd"
    atomicOr = "atomicOr"
    atomicXor = "atomicXor"
    atomicExchange = "atomicExchange"
    atomicCompSwap = "atomicCompSwap"
    imageSize = "imageSize"
    imageSamples = "imageSamples"
    imageLoad = "imageLoad"
    imageStore = "imageStore"
    dFdx = "dFdx"
    dFdy = "dFdy"
    dFdxFine = "dFdxFine"
    dFdyFine = "dFdyFine"
    dFdxCoarse = "dFdxCoarse"
    dFdyCoarse = "dFdyCoarse"
    fwidth = "fwidth"
    fwidthFine = "fwidthFine"
    fwidthCoarse = "fwidthCoarse"
    interpolateAtCentroid = "interpolateAtCentroid"
    interpolateAtSample = "interpolateAtSample"
    interpolateAtOffset = "interpolateAtOffset"
    noise1 = "noise1"
    noise2 = "noise2"
    noise3 = "noise3"
    noise4 = "noise4"

class BuiltinFunctionCallExpression(Expression):
    def __init__(self,
        name: BuiltinFunctionId,
        arguments: Expression,             
    ) -> None:
        super().__init__()

        self.name = name
        self.arguments = arguments

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'Builtin Function Call @ {}\n{}'.format(
            self.name.value,
            self.arguments.toString(depth + 1),
        )
    
class ReturnStatement:
    def __init__(self,
        expression: Expression,
    ) -> None:
        self.expression = expression

    def toString(self, depth: int) -> str:
        return ' ' * depth + 'Return Statement @{}'.format(
            ('\n' + self.expression.toString(depth + 1)) if 'toString' in dir(self.expression) else ' {}\n'.format(str(self.expression)),
        )

class BuiltinVariable(Enum):
    FragCoord = 'gl_FragCoord'

class CreateObjectVisitor(NodeVisitor):
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
        children = CreateObjectVisitor.Existing(visited_children)
        if node.expr_name in CreateObjectVisitor.StringLikeRules:
            return str(node.text)
        elif node.expr_name in CreateObjectVisitor.BinaryReductionBasedRules:
            return Appendix(children[0], children[1])
        elif node.expr_name in CreateObjectVisitor.BinaryRootBasedRules:
            if len(children) == 0:
                return None
            elif len(children) == 1:
                return children[0]
            if type(children[1]) is not list:
                children[1] = [children[1]]
            return BinaryExpression(children[0], children[1])
        elif node.expr_name in CreateObjectVisitor.LogicalOperatorBasedRules:
            return LogicalOperator(node.text)
        elif node.expr_name in CreateObjectVisitor.BitWiseOperatorBasedRules:
            return BitWiseOperator(node.text)
        elif node.expr_name in CreateObjectVisitor.JoinChildrenBasedRules:
            return ''.join(children)
        elif len(children) == 1:
            return children[0]
        elif len(children) == 0:
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
        return AssignmentOperator(CreateObjectVisitor.Existing(visited_children)[0])

    def visit_bitWiseShiftOperator(self, node, visited_children):
        return BitWiseOperator(CreateObjectVisitor.Existing(visited_children)[0])
    
    def visit_unaryPrefixOperator(self, node, visited_children):
        return UnaryPrefixOperator(CreateObjectVisitor.Existing(visited_children)[0])
    
    def visit_unaryPostfixOperator(self, node, visited_children):
        return UnaryPostfixOperator(CreateObjectVisitor.Existing(visited_children)[0])

    def visit_arraySubscriptAppendix(self, node, visited_children):
        return ArraySubscriptAppendix(CreateObjectVisitor.Existing(visited_children)[0])

    def visit_swizzleAppendix(self, node, visited_children):
        return SwizzleAppendix(CreateObjectVisitor.Existing(visited_children)[0])
    
    def visit_fieldSelectorAppendix(self, node, visited_children):
        return FieldSelectorAppendix(CreateObjectVisitor.Existing(visited_children)[0])

    def visit_unaryPrefixExpression(self, node, visited_children):
        children = CreateObjectVisitor.Existing(visited_children)
        
        return UnaryPrefixExpression(children[0], children[1])

    def visit_primaryAppendix(self, node, visited_children):
        children = CreateObjectVisitor.Existing(visited_children)
        if type(children[0]) is UnaryPostfixOperator:
            return UnaryPostfixAppendix(children[0])
        return children[0]

    def visit_primaryExpressionRoot(self, node, visited_children):
        children = CreateObjectVisitor.Existing(visited_children)
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
        children = CreateObjectVisitor.Existing(visited_children)
        
        return VersionDirective(children[0], len(children) == 2)

    def visit_statement(self, node, visited_children):
        children = CreateObjectVisitor.Existing(visited_children)

        if len(children) == 1:
            return Statement(children[0])
        return Statement(None)
    
    def visit_typeQualifier(self, node, visited_children):
        return TypeQualifier(node.text)
    
    def visit_typeSpecifier(self, node, visited_children):
        return TypeSpecifier(node.text)

    def visit_variableDeclaration(self, node, visited_children):
        children = CreateObjectVisitor.Existing(visited_children)

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
        children = CreateObjectVisitor.Existing(visited_children)

        layoutExtension = None
        variableDeclaration = None

        for child in children:
            if type(child) is LayoutExtension:
                layoutExtension = child
            else:
                variableDeclaration = child

        return UniformDeclaration(layoutExtension, variableDeclaration)

    def visit_layoutExtension(self, node, visited_children):
        return LayoutExtension(CreateObjectVisitor.Existing(visited_children)[0])

    def visit_block(self, node, visited_children):
        children = CreateObjectVisitor.Existing(visited_children)
        
        result = []
        for child in children:
            if type(child) is list:
                result.append(Block(child))
            else:
                result.append(child)

        return Block(result)

    def visit_forConstruct(self, node, visited_children):
        children = CreateObjectVisitor.Existing(visited_children)
        return ForConstruct(children[0], children[1], children[2], children[3] if len(children) == 4 else None)

    def visit_ifCase(self, node, visited_children):
        children = CreateObjectVisitor.Existing(visited_children)
        return IfCase(children[0], children[1])
    
    def visit_elseIfCase(self, node, visited_children):
        children = CreateObjectVisitor.Existing(visited_children)
        return IfCase(children[0], children[1])
    
    def visit_elseCase(self, node, visited_children):
        return IfCase(None, CreateObjectVisitor.Existing(visited_children)[0])

    def visit_ifConstruct(self, node, visited_children):
        children = CreateObjectVisitor.Existing(visited_children)
        if len(children) == 1:
            return IfConstruct(children[0], [], None)
        elif len(children) == 2:
            return IfConstruct(children[0], [], children[1])
        return IfConstruct(children[0], children[1] if type(children[1]) is list else [children[1]], children[2])

    def visit_whileConstruct(self, node, visited_children):
        children = CreateObjectVisitor.Existing(visited_children)
        return WhileConstruct(children[0], children[1])

    def visit_defineDirective(self, node, visited_children):
        children = CreateObjectVisitor.Existing(visited_children)
        
        arguments = None
        body = ""
        for child in children[1:]:
            if type(child) is list:
                arguments = child
            else:
                body = child
        
        return MacroDefinition(children[0], arguments, body)

    def visit_macroArgumentListAppendix(self, node, visited_children):
        return CreateObjectVisitor.Existing(visited_children)[1]

    def visit_macroArgument(self, node, visited_children):
        children = CreateObjectVisitor.Existing(visited_children)
        return children[0]

    # TODO: try and remove the code duplication below.
    def visit_macroArgumentList(self, node, visited_children):
        children = CreateObjectVisitor.Existing(visited_children)
        
        if len(children) == 0:
            return None
        elif len(children) == 1:
            return children[0]
        return [children[0]] + children[1]
    
    def visit_functionPrototypeParameterListAppendix(self, node, visited_children):
        children = CreateObjectVisitor.Existing(visited_children)
        
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
        children = CreateObjectVisitor.Existing(visited_children)

        if len(children) == 0:
            return None
        elif len(children) == 1:
            return children[0]
        return [children[0]] + children[1]

    def visit_functionPrototypeParameterListEntry(self, node, visited_children):
        children = CreateObjectVisitor.Existing(visited_children)
        
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
        children = CreateObjectVisitor.Existing(visited_children)
        return FunctionPrototype(children[0], children[1], children[2])
    
    def visit_functionParameterListAppendix(self, node, visited_children):
        children = CreateObjectVisitor.Existing(visited_children)
        
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
        children = CreateObjectVisitor.Existing(visited_children)

        if len(children) == 0:
            return None
        elif len(children) == 1:
            return children[0]
        return [children[0]] + children[1]

    def visit_functionParameterListEntry(self, node, visited_children):
        children = CreateObjectVisitor.Existing(visited_children)
        
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
        children = CreateObjectVisitor.Existing(visited_children)

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
        children = CreateObjectVisitor.Existing(visited_children)
        return ShaderFile(children[0], children[1])
    
    def visit_functionCallExpression(self, node, visited_children):
        children = CreateObjectVisitor.Existing(visited_children)
        return FunctionCallExpression(children[0], children[1] if type(children[1]) is list else [children[1]])

    def visit_returnStatement(self, node, visited_children):
        children = CreateObjectVisitor.Existing(visited_children)
        return ReturnStatement(children[0])

if __name__ == '__main__':
    source = None
    with open("poc.frag", "rt") as f:
        source = f.read()
        f.close()

    assert source != None

    grammarSource = None
    with open("grammar.peg", "rt") as f:
        grammarSource = f.read()
        f.close()

    assert grammarSource != None

    grammar = Grammar(grammarSource)
    ast = grammar.parse(source)

    uglificationVisitor = CreateObjectVisitor()
    uglifiedSource = uglificationVisitor.visit(ast)
    print(uglifiedSource.toString(0))
