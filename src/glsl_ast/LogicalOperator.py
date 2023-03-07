from enum import Enum

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
