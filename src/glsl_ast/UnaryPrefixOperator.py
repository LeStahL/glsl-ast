from enum import Enum

class UnaryPrefixOperator(Enum):
    Increment = '++'
    Decrement = '--'
    Plus = '+'
    Minus = '-'
    BitWiseNot = '~'
    LogicalNot = '!'
