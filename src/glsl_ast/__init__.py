from os.path import dirname, join
from parsimonious.grammar import Grammar

grammarSource = None
with open(join(dirname(__file__), "grammar.peg"), "rt") as f:
    grammarSource = f.read()
    f.close()
assert grammarSource is not None

grammar = Grammar(grammarSource)
assert grammar is not None
