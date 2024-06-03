import commentjson as cjson
import json
from symbol_library import SymType
from tree import Node, BatchedNode


def read_expressions(filepath):
    expressions = []
    with open(filepath, "r") as file:
        for line in file:
            expressions.append(line.strip().split(" "))
    return expressions


def read_expressions_json(filepath):
    with open(filepath, "r") as file:
        return [Node.from_dict(d) for d in json.load(file)]


def tokens_to_tree(tokens, symbols):
    """
    tokens : list of string tokens
    symbols: dictionary of possible tokens -> attributes, each token must have attributes: nargs (0-2), order
    """
    num_tokens = len([t for t in tokens if t != "(" and t != ")"])
    expr_str = ''.join(tokens)
    tokens = ["("] + tokens + [")"]
    operator_stack = []
    out_stack = []
    for token in tokens:
        if token == "(":
            operator_stack.append(token)
        elif token in symbols and symbols[token]["type"].value in [SymType.Var.value, SymType.Const.value, SymType.Literal.value]:
            out_stack.append(Node(token))
        elif token in symbols and symbols[token]["type"].value is SymType.Fun.value:
            if symbols[token]["precedence"] <= 0:
                out_stack.append(Node(token, left=out_stack.pop()))
            else:
                operator_stack.append(token)
        elif token in symbols and symbols[token]["type"].value is SymType.Operator.value:
            while len(operator_stack) > 0 and operator_stack[-1] != '(' \
                    and symbols[operator_stack[-1]]["precedence"] > symbols[token]["precedence"]:
                if symbols[operator_stack[-1]]["type"].value is SymType.Fun.value:
                    out_stack.append(Node(operator_stack.pop(), left=out_stack.pop()))
                else:
                    out_stack.append(Node(operator_stack.pop(), out_stack.pop(), out_stack.pop()))
            operator_stack.append(token)
        else:
            while len(operator_stack) > 0 and operator_stack[-1] != '(':
                if symbols[operator_stack[-1]]["type"].value is SymType.Fun.value:
                    out_stack.append(Node(operator_stack.pop(), left=out_stack.pop()))
                else:
                    out_stack.append(Node(operator_stack.pop(), out_stack.pop(), out_stack.pop()))
            operator_stack.pop()
            if len(operator_stack) > 0 and operator_stack[-1] in symbols \
                    and symbols[operator_stack[-1]]["type"].value is SymType.Fun.value:
                out_stack.append(Node(operator_stack.pop(), left=out_stack.pop()))
    if len(out_stack[-1]) == num_tokens:
        return out_stack[-1]
    else:
        raise Exception(f"Error while parsing expression {expr_str}.")


def load_config_file(path):
    with open(path, "r") as file:
        jo = cjson.load(file)
    return jo


def create_batch(trees):
    t = BatchedNode(trees=trees)
    t.create_target()
    return t


def expression_set_to_json(expressions, symbols, out_path):
    dicts = []
    so = {s["symbol"]: s for s in symbols}
    for expr in expressions:
        dicts.append(tokens_to_tree(expr, so).to_dict())

    with open(out_path, "w") as file:
        json.dump(dicts, file)
