import re
import numpy as np

def binop(op, operand_1, operand_2):
    if op == "+":
        return operand_1 + operand_2
    elif op == "-":
        return operand_1 - operand_2
    elif op == '*':
        return operand_1 * operand_2
    elif op == '/':
        return operand_1 / operand_2
    elif op == '^':
        return operand_1 ** operand_2
    
def tokenize(expression: str):
    pattern = r"\s*(-*\d*\.*\d+)|(\w+)|([-+*/^])|([(),])"
    tokens = re.findall(pattern, expression)
    return [t[0] or t[1] or t[2] or t[3] for t in tokens]