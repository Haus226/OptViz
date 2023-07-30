import os
import re
from Var import Var, FUNC, CONSTANT, MATH_FUNC
from VarMath import *

# For visualization, make sure graphviz is installed and added into environment path
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'


# def tokenize_(expression, variables):
#     pattern = r"\s*(\d*\.*\d+)|(\w+)|([-+*/^])|([(),])"
#     tokens = re.findall(pattern, expression)
#     token = []
#     vars = variables
#     for t in tokens:
#         if t[0]:
#             token.append(Var(val=float(t[0]), type="number", info=t[0], exp=t[0]))
#         elif t[1]:
#             if len(t[1]) > 1:
#                 token.append(Var(type="func", info=t[1], exp=t[1]))
#             else:
#                 token.append(vars[t[1]])
#         elif t[2]:
#             token.append(Var(type="op", info=t[2]))
#         elif t[3]:
#             token.append(Var(type="other", info=t[3]))
#     return token, vars

# def ShuntingYard_(tokens:list[Var]):
#     queue = []
#     stack = []
#     for token in tokens:
#         print("Current Token : ", token.info)

#         for t in stack:
#             print(t.info, end=" ")
#         print()
#         if token.type == "number" or token.type == "var":
#             queue.append(token)
#         elif token.type == "func":
#             stack.append(token)
#         elif token.type == "op":
#             while (len(stack) and stack[-1].type == "op" \
#                    and (PRECEDENCE[token.info] <=PRECEDENCE[stack[-1].info])):
#                 queue.append(stack.pop())
#             stack.append(token)
#         elif token.info == ",":
#             while (len(stack) and stack[-1].info != "("):
#                 queue.append(stack.pop())
#         elif token.info == "(":
#             stack.append(token)
#         elif token.info == ")":
#             while (len(stack) and stack[-1].info != "("):
#                 queue.append(stack.pop())
#             if len(stack):
#                 if stack[-1].info == "(":
#                     stack.pop()
#                 if stack[-1].type == "func":
#                     queue.append(stack.pop())
#     while len(stack):
#         if stack[-1].info == "(":
#             raise ValueError("Mismatched Parenthesis")  
#         queue.append(stack.pop()) 
#     # print("Output Queue : ")  
#     # for t in queue:
#     #     print(t.info)
#     return queue

# def postfix_evaluation_(tokens):
#     def apply_binop(op:Var, operand_1:Var, operand_2:Var):
#         if op.info == "+":
#             return operand_1 + operand_2
#         elif op.info =="-":
#             return operand_1 - operand_2
#         elif op.info == '*':
#             return operand_1 * operand_2
#         elif op.info == '/':
#             return operand_1 / operand_2
#         elif op.info == '^':
#             return operand_1 ** operand_2
#     stack = []
#     for token in tokens:
#         # print("Stack : ")
#         # for s in stack:
#         #     print(s.info)
#         if token.type == "number" or token.type == "var":
#             stack.append(token)
#         elif token.type in ("op", "func"):
#             if token.info in FUNC_:
#                 # print(token.info)
#                 operand = stack.pop()
#                 result = FUNC_[token.info](operand)
#             else:
#                 operand_2 = stack.pop()
#                 operand_1 = stack.pop()
#                 result = apply_binop(token, operand_1, operand_2)
#             stack.append(result)
#         else:
#             raise ValueError("Invalid token in the expression")
#     return stack[-1]

def tokenize(expression: str):
    pattern = r"\s*(-*\d*\.*\d+)|(\w+)|([-+*/^])|([(),])"
    tokens = re.findall(pattern, expression)
    return [t[0] or t[1] or t[2] or t[3] for t in tokens]


def ShuntingYard(expression):
    tokens = tokenize(expression)
    queue = []
    stack = []
    for token in tokens:
        # print("Current Token : ", token)
        # print(stack)
        if re.match(r'-*\d*\.*\d+', token) or re.match(CONSTANT, token.lower()):
            queue.append(token)
        elif re.match(FUNC, token):
            stack.append(token)
        elif re.match(r'[a-zA-Z]', token):
            queue.append(token)
        elif re.match(r'[-+*/^]', token):
            while len(stack) and re.match(r'[-+*/^]', stack[-1]) and (PRECEDENCE[token] <= PRECEDENCE[stack[-1]]):
                queue.append(stack.pop())
            stack.append(token)
        elif token == ",":
            while len(stack) and stack[-1] != "(":
                queue.append(stack.pop())
        elif token == "(":
            stack.append(token)
        elif token == ")":
            while len(stack) and stack[-1] != "(":
                queue.append(stack.pop())
            if len(stack) and stack[-1] == "(":
                stack.pop()
            if len(stack) and re.match(FUNC, stack[-1]):
                queue.append(stack.pop())
    while len(stack):
        if stack[-1] == "(":
            raise ValueError("Mismatched Parenthesis")
        queue.append(stack.pop())  # print("Output Queue : ")
    # for t in queue:
    #     print(t.info)
    return queue


def evaluateVar(tokens: list, vars):
    def apply_binop(op, operand_1: Var, operand_2: Var):
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

    stack = []
    for token in tokens:
        if re.match(r'-*\d*\.*\d+', token):
            stack.append(Var(float(token), type="number", info=token, exp=token))
        elif re.match(CONSTANT, token.lower()):
            stack.append(Var(CONSTANT_[token.lower()], type="constant", info=token.lower(), exp=token.lower()))
        elif re.match(FUNC, token):
            operand = stack.pop()
            result = FUNC_[token](operand)
            stack.append(result)
        elif re.match(r'[a-zA-Z]', token):
            stack.append(vars[token])
        elif re.match(r'[-+*/^]', token):
            operand_2 = stack.pop()
            operand_1 = stack.pop()
            result = apply_binop(token, operand_1, operand_2)
            stack.append(result)
        else:
            raise ValueError("Invalid token in the expression")
    return stack[-1]

def evaluateFunc(tokens: list, vars):
    def apply_binop(op, operand_1, operand_2):
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

    stack = []
    for token in tokens:
        if re.match(r'-*\d*\.*\d+', token):
            stack.append(float(token))
        elif re.match(CONSTANT, token.lower()):
            stack.append(CONSTANT_[token.lower()])
        elif re.match(FUNC, token):
            operand = stack.pop()
            result = MATH_FUNC[token][0](operand)
            stack.append(result)
        elif re.match(r'[a-zA-Z]', token):
            stack.append(vars[token])
        elif re.match(r'[-+*/^]', token):
            operand_2 = stack.pop()
            operand_1 = stack.pop()
            result = apply_binop(token, operand_1, operand_2)
            stack.append(result)
        else:
            raise ValueError("Invalid token in the expression")
    return stack[-1]

# Return two functions one for AD while one for normal math evaluation of the expression given
def expression_to_function(expression):
    tokens = ShuntingYard(expression)

    def func(**kwargs):
        variables = kwargs
        for k in kwargs:
            variables[k] = Var(kwargs[k], exp=k, type="var", info=k)
        return evaluateVar(tokens, variables), variables

    def f(**kwargs):
        variables = kwargs
        return evaluateFunc(tokens, variables)

    return func, f


if __name__ == "__main__":
    d = {
        "x": Var(.12, type="var", exp="x", info="x"),
        "y": Var(.13, type="var", exp="y", info="y"),
        "z": Var(2.901, type="var", exp="y", info="z"),
    }
    # e = "sin ( cos(x + 5) / 3 * 3.142 )"
    # e = "sin (x * z ^ 2) / cos(y) + cos(y ^ exp(sin(z))) + 1.09132 * x"
    e = "(A+B) * (C+D)"
    token = tokenize(e)
    print(token)
    pos = ShuntingYard(token)
    print(
        pos)  # f = expression_to_function("sin (x * z^2) / cos(y) + cos(y ^ exp(sin(2.901))) + 1.09132 * x")  #  # _, vars = f(x=0.12, y=0.13, z=2.901)  # _.backward()  # draw_dag(_, "aaa")  # print(_)  # print(vars)

    # for v in vars:  #     # print(v)  #     print(vars[v])  #     print(f"df/d{vars[v].info} : ", vars[v].grad)
