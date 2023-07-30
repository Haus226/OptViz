from Var import Var, FUNC, CONSTANT, MATH_FUNC, draw_dag
from VarMath import *
import re, os
from graphviz import Digraph

# For visualization, make sure graphviz is installed and added into environment path
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'



def tokenize(expression):
    pattern = r"\s*(-*\d*\.*\d+)|(\w+)|([-+*/^()])"
    tokens = re.findall(pattern, expression)
    return [t[0] or t[1] or t[2] for t in tokens]

def parse(exp):
    current_token_index = 0
    tokens = tokenize(exp)
    # print(tokens)

    def expression(min_precedence=0):
        left_side = term()
        # Lead to the problem of precedence:
        # if current_token_index < len(tokens) and tokens[current_token_index] in "+-":
        operator = ""
        while (
            current_token_index < len(tokens)
            and tokens[current_token_index] in PRECEDENCE
            and PRECEDENCE[tokens[current_token_index]] >= min_precedence
        ):
            operator = tokens[current_token_index]
            advance()
            left_side = {"type": "binop", "op": operator, "left": left_side, "right": expression(PRECEDENCE[operator] + 1)}
        return left_side
    
    def term():
        left_side = factor()
        # if current_token_index < len(tokens) and tokens[current_token_index] in '*/^':
        while (
            current_token_index < len(tokens)
            and tokens[current_token_index] in PRECEDENCE
            and PRECEDENCE[tokens[current_token_index]] >= 3
        ):
            operator = tokens[current_token_index]
            advance()
            left_side = {'type': 'binop', 'op': operator, 'left': left_side, 'right': term()}
        return left_side

    def factor():
        if current_token_index < len(tokens):
            if tokens[current_token_index] == '(':
                advance()
                expr_val = expression()
                if tokens[current_token_index] == ')':
                    advance()
                    return expr_val
            elif re.match(FUNC, tokens[current_token_index]):
                function_exp = tokens[current_token_index]
                advance()
                if tokens[current_token_index] == '(':
                    advance()
                    args = []
                    while tokens[current_token_index] != ')':
                        args.append(expression())
                        if tokens[current_token_index] == ',':
                            advance()
                        else:
                            break
                    if tokens[current_token_index] == ')':
                        advance()
                        return {'type': 'function_call', 'func': function_exp, 'args': args}
            return parse_number_or_variable()

    def parse_number_or_variable():
        # print(tokens[current_token_index])
        if current_token_index < len(tokens):
            if re.match(r'-*\d*\.*\d+', tokens[current_token_index]):
                num = float(tokens[current_token_index])
                advance()
                return {'type': 'number', 'value': num}
            elif re.match(CONSTANT, tokens[current_token_index].lower()):
                advance()
                return {"type": "constant", "value": CONSTANT_[tokens[current_token_index - 1].lower()], "cons_name":tokens[current_token_index - 1].lower()}
            elif re.match(r'\w+', tokens[current_token_index]):
                var_exp = tokens[current_token_index]
                advance()
                return {'type': 'variable', 'var': var_exp}
        raise ValueError("Invalid expression")

    def advance():
        nonlocal current_token_index
        current_token_index += 1

    return expression()

def evaluateVar(ast, variables):
    if ast['type'] == 'number':
        return Var(ast['value'], exp=f"{ast['value']}", type="number")
    elif ast['type'] == 'constant':
        return Var(ast['value'], exp=f"{ast['cons_name'].lower()}", type="constant", info=ast["cons_name"])
    elif ast['type'] == 'variable':
        return variables[ast["var"]]
    elif ast['type'] == 'binop':
        left_val = evaluateVar(ast['left'], variables)
        right_val = evaluateVar(ast['right'], variables)
        if ast['op'] == '+':
            return left_val + right_val
        elif ast['op'] == '-':
            return left_val - right_val
        elif ast['op'] == '*':
            return left_val * right_val
        elif ast['op'] == '/':
            return left_val / right_val
        elif ast['op'] == '^':
            # if ast["right"]["type"] == "variable" or ast["right"]["type"] == "function_call":
                # return left_val @ right_val
            # print("Left : ", left_val)
            # print("Right : ", right_val)
            return left_val ** right_val
    elif ast['type'] == 'function_call':
        function_exp = ast['func']
        args = [evaluateVar(arg, variables) for arg in ast['args']]
        return FUNC_[function_exp](args[0])
    raise ValueError("Invalid AST node")

def evaluateFunc(ast, variables):
    if ast['type'] == 'number':
        return float(ast['value'])
    elif ast['type'] == 'variable':
        return variables[ast["var"]]
    elif ast['type'] == 'binop':
        left_val = evaluateFunc(ast['left'], variables)
        right_val = evaluateFunc(ast['right'], variables)
        if ast['op'] == '+':
            return left_val + right_val
        elif ast['op'] == '-':
            return left_val - right_val
        elif ast['op'] == '*':
            return left_val * right_val
        elif ast['op'] == '/':
            return left_val / right_val
        elif ast['op'] == '^':
            # if ast["right"]["type"] == "variable" or ast["right"]["type"] == "function_call":
                # return left_val @ right_val
            # print("Left : ", left_val)
            # print("Right : ", right_val)

            return left_val ** right_val
    elif ast['type'] == 'function_call':
        function_exp = ast['func']
        args = [evaluateFunc(arg, variables) for arg in ast['args']]
        # print(args)
        return MATH_FUNC[function_exp][0](args[0])
    raise ValueError("Invalid AST node")

def to_latex(ast):
    if ast["type"] == "binop":
        if ast["op"] == "+":
            return f"{to_latex(ast['left'])} + {to_latex(ast['right'])}"
        elif ast["op"] == "-":
            return f"{to_latex(ast['left'])} - {to_latex(ast['right'])}"
        elif ast["op"] == "*":
            return f"{to_latex(ast['left'])} * {to_latex(ast['right'])}"
        elif ast["op"] == "/":
            return f"\\frac{{{to_latex(ast['left'])}}}{{{to_latex(ast['right'])}}}"
        elif ast["op"] == "^":
            return f"{to_latex(ast['left'])} ^ {{{to_latex(ast['right'])}}}"
    elif ast["type"] == 'number':
        return str(ast['value'])
    elif ast["type"] == 'variable':
        return ast["var"]
    elif ast["type"] == 'function_call':
        args = [to_latex(arg) for arg in ast['args']]
        return f"\{ast['func']}({args[0]})"

def draw_ast(ast, filename="ast"):
    dot = Digraph(comment="Abstract Syntax Tree", format="png")

    def traverse(node):
        if node['type'] == "number":
            dot.node(str(id(node)), label=str(node['type']) + "\n" + str(node["value"]))
        elif node["type"] == "constant":
            dot.node(str(id(node)), label=str(node['type']) + "\n" + str(node["cons_name"]))

        elif node['type'] == "variable":
            dot.node(str(id(node)), label=str(node['type']) + "\n" + str(node["var"]))
        elif node['type'] == "function_call":
            dot.node(str(id(node)), label=str(node['type']) + "\n" + str(node["func"]))
            for arg in node["args"]:
                traverse(arg)
                dot.edge(str(id(node)), str(id(arg)))
        elif node['type'] == 'binop':
            dot.node(str(id(node)), label=str(node['type']) + '|' + str(node['op']), shape="record")
            traverse(node['left'])
            traverse(node['right'])
            dot.edge(str(id(node)), str(id(node["left"])))
            dot.edge(str(id(node)), str(id(node["right"])))

    traverse(ast)
    dot.render(filename)

# Return two functions one for AD while one for normal math evaluation of the expression given
def expression_to_function(expr_str):
    ast = parse(expr_str)
    def func(**kwargs):
        variables = kwargs
        for k in kwargs:
            variables[k] = Var(kwargs[k], exp=k, type="var", info=k)
        return evaluateVar(ast, variables), variables

    def f(**kwargs):
        variables = kwargs
        return evaluateFunc(ast, variables)

    return func, f

if __name__ == "__main__":
    def test_sanity_check():
        x = Var(.12)
        y = Var(.13)
        t = Var(.14)

        z = sin(x ** 2) + log(y @ sin(x)) + exp(t)

        z.backward()
        # draw_dag(z)
        print(y)
        print(x)
        print(t)
        print(z)

    # test_sanity_check()
    # test_sanity_check()
    # f = expression_to_function("tan(sin(x * sech(z ^ sinh(2 + csch(z * cosech(y))))) / cos(y * coth(z)) + cos(y ^ exp(sin(z + 0.156))) + 1.09132 * arcsin(x * arcot(z / coth(x))))")
    f = expression_to_function("sin (x * z^2) / cos(y) + cos(y ^ exp(sin(z))) + 1.09132 * x")

    _, func, vars = f(x=0.12, y=0.13, z=2.901)
    print(func, _.v)

