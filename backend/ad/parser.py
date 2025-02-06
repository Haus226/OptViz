import re
from .var import Var, FUNC, CONSTANT, MATH_FUNC, CONSTANT_, PRECEDENCE
from .utils import binop, tokenize
# from graphviz import Digraph
# For visualization, make sure graphviz is installed and added into environment path

class Parser:

    def parse(self, exp):
        tokens = tokenize(exp)
        queue = []
        stack = []
        for token in tokens:

            if re.match(r'\d*\.*\d+', token) or re.match(CONSTANT, token.lower()):
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
            queue.append(stack.pop())  
        return queue
    
    # For debug purpose
    def postfix2tree(self, exp) -> dict:
        tokens = self.parse(exp)
        stack = []
        
        for token in tokens:
            if re.match(r'-*\d*\.*\d+', token):
                stack.append({'type': 'number', 'value': float(token)})
            elif re.match(CONSTANT, token.lower()):
                stack.append({
                    'type': 'constant',
                    'value': CONSTANT_[token.lower()],
                    'cons_name': token.lower()
                })
            elif re.match(FUNC, token):
                arg = stack.pop()
                stack.append({
                    'type': 'function_call',
                    'func': token,
                    'args': [arg]
                })
            elif re.match(r'[a-zA-Z]', token):
                stack.append({'type': 'variable', 'var': token})
            elif re.match(r'[-+*/^]', token):
                right = stack.pop()
                left = stack.pop()
                node = {
                    'type': 'binop',
                    'op': token,
                    'left': left,
                    'right': right
                }
                stack.append(node)
        return stack[0]
    
    def print_tree(self, tree: dict, indent="", is_root=True, is_last=True) -> str:
        
        branch = "    ROOT\n    " if is_root else ("└── " if is_last else "├── ")        
        if tree['type'] == 'number':
            node = f"NUM({tree['value']})"
        elif tree['type'] == 'constant':
            node = f"CONST({tree['cons_name']})"
        elif tree['type'] == 'variable':
            node = f"VAR({tree['var']})"
        elif tree['type'] == 'function_call':
            node = f"FUNC({tree['func']})"
        else:
            node = f"OP({tree['op']})"
        result = indent + branch + node + "\n"
        
        # Handle children
        next_indent = indent + ("    " if is_last else "│   ")
        
        if tree['type'] == 'function_call':
            args = tree['args']
            for i, arg in enumerate(args):
                result += self.print_tree(arg, next_indent, False, i == len(args)-1)
        elif tree['type'] == 'binop':
            result += self.print_tree(tree['left'], next_indent, False, False)
            result += self.print_tree(tree['right'], next_indent, False, True)
            
        return result

    # @staticmethod
    # def draw_ast(ast, filename="ast"):
    #     dot = Digraph(comment="Abstract Syntax Tree", format="png")

    #     def traverse(node):
    #         if node['type'] == "number":
    #             dot.node(str(id(node)), label=str(node['type']) + "\n" + str(node["value"]))
    #         elif node["type"] == "constant":
    #             dot.node(str(id(node)), label=str(node['type']) + "\n" + str(node["cons_name"]))

    #         elif node['type'] == "variable":
    #             dot.node(str(id(node)), label=str(node['type']) + "\n" + str(node["var"]))
    #         elif node['type'] == "function_call":
    #             dot.node(str(id(node)), label=str(node['type']) + "\n" + str(node["func"]))
    #             for arg in node["args"]:
    #                 traverse(arg)
    #                 dot.edge(str(id(node)), str(id(arg)))
    #         elif node['type'] == 'binop':
    #             dot.node(str(id(node)), label=str(node['type']) + '|' + str(node['op']), shape="record")
    #             traverse(node['left'])
    #             traverse(node['right'])
    #             dot.edge(str(id(node)), str(id(node["left"])))
    #             dot.edge(str(id(node)), str(id(node["right"])))

    #     traverse(ast)
    #     dot.render(filename)

    def __evalVar(self, tokens: list, vars):
        stack = []
        for token in tokens:
            if re.match(r'\d*\.*\d+', token):
                stack.append(Var(float(token), type="number", info=token, exp=token))
            elif re.match(CONSTANT, token.lower()):
                stack.append(Var(CONSTANT_[token.lower()], type="constant", info=token.lower(), exp=token.lower()))
            elif re.match(FUNC, token):
                operand = stack.pop()
                result = Var(MATH_FUNC[token][0](operand.v), parents=[operand], type="func", info=token, exp=f"{token}(" + operand.exp + ")")
                stack.append(result)
            elif re.match(r'[a-zA-Z]', token):
                stack.append(vars[token])
            elif re.match(r'[-+*/^]', token):
                operand_2 = stack.pop()
                operand_1 = stack.pop()
                result = binop(token, operand_1, operand_2)
                stack.append(result)
            else:
                raise ValueError("Invalid token in the expression")
        return stack[-1]

    def __evalFunc(self, tokens: list, vars):
        stack = []
        for token in tokens:
            if re.match(r'\d*\.*\d+', token):
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
                result = binop(token, operand_1, operand_2)
                stack.append(result)
            else:
                raise ValueError("Invalid token in the expression")
        return stack[-1]
    
    def exp2func(self, exp):
        tokens = self.parse(exp)

        def func(**kwargs):
            variables = {k: Var(v, exp=k, type="var", info=k) for k, v in kwargs.items()}
            return self.__evalVar(tokens, variables), variables

        def f(**kwargs):
            variables = kwargs
            return self.__evalFunc(tokens, variables)


        return func, f