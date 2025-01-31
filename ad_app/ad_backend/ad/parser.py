from abc import ABC, abstractmethod
import re
from .var import Var, FUNC, CONSTANT, MATH_FUNC, CONSTANT_, PRECEDENCE
from .utils import binop, tokenize
from .visualizer import ASTVisualizer

class Parser(ABC):
    
    def __call__(self, exp):
        return self.exp2func(exp)
    
    @abstractmethod
    def parse(self, exp):
        pass

    @abstractmethod
    def exp2func(self, exp):
        pass

class ASTParser(Parser):
    
    def parse(self, exp):
        current_token_index = 0
        tokens = tokenize(exp)

        def expression(min_precedence=0):
            left_side = term()
            while (
                current_token_index < len(tokens)
                and tokens[current_token_index] in PRECEDENCE
                and PRECEDENCE[tokens[current_token_index]] >= min_precedence
            ):
                operator = tokens[current_token_index]
                step()
                left_side = {"type": "binop", "op": operator, "left": left_side, "right": expression(PRECEDENCE[operator] + 1)}
            return left_side
        
        def term():
            left_side = factor()
            while (
                current_token_index < len(tokens)
                and tokens[current_token_index] in PRECEDENCE
                and PRECEDENCE[tokens[current_token_index]] >= 3
            ):
                
                operator = tokens[current_token_index]
                step()
                left_side = {'type': 'binop', 'op': operator, 'left': left_side, 'right': term()}
            return left_side

        def factor():
            if current_token_index < len(tokens):
                if tokens[current_token_index] == '(':
                    step()
                    expr_val = expression()
                    if tokens[current_token_index] == ')':
                        step()
                        return expr_val
                elif re.match(FUNC, tokens[current_token_index]):
                    function_exp = tokens[current_token_index]
                    step()
                    if tokens[current_token_index] == '(':
                        step()
                        args = []
                        while tokens[current_token_index] != ')':
                            args.append(expression())
                            if tokens[current_token_index] == ',':
                                step()
                            else:
                                break
                        if tokens[current_token_index] == ')':
                            step()
                            return {'type': 'function_call', 'func': function_exp, 'args': args}
                return parse_number_or_variable()

        def parse_number_or_variable():
            if current_token_index < len(tokens):
                if re.match(r'-*\d*\.*\d+', tokens[current_token_index]):
                    num = float(tokens[current_token_index])
                    step()
                    return {'type': 'number', 'value': num}
                elif re.match(CONSTANT, tokens[current_token_index].lower()):
                    step()
                    return {"type": "constant", "value": CONSTANT_[tokens[current_token_index - 1].lower()], "cons_name":tokens[current_token_index - 1].lower()}
                elif re.match(r'\w+', tokens[current_token_index]):
                    var_exp = tokens[current_token_index]
                    step()
                    return {'type': 'variable', 'var': var_exp}
            raise ValueError("Invalid expression")

        def step():
            nonlocal current_token_index
            current_token_index += 1

        return expression()

    def __evalVar(self, ast, variables):
        if ast['type'] == 'number':
            return Var(ast['value'], exp=f"{ast['value']}", type="number")
        elif ast['type'] == 'constant':
            return Var(ast['value'], exp=f"{ast['cons_name'].lower()}", type="constant", info=ast["cons_name"])
        elif ast['type'] == 'variable':
            return variables[ast["var"]]
        elif ast['type'] == 'binop':
            left_val = self.__evalVar(ast['left'], variables)
            right_val = self.__evalVar(ast['right'], variables)
            op = ast["op"]
            return binop(op, left_val, right_val)
        elif ast['type'] == 'function_call':
            function_exp = ast['func']
            args = [self.__evalVar(arg, variables) for arg in ast['args']]
            return Var(MATH_FUNC[function_exp][0](args[0].v), 
                    parents=args, 
                    type="func", info=function_exp, exp=f"{function_exp}(" + args[0].exp + ")")
        raise ValueError("Invalid AST node")

    def exp2func(self, exp) -> callable:
        tokens = self.parse(exp)

        def func(**kwargs):
            variables = {k: Var(v, exp=k, type="var", info=k) for k, v in kwargs.items()}
            return self.__evalVar(tokens, variables), variables
        
        return func
    
    def to_latex(self, exp) -> str:
        ast = self.parse(exp)
        return ASTVisualizer.to_latex(ast)
    
    def draw_ast(self, exp, filename) -> None:
        ast = self.parse(exp)
        ASTVisualizer.draw_ast(ast, filename)
    
class SYParser(Parser):

    def parse(self, exp):
        tokens = tokenize(exp)
        queue = []
        stack = []
        for token in tokens:
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
            queue.append(stack.pop())  
        return queue

    def __evalVar(self, tokens: list, vars):
        stack = []
        for token in tokens:
            if re.match(r'-*\d*\.*\d+', token):
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

if __name__ == "__main__":
    exp = "sin (x * z^2) / cos(y) + cos(y ^ exp(sin(2.901))) + 1.09132 * x"
    sy_parser = SYParser()
    func, f = sy_parser(exp)
    _, vars = func(x=0.12, y=0.13, z=2.901)
    _.backward()  
    # print(_)  

    # for v in vars:  
    #     print(vars[v])  
    #     print(f"df/d{vars[v].info} : ", vars[v].grad)

    ast_parser = ASTParser()
    func = ast_parser(exp)
    _, vars = func(x=0.12, y=0.13, z=2.901)
    _.backward()  
    print(_)  

    for v in vars:  
        print(vars[v], id(vars[v]))  
        print(f"df/d{vars[v].info} : ", vars[v].grad)

