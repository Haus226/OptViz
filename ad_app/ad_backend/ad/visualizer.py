from .var import Var


# For visualization, make sure graphviz is installed and added into environment path
# import os
# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'


class ASTVisualizer:

    @staticmethod
    def to_latex(ast: Var):
        if ast["type"] == "binop":
            if ast["op"] == "+":
                return f"{ASTVisualizer.to_latex(ast['left'])} + {ASTVisualizer.to_latex(ast['right'])}"
            elif ast["op"] == "-":
                return f"{ASTVisualizer.to_latex(ast['left'])} - {ASTVisualizer.to_latex(ast['right'])}"
            elif ast["op"] == "*":
                return f"{ASTVisualizer.to_latex(ast['left'])} * {ASTVisualizer.to_latex(ast['right'])}"
            elif ast["op"] == "/":
                return f"\\frac{{{ASTVisualizer.to_latex(ast['left'])}}}{{{ASTVisualizer.to_latex(ast['right'])}}}"
            elif ast["op"] == "^":
                return f"{ASTVisualizer.to_latex(ast['left'])} ^ {{{ASTVisualizer.to_latex(ast['right'])}}}"
        elif ast["type"] == 'number':
            return str(ast['value'])
        elif ast["type"] == 'variable':
            return ast["var"]
        elif ast["type"] == 'function_call':
            args = [ASTVisualizer.to_latex(arg) for arg in ast['args']]
            return f"\\{ast['func']}({args[0]})"

    @staticmethod
    def draw_ast(ast: Var, filename="ast"):
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