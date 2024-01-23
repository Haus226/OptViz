import re
from typing import Union
import numpy as np
import os
from graphviz import Digraph

# For visualization, make sure graphviz is installed and added into environment path
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'

TRIGO_FUNC = "exp|log|sin|cos|tan|arcsin|arccos|arctan|asin|acos|atan|cot|sec|cosec|csc|arcsec|arccosec|arcsc|arcot|asec|acosec|acsc|acot"
HYPERBOLIC_FUNC = "sinh|cosh|tanh|arcsinh|arccosh|arctanh|asinh|acosh|atanh|coth|sech|cosech|csch|arcoth|arcsech|arccosech|arccsch|acoth|asech|acosech|acsch"
FUNC = rf"{TRIGO_FUNC + HYPERBOLIC_FUNC}"
CONSTANT = r'pi|\be\b'
PRECEDENCE = {'^': 4, '*': 3, '/': 3, '+': 2, '-': 2, }
CONSTANT_ = {"pi": np.pi, "e": np.e}
MATH_FUNC = {
    'log': [np.log, lambda x: 1.0 / x],
    'exp': [np.exp, np.exp],
    'sin': [np.sin, np.cos],
    'cos': [np.cos, lambda x: -np.sin(x)],
    'tan': [np.tan, lambda x: (1.0 / np.cos(x)) ** 2],
    'sec': [lambda x: 1.0 / np.cos(x), lambda x:(1.0 / np.cos(x)) * np.tan(x)], 
    'cosec': [lambda x: 1.0 / np.sin(x), lambda x: -(1.0 / np.sin(x) ** 2) * np.cos(x)],  
    'cot': [lambda x: 1.0 / np.tan(x), lambda x: -(1.0 / np.sin(x) ** 2)],
    'arcsin': [np.arcsin, lambda x: 1.0 / np.sqrt(1 - x ** 2)],
    'asin': [np.arcsin, lambda x: 1.0 / np.sqrt(1 - x ** 2)],
    'arccos': [np.arccos, lambda x: -1.0 / np.sqrt(1 - x ** 2)],
    'acos': [np.arccos, lambda x: -1.0 / np.sqrt(1 - x ** 2)],
    'arctan': [np.arctan, lambda x: 1.0 / (1 + x ** 2)],
    'atan': [np.arctan, lambda x: 1.0 / (1 + x ** 2)],
    'arccsc': [lambda x: np.arcsin(1.0 / x),  lambda x: 1.0 / (abs(x) * np.sqrt(x ** 2 - 1))], # Inverse of cosec is arcsin(1/x)
    'acsc': [lambda x: np.arcsin(1.0 / x),  lambda x: 1.0 / (abs(x) * np.sqrt(x ** 2 - 1))], 
    'arcsec': [lambda x: np.arcsin(1.0 / x),  lambda x: 1.0 / (abs(x) * np.sqrt(x ** 2 - 1))], 
    'asec': [lambda x: np.arccos(1.0 / x),  lambda x: 1.0 / (abs(x) * np.sqrt(x ** 2 - 1))], 
    'arcot': [lambda x: np.arctan(1.0 / x), lambda x:-1.0 / (1 + x ** 2)], # Inverse of cot is arctan(1/x)
    'acot': [lambda x: np.arctan(1.0 / x), lambda x:-1.0 / (1 + x ** 2)],
    'sinh': [np.sinh, np.cosh],
    'cosh': [np.cosh, np.sinh],
    'tanh': [np.tanh, lambda x: 1.0 / np.cosh(x) ** 2],
    'arcsinh': [np.arcsinh, lambda x: 1.0 / np.sqrt(x ** 2 + 1)],
    'asinh': [np.arcsinh, lambda x: 1.0 / np.sqrt(x ** 2 + 1)],
    'arccosh': [np.arccosh, lambda x: 1.0 / np.sqrt(x ** 2 - 1)],
    'acosh': [np.arccosh, lambda x: 1.0 / np.sqrt(x ** 2 - 1)],
    'arctanh': [np.arctanh, lambda x: 1.0 / (1 - x ** 2)],
    'atanh': [np.arctanh, lambda x: 1.0 / (1 - x ** 2)],
    'sech': [lambda x: 1.0 / np.cosh(x), lambda x: -np.tanh(x) / np.cosh(x)],  # Define sech as 1/cosh(x)
    'cosech': [lambda x: 1.0 / np.sinh(x),  lambda x: -1.0 / (np.sinh(x) ** 2) * np.tanh(x)], # Define cosech as 1/sinh(x)
    'csch': [lambda x: 1.0 / np.sinh(x),  lambda x: -1.0 / (np.sinh(x) ** 2) * np.tanh(x)], # Define csch as 1/sinh(x)
    'coth': [lambda x: 1.0 / np.tanh(x), lambda x: -1.0 / np.sinh(x) ** 2], # Define coth as 1/tanh(x)
    'arccsch': lambda x: np.arcsinh(1.0 / x),  # Inverse of cosec is arcsin(1/x)
    'acsch': [lambda x: np.acosh(1.0 / x), lambda x: -1.0 / (abs(x) * np.sqrt(1 - x ** 2))],
    'arcsech': [lambda x: np.acosh(1.0 / x), lambda x: -1.0 / (abs(x) * np.sqrt(1 - x ** 2))],  # Inverse of sec is arccos(1/x)
    'asech': [lambda x: np.arccosh(1.0 / x), lambda x: -1.0 / (abs(x) * np.sqrt(1 - x** 2))],
    'arcoth': [lambda x: np.arctanh(1.0 / x), lambda x: 1.0 / (1 - x ** 2)],  # Inverse of cot is arctan(1/x)
    'acoth': [lambda x: np.arctanh(1.0 / x), lambda x: 1.0 / (1 - x ** 2)]
}



class Var:

    def __init__(self, val: Union[float, int]=None, parents=None, type=None, info=None, exp:str=None):
        if parents is None:
            parents = []
        self.v = val
        self.parents = parents
        self.grad = 0.0
        self.info = info
        self.type = type
        self.exp = exp

    def forward(self):
        if len(self.parents) > 0:
            if len(self.parents) == 1:
                self.parents[0][0].forward()
                self.v = MATH_FUNC[self.info][0](self.parents[0][0].v)
                self.parents[0][1] = MATH_FUNC[self.info][1](self.parents[0][0].v)
                self.parents[0][0].grad = 0.0
                return
            elif len(self.parents) == 2:
                self.parents[0][0].forward()
                self.parents[1][0].forward()
                if self.info == "+":
                    self.v = self.parents[0][0].v + self.parents[1][0].v
                    self.parents[0][0].grad = 0.0
                    self.parents[1][0].grad = 0.0
                elif self.info == "-":
                    self.v = self.parents[0][0].v - self.parents[1][0].v
                    self.parents[0][0].grad = 0.0
                    self.parents[1][0].grad = 0.0
                elif self.info == "/":
                    self.v = self.parents[0][0].v / self.parents[1][0].v
                    self.parents[0][1] = 1 / self.parents[1][0].v
                    self.parents[1][1] = - self.parents[0][0].v / self.parents[1][0].v ** 2
                    self.parents[0][0].grad = 0.0
                    self.parents[1][0].grad = 0.0
                elif self.info == "*":
                    self.v = self.parents[0][0].v * self.parents[1][0].v
                    self.parents[0][1] = self.parents[1][0].v
                    self.parents[1][1] = self.parents[0][0].v
                    self.parents[0][0].grad = 0.0
                    self.parents[1][0].grad = 0.0
                elif self.info == "^":
                    self.v = self.parents[0][0].v ** self.parents[1][0].v
                    self.parents[0][1] = self.parents[1][0].v * self.parents[0][0].v ** (self.parents[1][0].v - 1)
                    self.parents[1][1] = (self.parents[0][0].v ** self.parents[1][0].v * np.log(self.parents[0][0].v)) if self.parents[0][0].v >= 1e-10 else 0.0
                    self.parents[0][0].grad = 0.0
                    self.parents[1][0].grad = 0.0
        self.grad = 0.0
        
    def getGrad(self):
        return self.grad
    
    def getValue(self):
        return self.v
    
    def getExp(self):
        return self.exp

    def backprop(self, bp):
        self.grad += bp
        for parent, grad in self.parents:
            parent.backprop(grad * bp)

    def backward(self):
        self.backprop(1.0)

    def trace(self):
        nodes, edges = set(), set()
        def build(node):
            if node not in nodes: # If node has not been seen before
                nodes.add(node) # Store node
                for parent, grad in node.parents: # For each parent of this node
                    edges.add((parent, node)) # Add a (parent, current node) edge
                    build(parent) # Recursively call build
        build(self)
        return nodes, edges

    # All the gradients below calculated as partial differentiate the equation respect to self and other.
    # Examples : 
    # __mul__ : self * other, differentiate respect to self gives us other while other when respect to self
    # __truediv__ : self / other, differentiate respect to self gives us 1 / other while -self / other ** 2 when respect to self

    def __add__(self: 'Var', other: 'Var') -> 'Var':

        return Var(self.v + other.v, [[self, 1.0], [other, 1.0]], type="op", info="+", exp=self.exp + " + " + other.exp)

    def __mul__(self: 'Var', other: 'Var') -> 'Var':
        return Var(self.v * other.v, [[self, other.v], [other, self.v]], type="op", info="*", exp=self.exp + " * " + other.exp)

    def __pow__(self, other: 'Var') -> 'Var':
        return Var(self.v ** other.v, [[self, other.v * self.v ** (other.v - 1)], [other, (self.v ** other.v * np.log(self.v)) if self.v >= 0 else 0.0]], type="op", info=f"^",
                   exp=self.exp + " ^ " + other.exp)
    
    # def __matmul__(self, other: 'Var') -> 'Var':
    #     assert isinstance(other, Var)
    #     return Var(self.v ** other.v, [(self, other.v * self.v ** (other.v - 1)), (other, self.v ** other.v * np.log(self.v))], type="op", info=f"^{round(other.v, 3)}",
    #                exp=self.exp + " ^ " + other.exp)

    def __neg__(self: 'Var') -> 'Var':
        return Var(-1.0) * self

    def __sub__(self: 'Var', other: 'Var') -> 'Var':
        return Var(self.v - other.v, [[self, 1.0], [other, -1.0]], type="op", info="-", exp=self.exp + " - " + other.exp)

    def __truediv__(self: 'Var', other: 'Var') -> 'Var':
        return Var(self.v / other.v, [[self, 1.0 / other.v], [other, -self.v / (other.v ** 2)]], type="op", info="/", exp=f"({self.exp} / {other.exp})")

    def __repr__(self):
        return "Var(v=%.4f, grad=%.4f, exp=%s)" % (self.v, self.grad, self.exp)

def draw_dag(root, filename, rankdir='RL'):
    assert rankdir in ['LR', 'TB', 'RL']
    nodes, edges = root.trace()

    dot = Digraph(format='png', graph_attr={'rankdir': rankdir})
    
    for n in nodes: 
        label = "{f : %.3f} | {d : %.3f}" % (n.v, n.grad)
        if n.type == "number":
            label = "{v : %.3f}" % (n.v)
        if n.type == "constant":
            label = n.info
        if n.type == "var":
            label += f" | {n.info}"        
        dot.node(name=str(id(n)), label=label, shape='record')
        if n.type == "op" or n.type == "func":
            dot.node(name=str(id(n)) + n.info, label=n.info)
            dot.edge(str(id(n)), str(id(n)) + n.info)  

    for n1, n2 in edges:
        dot.edge(str(id(n2)) + n2.info, str(id(n1)))
    dot.render(filename=filename, directory='').replace('\\', '/')
    return dot
    



