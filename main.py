import matplotlib.pyplot
import lib.graDescending
from lib.Var import Var, draw_dag
import lib.ParserPostfix
import lib.ParserAST

himmelblau = "(x^2 + y - 11)^2 + (x+ y ^ 2 - 7) ^2"
ackley = "-20 * exp(-0.2 * ((x^2+y^2)*0.5)^0.5) - exp(0.5*(cos(2 * x * pi) + cos (2 * y * pi))) + e + 20"
rastrigin = "20 + x^2 - (10 * cos(2 * pi * x)) + y^2 - (10 * cos(2 * PI * y))"
rosenbrock = "(1 - x)^2 + 100 * (y-x^2)^2"
sphere = "x^2 + y ^ 2"

func = [
    sphere,
    himmelblau,
    ackley,
    rastrigin,
    rosenbrock
]
init = [
    {"x":0.5, "y":3},
    {"x":-3, "y":-5},
    {"x":.5, "y":.5},
    {"x":3.2, "y":-2.2},
    {"x":.1, "y":.1}
]
name = [
    "sphere",
    "himmelblau",
    "ackley",
    "rastrigin",
    "rosenbrock"
]

# Test (Visualize DAG and AST)
# for t, i, n in zip(func, init, name):
#     print(n)
#     variables = i.copy()
#     # Prepare Var objects
#     for v in variables:
#         variables[v] = Var(i[v], exp=v, type="var", info=v)
#     ast = ParserAST.parse(t)
#     ParserAST.draw_ast(ast, n + "_AST")
#     _ = ParserAST.evaluateVar(ast, variables)

#     # Perform backward propagation to calculate the gradient
#     _.backward()

#     draw_dag(_, n + "_DAG_AST", "TB")
# for t, i, n in zip(func, init, name):
#     print(n)
#     variables = i.copy()
#     # Prepare Var objects
#     for v in variables:
#         variables[v] = Var(i[v], exp=v, type="var", info=v)
#     tokens = lib.ParserPostfix.ShuntingYard(t)
#     _ = lib.ParserPostfix.evaluateVar(tokens, variables)
#     # Perform backward propagation to calculate the gradient
#     _.backward()
#     draw_dag(_, n + "_DAG_PF", "TB")


# Test (Optimize functions using different Gradient Descend based method)
# import time
# iter = 100
# b = time.time()
# for t, i, n in zip(func, init, name):
#     print("-" * 50 + n + "-" * 50)
#     f_var, f = lib.ParserPostfix.expression_to_function(t)

#     print("Vanilla")
#     coords, evaluated = lib.graDescending.GradientDescending(f_var, i, lr=0.01, iter=iter)
#     lib.graDescending.animate(coords, f, title=f"Vanilla_{n}", filename=f"Vanilla_{n}.png")

#     print("Adam")
#     coords, evaluated = lib.graDescending.Adam(f_var, i, lr=0.1, iter=iter)
#     lib.graDescending. animate(coords, f, title=f"Adam_{n}", filename=f"Adam_{n}.png")

#     print("AdaGrad")
#     coords, evaluated = lib.graDescending.AdaGrad(f_var, i, lr=0.1, iter=iter)
#     lib.graDescending.animate(coords, f, title=f"AdaGrad_{n}", filename=f"AdaGrad_{n}.png")

#     print("RMSprop")
#     coords, evaluated = lib.graDescending.RMSprop(f_var, i, lr=0.1, iter=iter)
#     lib.graDescending.animate(coords, f, title=f"RMSprop_{n}", filename=f"RMSprop_{n}.png")

#     print("Momentum")
#     coords, evaluated = lib.graDescending.Momentum_(f_var, i, lr=0.001, iter=iter)
#     lib.graDescending.animate(coords, f, title=f"Momentum_{n}", filename=f"Momentum_{n}.png")

#     print("Nesterov")
#     coords, evaluated = lib.graDescending.Nesterov(f_var, i, lr=0.001, iter=iter)
#     lib.graDescending.animate(coords, f, title=f"Nesterov_{n}", filename=f"Nesterov_{n}.png")

#     print("BFGS")
#     coords, evaluated = lib.graDescending.BFGS(f_var, i, iter=iter)
#     lib.graDescending.animate(coords, f, title=f"BFGS_{n}", filename=f"BFGS_{n}.png")