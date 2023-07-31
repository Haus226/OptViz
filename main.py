import matplotlib.pyplot
import graDescending
from Var import Var, draw_dag
import ParserPostfix
import ParserAST

himmelblau = "(x^2 + y - 11)^2 + (x+ y ^ 2 - 7) ^2"
ackley = "-20 * exp(-0.2 * ((x^2+y^2)*0.5)^0.5) - exp(0.5*(cos(2 * x * pi) + cos (2 * y * pi))) + e + 20"
rastrigin = "20 + x^2 - (10 * cos(2 * pi * x)) + y^2 - (10 * cos(2 * PI * y))"
rosenbrock = "(1 - x)^2 + 100 * (y-x^2)^2"
sphere = "x^2+y^2"

func = [
    sphere,
    himmelblau,
    ackley,
    rastrigin,
    rosenbrock
]
init = [
    {"x":2, "y":3},
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
#     tokens = ParserPostfix.ShuntingYard(t)
#     _ = ParserPostfix.evaluateVar(tokens, variables)
#     # Perform backward propagation to calculate the gradient
#     _.backward()
#     draw_dag(_, n + "_DAG_PF", "TB")


# Test (Optimize functions using different Gradient Descend based method)
import time
iter = 100
b = time.time()
for t, i, n in zip(func, init, name):
    print("-" * 50 + n + "-" * 50)
    f_var, f = ParserPostfix.expression_to_function(t)
    print("Vanilla")
    coords, evaluated = graDescending.GradientDescending(f_var, i, lr=0.01, iter=iter)
    graDescending.animate(coords, f, title=f"Vanilla_{n}", filename=f"Vanilla_{n}.png")
    # print(coords[-10:-1])
    print("Adam")
    coords, evaluated = graDescending.Adam(f_var, i, lr=0.1, iter=iter)
    graDescending. animate(coords, f, title=f"Adam_{n}", filename=f"Adam_{n}.png")
    # print(coords[-10:-1])
    print("RMSprop")
    coords, evaluated = graDescending.RMSprop(f_var, i, lr=0.1, iter=iter)
    graDescending.animate(coords, f, title=f"RMSprop_{n}", filename=f"RMSprop_{n}.png")
    # print(coords[-10:-1])
    print("Momentum")
    coords, evaluated = graDescending.Momentum(f_var, i, lr=0.001, iter=iter)
    graDescending.animate(coords, f, title=f"Momentum_{n}", filename=f"Momentum_{n}.png")
    # print(coords[-10:-1])
    print("BFGS")
    coords, evaluated = graDescending.BFGS(f_var, i, iter=iter)
    graDescending.animate(coords, f, title=f"BFGS_{n}", filename=f"BFGS_{n}.png")
    # print(coords[-10:-1])
    matplotlib.pyplot.close()
e = time.time()
print(e - b)
#
