# from ParserAST import expression_to_function
from lib.ParserPostfix import expression_to_function
from lib.Var import draw_dag
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def GradientDescending(f:callable, init:dict, lr:float=0.01, iter:int=100, tol:float=1e-10):
    _, vars = f(**init)
    _.backward()
    coords = np.array([[var.v for var in vars.values()]])
    evaluated = np.array([_.v])
    prev_f = _.v
    prev_grad_norm = np.linalg.norm(np.array([var.grad for var in vars.values()]))
    for cnt in range(iter):
        for var in vars.values():
            var.v -= lr * var.grad
        _.forward()
        print("v:", _.v)
        if (np.abs(prev_f - _.v) < tol):
            print(f"Terminate by f_value after {cnt} iterations")
            break
        _.backward()

        grad_norm = np.linalg.norm(np.array([var.grad for var in vars.values()]))
        if np.abs(grad_norm - prev_grad_norm) < tol:
            print(f"Terminate by gradient after {cnt} iterations")
            break
        coords = np.append(coords, np.array([[var.v for var in vars.values()]]), axis=0)
        evaluated = np.append(evaluated, np.array([_.v]), axis=0)
        prev_f = _.v
        prev_grad_norm = grad_norm
    
    # print(coords)
    # print(evaluated)
    return coords, evaluated

def Momentum(f:callable, init:dict, lr:float=0.01, iter:int=100, tol:float=1e-10, beta=0.9):
    v = np.zeros_like(list(init), dtype=float)
    grad = np.empty_like(list(init), dtype=float)
    _, vars = f(**init)
    _.backward()
    coords = np.array([[var.v for var in vars.values()]])
    evaluated = np.array([_.v])
    prev_f = _.v
    prev_grad_norm = np.linalg.norm(np.array([var.grad for var in vars.values()]))
    for cnt in range(iter):
        for idx, k in enumerate(vars):
            grad[idx] = vars[k].grad
        v = beta * v - lr *  grad
        for idx, var in enumerate(vars.values()):
            var.v += v[idx]
        _.forward()
        if (np.abs(prev_f - _.v) < tol):
            print(f"Terminate by f_value after {cnt} iterations")
            break
        _.backward()
        grad_norm = np.linalg.norm(np.array([var.grad for var in vars.values()]))
        if np.abs(grad_norm - prev_grad_norm) < tol:
            print(f"Terminate by gradient after {cnt} iterations")
            break
        coords = np.append(coords, np.array([[var.v for var in vars.values()]]), axis=0)
        evaluated = np.append(evaluated, np.array([_.v]), axis=0)
        prev_f = _.v
        prev_grad_norm = grad_norm
    # print(coords)
    # print(evaluated)
    return coords, evaluated

def Momentum_(f:callable, init:dict, lr:float=0.01, iter:int=100, tol:float=1e-10, beta=0.9):
    
    v = np.zeros_like(list(init), dtype=float)
    grad = np.empty_like(list(init), dtype=float)
    _, vars = f(**init)
    _.backward()
    coords = np.array([[init[k] for k in init]])
    evaluated = np.array([_.v])
    prev_f = _.v
    prev_grad_norm = np.linalg.norm(np.array([vars[k].grad for k in init]))
    for cnt in range(iter):
        for idx, k in enumerate(vars):
            grad[idx] = vars[k].grad
        v = beta * v + grad

        for idx, var in enumerate(vars.values()):
            var.v -= lr * v[idx]

        _.forward()
        if (np.abs(prev_f - _.v) < tol):
            print(f"Terminate by f_value after {cnt} iterations")
            break
        _.backward()
        grad_norm = np.linalg.norm(np.array([vars[k].grad for k in init]))
        if np.abs(grad_norm - prev_grad_norm) < tol:
            print(f"Terminate by gradient after {cnt} iterations")
            break
        coords = np.append(coords, np.array([[init[k] for k in init]]), axis=0)
        evaluated = np.append(evaluated, np.array([_.v]), axis=0)
        prev_f = _.v
        prev_grad_norm = grad_norm
    # print(coords)
    # print(evaluated)
    return coords, evaluated

def Momentum__(f:callable, init:dict, lr:float=0.01, iter:int=100, tol:float=1e-10, beta=0.9):
    
    v = np.zeros_like(list(init), dtype=float)
    grad = np.empty_like(list(init), dtype=float)
    _, vars = f(**init)
    _.backward()
    coords = np.array([[init[k] for k in init]])
    evaluated = np.array([_.v])
    prev_f = _.v
    prev_grad_norm = np.linalg.norm(np.array([vars[k].grad for k in init]))
    for cnt in range(iter):
        for idx, k in enumerate(vars):
            grad[idx] = vars[k].grad
        v = beta * v + (1 - beta) *  grad
        for idx, var in enumerate(vars.values()):
            var -= lr * v[idx]
        _.forward()
        if (np.abs(prev_f - _.v) < tol):
            print(f"Terminate by f_value after {cnt} iterations")
            break
        _.backward()
        grad_norm = np.linalg.norm(np.array([vars[k].grad for k in init]))
        if np.abs(grad_norm - prev_grad_norm) < tol:
            print(f"Terminate by gradient after {cnt} iterations")
            break
        coords = np.append(coords, np.array([[init[k] for k in init]]), axis=0)
        evaluated = np.append(evaluated, np.array([_.v]), axis=0)
        prev_f = _.v
        prev_grad_norm = grad_norm
    # print(coords)
    # print(evaluated)
    return coords, evaluated

# def Nesterov(f:callable, init:dict, lr:float=0.01, iter:int=100, tol:float=1e-10, beta=0.9):
#     
#     v = np.zeros_like(list(init), dtype=float)
#     grad = np.empty_like(list(init), dtype=float)
#     _, vars = f(**init)
#     _.backward()
#     coords = np.array([[init[k] for k in init]])
#     evaluated = np.array([_.v])
#     prev_f = _.v
#     prev_grad_norm = np.linalg.norm(np.array([vars[k].grad for k in init]))
#     for cnt in range(iter):
#         for idx, k in enumerate(vars):
#             grad[idx] = vars[k].grad
#         v = beta * v - lr * grad
#         for idx, k in enumerate(init):
#             init[k] = init[k] + (beta ** 2) * v[idx] - (1 + beta) * lr * grad[idx]
#             vars[k].v = init[k]
#         _.forward()
#         if (np.abs(prev_f - _.v) < tol):
#             print(f"Terminate by f_value after {cnt} iterations")
#             break
#         _.backward()
#         grad_norm = np.linalg.norm(np.array([vars[k].grad for k in init]))
#         if np.abs(grad_norm - prev_grad_norm) < tol:
#             print(f"Terminate by gradient after {cnt} iterations")
#             break
#         coords = np.append(coords, np.array([[init[k] for k in init]]), axis=0)
#         evaluated = np.append(evaluated, np.array([_.v]), axis=0)
#         prev_f = _.v
#         prev_grad_norm = grad_norm
#     # print(coords)
#     # print(evaluated)
#     return coords, evaluated

def Nesterov(f:callable, init:dict, lr:float=0.01, iter:int=100, tol:float=1e-10, beta=0.9):
    v = np.zeros_like(list(init), dtype=float)
    grad = np.empty_like(list(init), dtype=float)

    _, vars = f(**init)
    _.backward()
    _next, _vars = f(**init)
    _next.backward()

    coords = np.array([[var.v for var in vars.values()]])
    evaluated = np.array([_.v])
    prev_f = _.v
    prev_grad_norm = np.linalg.norm(np.array([var.grad for var in vars.values()]))
    for cnt in range(iter):
        for idx, k in enumerate(_vars):
            _vars[k].v = vars[k].v + beta * v[idx]
        _next.forward()
        _next.backward()
        for idx, var in enumerate(_vars.values()):
            grad[idx] = var.grad
        v = beta * v - lr * grad
        for idx, var in enumerate(vars.values()):
            var.v += v[idx]
        _.forward()
        if (np.abs(prev_f - _.v) < tol):
            print(f"Terminate by f_value after {cnt} iterations")
            break
        _.backward()
        grad_norm = np.linalg.norm(np.array([var.grad for var in vars.values()]))
        if np.abs(grad_norm - prev_grad_norm) < tol:
            print(f"Terminate by gradient after {cnt} iterations")
            break
        coords = np.append(coords, np.array([[var.v for var in vars.values()]]), axis=0)
        evaluated = np.append(evaluated, np.array([_.v]), axis=0)
        prev_f = _.v
        prev_grad_norm = grad_norm
    # print(coords)
    # print(evaluated)
    return coords, evaluated

def AdaGrad(f:callable, init:dict, lr:float=0.01, iter:int=100, tol:float=1e-10, epsilon=1e-8):
    v = np.zeros_like(list(init), dtype=float)
    grad = np.empty_like(list(init), dtype=float)
    _, vars = f(**init)
    _.backward()
    coords = np.array([[var.v for var in vars.values()]])
    evaluated = np.array([_.v])
    prev_f = _.v
    prev_grad_norm = np.linalg.norm(np.array([var.grad for var in vars.values()]))
    for cnt in range(iter):
        for idx, var in enumerate(vars.values()):
            grad[idx] = var.grad
        v += np.square(grad)
        for idx, var in enumerate(vars.values()):
            var.v -= lr * grad[idx] / (np.sqrt(v[idx]) + epsilon)
        _.forward()
        if (np.abs(prev_f - _.v) < tol):
            print(f"Terminate by f_value after {cnt} iterations")
            break
        _.backward()
        grad_norm = np.linalg.norm(np.array([var.grad for var in vars.values()]))
        if np.abs(grad_norm - prev_grad_norm) < tol:
            print(f"Terminate by gradient after {cnt} iterations")
            break
        coords = np.append(coords, np.array([[var.v for var in vars.values()]]), axis=0)
        evaluated = np.append(evaluated, np.array([_.v]), axis=0)
        prev_f = _.v
        prev_grad_norm = grad_norm
    # print(coords)
    # print(evaluated)
    return coords, evaluated

def RMSprop(f:callable, init:dict, lr:float=0.01, iter:int=100, tol:float=1e-10, beta=0.9, epsilon=1e-8):
    
    v = np.zeros_like(list(init), dtype=float)
    grad = np.empty_like(list(init), dtype=float)
    _, vars = f(**init)
    _.backward()
    coords = np.array([[var.v for var in vars.values()]])
    evaluated = np.array([_.v])
    prev_f = _.v
    prev_grad_norm = np.linalg.norm(np.array([var.grad for var in vars.values()]))
    for cnt in range(iter):
        for idx, var in enumerate(vars.values()):
            grad[idx] = var.grad
        v = beta * v + (1 - beta) * np.square(grad)
        for idx, var in enumerate(vars.values()):
            var.v -= lr * grad[idx] / (np.sqrt(v[idx]) + epsilon)
        _.forward()
        if (np.abs(prev_f - _.v) < tol):
            print(f"Terminate by f_value after {cnt} iterations")
            break
        _.backward()
        grad_norm = np.linalg.norm(np.array([var.grad for var in vars.values()]))
        if np.abs(grad_norm - prev_grad_norm) < tol:
            print(f"Terminate by gradient after {cnt} iterations")
            break
        coords = np.append(coords, np.array([[var.v for var in vars.values()]]), axis=0)
        evaluated = np.append(evaluated, np.array([_.v]), axis=0)
        prev_f = _.v
        prev_grad_norm = grad_norm
    # print(coords)
    # print(evaluated)
    return coords, evaluated

def Adam(f:callable, init:dict, lr:float=0.01, iter:int=100, tol:float=1e-10, beta1=0.9, beta2=0.999, epsilon=1e-8):
    
    m = np.zeros_like(list(init), dtype=float)
    v = np.zeros_like(list(init), dtype=float)
    grad = np.empty_like(list(init), dtype=float)
    _, vars = f(**init)
    _.backward()
    coords = np.array([[var.v for var in vars.values()]])
    evaluated = np.array([_.v])
    prev_f = _.v
    prev_grad_norm = np.linalg.norm(np.array([var.grad for var in vars.values()]))
    for cnt in range(iter):
        for idx, var in enumerate(vars.values()):
            grad[idx] = var.grad
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * np.square(grad)
        m_hat = m / (1 - np.power(beta1, cnt + 1))
        v_hat = v / (1 - np.power(beta2, cnt + 1))
        for idx, var in enumerate(vars.values()):
            var.v -= lr * m_hat[idx] / (np.sqrt(v_hat[idx]) + epsilon)
        _.forward()
        if (np.abs(prev_f - _.v) < tol):
            print(f"Terminate by f_value after {cnt} iterations")
            break
        _.backward()
        grad_norm = np.linalg.norm(np.array([var.grad for var in vars.values()]))
        if np.abs(grad_norm - prev_grad_norm) < tol:
            print(f"Terminate by gradient after {cnt} iterations")
            break
        coords = np.append(coords, np.array([[var.v for var in vars.values()]]), axis=0)
        evaluated = np.append(evaluated, np.array([_.v]), axis=0)
        prev_f = _.v
        prev_grad_norm = grad_norm
    # print(coords)
    # print(evaluated)
    return coords, evaluated


def BFGS(f, init:dict, iter, tol=1e-10, alpha_=1.0, rho=0.8, min_alpha=1e-8):
    '''
    :param rho : The shrinking factor of step size during linear search
    :param alpha : Act like "learning_rate" in gradient descent
    '''

    _, vars = f(**init)
    _.backward()
    _next, vars_next = f(**init)

    coords = np.array([[var.v for var in vars.values()]])
    evaluated = np.array([_.v])
    I = np.eye(len(list(init)))
    H_k = I

    # Constants used in linear search for step size alpha
    c1 = 1e-4
    c2 = 0.9

    for cnt in range(iter):
        # Compute gradient and search direction
        g_k = np.array([var.grad for var in vars.values()])
        p_k = -np.dot(H_k, g_k)
        # Perform line search
        alpha = alpha_
        for idx, k in enumerate(vars):
            vars_next[k].v = vars[k].v + alpha * p_k[idx]
        _next.forward()
        _next.backward()
        g_next = np.array([var.grad for var in vars_next.values()])

        # Check strong Wolfe's Condition
        while _next.v >= _.v + c1 * alpha * np.dot(g_k, p_k) or np.dot(g_next, p_k) <= c2 * np.dot(g_k, p_k):
            alpha *= rho
            for idx, k in enumerate(vars):
                vars_next[k].v = vars[k].v + alpha * p_k[idx]
            _next.forward()
            _next.backward()
            g_next = np.array([var.grad for var in vars_next.values()])
            if alpha < min_alpha:
                break

        # Update parameters and gradients
        x_prev = np.array([var.v for var in vars.values()])
        g_prev = g_k.copy()
        f_prev = _.v

        for idx, var in enumerate(vars.values()):
            var.v += alpha * p_k[idx]
        _.forward()
        if (np.abs(f_prev - _.v) < tol):
            print(f"Terminate by f_value after {cnt} iterations")
            break
        _.backward()
        g_k = np.array([var.grad for var in vars.values()])
        if np.abs(np.linalg.norm(g_k) - np.linalg.norm(g_prev)) < tol:
            print(f"Terminate by gradient after {cnt} iterations")
            break

        coords = np.append(coords, np.array([[var.v for var in vars.values()]]), axis=0)
        evaluated = np.append(evaluated, np.array([_.v]), axis=0)

        # Compute difference vectors
        s_k = alpha * p_k
        y_k = g_k - g_prev
        rk_inv = np.dot(y_k, s_k)
        if rk_inv == 0: # Prevent zero division
            r_k = 1000.0
        else:
            r_k = 1. / rk_inv
        H_k = np.dot((I - r_k * np.outer(s_k, y_k)), np.dot(H_k, (I - r_k * np.outer(y_k, s_k)))) + r_k * np.outer(s_k, s_k)

    return coords, evaluated

def animate(coords, f, title="Visualization", filename="animate.gif", fps=2):
    fig, ax = plt.subplots(figsize=(15, 15), dpi=160)
    x = np.arange(np.min(coords[:, 0] - .5), np.max(coords[:, 0] + .5), 0.05)
    y = np.arange(np.min(coords[:, 1] - .5), np.max(coords[:, 1] + 0.5), 0.05)
    x, y = np.meshgrid(x, y)
    contour = ax.contour(x, y, f(x=x, y=y))
    plt.clabel(contour, inline=True, fontsize=15)
    plt.title(f"{title}", fontsize=40)
    plt.xlabel("x", fontsize=30)
    plt.ylabel("y", fontsize=30)
    length = coords.shape[0]

    def animate_(i):
        x = coords[i, [0]]
        y = coords[i, [1]]
        plt.scatter(x, y, c="green", s=250)
        if i:
            prev_x = coords[i - 1, [0]]
            prev_y = coords[i - 1, [1]]
            plt.scatter(prev_x, prev_y, c="red", s=250)
            plt.plot([prev_x, x], [prev_y, y], c="black")
        if i == coords.shape[0] - 1:
            x = coords[i, [0]]
            y = coords[i, [1]]
            plt.scatter(x, y, c="blue", s=250)
    if filename[-3:] in ["png", "jpg"]:
        for idx in range(length):
            animate_(idx)
        plt.savefig(filename)
    elif filename[-3:] == "gif":
        anim = animation.FuncAnimation(fig, animate_, frames=length)
        writergif = animation.PillowWriter(fps=fps)
        anim.save(f"{filename}", writer=writergif)


if __name__ == "__main__":
    # f = expression_to_function("x^2")
    # f_var, f = expression_to_function("20 + x^2 - 10 * cos(2 * 3.141592653589793238462643383279502884197 * x) + y^2 - 10 * cos(2 * 3.141592653589793238462643383279502884197 * y)")
    # init = {"x":1.0, "y":1.0}
    himmelblau = "(x^2 + y - 11)^2 + (x+ y ^ 2 - 7) ^2"
    # sphere = "x^2+y^2"
    f_var, f = expression_to_function(himmelblau)
    init =    {"x":1, "y":1}
    iter = 10
    from datetime import datetime
    begin = datetime.now()
    # c, e = GradientDescending(f_var, init, iter=iter)
    c, e = BFGS(f_var, init, iter=iter)
    print(c)
    print(e)
    # Adam(f, init, iter=iter)
    # RMSprop(f, init, iter=iter)
    # Momentum(f, init, lr=0.001, iter=iter)
    end = datetime.now()
    print(end - begin)
    animate(c, f)
