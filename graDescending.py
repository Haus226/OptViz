from Var import Var, draw_dag
from ParserAST import expression_to_function
# from ParserPostfix import expression_to_function
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def GradientDescending(f:callable, init:dict, lr:float=0.01, iter:int=100, tol:float=1e-10):
    cur = init.copy()
    _, vars = f(**cur)
    _.backward()
    coords = np.array([[cur[k] for k in cur]])
    evaluated = np.array([_.v])
    prev_f = _.v
    prev_grad_norm = np.linalg.norm(np.array([vars[k].grad for k in cur]))
    for cnt in range(iter):
        for k in cur:
            cur[k] = cur[k] - lr * vars[k].grad
            vars[k].v = cur[k]
        _.forward()
        _.backward()
        if (np.abs(prev_f - _.v) < tol):
            print(f"Terminate by f_value after {cnt} iterations")
            break
        grad_norm = np.linalg.norm(np.array([vars[k].grad for k in cur]))
        if np.abs(grad_norm - prev_grad_norm) < tol:
            print(f"Terminate by gradient after {cnt} iterations")
            break
        coords = np.append(coords, np.array([[cur[k] for k in cur]]), axis=0)
        evaluated = np.append(evaluated, np.array([_.v]), axis=0)
        prev_f = _.v
        prev_grad_norm = grad_norm
    # print(coords)
    # print(evaluated)
    return coords, evaluated

def Adam(f:callable, init:dict, lr:float=0.01, iter:int=100, tol:float=1e-10, beta1=0.9, beta2=0.999, epsilon=1e-8):
    cur = init.copy()
    m = np.zeros_like(list(cur), dtype=float)
    v = np.zeros_like(list(cur), dtype=float)
    grad = np.empty_like(list(cur), dtype=float)
    _, vars = f(**cur)
    _.backward()
    coords = np.array([[cur[k] for k in cur]])
    evaluated = np.array([_.v])
    prev_f = _.v
    prev_grad_norm = np.linalg.norm(np.array([vars[k].grad for k in cur]))
    for cnt in range(iter):
        for idx, k in enumerate(vars):
            grad[idx] = vars[k].grad
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * np.square(grad)
        m_hat = m / (1 - np.power(beta1, cnt + 1))
        v_hat = v / (1 - np.power(beta2, cnt + 1))
        for idx, k in enumerate(cur):
            cur[k] = cur[k] - lr * m_hat[idx] / (np.sqrt(v_hat[idx]) + epsilon)
            vars[k].v = cur[k]
        _.forward()
        _.backward()
        if (np.abs(prev_f - _.v) < tol):
            print(f"Terminate by f_value after {cnt} iterations")
            break
        grad_norm = np.linalg.norm(np.array([vars[k].grad for k in cur]))
        if np.abs(grad_norm - prev_grad_norm) < tol:
            print(f"Terminate by gradient after {cnt} iterations")
            break
        coords = np.append(coords, np.array([[cur[k] for k in cur]]), axis=0)
        evaluated = np.append(evaluated, np.array([_.v]), axis=0)
        prev_f = _.v
        prev_grad_norm = grad_norm
    # print(coords)
    # print(evaluated)
    return coords, evaluated

def RMSprop(f:callable, init:dict, lr:float=0.01, iter:int=100, tol:float=1e-10, beta=0.9, epsilon=1e-8):
    cur = init.copy()
    v = np.zeros_like(list(cur), dtype=float)
    grad = np.empty_like(list(cur), dtype=float)
    _, vars = f(**cur)
    _.backward()
    coords = np.array([[cur[k] for k in cur]])
    evaluated = np.array([_.v])
    prev_f = _.v
    prev_grad_norm = np.linalg.norm(np.array([vars[k].grad for k in cur]))
    for cnt in range(iter):
        for idx, k in enumerate(vars):
            grad[idx] = vars[k].grad
        v = beta * v + (1 - beta) * np.square(grad)
        for idx, k in enumerate(cur):
            cur[k] = cur[k] - lr * grad[idx] / (np.sqrt(v[idx]) + epsilon)
            vars[k].v = cur[k]
        _.forward()
        _.backward()
        if (np.abs(prev_f - _.v) < tol):
            print(f"Terminate by f_value after {cnt} iterations")
            break
        grad_norm = np.linalg.norm(np.array([vars[k].grad for k in cur]))
        if np.abs(grad_norm - prev_grad_norm) < tol:
            print(f"Terminate by gradient after {cnt} iterations")
            break
        coords = np.append(coords, np.array([[cur[k] for k in cur]]), axis=0)
        evaluated = np.append(evaluated, np.array([_.v]), axis=0)
        prev_f = _.v
        prev_grad_norm = grad_norm
    # print(coords)
    # print(evaluated)
    return coords, evaluated

def Momentum(f:callable, init:dict, lr:float=0.01, iter:int=100, tol:float=1e-10, beta=0.9):
    cur = init.copy()
    v = np.zeros_like(list(cur), dtype=float)
    grad = np.empty_like(list(cur), dtype=float)
    _, vars = f(**cur)
    _.backward()
    coords = np.array([[cur[k] for k in cur]])
    evaluated = np.array([_.v])
    prev_f = _.v
    prev_grad_norm = np.linalg.norm(np.array([vars[k].grad for k in cur]))
    for cnt in range(iter):
        for idx, k in enumerate(vars):
            grad[idx] = vars[k].grad
        v = beta * v + lr * grad
        for idx, k in enumerate(cur):
            cur[k] = cur[k] - v[idx]
            vars[k].v = cur[k]
        _.forward()
        _.backward()
        if (np.abs(prev_f - _.v) < tol):
            print(f"Terminate by f_value after {cnt} iterations")
            break
        grad_norm = np.linalg.norm(np.array([vars[k].grad for k in cur]))
        if np.abs(grad_norm - prev_grad_norm) < tol:
            print(f"Terminate by gradient after {cnt} iterations")
            break
        coords = np.append(coords, np.array([[cur[k] for k in cur]]), axis=0)
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
    x_k = init.copy()
    x_next = init.copy()

    _, vars = f(**x_k)
    _.backward()
    _next, vars_next = f(**x_next)

    coords = np.array([[x_k[k] for k in x_k]])
    evaluated = np.array([_.v])
    I = np.eye(len(list(init)))
    H_k = I

    # Constants used in linear search for step size alpha
    c1 = 1e-4
    c2 = 0.9

    for cnt in range(iter):
        # Compute gradient and search direction
        g_k = np.array([vars[k].grad for k in x_k])
        p_k = -np.dot(H_k, g_k)
        # Perform line search
        alpha = alpha_
        for idx, k in enumerate(x_k):
            x_next[k] = x_k[k] + alpha * p_k[idx]
            vars_next[k].v = x_next[k]
        _next.forward()
        _next.backward()
        g_next = np.array([vars_next[k].grad for k in x_next])

        # Check strong Wolfe's Condition
        while _next.v >= _.v + c1 * alpha * np.dot(g_k, p_k) or np.dot(g_next, p_k) <= c2 * np.dot(g_k, p_k):
            alpha *= rho
            for idx, k in enumerate(x_k):
                x_next[k] = x_k[k] + alpha * p_k[idx]
                vars_next[k].v = x_next[k]
            _next.forward()
            _next.backward()
            g_next = np.array([vars_next[k].grad for k in x_k])

            if alpha < min_alpha:
                break

        # Update parameters and gradients
        x_prev = x_k.copy()
        g_prev = g_k.copy()
        f_prev = _.v
        for idx, k in enumerate(x_k):
            x_k[k] = x_k[k] + alpha * p_k[idx]
            vars[k].v = x_k[k]
        _.forward()

        if (np.abs(f_prev - _.v) < tol):
            print(f"Terminate by f_value after {cnt} iterations")
            break

        _.backward()
        g_k = np.array([vars[k].grad for k in x_k])

        if np.abs(np.linalg.norm(g_k) - np.linalg.norm(g_prev)) < tol:
            print(f"Terminate by gradient after {cnt} iterations")
            break

        coords = np.append(coords, np.array([[x_k[k] for k in x_k]]), axis=0)
        evaluated = np.append(evaluated, np.array([_.v]), axis=0)

        # Compute difference vectors
        s_k = np.array([x_k[k] for k in x_k]) - np.array([x_prev[k] for k in x_prev])
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
    contour = ax.contour(x, y, f(x=x, y=y), levels=15)
    plt.clabel(contour, inline=True, fontsize=15)
    plt.title(f"{title}", fontsize=40)
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
