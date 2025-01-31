import numpy as np


# TODO:
# LION
# Tiger
# CAME
# Lamb
# Shampoo
# Muon
# SOAP
# MARS
# APOLLO
# signSGD
# Lookahead
# NAdam
# Adan
# Adai
# AdaBelief



class GradDescent:
    def __init__(self, func:callable, init_p:dict, lr:float=0.01):
        self.func = func
        self.lr = lr
        self.var, self.var_dict = self.func(**init_p)
        self.coords = np.array([[var.v for var in self.var_dict.values()]])
        self.evaluated = np.array([self.var.v])

    def step(self) -> None:
        self.var.backward()
        if hasattr(self, "grad"):
            for idx, var in enumerate(self.var_dict.values()):
                self.grad[idx] = var.grad
        elif hasattr(self, "lookahead_var"):
            self.nesterov()
        self.update()
        self.var.forward()
        self.coords = np.append(self.coords, np.array([[var.v for var in self.var_dict.values()]]), axis=0)
        self.evaluated = np.append(self.evaluated, np.array([self.var.v]), axis=0)

    def update(self):
        for var in self.var_dict.values():
            var.v -= self.lr * var.grad

    def nesterov(self):
        pass

class AdaGrad(GradDescent):
    def __init__(self, func, init_p, lr=0.01, epsilon=1e-8):
        super().__init__(func, init_p, lr)
        self.v = np.zeros_like(list(init_p), dtype=float)
        self.epsilon = epsilon

    def update(self):
        self.v += np.square(self.grad)
        for idx, var in enumerate(self.var_dict.values()):
            var.v -= self.lr * var.grad / (np.sqrt(self.v[idx]) + self.epsilon)

class RMSprop(GradDescent):
    def __init__(self, func, init_p, lr=0.01, beta=0.9, epsilon=1e-8):
        super().__init__(func, init_p, lr)
        self.v = np.zeros_like(list(init_p), dtype=float)
        self.beta = beta
        self.epsilon = epsilon
        self.grad = np.empty_like(list(init_p), dtype=float)

    
    def update(self):
        self.v = self.beta * self.v + (1 - self.beta) * np.square(self.grad)
        for idx, var in enumerate(self.var_dict.values()):
            var.v -= self.lr * var.grad / (np.sqrt(self.v[idx]) + self.epsilon)

class Momentum(GradDescent):
    def __init__(self, func, init_p, lr = 0.01, beta=0.9, mode=0):
        super().__init__(func, init_p, lr)
        self.beta = beta
        self.m = np.zeros_like(list(init_p), dtype=float)
        self.grad = np.empty_like(list(init_p), dtype=float)
        self.mode = mode

    def update(self):
        match self.mode:
            case 0 | "heavy_ball":
                self.m = self.beta * self.m - self.lr *  self.grad
                for idx, var in enumerate(self.var_dict.values()):
                    var.v += self.m[idx]
            case 1 | "classic":
                self.m = self.beta * self.m + self.grad
                for idx, var in enumerate(self.var_dict.values()):
                    var.v -= self.lr * self.m[idx]
            case 2 | "emwa":
                self.m = self.beta * self.m + (1 - self.beta) * self.grad
                for idx, var in enumerate(self.var_dict.values()):
                    var.v -= self.lr * self.m[idx]

class Adam(GradDescent):

    def __init__(self, func, init_p, lr = 0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        super().__init__(func, init_p, lr)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = epsilon
        self.m = np.zeros_like(list(init_p), dtype=float)
        self.v = np.zeros_like(list(init_p), dtype=float)
        self.grad = np.empty_like(list(init_p), dtype=float)
        self.t = 0

    def update(self):
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * self.grad
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * np.square(self.grad)
        m_hat = self.m / (1 - np.power(self.beta_1, self.t + 1))
        v_hat = self.v / (1 - np.power(self.beta_2, self.t + 1))
        for idx, var in enumerate(self.var_dict.values()):
            var.v -= self.lr * m_hat[idx] / (np.sqrt(v_hat[idx]) + self.eps)
        self.t += 1
            
class Nesterov(Momentum):
    def __init__(self, func, init_p, lr=0.01, beta=0.9):
        super().__init__(func, init_p, lr, beta)
        self.lookahead_var, self.lookahead_var_dict = self.func(**init_p)
        
    def nesterov(self):
        for idx, k in enumerate(self.var_dict.keys()):
            self.lookahead_var_dict[k].v = self.var_dict[k].v + self.beta * self.m[idx]
        self.lookahead_var.forward()
        self.lookahead_var.backward()
        for idx, var in enumerate(self.lookahead_var_dict.values()):
            self.grad[idx] = var.grad

    def update(self):
        self.m = self.beta * self.m - self.lr * self.grad
        for idx, var in enumerate(self.var_dict.values()):
            var.v += self.m[idx]

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