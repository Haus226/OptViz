
from Var import Var
import numpy as np



def log(x: Var):
    return Var(np.log(x.v), [[x, 1.0 / x.v]], type="func", info="ln", exp="log(" + x.exp + ")")


def exp(x: Var):
    return Var(np.exp(x.v), [[x, np.exp(x.v)]], type="func", info="exp", exp="exp(" + x.exp + ")")


def sin(x: Var):
    return Var(np.sin(x.v), [[x, np.cos(x.v)]], type="func", info="sin", exp="sin(" + x.exp + ")")


def cos(x: Var):
    return Var(np.cos(x.v), [[x, -np.sin(x.v)]], type="func", info="cos", exp="cos(" + x.exp + ")")


def tan(x: Var):
    return Var(np.tan(x.v), [[x, (1.0 / np.cos(x.v)) ** 2]], type="func", info="tan", exp="tan(" + x.exp + ")")


def sec(x: Var):
    return Var(1.0 / np.cos(x.v), [[x, (1.0 / np.cos(x.v)) * np.tan(x.v)]], type="func", info="sec", exp="sec(" + x.exp + ")")


def cosec(x: Var):
    return Var(1.0 / np.sin(x.v), [[x, -(1.0 / np.sin(x.v) ** 2) * np.cos(x.v)]], type="func", info="cosec", exp="cosec(" + x.exp + ")")


def cot(x: Var):
    return Var(1.0 / np.tan(x.v), [[x, -(1.0 / np.sin(x.v) ** 2)]], type="func", info="cosec", exp="cosec(" + x.exp + ")")


def arccos(x: Var):
    return Var(np.acos(x.v), [[x, -1.0 / np.sqrt(1 - x.v ** 2)]], type="func", info="arccos", exp="arccos(" + x.exp + ")")


def arcsin(x: Var):
    return Var(np.asin(x.v), [[x, 1.0 / np.sqrt(1 - x.v ** 2)]], type="func", info="arcsin", exp="arcsin(" + x.exp + ")")


def arctan(x: Var):
    return Var(np.atan(x.v), [[x, 1.0 / (1 + x.v ** 2)]], type="func", info="arctan", exp="arctan(" + x.exp + ")")


def arcsec(x: Var):
    return Var(np.acos(1.0 / x.v), [[x, -1.0 / (abs(x.v) * np.sqrt(x.v ** 2 - 1))]], type="func", info="arcsec", exp="arcsec(" + x.exp + ")")


def arccosec(x: Var):
    return Var(np.asin(1.0 / x.v), [[x, 1.0 / (abs(x.v) * np.sqrt(x.v ** 2 - 1))]], type="func", info="arccosec", exp="arccosec(" + x.exp + ")")


def arcot(x: Var):
    return Var(np.atan(1.0 / x.v), [[x, -1.0 / (1 + x.v ** 2)]], type="func", info="arcot", exp="arcot(" + x.exp + ")")


def sinh(x: Var):
    return Var(np.sinh(x.v), [[x, np.cosh(x.v)]], type="func", info="sinh", exp="sinh(" + x.exp + ")")


def cosh(x: Var):
    return Var(np.cosh(x.v), [[x, np.sinh(x.v)]], type="func", info="cosh", exp="cosh(" + x.exp + ")")


def tanh(x: Var):
    return Var(np.tanh(x.v), [[x, 1.0 / np.cosh(x.v) ** 2]], type="func", info="tanh", exp="tanh(" + x.exp + ")")


def arccosh(x: Var):
    return Var(np.acosh(x.v), [[x, 1.0 / np.sqrt(x.v ** 2 - 1)]], type="func", info="arccosh", exp="arccosh(" + x.exp + ")")


def arcsinh(x: Var):
    return Var(np.asinh(x.v), [[x, 1.0 / np.sqrt(x.v ** 2 + 1)]], type="func", info="arcsinh", exp="arcsinh(" + x.exp + ")")


def arctanh(x: Var):
    return Var(np.atanh(x.v), [[x, 1.0 / (1 - x.v ** 2)]], type="func", info="arctanh", exp="arctanh(" + x.exp + ")")


def cosech(x: Var):
    return Var(1.0 / np.sinh(x.v), [[x, -1.0 / (np.sinh(x.v) ** 2) * np.tanh(x.v)]], type="func", info="cosech", exp="cosech(" + x.exp + ")")


def arccsch(x: Var):
    return Var(np.asinh(1.0 / x.v), [[x, -1.0 / (abs(x.v) * np.sqrt(x.v ** 2 + 1))]], type="func", info="arccsch", exp="arccsch(" + x.exp + ")")


def sech(x: Var):
    return Var(1.0 / np.cosh(x.v), [[x, -np.tanh(x.v) / np.cosh(x.v)]], type="func", info="sech", exp="sech(" + x.exp + ")")


def arcsech(x: Var):
    return Var(np.acosh(1.0 / x.v), [[x, -1.0 / (abs(x.v) * np.sqrt(1 - x.v ** 2))]], type="func", info="arcsech", exp="arcsech(" + x.exp + ")")


def coth(x: Var):
    return Var(1.0 / np.tanh(x.v), [[x, -1.0 / np.sinh(x.v) ** 2]], type="func", info="coth", exp="coth(" + x.exp + ")")


def arcoth(x: Var):
    return Var(0.5 * np.log((x.v + 1) / (x.v - 1)), [[x, 1.0 / (1 - x.v ** 2)]], type="func", info="arcoth", exp="arcoth(" + x.exp + ")")


FUNC_ = {'log': log, 'exp': exp, 'sin': sin, 'cos': cos, 'tan': tan, 'arcsin': arcsin, 'asin': arcsin, 'arccos': arccos, 'acos': arccos, 'arctan': arctan, 'atan': arctan,
    'arccosec': arccosec, 'arccsc': arccosec, 'acsc': arccosec, 'arcsec': arcsec, 'asec': arcsec, 'arcot': arcot, 'acot': arcot, 'sinh': sinh, 'cosh': cosh, 'tanh': tanh,
    'arcsinh': arcsinh, 'asinh': arcsinh, 'arccosh': arccosh, 'acosh': arccosh, 'arctanh': arctanh, 'atanh': arctanh, 'sec': sec, 'cosec': cosec, 'cot': cot, 'sech': sech,
    'cosech': cosech, 'csch': cosech, 'coth': coth, 'arccosech': arccsch, 'arccsch': arccsch, 'acsch': arccsch, 'arcsech': arcsech, 'asech': arcsech, 'arcoth': arcoth,
    'acoth': arcoth, }

PRECEDENCE = {'^': 4, '*': 3, '/': 3, '+': 2, '-': 2, }

CONSTANT_ = {"pi": np.pi, "e": np.e}

