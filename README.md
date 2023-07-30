# Automatic Differentiation Library (Reverse Mode)
## Introduction
The purpose of this project is to implement an Automatic Differentiation (AD) library using the reverse mode, a powerful technique that efficiently computes gradients of functions. The motivation behind this implementation is to address the limitations of numeric difference methods, which can introduce errors in gradient calculations. The AD library enables us to accurately and efficiently compute gradients, making it suitable for optimization algorithms like gradient descent.

## Features
- Reverse Mode Automatic Differentiation: The library implements reverse mode AD, a technique that efficiently computes gradients for a given mathematical function.
- Math Expression Parser: The library provides two parsers to read mathematical expressions in string literals and transform them into functions compatible with automatic differentiation. One parser is based on Abstract Syntax Tree (AST), and the other one uses the Shunting Yard Algorithm and Postfix Evaluation.
- Optimization Algorithms: Apply the automatic differentiation library to implement various optimization algorithms, including:
    - Vanilla Gradient Descent
    - Adam Optimizer
    - RMSprop Optimizer
    - Momentum-based Gradient Descent
    - BFGS Algorithm

## Implementation Details
### Reverse Mode Automatic Differentiation
I define a class called Var to represent variables or constant and their gradients during the forward and backward passes. The core of the library involves overloading arithmetic operators (+, -, *, /, **) to support the computation of gradients during automatic differentiation.
At the same time, common used math function (Var version) such as sin, cos, tan are also being implemented. The library also provides a function "draw_dag" to visualize the
relationship between the gradient of the variables etc. (Graphviz need to be installed)

### Math Expression Parsers
- AST-based
- Shunting Yard Algorithm and Postfix Evaluation-based 
Both parsers are designed to tokenize the mathematical expression, breaking it down into individual components such as math functions, math constants, operators, parentheses, and variables (a-zA-Z). Once the expression is tokenized, both parsers generate two Python functions: one for automatic differentiation and another for normal mathematical calculations.

The automatic differentiation function leverages the reverse mode of automatic differentiation, enabling efficient computation of gradients for the given mathematical expression. This is particularly beneficial for optimization algorithms, as it accurately calculates gradients without relying on numerical difference methods, which can introduce errors.

On the other hand, the second Python function produced by the parser performs standard mathematical calculations for the given expression. This function is suitable for evaluating the expression and obtaining the final result without the need for gradient information.

By providing both automatic differentiation and standard calculation functions, the parsers offer a comprehensive solution for handling mathematical expressions. Users can easily choose between the two functions depending on their specific requirements, whether it's optimizing the function using gradient-based algorithms or simply obtaining the result of the expression.

In summary, these parsers efficiently tokenize and transform mathematical expressions into Python functions, empowering users to seamlessly incorporate automatic differentiation capabilities into their optimization algorithms and perform accurate mathematical computations.
Optimization Algorithms
We leverage the automatic differentiation library to implement various optimization algorithms, making it easier for users to optimize their custom mathematical functions. The implemented optimization algorithms include popular methods like vanilla gradient descent, Adam optimizer, RMSprop optimizer, momentum-based gradient descent, and the BFGS algorithm.

### Optimization Algorithms
Leverage the automatic differentiation library to implement various optimization algorithms, making it easier for users to optimize their custom mathematical functions. The implemented optimization algorithms include popular methods like vanilla gradient descent, momentum-based gradient descent, RMSprop optimizer, Adam optimizer and the BFGS algorithm.

### Structure of the repository
- Var.py
- VarMath.py
- graDescending.py
- main.py

### Examples

