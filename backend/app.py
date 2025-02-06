from flask import Flask, jsonify, request
from ad.parser import Parser
from ad.grad import GradDescent, AdaGrad, RMSprop, Momentum, Adam, Nesterov, SignSGD, AdaDelta, NAdam, AdaBelief, AdamW, Lion, Tiger
from flask_cors import CORS
import numpy as np
from heu.algo import DE, PSO, HS

app = Flask(__name__)
CORS(app, resources={
    r"/*": {  # Allow all routes
        "origins": ["https://optviz.up.railway.app"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})
# CORS(app)
parser = Parser()

@app.route('/gd', methods=['POST'])
def gradient_descent():
    data = request.json
    print(data)
    equation = data.get('equation')
    startX = float(data.get('startX'))
    startY = float(data.get('startY'))
    learning_rate = float(data.get('learningRate'))
    steps = int(data.get('steps'))
    optimizer = data.get('optimizer')
    hyperparameters = data.get('hyperparameters', {})

    # Parse the equation into a callable function
    func, f = parser.exp2func(equation)

    # Initialize the optimizer
    init_p = {'x': startX, 'y': startY}
    if optimizer == 'gradient_descent':
        optimizer = GradDescent(func, init_p, lr=learning_rate)
    elif optimizer == 'signsgd':
        optimizer = SignSGD(func, init_p, lr=learning_rate)
    elif optimizer == 'adadelta':
        beta = hyperparameters.get('beta', 0.95)
        optimizer = AdaDelta(func, init_p, lr=learning_rate, beta=beta)
    elif optimizer == 'adagrad':
        optimizer = AdaGrad(func, init_p, lr=learning_rate)
    elif optimizer == 'rmsprop':
        beta = hyperparameters.get('beta', 0.9)
        optimizer = RMSprop(func, init_p, lr=learning_rate, beta=beta)
    elif optimizer == 'momentum':
        beta = hyperparameters.get('beta', 0.9)
        mode = hyperparameters.get('mode', 'heavy_ball')
        optimizer = Momentum(func, init_p, lr=learning_rate, beta=beta, mode=mode)
    elif optimizer == 'nesterov':
        beta = hyperparameters.get('beta', 0.9)
        optimizer = Nesterov(func, init_p, lr=learning_rate, beta=beta)
    elif optimizer == 'adam':
        beta_1 = hyperparameters.get('beta1', 0.9)
        beta_2 = hyperparameters.get('beta2', 0.999)
        epsilon = hyperparameters.get('epsilon', 1e-8)
        optimizer = Adam(func, init_p, lr=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
    elif optimizer == 'nadam':
        beta_1 = hyperparameters.get('beta1', 0.9)
        beta_2 = hyperparameters.get('beta2', 0.999)
        epsilon = hyperparameters.get('epsilon', 1e-8)
        optimizer = NAdam(func, init_p, lr=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
    elif optimizer == 'adabelief':
        beta_1 = hyperparameters.get('beta1', 0.9)
        beta_2 = hyperparameters.get('beta2', 0.999)
        epsilon = hyperparameters.get('epsilon', 1e-8)
        optimizer = AdaBelief(func, init_p, lr=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
    elif optimizer == 'adamw':
        beta_1 = hyperparameters.get('beta1', 0.9)
        beta_2 = hyperparameters.get('beta2', 0.999)
        epsilon = hyperparameters.get('epsilon', 1e-8)
        weight_decay = hyperparameters.get('weight_decay', 0.01)
        optimizer = AdamW(func, init_p, lr=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, weight_decay=weight_decay)
    elif optimizer == 'lion':
        beta_1 = hyperparameters.get('beta1', 0.9)
        beta_2 = hyperparameters.get('beta2', 0.999)
        weight_decay = hyperparameters.get('weight_decay', 0.01)
        optimizer = Lion(func, init_p, lr=learning_rate, beta_1=beta_1, beta_2=beta_2, weight_decay=weight_decay)
    elif optimizer == 'tiger':
        beta = hyperparameters.get('beta', 0.945)
        weight_decay = hyperparameters.get('weight_decay', 0.01)
        optimizer = Tiger(func, init_p, lr=learning_rate, beta=beta, weight_decay=weight_decay)
    else:
        return jsonify({'error': 'Invalid optimizer'}), 400

    # Perform gradient descent steps
    for _ in range(steps):
        optimizer.step()

    # Return the results
    return jsonify({
        'coords': optimizer.coords.tolist(),  
        'evaluated': optimizer.evaluated.tolist(),
    })

@app.route('/mh', methods=['POST'])
def meta_heuristic():
    data = request.json
    print(data)
    equation = data.get('equation')
    optimizer = data.get('optimizer', 'differential_evolution')
    pop_size = int(data.get('population_size', 20))
    steps = int(data.get('steps', 100))
    x_min, x_max, y_min, y_max = data.get('xMin'), data.get('xMax'), data.get('yMin'), data.get('yMax')

    bounds = np.array([[x_min, x_max], [y_min, y_max]])
    ttl = int(data.get('ttl', 10))
    
    # Parse equation
    func, f = parser.exp2func(equation)
    
    var_dict = {'x': 0, 'y': 0}

    if optimizer == 'differential_evolution':
        mut_1 = float(data.get('mut_1', 0.9))
        mut_2 = float(data.get('mut_2', 0.9))
        cross_p = float(data.get('cross_p', 0.95))
        optimizer = DE(f, var_dict, bounds, pop_size, ttl, mut_1, mut_2, cross_p)
    elif optimizer == 'harmony_search':
        HMCR = float(data.get('hmcr', 0.7))
        PAR = float(data.get('pcr', 0.3))
        BW = data.get('bw', None)
        if BW is not None: BW = float(BW)
        optimizer = HS(f, var_dict, bounds, pop_size, ttl, HMCR, PAR, BW)
    elif optimizer == 'particle_swarm':
        inertia = float(data.get('inertia', 0.5))
        cognitive = float(data.get('cognitive', 1.5))
        social = float(data.get('social', 1.5))
        optimizer = PSO(f, var_dict, bounds, pop_size, ttl, inertia, cognitive, social)
    else:
        return jsonify({'error': 'Invalid algorithm'}), 400

    # Storage for population history
    population_history = []
    fitness_history = []
    best_history = []

    # Perform optimization steps
    for _ in range(steps):
        optimizer.step()
        population_history.append(optimizer.pop.tolist())
        fitness_history.append(optimizer.fitness.tolist())
        best_history.append([float(optimizer.optimal_coords[0]), 
                        float(optimizer.optimal_coords[1]), 
                        float(optimizer.optimal_evaluated)])

    return jsonify({
        'population_history': population_history,
        'fitness_history': fitness_history,
        'best_history': best_history,
    })

if __name__ == '__main__':
    app.run()