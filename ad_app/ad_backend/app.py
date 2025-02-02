from flask import Flask, jsonify, request
from ad.parser import SYParser
from ad.grad import GradDescent, AdaGrad, RMSprop, Momentum, Adam, Nesterov, SignSGD, AdaDelta, NAdam, AdaBelief, AdamW, Lion, Tiger
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={
    r"/*": {  # Allow all routes
        "origins": ["https://automaticdifferentiation-frontend.up.railway.app"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})
parser = SYParser()

@app.route('/', methods=['POST'])
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
        beta_1 = hyperparameters.get('beta', 0.945)
        weight_decay = hyperparameters.get('weight_decay', 0.01)
        optimizer = Tiger(func, init_p, lr=learning_rate, beta=beta, weight_decay=weight_decay)
    else:
        return jsonify({'error': 'Invalid optimizer'}), 400

    # Perform gradient descent steps
    for _ in range(steps):
        optimizer.step()

    # Return the results
    return jsonify({
        'coords': optimizer.coords.tolist(),  # List of coordinates
        'evaluated': optimizer.evaluated.tolist()  # List of function values
    })

if __name__ == '__main__':
    app.run()