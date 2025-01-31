from flask import Flask, jsonify, request
from ad.parser import SYParser
from ad.grad import GradDescent, AdaGrad, RMSprop, Momentum, Adam, Nesterov
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

@app.route('/gradient-descent', methods=['POST'])
def gradient_descent():
    data = request.json
    print(data)
    equation = data.get('equation')  # User-provided equation
    startX = float(data.get('startX'))  # Starting X value
    startY = float(data.get('startY'))  # Starting Y value
    learning_rate = float(data.get('learningRate'))  # Learning rate
    steps = int(data.get('steps'))  # Number of steps
    optimizer = data.get('optimizer', 'gradient_descent')  # Optimizer type

    # Parse the equation into a callable function
    func, f = parser.exp2func(equation)

    # Initialize the optimizer
    init_p = {'x': startX, 'y': startY}
    print(init_p)
    if optimizer == 'gradient_descent':
        optimizer = GradDescent(func, init_p, lr=learning_rate)
    elif optimizer == 'adagrad':
        optimizer = AdaGrad(func, init_p, lr=learning_rate)
    elif optimizer == 'rmsprop':
        optimizer = RMSprop(func, init_p, lr=learning_rate)
    elif optimizer == 'momentum':
        optimizer = Momentum(func, init_p, lr=learning_rate)
    elif optimizer == 'adam':
        optimizer = Adam(func, init_p, lr=learning_rate)
    elif optimizer == 'nesterov':
        optimizer = Nesterov(func, init_p, lr=learning_rate)
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