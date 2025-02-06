let history = [];
let currentStep = 0;
let animationInterval;
let currentFrame = 0;
let isPaused = false;
let inputsChanged = false;
let lastOptimizationParams = null;
let animationSpeed = 1;
let isDragging = false;
const SPEED_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 5096];

const GRADIENT_OPTIMIZERS = new Set([
    'gradient_descent', 'signsgd', 'adadelta', 'adagrad', 'rmsprop',
    'momentum', 'nesterov', 'adam', 'nadam', 'adabelief', 'adamw',
    'lion', 'tiger'
]);

const META_HEURISTIC_OPTIMIZERS = new Set([
    'differential_evolution', 'harmony_search', 'particle_swarm'
]);

function evaluateFunction(x, y, equation) {
    const expr = math.compile(equation);
    return expr.evaluate({ x: x, y: y });
}

function generateSurfaceData() {
    const equation = document.getElementById('equation').value;
    const xMin = parseFloat(document.getElementById('xMin').value);
    const xMax = parseFloat(document.getElementById('xMax').value);
    const yMin = parseFloat(document.getElementById('yMin').value);
    const yMax = parseFloat(document.getElementById('yMax').value);
    const xValues = math.range(xMin, xMax, (xMax - xMin) / 100, true).toArray();
    const yValues = math.range(yMin, yMax, (yMax - yMin) / 100, true).toArray();
    const z = [];

    for (let i = 0; i < yValues.length; i++) {
        const row = [];
        for (let j = 0; j < xValues.length; j++) {
            row.push(evaluateFunction(xValues[j], yValues[i], equation));
        }
        z.push(row);
    }

    return { x: xValues, y: yValues, z: z };
}

function updateOptimizerControls() {

    const optimizer = document.getElementById('optimizer').value;
    const gradientParams = document.querySelector('.gradient-params');
    const metaHeuristicParams = document.querySelector('.meta-heuristic-params');
    // Hide all parameter groups first, this must be above the gradient and heuristic params
    // since the class name contains substring param.
    document.querySelectorAll('[class*="-param"]').forEach(el => {
        el.style.display = 'none';
    });

    // Show/hide main parameter groups
    if (GRADIENT_OPTIMIZERS.has(optimizer)) {
        gradientParams.style.display = 'block';
        metaHeuristicParams.style.display = 'none';
    } else {
        gradientParams.style.display = 'none';
        metaHeuristicParams.style.display = 'block';
    }


    // Show relevant parameters based on optimizer
    switch (optimizer) {
        case 'differential_evolution':
            document.querySelectorAll('.de-param').forEach(el => {
                el.style.display = 'block';
            });
            break;
        case "particle_swarm":
            document.querySelectorAll('.pso-param').forEach(el => {
                el.style.display = 'block';
            });
            break;
        case "harmony_search":
            document.querySelectorAll('.hs-param').forEach(el => {
                el.style.display = 'block';
            });
            break;
        case 'momentum':
            document.querySelectorAll('.momentum-param').forEach(el => {
                el.style.display = 'block';
            });
            break;
        case 'nesterov':
            document.querySelectorAll('.nesterov-param').forEach(el => {
                el.style.display = 'block';
            });
            break;
        case 'adam':
        case 'nadam':
        case 'adabelief':
            document.querySelectorAll('.adam-param').forEach(el => {
                el.style.display = 'block';
            });
            break;
        case 'adamw':
            document.querySelectorAll('.adam-param, .weight-decay-param').forEach(el => {
                el.style.display = 'block';
            });
            break;
        case 'lion':
            document.querySelectorAll('.lion-param').forEach(el => {
                el.style.display = 'block';
            });
            break;
        case 'tiger':
            document.querySelectorAll('.tiger-param').forEach(el => {
                el.style.display = 'block';
            });
            break;
        case 'rmsprop':
            document.querySelectorAll('.rmsprop-param').forEach(el => {
                el.style.display = 'block';
            });
            break;
        case 'adadelta':
            document.querySelectorAll('.adadelta-param').forEach(el => {
                el.style.display = 'block';
            });
            break;
    }
}

async function fetchData() {
    const optimizer = document.getElementById('optimizer').value;
    const hyperparameters = {};

    // Add relevant hyperparameters based on optimizer
    switch (optimizer) {
        case 'differential_evolution':
            hyperparameters.mut_1 = parseFloat(document.getElementById('mut-1').value);
            hyperparameters.mut_2 = parseFloat(document.getElementById('mut-2').value);
            hyperparameters.cross_p = parseFloat(document.getElementById('cross-p').value);
            break;
        case 'particle_swarm':
            hyperparameters.inertia = parseFloat(document.getElementById('inertia').value);
            hyperparameters.cognitive = parseFloat(document.getElementById('cognitive').value);
            hyperparameters.social = parseFloat(document.getElementById('social').value);
            break;
        case 'harmony_search':
            hyperparameters.hmcr = parseFloat(document.getElementById('hmcr').value);
            hyperparameters.par = parseFloat(document.getElementById('par').value);
            hyperparameters.bw = parseFloat(document.getElementById('bw').value);
            break;
        case 'momentum':
            hyperparameters.beta = parseFloat(document.getElementById('momentum-beta').value);
            hyperparameters.mode = document.getElementById('momentum-mode').value;
            break;
        case 'nesterov':
            hyperparameters.beta = parseFloat(document.getElementById('nesterov-beta').value);
            break;
        case 'adam':
        case 'nadam':
        case 'adabelief':
            hyperparameters.beta1 = parseFloat(document.getElementById('adam-beta1').value);
            hyperparameters.beta2 = parseFloat(document.getElementById('adam-beta2').value);
            hyperparameters.epsilon = parseFloat(document.getElementById('adam-epsilon').value);
            break;
        case 'adamw':
            hyperparameters.beta1 = parseFloat(document.getElementById('adam-beta1').value);
            hyperparameters.beta2 = parseFloat(document.getElementById('adam-beta2').value);
            hyperparameters.epsilon = parseFloat(document.getElementById('adam-epsilon').value);
            hyperparameters.weight_decay = parseFloat(document.getElementById('weight-decay').value);
            break;
        case 'lion':
            hyperparameters.beta1 = parseFloat(document.getElementById('lion-beta1').value);
            hyperparameters.beta2 = parseFloat(document.getElementById('lion-beta2').value);
            hyperparameters.weight_decay = parseFloat(document.getElementById('lion-weight-decay').value);
            break;
        case 'tiger':
            hyperparameters.beta = parseFloat(document.getElementById('tiger-beta').value);
            hyperparameters.weight_decay = parseFloat(document.getElementById('tiger-weight-decay').value);
            break;
        case 'rmsprop':
            hyperparameters.beta = parseFloat(document.getElementById('rmsprop-beta').value);
            break;
        case 'adadelta':
            hyperparameters.beta = parseFloat(document.getElementById('adadelta-beta').value);
            break;
    }




    // Add hyperparameters to the request body
    const requestBody = {
        equation: document.getElementById('equation').value,
        optimizer: optimizer,
        steps: parseInt(document.getElementById('steps').value),
        xMin: parseFloat(document.getElementById('xMin').value),
        xMax: parseFloat(document.getElementById('xMax').value),
        yMin: parseFloat(document.getElementById('yMin').value),
        yMax: parseFloat(document.getElementById('yMax').value),
        hyperparameters: hyperparameters
    };

    if (GRADIENT_OPTIMIZERS.has(optimizer)) {
        requestBody.startX = parseFloat(document.getElementById('startX').value);
        requestBody.startY = parseFloat(document.getElementById('startY').value);
        requestBody.learningRate = parseFloat(document.getElementById('learningRate').value);
    } else {
        requestBody.population_size = parseInt(document.getElementById('population-size').value);
        requestBody.ttl = parseInt(document.getElementById('ttl').value);

    }
    updateStatus('running');

    try {
        var url = 'https://optviz-226.up.railway.app/';
        // var url = "http://127.0.0.1:5000/";
        if (GRADIENT_OPTIMIZERS.has(optimizer)) {
            url += 'gd';
        } else {
            url += 'mh';
        }
        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });
        if (response.ok) {
            history = await response.json();
            currentStep = 0;
            inputsChanged = false;
            initializePlots();
        }
    } catch (error) {
        console.error('Error:', error);
        updateStatus('error');
    }
}

function updateStatus(state) {
    const statusEl = document.querySelector('.status');
    const progressBar = document.querySelector('.progress-bar-fill');

    statusEl.className = 'status ' + state;

    switch (state) {
        case 'running':
            statusEl.textContent = 'Animation in progress...';
            break;
        case 'paused':
            statusEl.textContent = 'Animation paused';
            break;
        case 'completed':
            statusEl.textContent = 'Animation completed!';
            progressBar.style.width = '100%';
            break;
        case 'error':
            statusEl.textContent = 'Error occurred';
            break;
        default:
            statusEl.textContent = 'Ready to start';
            progressBar.style.width = '0%';
    }
}

function initializePlots() {
    const { x, y, z } = generateSurfaceData();
    const xMin = parseFloat(document.getElementById('xMin').value);
    const xMax = parseFloat(document.getElementById('xMax').value);
    const yMin = parseFloat(document.getElementById('yMin').value);
    const yMax = parseFloat(document.getElementById('yMax').value);
    const optimizer = document.getElementById('optimizer').value;

    // Base surface and contour configurations
    const surface = {
        type: 'surface',
        x: x,
        y: y,
        z: z,
        colorscale: 'Rainbow',
        showscale: false
    };

    const contour = {
        type: 'contour',
        x: x,
        y: y,
        z: z,
        colorscale: 'Rainbow',
        contours: {
            coloring: 'heatmap'
        },
        showscale: false
    };

    // Add initial traces based on optimizer type
    let initial3DTraces = [surface];
    let initialContourTraces = [contour];


    if (META_HEURISTIC_OPTIMIZERS.has(optimizer)) {
        initial3DTraces.push(
            {
                type: 'scatter3d',
                mode: 'markers',
                x: history.population_history[0].map(p => p[0]),
                y: history.population_history[0].map(p => p[1]),
                z: history.fitness_history[0],
                marker: { size: 8, color: 'yellow', },
                name: 'Population'
            },

            {
                type: 'scatter3d',
                mode: 'markers',
                x: [history.best_history[0][0]],
                y: [history.best_history[0][1]],
                z: [history.best_history[0][2]],
                marker: { size: 12, color: 'red', symbol: 'square' },
                name: 'Best Point'
            }
        );

        initialContourTraces.push(
            {
                type: 'scatter',
                mode: 'markers',
                x: history.population_history[0].map(p => p[0]),
                y: history.population_history[0].map(p => p[1]),
                marker: { size: 8, color: 'yellow', },
                name: 'Population'
            },
            {
                type: 'scatter',
                mode: 'markers',
                x: [history.best_history[0][0]],
                y: [history.best_history[0][1]],
                marker: { size: 12, color: 'red', symbol: 'star' },
                name: 'Best Point'
            }
        );
    } else {
    //     // Initial path trace
        initial3DTraces.push(
            {
                type: 'scatter3d',
                mode: 'lines+markers',
                x: [history.coords[0][0]],
                y: [history.coords[0][1]],
                z: [history.evaluated[0]],
                line: { color: 'yellow', width: 3 },
                marker: { size: 8, color: 'yellow', },
                name: 'Path'
            },
            {
                type: 'scatter3d',
                mode: 'markers',
                x: [history.coords[0][0]],
                y: [history.coords[0][1]],
                z: [history.evaluated[0]],
                marker: { size: 12, color: 'red', symbol: 'square' },
                name: 'Latest Point'
            },
            {
                type: 'scatter3d',
                mode: 'markers',
                x: [history.coords[0][0]],
                y: [history.coords[0][1]],
                z: [history.evaluated[0]],
                marker: { size: 12, color: 'orange', symbol: 'diamond' },
                name: 'Current Best Point'
            }
        );

        initialContourTraces.push(
            {
                type: 'scatter',
                mode: 'lines+markers',
                x: [history.coords[0][0]],
                y: [history.coords[0][1]],
                line: { color: 'yellow', width: 3 },
                marker: { size: 8, color: 'yellow', },
                name: 'Path'
            },
            {
                type: 'scatter',
                mode: 'markers',
                x: [history.coords[0][0]],
                y: [history.coords[0][1]],
                marker: { size: 12, color: 'red', symbol: 'star' },
                name: 'Current Best Point'
            },
            {
                type: 'scatter',
                mode: 'markers',
                x: [history.coords[0][0]],
                y: [history.coords[0][1]],
                marker: { size: 12, color: 'orange', symbol: 'diamond' },
                name: 'Latest Point'
            }
        );
    }

    // Create plots with initial traces
    Plotly.newPlot('surface-plot', initial3DTraces, {
        title: '$' + math.parse(document.getElementById('equation').value.toLowerCase()).toTex() + '$',
        autosize: true,
        scene: {
            camera: {
                up: { x: 0, y: 0, z: 1 },
                center: { x: 0, y: 0, z: 0 },
                eye: { x: 1.5, y: 1.5, z: 1.5 }
            },
            xaxis: { range: [xMin, xMax], title: 'X' },
            yaxis: { range: [yMin, yMax], title: 'Y' },
            zaxis: { title: 'Value' }
        },
        margin: { l: 0, r: 0, t: 30, b: 0 },
        showlegend: true
    });

    Plotly.newPlot('contour-plot', initialContourTraces, {
        title: 'Contour Plot',
        xaxis: { range: [xMin, xMax], title: 'X' },
        yaxis: { range: [yMin, yMax], title: 'Y' },
        margin: { l: 50, r: 50, t: 30, b: 50 },
        showlegend: true
    });

    if (history && Object.keys(history).length > 0) {
        startAnimation();
    }
}

function animate() {
    const optimizer = document.getElementById('optimizer').value;
    const isMetaHeuristic = META_HEURISTIC_OPTIMIZERS.has(optimizer);
    const maxSteps = isMetaHeuristic ? history.population_history.length : history.coords.length;

    if (currentStep >= maxSteps) {
        clearInterval(animationInterval);
        updateStatus('completed');
        return;
    }

    if (!isDragging) {
        const progress = (currentStep / maxSteps) * 100;
        document.querySelector('.progress-bar-fill').style.width = `${progress}%`;
    }
    const stepsToAdvance = Math.min(animationSpeed, maxSteps - currentStep);
    currentStep += stepsToAdvance;

    if (isMetaHeuristic) {
        // Meta-heuristic visualization
        const currentPopulation = history.population_history[currentStep - 1];
        const currentBest = history.best_history[currentStep - 1];
        const initialBest = history.best_history[0];

        Plotly.update('surface-plot', {
            x: [currentPopulation.map(p => p[0])],
            y: [currentPopulation.map(p => p[1])],
            z: [history.fitness_history[currentStep - 1]],
        }, {}, [1]);  

        Plotly.update('contour-plot', {
            x: [currentPopulation.map(p => p[0])],
            y: [currentPopulation.map(p => p[1])]
        }, {}, [1]);  

        Plotly.update('surface-plot', {
            x: [[currentBest[0]]],
            y: [[currentBest[1]]],
            z: [[currentBest[2]]],
            marker: { size: 12, color: 'red', symbol: 'square'}
        }, {}, [2]);

        Plotly.update('contour-plot', {
            x: [[currentBest[0]]],
            y: [[currentBest[1]]],
            marker: { size: 12, color: 'red', symbol: 'star' }
        }, {}, [2]);  

        updateInfo(
            initialBest[0], initialBest[1], initialBest[2],
            currentBest[0], currentBest[1], currentBest[2], 
            currentStep);
    } else {
        // Gradient-based visualization
        const currentCoords = history.coords.slice(0, currentStep);
        const currentValues = history.evaluated.slice(0, currentStep);
        const latestPoint = currentCoords[currentCoords.length - 1];
        const latestValue = currentValues[currentValues.length - 1];

        // Find the current best point (minimum value in the current array)
        let minIndex = 0;
        for (let i = 1; i < currentValues.length; i++) {
            if (currentValues[i] < currentValues[minIndex]) {
                minIndex = i;
            }
        }
        const currentBestPoint = currentCoords[minIndex];
        const currentBestValue = currentValues[minIndex];

        // Update trace 1, the path
        Plotly.update('surface-plot', {
            x: [currentCoords.map(c => c[0])],
            y: [currentCoords.map(c => c[1])],
            z: [currentValues]
        }, {}, [1]);  

        Plotly.update('contour-plot', {
            x: [currentCoords.map(c => c[0])],
            y: [currentCoords.map(c => c[1])]
        }, {}, [1]);  

        // Update trace 2, the current best point
        Plotly.update('surface-plot', {
            x: [[currentBestPoint[0]]],
            y: [[currentBestPoint[1]]],
            z: [[currentBestValue]],
            marker: { size: 12, color: 'orange', symbol: 'diamond' }
        }, {}, [3]);  

        Plotly.update('contour-plot', {
            x: [[currentBestPoint[0]]],
            y: [[currentBestPoint[1]]],
            marker: { size: 12, color: 'orange', symbol: 'diamond' }
        }, {}, [3]);  

        // Update trace 3, the lastest point
        Plotly.update('surface-plot', {
            x: [[latestPoint[0]]],
            y: [[latestPoint[1]]],
            z: [[latestValue]],
            marker: { size: 12, color: 'red', symbol: 'square', }
        }, {}, [3]);  

        Plotly.update('contour-plot', {
            x: [[latestPoint[0]]],
            y: [[latestPoint[1]]],
            marker: { size: 12, color: 'red', symbol: 'star' }
        }, {}, [3]);  

        updateInfo(
            latestPoint[0],
            latestPoint[1],
            latestValue,
            currentBestPoint[0], currentBestPoint[1], currentBestValue,
            currentStep - 1
        );
    }
}

function startAnimation() {
    clearInterval(animationInterval);
    const baseInterval = 100; // Base interval in milliseconds
    animationInterval = setInterval(animate, baseInterval / animationSpeed);
}

function updateAnimationSpeed(speed) {
    animationSpeed = speed;
    if (!isPaused && animationInterval) {
        startAnimation();
    }

    // Update button styles
    document.querySelectorAll('.speed-button').forEach(button => {
        if (parseInt(button.dataset.speed) === speed) {
            button.classList.add('active');
        } else {
            button.classList.remove('active');
        }
    });
}

function updateInfo(x, y, value, best_x, best_y, best_f, currentStep) {
    document.getElementById('iteration').textContent = currentStep;
    document.getElementById('coordinates').textContent = `(${x.toFixed(4)}, ${y.toFixed(4)})`;
    document.getElementById('function-value').textContent = value.toFixed(4);
    document.getElementById('best-coordinates').textContent = `(${best_x.toFixed(4)}, ${best_y.toFixed(4)})`;
    document.getElementById('best-function-value').textContent = best_f.toFixed(4);
}

function updateProgressBar(e) {
    const progressBar = document.querySelector('.progress-bar');
    const rect = progressBar.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const width = rect.width;
    const progress = Math.min(Math.max((x / width) * 100, 0), 100);

    const optimizer = document.getElementById('optimizer').value;
    const isMetaHeuristic = META_HEURISTIC_OPTIMIZERS.has(optimizer);
    const maxSteps = isMetaHeuristic ? history.population_history.length : history.coords.length;

    // Calculate new step based on progress
    currentStep = Math.floor((progress / 100) * maxSteps);

    // Update progress bar visual
    document.querySelector('.progress-bar-fill').style.width = `${progress}%`;

    // Trigger animation frame
    animate();
}

document.addEventListener('DOMContentLoaded', () => {
    const progressBar = document.querySelector('.progress-bar');
    isDragging = false;

    progressBar.addEventListener('mousedown', (e) => {
        isDragging = true;
        updateProgressBar(e);
    });

    document.addEventListener('mousemove', (e) => {
        if (isDragging) {
            updateProgressBar(e);
        }
    });

    document.addEventListener('mouseup', () => {
        isDragging = false;
    });
    
    // Modified input change handlers
    const inputs = document.querySelectorAll('input, select');

    inputs.forEach(input => {
        input.addEventListener('change', () => {
            inputsChanged = true;
            const startButton = document.getElementById('start');
            startButton.textContent = 'Start';
            startButton.classList.remove('resume');
            updateStatus('ready');
        });
    });


    document.getElementById('optimizer').addEventListener('change', updateOptimizerControls);



    document.getElementById('speed-slider').addEventListener('input', (e) => {
        const speedIndex = parseInt(e.target.value);
        const speed = SPEED_VALUES[speedIndex];
        document.getElementById('speed-value').textContent = speed + 'x';
        updateAnimationSpeed(speed);
    });

    // Start button
    document.getElementById('start').addEventListener('click', () => {
        const startButton = document.getElementById('start');

        if (isPaused && !inputsChanged) {
            // Resume animation
            startAnimation();
            startButton.textContent = 'Start';
            startButton.classList.remove('resume');
            isPaused = false;
            updateStatus('running');
        } else {
            // Start new animation
            clearInterval(animationInterval);
            fetchData().then(() => {
                startButton.textContent = 'Start';
                startButton.classList.remove('resume');
                isPaused = false;
            });
        }
    });

    // Pause button
    document.getElementById('pause').addEventListener('click', () => {
        clearInterval(animationInterval);
        isPaused = true;
        const startButton = document.getElementById('start');
        startButton.textContent = 'Resume';
        startButton.classList.add('resume');
        updateStatus('paused');
    });

    // Reset button
    document.getElementById('reset').addEventListener('click', () => {
        clearInterval(animationInterval);
        currentStep = 0;
        currentFrame = 0;
        history = [];
        isPaused = false;
        inputsChanged = false;
        const startButton = document.getElementById('start');
        startButton.textContent = 'Start';
        startButton.classList.remove('resume');
        updateStatus('ready');
        initializePlots(true);
    });
    

});


// Initial plot


