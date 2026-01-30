// Simulated Sklearn Digits Dataset (8x8 pixel images)
// Each image is flattened to 64 features
const DIGITS_DATA = generateDigitsDataset();

// Global variables
let trainData = [];
let testData = [];
let trainLabels = [];
let testLabels = [];
let scaledTrainData = [];
let scaledTestData = [];
let scaler = null;
let accuracyResults = {};
let confusionMatrixData = null;
let currentPredictions = [];

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeDataset();
    displaySampleDigits();
    setupEventListeners();
});

// Generate a simulated digits dataset (similar to sklearn digits)
function generateDigitsDataset() {
    const dataset = [];
    const samplesPerDigit = 180; // Total: 1800 samples
    
    for (let digit = 0; digit < 10; digit++) {
        for (let i = 0; i < samplesPerDigit; i++) {
            const image = generateDigitImage(digit);
            dataset.push({
                pixels: image,
                label: digit
            });
        }
    }
    
    // Shuffle dataset
    return shuffleArray(dataset);
}

// Generate a digit image (8x8 grid) with some variation
function generateDigitImage(digit) {
    const basePatterns = {
        0: [0,1,1,1,1,1,1,0, 1,1,1,0,0,1,1,1, 1,1,0,0,0,0,1,1, 1,1,0,0,0,0,1,1, 1,1,0,0,0,0,1,1, 1,1,0,0,0,0,1,1, 1,1,1,0,0,1,1,1, 0,1,1,1,1,1,1,0],
        1: [0,0,0,1,1,0,0,0, 0,0,1,1,1,0,0,0, 0,1,1,1,1,0,0,0, 0,0,0,1,1,0,0,0, 0,0,0,1,1,0,0,0, 0,0,0,1,1,0,0,0, 0,0,0,1,1,0,0,0, 0,1,1,1,1,1,1,0],
        2: [0,1,1,1,1,1,1,0, 1,1,1,0,0,1,1,1, 0,0,0,0,0,1,1,1, 0,0,0,0,1,1,1,0, 0,0,1,1,1,0,0,0, 0,1,1,1,0,0,0,0, 1,1,1,0,0,0,0,0, 1,1,1,1,1,1,1,1],
        3: [0,1,1,1,1,1,1,0, 1,1,1,0,0,1,1,1, 0,0,0,0,0,1,1,1, 0,0,0,1,1,1,1,0, 0,0,0,1,1,1,1,0, 0,0,0,0,0,1,1,1, 1,1,1,0,0,1,1,1, 0,1,1,1,1,1,1,0],
        4: [0,0,0,0,1,1,1,0, 0,0,0,1,1,1,1,0, 0,0,1,1,0,1,1,0, 0,1,1,0,0,1,1,0, 1,1,0,0,0,1,1,0, 1,1,1,1,1,1,1,1, 0,0,0,0,0,1,1,0, 0,0,0,0,0,1,1,0],
        5: [1,1,1,1,1,1,1,1, 1,1,1,0,0,0,0,0, 1,1,1,0,0,0,0,0, 1,1,1,1,1,1,1,0, 0,0,0,0,0,1,1,1, 0,0,0,0,0,1,1,1, 1,1,1,0,0,1,1,1, 0,1,1,1,1,1,1,0],
        6: [0,0,1,1,1,1,1,0, 0,1,1,1,0,0,0,0, 1,1,1,0,0,0,0,0, 1,1,1,1,1,1,1,0, 1,1,1,0,0,1,1,1, 1,1,1,0,0,1,1,1, 1,1,1,0,0,1,1,1, 0,1,1,1,1,1,1,0],
        7: [1,1,1,1,1,1,1,1, 1,1,1,0,0,1,1,1, 0,0,0,0,1,1,1,0, 0,0,0,0,1,1,0,0, 0,0,0,1,1,1,0,0, 0,0,0,1,1,0,0,0, 0,0,1,1,1,0,0,0, 0,0,1,1,0,0,0,0],
        8: [0,1,1,1,1,1,1,0, 1,1,1,0,0,1,1,1, 1,1,1,0,0,1,1,1, 0,1,1,1,1,1,1,0, 0,1,1,1,1,1,1,0, 1,1,1,0,0,1,1,1, 1,1,1,0,0,1,1,1, 0,1,1,1,1,1,1,0],
        9: [0,1,1,1,1,1,1,0, 1,1,1,0,0,1,1,1, 1,1,1,0,0,1,1,1, 0,1,1,1,1,1,1,1, 0,0,0,0,0,1,1,1, 0,0,0,0,0,1,1,1, 0,0,0,0,1,1,1,0, 0,1,1,1,1,1,0,0]
    };
    
    const pattern = [...basePatterns[digit]];
    
    // Add random noise for variation
    for (let i = 0; i < pattern.length; i++) {
        const noise = (Math.random() - 0.5) * 0.3;
        pattern[i] = Math.max(0, Math.min(1, pattern[i] + noise));
        pattern[i] = pattern[i] * 16; // Scale to 0-16 range
    }
    
    return pattern;
}

// Fisher-Yates shuffle
function shuffleArray(array) {
    const shuffled = [...array];
    for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
}

// Initialize and split dataset
function initializeDataset() {
    const splitIndex = Math.floor(DIGITS_DATA.length * 0.8);
    
    const trainSet = DIGITS_DATA.slice(0, splitIndex);
    const testSet = DIGITS_DATA.slice(splitIndex);
    
    trainData = trainSet.map(d => d.pixels);
    trainLabels = trainSet.map(d => d.label);
    testData = testSet.map(d => d.pixels);
    testLabels = testSet.map(d => d.label);
    
    // Update UI
    document.getElementById('trainSize').textContent = `${trainData.length} samples`;
    document.getElementById('testSize').textContent = `${testData.length} samples`;
    
    // Apply feature scaling
    applyStandardScaling();
}

// StandardScaler implementation
function applyStandardScaling() {
    const numFeatures = trainData[0].length;
    const means = new Array(numFeatures).fill(0);
    const stds = new Array(numFeatures).fill(0);
    
    // Calculate means
    for (let i = 0; i < trainData.length; i++) {
        for (let j = 0; j < numFeatures; j++) {
            means[j] += trainData[i][j];
        }
    }
    means.forEach((sum, i) => means[i] = sum / trainData.length);
    
    // Calculate standard deviations
    for (let i = 0; i < trainData.length; i++) {
        for (let j = 0; j < numFeatures; j++) {
            stds[j] += Math.pow(trainData[i][j] - means[j], 2);
        }
    }
    stds.forEach((sum, i) => stds[i] = Math.sqrt(sum / trainData.length) || 1);
    
    scaler = { means, stds };
    
    // Transform train data
    scaledTrainData = trainData.map(sample => 
        sample.map((val, i) => (val - means[i]) / stds[i])
    );
    
    // Transform test data
    scaledTestData = testData.map(sample => 
        sample.map((val, i) => (val - means[i]) / stds[i])
    );
}

// Display sample digit images
function displaySampleDigits() {
    const container = document.getElementById('digitSamples');
    container.innerHTML = '';
    
    // Show 10 random samples
    const samples = [];
    for (let digit = 0; digit < 10; digit++) {
        const indices = trainLabels.map((label, idx) => label === digit ? idx : -1).filter(idx => idx !== -1);
        if (indices.length > 0) {
            const randomIdx = indices[Math.floor(Math.random() * indices.length)];
            samples.push({ pixels: trainData[randomIdx], label: digit });
        }
    }
    
    samples.forEach(sample => {
        const digitDiv = document.createElement('div');
        digitDiv.className = 'digit-item';
        
        const canvas = document.createElement('canvas');
        canvas.className = 'digit-canvas';
        canvas.width = 8;
        canvas.height = 8;
        
        drawDigit(canvas, sample.pixels);
        
        const label = document.createElement('div');
        label.className = 'digit-label';
        label.textContent = `Digit: ${sample.label}`;
        
        digitDiv.appendChild(canvas);
        digitDiv.appendChild(label);
        container.appendChild(digitDiv);
    });
}

// Draw digit on canvas
function drawDigit(canvas, pixels) {
    const ctx = canvas.getContext('2d');
    const size = 8;
    
    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            const idx = i * size + j;
            const intensity = Math.floor((pixels[idx] / 16) * 255);
            ctx.fillStyle = `rgb(${intensity}, ${intensity}, ${intensity})`;
            ctx.fillRect(j, i, 1, 1);
        }
    }
}

// Setup event listeners
function setupEventListeners() {
    document.getElementById('trainBtn').addEventListener('click', trainSingleK);
    document.getElementById('trainAllBtn').addEventListener('click', trainAllKValues);
    document.getElementById('resetBtn').addEventListener('click', resetApplication);
}

// Euclidean distance calculation
function euclideanDistance(point1, point2) {
    let sum = 0;
    for (let i = 0; i < point1.length; i++) {
        sum += Math.pow(point1[i] - point2[i], 2);
    }
    return Math.sqrt(sum);
}

// KNN prediction for a single test point
function knnPredict(testPoint, k) {
    // Calculate distances to all training points
    const distances = scaledTrainData.map((trainPoint, idx) => ({
        distance: euclideanDistance(testPoint, trainPoint),
        label: trainLabels[idx]
    }));
    
    // Sort by distance and get k nearest
    distances.sort((a, b) => a.distance - b.distance);
    const kNearest = distances.slice(0, k);
    
    // Vote: most common label among k nearest
    const votes = {};
    kNearest.forEach(neighbor => {
        votes[neighbor.label] = (votes[neighbor.label] || 0) + 1;
    });
    
    let maxVotes = 0;
    let prediction = 0;
    for (let label in votes) {
        if (votes[label] > maxVotes) {
            maxVotes = votes[label];
            prediction = parseInt(label);
        }
    }
    
    return prediction;
}

// Train KNN with single K value
function trainSingleK() {
    const k = parseInt(document.getElementById('kValue').value);
    document.getElementById('currentK').textContent = k;
    document.getElementById('modelStatus').textContent = 'Training...';
    document.getElementById('modelStatus').classList.add('loading');
    
    disableControls();
    
    setTimeout(() => {
        const startTime = performance.now();
        
        // Make predictions
        const predictions = scaledTestData.map(testPoint => knnPredict(testPoint, k));
        currentPredictions = predictions;
        
        // Calculate accuracy
        let correct = 0;
        for (let i = 0; i < predictions.length; i++) {
            if (predictions[i] === testLabels[i]) correct++;
        }
        const accuracy = (correct / testLabels.length) * 100;
        
        const endTime = performance.now();
        const trainingTime = ((endTime - startTime) / 1000).toFixed(3);
        
        // Store results
        accuracyResults[k] = accuracy;
        
        // Update UI
        document.getElementById('currentAccuracy').textContent = `${accuracy.toFixed(2)}%`;
        document.getElementById('trainingTime').textContent = `${trainingTime}s`;
        document.getElementById('modelStatus').textContent = 'Trained ✓';
        document.getElementById('modelStatus').classList.remove('loading');
        
        // Update best K if applicable
        updateBestK();
        
        // Generate confusion matrix
        generateConfusionMatrix(predictions);
        
        // Display predictions
        displayPredictions(predictions);
        
        // Update chart
        updateAccuracyChart();
        
        enableControls();
    }, 100);
}

// Train all K values
function trainAllKValues() {
    const kValues = [3, 5, 7, 9, 11, 13, 15];
    document.getElementById('modelStatus').textContent = 'Training all K values...';
    document.getElementById('modelStatus').classList.add('loading');
    
    disableControls();
    
    let currentIndex = 0;
    
    function trainNext() {
        if (currentIndex >= kValues.length) {
            document.getElementById('modelStatus').textContent = 'All K values trained ✓';
            document.getElementById('modelStatus').classList.remove('loading');
            updateBestK();
            enableControls();
            return;
        }
        
        const k = kValues[currentIndex];
        const predictions = scaledTestData.map(testPoint => knnPredict(testPoint, k));
        
        let correct = 0;
        for (let i = 0; i < predictions.length; i++) {
            if (predictions[i] === testLabels[i]) correct++;
        }
        const accuracy = (correct / testLabels.length) * 100;
        
        accuracyResults[k] = accuracy;
        
        if (currentIndex === kValues.length - 1) {
            currentPredictions = predictions;
            generateConfusionMatrix(predictions);
            displayPredictions(predictions);
        }
        
        updateAccuracyChart();
        
        currentIndex++;
        setTimeout(trainNext, 200);
    }
    
    trainNext();
}

// Update best K value
function updateBestK() {
    let bestK = null;
    let bestAcc = 0;
    
    for (let k in accuracyResults) {
        if (accuracyResults[k] > bestAcc) {
            bestAcc = accuracyResults[k];
            bestK = k;
        }
    }
    
    if (bestK !== null) {
        document.getElementById('bestK').textContent = bestK;
        document.getElementById('bestAccuracy').textContent = `${bestAcc.toFixed(2)}%`;
    }
}

// Generate confusion matrix
function generateConfusionMatrix(predictions) {
    const matrix = Array(10).fill(0).map(() => Array(10).fill(0));
    
    for (let i = 0; i < predictions.length; i++) {
        matrix[testLabels[i]][predictions[i]]++;
    }
    
    confusionMatrixData = matrix;
    displayConfusionMatrix(matrix);
}

// Display confusion matrix
function displayConfusionMatrix(matrix) {
    const container = document.getElementById('confusionMatrix');
    container.innerHTML = '';
    
    const table = document.createElement('table');
    table.className = 'confusion-matrix';
    
    // Header row
    const headerRow = document.createElement('tr');
    headerRow.innerHTML = '<th>True \\ Pred</th>';
    for (let i = 0; i < 10; i++) {
        headerRow.innerHTML += `<th>${i}</th>`;
    }
    table.appendChild(headerRow);
    
    // Find max value for color scaling
    const maxVal = Math.max(...matrix.map(row => Math.max(...row)));
    
    // Data rows
    for (let i = 0; i < 10; i++) {
        const row = document.createElement('tr');
        row.innerHTML = `<th>${i}</th>`;
        for (let j = 0; j < 10; j++) {
            const cell = document.createElement('td');
            cell.textContent = matrix[i][j];
            const intensity = matrix[i][j] / maxVal;
            const color = `rgba(102, 126, 234, ${intensity})`;
            cell.style.backgroundColor = color;
            if (intensity > 0.5) cell.style.color = 'white';
            row.appendChild(cell);
        }
        table.appendChild(row);
    }
    
    container.appendChild(table);
    
    // Update legend
    document.getElementById('matrixLegend').innerHTML = `
        <div class="legend-gradient"></div>
        <div class="legend-labels">
            <span>Low (0)</span>
            <span>High (${maxVal})</span>
        </div>
    `;
}

// Display predictions
function displayPredictions(predictions) {
    const container = document.getElementById('predictions');
    container.innerHTML = '';
    
    // Show 15 random test samples
    const sampleIndices = [];
    while (sampleIndices.length < Math.min(15, testData.length)) {
        const idx = Math.floor(Math.random() * testData.length);
        if (!sampleIndices.includes(idx)) {
            sampleIndices.push(idx);
        }
    }
    
    sampleIndices.forEach(idx => {
        const predDiv = document.createElement('div');
        predDiv.className = 'prediction-item';
        
        const canvas = document.createElement('canvas');
        canvas.className = 'prediction-canvas';
        canvas.width = 8;
        canvas.height = 8;
        
        drawDigit(canvas, testData[idx]);
        
        const isCorrect = predictions[idx] === testLabels[idx];
        const info = document.createElement('div');
        info.className = 'prediction-info';
        info.innerHTML = `
            <p><strong>True:</strong> ${testLabels[idx]}</p>
            <p><strong>Predicted:</strong> ${predictions[idx]}</p>
            <p class="${isCorrect ? 'correct' : 'incorrect'}">
                ${isCorrect ? '✓ Correct' : '✗ Wrong'}
            </p>
        `;
        
        predDiv.appendChild(canvas);
        predDiv.appendChild(info);
        container.appendChild(predDiv);
    });
}

// Update accuracy chart
function updateAccuracyChart() {
    const canvas = document.getElementById('accuracyChart');
    const ctx = canvas.getContext('2d');
    
    // Set canvas size
    canvas.width = canvas.offsetWidth;
    canvas.height = 400;
    
    const kValues = Object.keys(accuracyResults).map(Number).sort((a, b) => a - b);
    if (kValues.length === 0) return;
    
    const accuracies = kValues.map(k => accuracyResults[k]);
    
    const padding = 60;
    const chartWidth = canvas.width - 2 * padding;
    const chartHeight = canvas.height - 2 * padding;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw axes
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, canvas.height - padding);
    ctx.lineTo(canvas.width - padding, canvas.height - padding);
    ctx.stroke();
    
    // Labels
    ctx.fillStyle = '#333';
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('K Value', canvas.width / 2, canvas.height - 20);
    
    ctx.save();
    ctx.translate(20, canvas.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Accuracy (%)', 0, 0);
    ctx.restore();
    
    // Find min and max accuracy for scaling
    const minAcc = Math.min(...accuracies);
    const maxAcc = Math.max(...accuracies);
    const accRange = maxAcc - minAcc || 1;
    
    // Draw grid lines
    ctx.strokeStyle = '#ddd';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
        const y = padding + (chartHeight / 5) * i;
        ctx.beginPath();
        ctx.moveTo(padding, y);
        ctx.lineTo(canvas.width - padding, y);
        ctx.stroke();
        
        const accValue = maxAcc - (accRange / 5) * i;
        ctx.fillStyle = '#555';
        ctx.textAlign = 'right';
        ctx.fillText(accValue.toFixed(1), padding - 10, y + 5);
    }
    
    // Draw line chart
    ctx.strokeStyle = '#667eea';
    ctx.lineWidth = 3;
    ctx.beginPath();
    
    kValues.forEach((k, idx) => {
        const x = padding + (chartWidth / (kValues.length - 1 || 1)) * idx;
        const y = canvas.height - padding - ((accuracies[idx] - minAcc) / accRange) * chartHeight;
        
        if (idx === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });
    ctx.stroke();
    
    // Draw points
    kValues.forEach((k, idx) => {
        const x = padding + (chartWidth / (kValues.length - 1 || 1)) * idx;
        const y = canvas.height - padding - ((accuracies[idx] - minAcc) / accRange) * chartHeight;
        
        ctx.fillStyle = '#667eea';
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fill();
        
        // Draw K values on x-axis
        ctx.fillStyle = '#555';
        ctx.textAlign = 'center';
        ctx.fillText(k, x, canvas.height - padding + 20);
    });
}

// Disable controls during training
function disableControls() {
    document.getElementById('trainBtn').disabled = true;
    document.getElementById('trainAllBtn').disabled = true;
    document.getElementById('resetBtn').disabled = true;
    document.getElementById('kValue').disabled = true;
}

// Enable controls after training
function enableControls() {
    document.getElementById('trainBtn').disabled = false;
    document.getElementById('trainAllBtn').disabled = false;
    document.getElementById('resetBtn').disabled = false;
    document.getElementById('kValue').disabled = false;
}

// Reset application
function resetApplication() {
    accuracyResults = {};
    confusionMatrixData = null;
    currentPredictions = [];
    
    document.getElementById('currentAccuracy').textContent = '--';
    document.getElementById('bestK').textContent = '--';
    document.getElementById('bestAccuracy').textContent = '--';
    document.getElementById('trainingTime').textContent = '--';
    document.getElementById('modelStatus').textContent = 'Ready';
    document.getElementById('currentK').textContent = '3';
    document.getElementById('kValue').value = '3';
    
    document.getElementById('confusionMatrix').innerHTML = '<p style="text-align: center; color: #999; padding: 40px;">Train the model to see confusion matrix</p>';
    document.getElementById('predictions').innerHTML = '<p style="text-align: center; color: #999; padding: 40px;">Train the model to see predictions</p>';
    
    const canvas = document.getElementById('accuracyChart');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#999';
    ctx.font = '16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Train the model to see accuracy chart', canvas.width / 2, canvas.height / 2);
    
    document.getElementById('matrixLegend').innerHTML = '';
}