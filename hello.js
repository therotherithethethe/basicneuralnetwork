class NeuralNetwork {
    constructor() {
        this.weightsInputHidden = [Array(2).fill().map(() => Math.random() * 0.1), Array(2).fill().map(() => Math.random() * 0.1)];
        this.weightsHiddenOutput = Array(2).fill().map(() => Math.random() * 0.1);

        // Initialize biases with a small value
        this.biasHidden = Array(2).fill(0.1);
        this.biasOutput = 0.1;
    }

    // Activation functions and their derivatives
    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    relu(x) {
        return Math.max(0, x);
    }

    sigmoidDerivative(x) {
        return x * (1 - x);
    }

    reluDerivative(x) {
        return x > 0 ? 1 : 0;
    }

    // Forward pass
    predict(input) {
        // From input to hidden layer
        let hidden = input.map((inp, idx) => this.relu(inp * this.weightsInputHidden[idx][0] + inp * this.weightsInputHidden[idx][1] + this.biasHidden[idx]));

        // From hidden layer to output
        let output = this.sigmoid(hidden[0] * this.weightsHiddenOutput[0] + hidden[1] * this.weightsHiddenOutput[1] + this.biasOutput);

        return output;
    }

    // Training function
    train(input, target) {
        const learningRate = 0.5;

        for (let i = 0; i < 2000; i++) {
            // Forward pass
            let hiddenInputs = input.map((inp, idx) => inp * this.weightsInputHidden[idx][0] + inp * this.weightsInputHidden[idx][1] + this.biasHidden[idx]);
            let hiddenOutputs = hiddenInputs.map(this.relu);
            let finalInput = hiddenOutputs[0] * this.weightsHiddenOutput[0] + hiddenOutputs[1] * this.weightsHiddenOutput[1] + this.biasOutput;
            let finalOutput = this.sigmoid(finalInput);

            // Error calculation
            let outputError = target - finalOutput;
            let hiddenErrors = this.weightsHiddenOutput.map(w => outputError * w);
            this.weightsHiddenOutput = this.weightsHiddenOutput.map((w, idx) => w + learningRate * hiddenOutputs[idx] * outputError * this.sigmoidDerivative(finalOutput));
            this.biasOutput += learningRate * outputError * this.sigmoidDerivative(finalOutput);

            hiddenErrors.forEach((error, idx) => {
                this.weightsInputHidden[idx] = this.weightsInputHidden[idx].map(w => w + learningRate * input[idx] * error * this.reluDerivative(hiddenOutputs[idx]));
                this.biasHidden[idx] += learningRate * error * this.reluDerivative(hiddenOutputs[idx]);
            });
        }
    }
    predictBinary(input) {
        const output = this.predict(input);
        return output > 0.5 ? 1 : 0;
    }
}

// Initialize the neural network
const nn = new NeuralNetwork();

// Training data for the AND function
const trainingData = [
    { input: [0, 0], target: 0 },
    { input: [0, 1], target: 0 },
    { input: [1, 0], target: 0 },
    { input: [1, 1], target: 1 }
];

// Train the network
trainingData.forEach(data => {
    nn.train(data.input, data.target);
});

// Test the nn
console.log("0 AND 0:", nn.predictBinary([0, 0]));
console.log("0 AND 1:", nn.predictBinary([0, 1]));
console.log("1 AND 0:", nn.predictBinary([1, 0]));
console.log("1 AND 1:", nn.predictBinary([1, 1]));
