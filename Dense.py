import numpy as np
class Dense:
    def __init__(self, input_size, output_size):
        # Initialize weights and biases
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))

    def forward(self, x):
        self.input = x  # Save input for backward pass
        self.output = np.dot(x, self.weights) + self.biases
        return self.output

    def backward(self, dout):
        # Compute gradients
        self.grad_weights = np.dot(self.input.T, dout)  # Gradient of weights
        self.grad_biases = np.sum(dout, axis=0, keepdims=True)  # Gradient of biases
        dx = np.dot(dout, self.weights.T)  # Gradient w.r.t input
        return dx
