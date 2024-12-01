import numpy as np

class BatchNorm:
    def __init__(self, num_features, momentum=0.9, epsilon=1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.epsilon = epsilon

        # Initialize gamma (scale) and beta (shift)
        self.gamma = np.ones((1, num_features, 1, 1))
        self.beta = np.zeros((1, num_features, 1, 1))

        # Running mean and variance for inference
        self.running_mean = np.zeros((1, num_features, 1, 1))
        self.running_var = np.ones((1, num_features, 1, 1))

    def forward(self, x, training=True):
        self.input = x

        if training:
            # Compute mean and variance
            self.batch_mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            self.batch_var = np.var(x, axis=(0, 2, 3), keepdims=True)

            # Normalize
            self.x_hat = (x - self.batch_mean) / np.sqrt(self.batch_var + self.epsilon)

            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var
        else:
            # Use running statistics for inference
            self.x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)

        # Scale and shift
        out = self.gamma * self.x_hat + self.beta
        return out

    def backward(self, dout):
        batch_size, _, height, width = dout.shape

        # Gradients for scale (gamma) and shift (beta)
        self.grad_gamma = np.sum(dout * self.x_hat, axis=(0, 2, 3), keepdims=True)
        self.grad_beta = np.sum(dout, axis=(0, 2, 3), keepdims=True)

        # Backpropagation through normalization
        dx_hat = dout * self.gamma
        dvar = np.sum(dx_hat * (self.input - self.batch_mean) * -0.5 * (self.batch_var + self.epsilon) ** -1.5, axis=(0, 2, 3), keepdims=True)
        dmean = np.sum(dx_hat * -1 / np.sqrt(self.batch_var + self.epsilon), axis=(0, 2, 3), keepdims=True) + dvar * np.sum(-2 * (self.input - self.batch_mean), axis=(0, 2, 3), keepdims=True) / (batch_size * height * width)
        dx = dx_hat / np.sqrt(self.batch_var + self.epsilon) + dvar * 2 * (self.input - self.batch_mean) / (batch_size * height * width) + dmean / (batch_size * height * width)
        return dx
