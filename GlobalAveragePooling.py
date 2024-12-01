import numpy as np
class GlobalAveragePooling:
    def __init__(self):
        pass

    def forward(self, x):
        # Compute mean across height and width for each channel
        self.input_shape = x.shape
        self.output = np.mean(x, axis=(2, 3), keepdims=False)  # Remove spatial dimensions
        return self.output

    def backward(self, dout):
        # Distribute gradient equally across the original spatial dimensions
        batch_size, num_channels, height, width = self.input_shape
        dx = dout[:, :, None, None] / (height * width)  # Broadcasting the gradient
        dx = np.tile(dx, (1, 1, height, width))  # Restore original shape
        return dx
