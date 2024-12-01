import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weights and biases
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        self.biases = np.zeros((out_channels, 1))

    def forward(self, x):
        self.input = x
        print(x.shape)
        batch_size, input_channels, input_height, input_width = x.shape
        output_height = (input_height - self.kernel_size+ 2 * self.padding) // self.stride + 1
        output_width = (input_width - self.kernel_size+2 * self.padding) // self.stride + 1

        if output_width<0 or output_height<0:
            print("problem")

        # Initialize output
        output = np.zeros((batch_size, self.out_channels, output_height, output_width))

        # Apply padding
        if self.padding > 0:
            x_padded = np.pad(
                x,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant',
            )
        else:
            x_padded = x



        # Apply convolution
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        # Extract region
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size

                        # Ensure the region shape matches the filter
                        region = x_padded[b, :, h_start:h_end, w_start:w_end]
                        #print(f"Region shape: {region.shape}")
                        #print(f"Weights shape: {self.weights[c_out].shape}")


                        output[b, c_out, i, j] = (np.sum(region * self.weights[c_out]) + self.biases[c_out])
                        #print(f"Output value: {output[b, c_out, i, j]}")

        return output

    def backward(self, dout):
        batch_size, _, out_height, out_width = dout.shape
        _, _, height, width = self.input.shape

        # Gradients for weights and biases
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_biases = np.zeros_like(self.biases)
        dx = np.zeros_like(self.input)

        padded_input = np.pad(self.input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                              mode='constant')
        padded_dx = np.zeros_like(padded_input)

        # Backpropagation
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for i in range(out_height):
                    for j in range(out_width):
                        region = padded_input[b, :, i * self.stride:i * self.stride + self.kernel_size,
                                 j * self.stride:j * self.stride + self.kernel_size]

                        # Accumulate gradients for weights
                        self.grad_weights[c_out] += dout[b, c_out, i, j] * region

                        # Accumulate gradients for biases
                        self.grad_biases[c_out] += dout[b, c_out, i, j]

                        # Accumulate gradient for input
                        padded_dx[b, :, i * self.stride:i * self.stride + self.kernel_size,
                        j * self.stride:j * self.stride + self.kernel_size] += dout[b, c_out, i, j] * self.weights[
                            c_out]

        # Remove padding from gradient
        if self.padding > 0:
            dx = padded_dx[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dx = padded_dx

        return dx




