import numpy as np
from Conv2D import Conv2D
from BatchNorm import BatchNorm
from ReLU import ReLU

def conv2d(x, kernel, stride=1, padding=0):
    """
    Implements 2D convolution for a single image and single kernel.
    """
    n, h, w, c_in = x.shape
    c_out, k_h, k_w, _ = kernel.shape
    h_out = (h + 2 * padding - k_h) // stride + 1
    w_out = (w + 2 * padding - k_w) // stride + 1

    # Add padding
    if padding > 0:
        x = np.pad(x, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')

    # Initialize output
    out = np.zeros((n, h_out, w_out, c_out))

    for i in range(h_out):
        for j in range(w_out):
            x_slice = x[:, i * stride:i * stride + k_h, j * stride:j * stride + k_w, :]

            for k in range(c_out):
                out[:, i, j, k] = np.sum(x_slice * kernel[k], axis=(1, 2, 3))
    return out


def relu(x):
    return np.maximum(0, x)


def batch_norm(x, gamma, beta, eps=1e-5):
    """
    Batch Normalization.
    """
    mean = np.mean(x, axis=(0, 1, 2), keepdims=True)
    var = np.var(x, axis=(0, 1, 2), keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta


def average_pool(x, size):
    """
    Implements Average Pooling.
    """
    n, h, w, c = x.shape
    pool_h, pool_w = size
    out = np.zeros((n, 1, 1, c))
    for i in range(c):
        out[:, 0, 0, i] = np.mean(x[:, :, :, i], axis=(1, 2))
    return out


def flatten(x):
    """
    Flatten the tensor into a 2D array.
    """
    return x.reshape(x.shape[0], -1)


def dense(x, weights, bias):
    """
    Fully connected layer.
    """
    return np.dot(x, weights) + bias


class ResNetBlock:
    def __init__(self, in_channels, out_channels, use_dropout=False, dropout_prob=0.5):
        self.use_dropout = use_dropout
        self.dropout_prob = dropout_prob
        self.conv1 = Conv2D(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = BatchNorm(out_channels)
        self.conv2 = Conv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm(out_channels)
        self.relu = ReLU()
        self.shortcut = None

        #if in_channels != out_channels:
            #self.shortcut = Conv2D(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        #else:
            #self.shortcut = None

    def forward(self, x, training=True):
        self.input = x
        out = self.conv1.forward(x)
        out = self.bn1.forward(out, training)
        out = self.relu.forward(out)

        if self.use_dropout:
            out, self.dropout_mask = self.dropout_forward(out, p=self.dropout_prob, training=training)

        out = self.conv2.forward(out)
        out = self.bn2.forward(out)

        # Handle shortcut connection
        if self.shortcut is None:
            # Adjust the dimensions of the shortcut to match `out` if needed
            if x.shape != out.shape:
                shortcut = self.adjust_shortcut(x, out.shape)
            else:
                shortcut = x
        else:
            shortcut = self.shortcut.forward(x)

        out += shortcut
        out = self.relu.forward(out)
        return out

    def adjust_shortcut(self, x, target_shape):
        """
        Adjust the shortcut connection to match the target shape.

        Parameters:
            x: The input tensor.
            target_shape: The desired shape for the shortcut.

        Returns:
            Adjusted tensor with the same shape as `target_shape`.
        """
        batch_size, channels, height, width = x.shape
        target_batch, target_channels, target_height, target_width = target_shape

        # Adjust channels using padding or slicing
        if channels < target_channels:
            pad_channels = target_channels - channels
            x = np.pad(x, ((0, 0), (0, pad_channels), (0, 0), (0, 0)), mode='constant')
        elif channels > target_channels:
            x = x[:, :target_channels, :, :]

        # Adjust height and width using pooling or padding
        if height != target_height or width != target_width:
            if height>target_height or width>target_width:
                x = x[:, :, :target_height, :target_width]
            else:
                x = np.pad(x, ((0, 0), (0, 0), (0, target_height - height), (0, target_width - width)), mode='constant')

        return x

    def backward(self, dout):
        dout = self.relu.backward(dout)
        shortcut_grad = dout

        dout = self.bn2.backward(dout)
        dout = self.conv2.backward(dout)

        if self.use_dropout:
            dout = self.dropout_backward(dout, self.dropout_mask, p=self.dropout_prob)

        dout = self.relu.backward(dout)
        dout = self.bn1.backward(dout)
        dout = self.conv1.backward(dout)

        if self.shortcut is not None:
            shortcut_grad = self.shortcut.backward(shortcut_grad)

        dout += shortcut_grad
        return dout

    def dropout_forward(self,x, p=0.5, training=True):
        """
        Apply dropout during the forward pass.
        x: Input data
        p: Dropout probability (fraction of neurons to drop)
        training: Boolean, apply dropout only during training
        """
        if training:
            # Generate a binary mask with probability (1 - p) of keeping a neuron
            mask = (np.random.rand(*x.shape) > p).astype(np.float32)
            out = x * mask  # Apply mask to the input
            out /= (1 - p)  # Scale during training
            return out, mask
        else:
            # During evaluation, no dropout is applied
            return x, None

    def dropout_backward(self,dout, mask, p=0.5):
        """
        Apply dropout during the backward pass.
        dout: Gradient of the loss w.r.t. the layer's output
        mask: Binary mask from the forward pass
        p: Dropout probability
        """
        if mask is not None:
            # Only propagate gradients for neurons that were active
            dx = dout * mask
            dx /= (1 - p)  # Scale during training
            return dx
        return dout

