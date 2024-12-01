import numpy as np
class ReLU:
    def __init__(self):
        pass

    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, dout):
        dx = dout * (self.input > 0)
        return dx
