from ResNetBlock import ResNetBlock
from GlobalAveragePooling import GlobalAveragePooling
from Dense import Dense
import numpy as np


class ResNet32:
    def __init__(self, num_classes=10, dropout_prob=0.5):
        self.stage1 = [ResNetBlock(3, 16, use_dropout=True, dropout_prob=dropout_prob)]
        self.stage1 += [ResNetBlock(16, 16, use_dropout=True, dropout_prob=dropout_prob) for _ in range(1)]

        self.stage2 = [ResNetBlock(16, 32, use_dropout=True, dropout_prob=dropout_prob)]
        self.stage2 += [ResNetBlock(32, 32, use_dropout=True, dropout_prob=dropout_prob) for _ in range(1)]

        self.stage3 = [ResNetBlock(32, 64, use_dropout=True, dropout_prob=dropout_prob)]
        self.stage3 += [ResNetBlock(64, 64, use_dropout=True, dropout_prob=dropout_prob) for _ in range(1)]

        self.global_avg_pool = GlobalAveragePooling()
        self.fc = Dense(64, num_classes)

    def forward(self, x, training=True):
        for block in self.stage1:
            x = block.forward(x, training)
        for block in self.stage2:
            x = block.forward(x, training)
        for block in self.stage3:
            x = block.forward(x, training)

        x = self.global_avg_pool.forward(x)
        x = self.fc.forward(x)
        return x

    def backward(self, dout):
        dout = self.fc.backward(dout)
        dout = self.global_avg_pool.backward(dout)

        for block in reversed(self.stage3):
            dout = block.backward(dout)
        for block in reversed(self.stage2):
            dout = block.backward(dout)

        for block in reversed(self.stage1):
            dout = block.backward(dout)

        return dout

    def compute_loss_and_gradients(self,y_pred, y_true):
        """
        Compute cross-entropy loss and its gradient.
        y_pred: Predicted probabilities (softmax output)
        y_true: One-hot encoded true labels
        """
        # Cross-entropy loss
        eps = 1e-8  # Avoid log(0)
        loss = -np.sum(y_true * np.log(y_pred + eps)) / y_true.shape[0]

        # Gradient of the loss w.r.t. predictions
        grad_y_pred = (y_pred - y_true) / y_true.shape[0]

        return loss, grad_y_pred



