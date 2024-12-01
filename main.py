import numpy as np
from keras.datasets import mnist
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from ResNet32 import ResNet32

import numpy as np


def train_model(model, x_train, y_train, x_val, y_val, epochs=10, batch_size=64, lr=0.01):
    """
    Train the ResNet-32 model on CIFAR-10.

    Parameters:
        model: The ResNet-32 model instance.
        x_train: Training data, shape (num_samples, height, width, channels).
        y_train: Training labels, shape (num_samples, num_classes).
        x_val: Validation data, shape (num_samples, height, width, channels).
        y_val: Validation labels, shape (num_samples, num_classes).
        epochs: Number of training epochs.
        batch_size: Size of each mini-batch.
        lr: Learning rate for parameter updates.
    """

    # Reshape x_train and x_val to match expected format
    x_train = np.transpose(x_train, (
        0, 3, 1, 2))  # (batch_size, height, width, channels) -> (batch_size, channels, height, width)
    x_val = np.transpose(x_val, (0, 3, 1, 2))

    num_train = x_train.shape[0]
    num_batches = num_train // batch_size

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        # Shuffle training data
        indices = np.random.permutation(num_train)
        x_train = x_train[indices]
        y_train = y_train[indices]



        # Training loop
        epoch_loss = 0.0
        correct = 0
        for i in range(0, num_train, batch_size):
            x_batch = x_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            # Forward pass
            outputs = model.forward(x_batch)
            print(f"batch {i}")

            # Compute loss (Cross-Entropy Loss)
            loss = -np.mean(np.sum(y_batch * np.log(outputs + 1e-8), axis=1))
            epoch_loss += loss

            # Compute accuracy
            correct += np.sum(np.argmax(outputs, axis=1) == np.argmax(y_batch, axis=1))

            # Backward pass
            dout = outputs - y_batch
            model.backward(dout)

            # Update parameters
            update_parameters(model, lr)

        # Calculate training accuracy
        train_acc = correct / num_train

        # Validation
        val_acc, val_loss = evaluate_model(model, x_val, y_val)

        # Print progress
        print(
            f"Loss: {epoch_loss / num_batches:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")


def evaluate_model(model, x_val, y_val):
    """
    Evaluate the model on the validation set.

    Parameters:
        model: The ResNet-32 model instance.
        x_val: Validation data, shape (num_samples, height, width, channels).
        y_val: Validation labels, shape (num_samples, num_classes).

    Returns:
        accuracy: Validation accuracy.
        loss: Validation loss.
    """
    num_samples = x_val.shape[0]
    correct = 0
    total_loss = 0.0

    for i in range(num_samples):
        output = model.forward(x_val[i:i + 1])  # Forward pass
        loss = -np.sum(y_val[i] * np.log(output + 1e-8))  # Cross-Entropy Loss
        total_loss += loss
        correct += np.argmax(output) == np.argmax(y_val[i])

    accuracy = correct / num_samples
    average_loss = total_loss / num_samples
    return accuracy, average_loss


def update_parameters(model, lr):
    """
    Update parameters of the ResNet-32 model using gradients computed in backward propagation.

    Args:
        model: The ResNet-32 model.
        lr: Learning rate for parameter updates.
    """
    # Iterate through all layers in the model
    for stage in [model.stage1, model.stage2, model.stage3]:
        for block in stage:
            # Update Conv2D layers in each block
            for layer in [block.conv1, block.conv2]:
                layer.weights -= lr * layer.grad_weights
                layer.biases -= lr * layer.grad_biases

            # Update BatchNorm layers
            for bn_layer in [block.bn1, block.bn2]:
                bn_layer.gamma -= lr * bn_layer.grad_gamma
                bn_layer.beta -= lr * bn_layer.grad_beta

    # Update the Dense layer at the end
    model.fc.weights -= lr * model.fc.grad_weights
    model.fc.biases -= lr * model.fc.grad_biases



if __name__ == "__main__":
    # Load and preprocess MNIST dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize images to [0, 1]
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    # Train-validation split
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # Initialize the model
    model = ResNet32(num_classes=10, dropout_prob=0.3)

    # Train the model
    train_model(model, x_train, y_train, x_val, y_val, epochs=10, batch_size=64, lr=0.001)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
