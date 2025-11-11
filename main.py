import numpy as np
import os
from struct import unpack


# Load MNIST data (your working version)
def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte')

    with open(labels_path, 'rb') as lbpath:
        magic, n = unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))


def sigmoid_derivative(x):
    return x * (1 - x)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# Neural Network Class with optimizations
class SimpleNeuralNetwork:
    def __init__(self, layer_sizes, activation='sigmoid', optimizer='sgd', learning_rate=0.1, momentum=0.9):
        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.weights = []
        self.biases = []

        # Initialize weights and biases
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2. / layer_sizes[i]))
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))

        # For momentum
        self.velocity_w = [np.zeros_like(w) for w in self.weights]
        self.velocity_b = [np.zeros_like(b) for b in self.biases]

    def forward(self, X):
        self.activations = [X]
        self.z_values = []

        current_activation = X

        for i in range(len(self.weights) - 1):
            z = np.dot(current_activation, self.weights[i]) + self.biases[i]
            self.z_values.append(z)

            if self.activation_name == 'sigmoid':
                current_activation = sigmoid(z)
            elif self.activation_name == 'relu':
                current_activation = relu(z)

            self.activations.append(current_activation)

        # Output layer (softmax)
        z_output = np.dot(current_activation, self.weights[-1]) + self.biases[-1]
        self.z_values.append(z_output)
        output_activation = softmax(z_output)
        self.activations.append(output_activation)

        return output_activation

    def backward(self, X, y):
        m = X.shape[0]
        y_one_hot = np.eye(10)[y]

        # Output layer error
        delta = self.activations[-1] - y_one_hot

        # Backpropagate
        for i in range(len(self.weights) - 1, -1, -1):
            # Calculate gradients
            dW = np.dot(self.activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m

            # Apply optimization
            if self.optimizer == 'momentum':
                self.velocity_w[i] = self.momentum * self.velocity_w[i] - self.learning_rate * dW
                self.velocity_b[i] = self.momentum * self.velocity_b[i] - self.learning_rate * db
                self.weights[i] += self.velocity_w[i]
                self.biases[i] += self.velocity_b[i]
            else:  # plain SGD
                self.weights[i] -= self.learning_rate * dW
                self.biases[i] -= self.learning_rate * db

            # Propagate error backwards (skip for input layer)
            if i > 0:
                if self.activation_name == 'sigmoid':
                    delta = np.dot(delta, self.weights[i].T) * sigmoid_derivative(self.activations[i])
                elif self.activation_name == 'relu':
                    delta = np.dot(delta, self.weights[i].T) * relu_derivative(self.activations[i])

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)

    def save_weights(self, filename):
        """Save weights and biases to a .npz file"""
        data = {}
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            data[f'weights_{i}'] = w
            data[f'biases_{i}'] = b

        # Save network configuration
        data['layer_sizes'] = np.array(self.layer_sizes)
        data['activation'] = self.activation_name
        data['optimizer'] = self.optimizer

        np.savez(filename, **data)
        print(f"✓ Weights saved to {filename}")

    def load_weights(self, filename):
        """Load weights and biases from a .npz file"""
        try:
            data = np.load(filename, allow_pickle=True)

            # Verify architecture matches
            loaded_sizes = data['layer_sizes'].tolist()
            if loaded_sizes != self.layer_sizes:
                print(f"Warning: Architecture mismatch! Loaded: {loaded_sizes}, Current: {self.layer_sizes}")
                return False

            self.weights = []
            self.biases = []

            for i in range(len(self.layer_sizes) - 1):
                self.weights.append(data[f'weights_{i}'])
                self.biases.append(data[f'biases_{i}'])

            print(f"✓ Weights loaded from {filename}")
            return True
        except Exception as e:
            print(f"✗ Error loading weights: {e}")
            return False


def get_next_filename():
    """Get next available weights filename"""
    base_name = "weights"
    counter = 0

    while True:
        if counter == 0:
            filename = f"{base_name}.npz"
        else:
            filename = f"{base_name}{counter}.npz"

        if not os.path.exists(filename):
            return filename
        counter += 1


# Enhanced training with learning rate scheduling
def train_with_early_stopping(network, X_train, y_train, X_test, y_test,
                              initial_epochs=20, learning_rate=0.1, patience=5):
    best_accuracy = 0
    epochs_without_improvement = 0
    epoch = 0

    # Learning rate scheduling
    current_lr = learning_rate
    lr_reduction_factor = 0.5
    lr_patience = 10

    print(f"Starting training with {network.optimizer} optimizer...")

    while True:
        # Update learning rate if using SGD
        if network.optimizer == 'sgd':
            network.learning_rate = current_lr

        # Forward and backward pass
        network.forward(X_train)
        network.backward(X_train, y_train)

        # Calculate accuracy
        train_acc = network.accuracy(X_train, y_train)
        test_acc = network.accuracy(X_test, y_test)

        if epoch % 10 == 0:  # Print every 10 epochs
            print(f"Epoch {epoch + 1}: Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}, LR = {current_lr:.6f}")

        # Check if we should continue training
        if epoch + 1 >= initial_epochs:
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                epochs_without_improvement = 0

            else:
                epochs_without_improvement += 1

            # Reduce learning rate if no improvement
            if epochs_without_improvement >= lr_patience:
                current_lr *= lr_reduction_factor
                epochs_without_improvement = 0
                print(f"Reducing learning rate to {current_lr:.6f}")

            if epochs_without_improvement >= patience:
                print(f"Stopping early at epoch {epoch + 1}. Best test accuracy: {best_accuracy:.4f}")

                # Load best weights
                network.load_weights("best_weights.npz")
                break

        epoch += 1

        # Safety stop
        if epoch > 1000:
            print("Reached maximum epochs (1000)")
            break

    return best_accuracy


# Main execution
if __name__ == "__main__":
    # Load data
    X_train, y_train = load_mnist('mnist_data/', 'train')
    X_test, y_test = load_mnist('mnist_data/', 't10k')

    # Normalize pixel values
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # Try to load existing weights first
    weights_file = "best_weights.npz"
    if os.path.exists(weights_file):
        print("Found existing weights file. Testing...")
        # Create network with same architecture
        nn = SimpleNeuralNetwork([784, 128, 10], activation='relu', optimizer='momentum', learning_rate=0.01)
        if nn.load_weights(weights_file):
            initial_acc = nn.accuracy(X_test, y_test)
            print(f"Loaded model accuracy: {initial_acc:.4f}")

        initial_epochs = 1

    else:
        # Create new network (try ReLU + Momentum for faster convergence)
        nn = SimpleNeuralNetwork([784, 128, 10], activation='relu', optimizer='momentum', learning_rate=0.01)
        initial_epochs = 20

    print("Initial test accuracy:", nn.accuracy(X_test, y_test))

    # Train with early stopping
    best_acc = train_with_early_stopping(nn, X_train, y_train, X_test, y_test,
                                         initial_epochs, learning_rate=0.01, patience=8)

    # Save final weights with auto-incrementing filename
    final_filename = get_next_filename()
    nn.save_weights(final_filename)

    # Final evaluation
    final_acc = nn.accuracy(X_test, y_test)
    print(f"Final test accuracy: {final_acc:.4f}")
    print(f"Weights saved to: {final_filename}")
