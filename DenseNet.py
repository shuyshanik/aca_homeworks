import numpy as np

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu_derivative(x):
    return np.where(x <= 0, 0, 1)

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def mean_squared_error_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

class DenseLayer:
    def __init__(self, input_size, output_size, activation=None, dropout_rate=0.0):
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.zeros((output_size, 1))
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.dropout_mask = None

    def forward(self, inputs, training=True):
        self.inputs = inputs
        self.z = np.dot(self.weights, inputs) + self.biases

        if self.activation is not None:
            self.a = self.activation(self.z)
        else:
            self.a = self.z

        if training and self.dropout_rate > 0:
            self.dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, size=self.a.shape) / (1 - self.dropout_rate)
            self.a *= self.dropout_mask

        return self.a

    def backward(self, grad):
        if self.activation is not None:
            grad *= self.activation_derivative(self.z)

        if self.dropout_mask is not None:
            grad *= self.dropout_mask

        self.grad_weights = np.dot(grad, self.inputs.T)
        self.grad_biases = np.sum(grad, axis=1, keepdims=True)
        self.grad_inputs = np.dot(self.weights.T, grad)

        return self.grad_inputs

class DenseNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs, training=True):
        for layer in self.layers:
            inputs = layer.forward(inputs, training=training)
        return inputs

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update_weights(self, learning_rate):
        for layer in self.layers:
            layer.weights -= learning_rate * layer.grad_weights
            layer.biases -= learning_rate * layer.grad_biases

    def train(self, x_train, y_train, epochs, learning_rate):
        for epoch in range(epochs):
            outputs = self.forward(x_train)
            loss = mean_squared_error(y_train, outputs)
            grad = mean_squared_error_derivative(y_train, outputs)
            self.backward(grad)
            self.update_weights(learning_rate)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}")

    def predict(self, x):
        return self.forward(x, training=False)
/