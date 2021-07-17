import numpy as np


class NeuralNetwork():

    def __init__(self, layer_sizes):
        # layer_sizes example: [4, 10, 2]
        input_layer_size = layer_sizes[0]
        hidden_layer_size = layer_sizes[1]
        output_layer_size = layer_sizes[2]
        self.activations = [None]
        self.z = [None, None, None]
        self.weights = [None]
        self.bias = [None]
        self.set_layer(hidden_layer_size, input_layer_size)
        self.set_layer(output_layer_size, hidden_layer_size)
        # TODO

    def set_layer(self, layer_size, last_layer_size):
        self.activations.append(np.zeros((layer_size, 1)))
        self.bias.append(np.zeros((layer_size, 1)))
        self.weights.append(np.random.randn(layer_size, last_layer_size))

    def activation(self, x):  # x is layer index
        self.z[x] = self.weights[x] @ self.activations[x - 1]
        self.z[x] = self.z[x] + self.bias[x]
        self.activations[x] = 1 / (1 + np.exp(-self.z[x]))

    def forward(self, x):
        self.activation[0] = x
        for layer_index in range(1, 3, 1):
            self.activation(layer_index)
