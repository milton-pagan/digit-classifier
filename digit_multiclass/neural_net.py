import numpy as np
from digit_multiclass.layer import Layer

class NeuralNetwork(object):

    def __init__(self, shape, inputs, output):
        self.inputs = inputs
        self.output = output

        self.shape = shape
        self.layers = []

        self.weights = [np.random.randn(y, x + 1) / 1000 for y, x in zip(self.shape[1:], self.shape[:-1])]

        self.reg_param = 0.1

        # Add layers
        for i in range(0, len(self.shape) - 1):
            activations_shape = (self.shape[i], len(self.inputs))
            self.layers.append(Layer(activations_shape))

        # Add input layer data
        self.layers[0].activations = self.inputs

        # Add output layer
        self.layers.append(Layer((self.shape[-1], len(self.inputs))))

    def feedforward(self, w):
        for i in range(1, len(self.layers)):
            self.layers[i].activations = self.layers[i - 1].calculate_activations(w[i - 1])

    def backprop(self, w):
        gradient = [np.zeros((w.shape)) for w in self.weights]

        deltas = [self.layers[-1].activations - self.output.transpose()]

        i = 0
        for w, l in zip(self.weights[::-1], self.layers[-2:0:-1]):
            a = np.insert(l.activations, 0, np.ones(l.activations_shape[1])).reshape((l.activations_shape[0] + 1, l.activations_shape[1]))

            sigmoid_prime = np.multiply(a, 1 - a)

            print(w.transpose().shape, deltas[i][1:, :].shape, a.shape)
            if i == 0:
                deltas.append(np.multiply(np.dot(w.transpose(), deltas[i]), sigmoid_prime))

            else:
                deltas.append(np.multiply(np.dot(w.transpose(), deltas[i][1:, :]), sigmoid_prime))

            i += 1

        deltas = deltas[::-1]

        print("\nDelta shapes: ")
        for a in deltas:
            print(a.shape)

    def cost_function(self, w):
        return (-1/len(self.inputs)) * np.sum(np.diagonal(np.nan_to_num(np.dot(np.log(self.layers[-1].activations), self.output) + np.dot(np.log(1 - self.layers[-1].activations), 1 - self.output)))) + (self.reg_param /(2 * len(self.inputs))) * np.sum(np.square(self.unroll_weights()))

    # Aids functionality with multiple sample subsets (mini-batches)
    def change_inputs(self, inputs):
        self.inputs = inputs
        self.layers[0].activations = self.inputs

    # Returns all weights in a single flattened ndarray (doesn't include weights for bias units by default)
    def unroll_weights(self, include_bias = False):

        if not include_bias:
            unrolled_weights = np.ravel(self.weights[0][:, 1:])

            for w in self.weights[1:]:
                np.concatenate((unrolled_weights, w[:, 1:].flatten()))

            return unrolled_weights

        unrolled_weights = np.ravel(self.weights[0])

        for w in self.weights[1:]:
            np.concatenate((unrolled_weights, w.flatten()))

        return unrolled_weights
