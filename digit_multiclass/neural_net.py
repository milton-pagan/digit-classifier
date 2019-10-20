import numpy as np
from digit_multiclass.layer import Layer

class NeuralNetwork(object):

    def __init__(self, shape, inputs, output):
        self.inputs = inputs
        self.output = output

        self.shape = shape
        self.layers = []

        self.weights = [np.random.randn(y, x + 1) / 1000 for y, x in zip(self.shape[1:], self.shape[:-1])]

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

    def cost_function(self, w):


    # Aids functionality with multiple sample subsets (mini-batches)
    def change_inputs(self, inputs):
        self.inputs = inputs
        self.layers[0].activations = self.inputs
