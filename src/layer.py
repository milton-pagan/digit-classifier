import numpy as np

class Layer(object):

    def __init__(self, neurons_shape, weights_shape):
        self.neurons_shape = neurons_shape
        self.weights_shape = weights_shape

        self.weights = np.random.rand(self.weights_shape)

        self.neurons = np.zeros(neurons_shape)
