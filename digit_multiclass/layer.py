import numpy as np

class Layer(object):

    def __init__(self, activations_shape):
        self.activations_shape = activations_shape

        self.activations = np.zeros(self.activations_shape)

    # Returns z = sig(w * a)
    def calculate_activations(self, w):

        # Insert bias units
        a = np.insert(self.activations, 0, np.ones(self.activations_shape[1])).reshape((self.activations_shape[0] + 1, self.activations_shape[1]))

        return self.sigmoid(np.dot(w, a))


    def sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))
