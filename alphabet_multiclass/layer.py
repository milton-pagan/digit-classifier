import numpy as np
from neuron import SigmoidNeuron

class Layer(object):

    def __init__(self, size):
        self.size = size
        self.weights = np.random.rand(size)

        self.neurons = [SigmoidNeuron() for i in range(size)]
