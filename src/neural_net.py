import numpy as np
from layer import Layer

class NeuralNetwork(object):

    def __init__(self, *shape, inputs, output):
        self.shape = shape
        self.layers = {}

        temp = 1
        for i in range(1, len(shape) - 1):
            layers['L' + str(temp)] = Layer((size, len(inputs.index)), (shape[i + 1], len(inputs.columns)))
