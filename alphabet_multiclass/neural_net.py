import numpy as np

class NeuralNetwork(object):

    def __init__(self, *shape):
        self.shape = shape
        self.layers = {}

        int temp = 1
        for size in shape:
            layers['L' + str(temp)] = Layer(size)
