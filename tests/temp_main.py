from digit_multiclass.neural_net import NeuralNetwork
import pandas as pd
import numpy as np

dataset = pd.read_csv("data/train.csv")

output = dataset['label'].to_numpy()
inputs = dataset.iloc[: , 1:].to_numpy()

# Prepare outputs
temp = np.zeros((output.shape[0], 10))

for sample, i in zip(temp, output) :
    sample[i] = 1

output = temp

ann = NeuralNetwork((784, 16, 16, 10), inputs, output)

# Check shapes
for weight in ann.weights:
    print(weight.shape)

#Feedforward
ann.feedforward(ann.weights)
print(ann.layers[-1].activations.shape)
print(output.shape)

#Cost Function
print(ann.cost_function(ann.weights))

#Backpropagation
ann.backprop(ann.weights)
