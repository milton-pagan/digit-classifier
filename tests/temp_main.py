from digit_multiclass.neural_net import NeuralNetwork
import pandas as pd

dataset = pd.read_csv("data/train.csv")

output = dataset['label'].to_numpy()
inputs = dataset.iloc[: , 1:].to_numpy()

ann = NeuralNetwork((784, 16, 16, 10), inputs, output)

# Check shapes

for weight in ann.weights:
    print(weight.shape)

#Feedforward
ann.feedforward(ann.weights)
print(ann.layers[-1].activations)
