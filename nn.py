# Neural Network Training Process
# 1. Take the inputs from the training example and put them through our formula to get the neuron's output
# 2. Calculate the error, which is the difference between the output we received and the actual output.
# 3. Depending on the severeness of the error, adjust the weights accordingly
# 4. Repeat this process 20,000 times (or number of times under range())
# this code extracted from this youtube link - https://www.youtube.com/watch?v=kft1AJ9WVDk

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

training_inputs = np.array([[0,0,1], 
                            [1,1,1], 
                            [1,0,1], 
                            [0,1,1]])

training_outputs = np.array([[0,1,1,0]]).T

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3,1)) - 1

print('Random starting synaptic weights: ')
print(synaptic_weights)

for iteration in range(100000):
    input_layer = training_inputs
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))
    error = training_outputs - outputs
    adjustments = error * sigmoid_derivative(outputs)
    synaptic_weights += np.dot(input_layer.T, adjustments)
        
print('Synaptic weights after training: ')
print(synaptic_weights)

print('Outputs after training: ')
print(outputs)
    
    