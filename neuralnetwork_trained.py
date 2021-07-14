# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 21:30:56 2021

@author: sacri
"""

# code extracted from this youtube link3 - https://www.youtube.com/watch?v=TEWy9vZcxW4&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=5
# code extracted from this youtube link4 - https://www.youtube.com/watch?v=gmjzbpSVY1A&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=7
import numpy as np

np.random.seed(0)

# actual trained data set - second half of link3
X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]



# create the hidden layer - as programmer we are not in charge how the layer change the value
# initialise weight using trained value - normalise the dataset to make the values between -1 to 1 (keep the value small for now)
# biases usually initalised as 0 to pass the value. However, we do not want it as it could assume the neuron to be "dead" only produce 0.
class Layer_Dense(): # two way to initalise layers
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) # shape of weights? find the answer on the video from link3 25:00
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
    
# print(0.10*np.random.randn(4,3))

layer1 = Layer_Dense(4, 5) # the n_neurons can be any number that you would like
layer2 = Layer_Dense(5, 2) # the n_neurons can be any number that you would like

layer1.forward(X)
#print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)

# link4 - first half of video - very good starting the ReLU(x) activation function explanation
# we can use step functions to be Activation Functions
# sigmoid activation functions = make more granual output of layer of neuron, it can calculate loss as well, and to be used for increase/decrease weights and biases
# y=ReLU(x) 
# this is the most popular hidden layer activation function technique
# linear vs non-linear activation function (ReLU activation)