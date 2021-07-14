# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 17:01:03 2021

@author: sacri
"""

# code extracted from this youtube link4 - https://www.youtube.com/watch?v=gmjzbpSVY1A&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=7

# link4 - first half of video - very good starting the ReLU(x) activation function explanation
# we can use step functions to be Activation Functions
# sigmoid activation functions = make more granual output of layer of neuron, it can calculate loss as well, and to be used for increase/decrease weights and biases
# y=ReLU(x) 
# this is the most popular hidden layer activation function technique
# linear vs non-linear activation function (ReLU activation)

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# np.random.seed(0)

# actual trained data set - second half of link3
X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

X, y = spiral_data(100, 3)

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []

'''
for i in inputs:
    if i > 0:
        output.append(i)
    elif i <= 0:
        output.append(0)

for i in inputs:
    output.append((max(0, i)))
      
print(output)
'''


class Layer_Dense(): # two way to initalise layers
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) # shape of weights? find the answer on the video from link3 25:00
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

layer1 = Layer_Dense(2, 5) # the n_neurons can be any number that you would like # Layer_Dense(number of input/ size of input data)
# layer2 = Layer_Dense(5, 2) # the n_neurons can be any number that you would like

activation1 = Activation_ReLU()

layer1.forward(X)
# print(layer1.output)

activation1.forward(layer1.output)
print(activation1.output)
# layer2.forward(layer1.output)
# print(layer2.output)