# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 20:07:57 2021

@author: sacri
"""

# code extracted from this youtube link5 - https://www.youtube.com/watch?v=omz_NdFgWyU&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=6
# code extracted from this youtube link6 - https://www.youtube.com/watch?v=dEXPMQXoiLc&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=7
# Softmax Activation second half of the video

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense(): # two way to initalise layers
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) # shape of weights? find the answer on the video from link3 25:00
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])
# largest from those 5 lines of output = 0.35303497
# from that, we can calculate how right or wrong of this model

# neural network did not output TRUE/FALSE clasification, they output the "probability distribution" - link6
# prediction usually 86% confidence
# to check this errors, we can calculate the Loss with Categorical Cross-Entropy = set the intended target value.
# Categorical Cross-Entropy = 
# One-hot encoding - link6 5:00
