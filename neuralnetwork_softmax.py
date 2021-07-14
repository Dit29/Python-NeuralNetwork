# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 18:52:12 2021

@author: sacri
"""

import math
import numpy as np
import nnfs

nnfs.init()

# code extracted from this youtube link5 - https://www.youtube.com/watch?v=omz_NdFgWyU&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=6
# Softmax Activation
# determine how wrong is this model? i.e. layer_outputs = [4.8, 4.79, 4.25] vs below layer_outputs

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

# utilise Euler's number to fix exponential issue (caused by more different inputs as we train the model)
# E = 2.71828182846
# E = math.e

''' 
exponential_values = []

for output in layer_outputs:
    exponential_values.append(E**output)
    
print (exponential_values)
'''


'''
# change the above code block to numpy format
exponential_values = np.exp(layer_outputs)
'''

'''
# normalisation should occur after the exponentiation
norm_base = sum(exponential_values)
norm_values = []

for value in exponential_values:
    norm_values.append(value / norm_base)
'''
'''
# change the above code block to numpy format
norm_values = exponential_values / np.sum(exponential_values)
    
print(norm_values)
print(sum(norm_values))
'''


# to summarise the code above - Input -> Exponentiate -> Normalise -> Output
# Exponentiate + Normalise = Softmax

exp_values = np.exp(layer_outputs)
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

print(norm_values)


# overflow prevention v = u - max u