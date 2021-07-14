# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 20:43:17 2021

@author: sacri
"""

# code extracted from this youtube link2 (using numpy - second half of the video) - https://www.youtube.com/watch?v=tMrbN67U9d4&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=4
# code extracted from this youtube link3 - https://www.youtube.com/watch?v=TEWy9vZcxW4&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=5
import numpy as np

np.random.seed(0)

'''
commented due to the code from link2
inputs = [1, 2, 3, 2.5]


# weights = [0.2, 0.8, -0.5, 1.0]
# bias = 2


weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]


# output = np.dot(inputs, weights) + bias (good for simple 1D vector, but not matrix)
output = np.dot(weights, inputs) + biases
print(output)
'''

# batches will help us for generalisation
# batch size will affect your standard deviation line
inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]


# adding extra layer
weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]


# ValueError: shapes (3,4) and (3,4) not aligned: 4 (dim 1) != 3 (dim 0)
# to correct the error above, we need to transpose the "weights" matrix - np.array(weights).T
# output = np.dot(inputs, np.array(weights).T) + biases
#print(output)

# adding extra layer on the output
layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs)