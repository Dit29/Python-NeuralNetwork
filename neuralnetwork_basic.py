# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 16:51:58 2021

@author: sacri
"""

# code extracted from this youtube link1 - https://www.youtube.com/watch?v=lGLto9Xd7bU&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=3
# code extracted from this youtube link2 (mostly optimisation) - https://www.youtube.com/watch?v=tMrbN67U9d4&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=4

# inputs could be input from the sensors link1
# the whole project under this file is using raw python

'''
# input layer (for this section only 4 inputs to 1 neuron)
inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2
'''

# input layer (for this section we will take 4 inputs to 3 neurons) - link1
inputs = [1, 2, 3, 2.5]

'''
weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

bias1 = 2
bias2 = 3
bias3 = 0.5
'''

# input layer (optimisation from above input layers - make it as a list) - link2
# the reason we are doing this because it would help us to make the code more dynamic and simpler
# weight is changing the magnitude, while bias is offsetting the input. Both could affect the input to be positive or negative
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

'''
# output layer - link1
output = [inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
          inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
          inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3]

print(output)
'''

# modified layer_outputs to cater the weights and biases above - link2
layer_outputs = [] # output of current layer
for neuron_weights, neuron_bias in zip(weights,biases):
    neuron_output = 0 # output of given neuron
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input*weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)
    
print(layer_outputs)
    
