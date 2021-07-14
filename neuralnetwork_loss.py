# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 20:27:39 2021

@author: sacri
"""

# code extracted from this youtube link6 - https://www.youtube.com/watch?v=dEXPMQXoiLc&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=7

# largest product from those 5 lines of output = 0.35303497 from neuralnetwork_softmax_outputlayer.py
# from that, we can calculate how right or wrong of this model

# neural network did not output TRUE/FALSE clasification, they output the "probability distribution" - link6
# prediction usually 86% confidence
# to check this errors, we can calculate the Loss with Categorical Cross-Entropy = set the intended target value.
# Categorical Cross-Entropy = 
# One-hot encoding - link6 5:00

'''
solving for x

e ** x = b
'''

'''
# numpy way (easy)
import numpy as np
import math

b = 5.2

print(np.log(b))
print(math.e**np.log(b))
'''

# non-numpy way (hard)
import math

softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0]

# target_class = 0 | if it is 0, one-hot output is the first digit, hence target_output = [1, 0, 0]

# calculate the loss
loss = -(math.log(softmax_output[0]) * target_output[0] + 
         math.log(softmax_output[1]) * target_output[1] + 
         math.log(softmax_output[2]) * target_output[2])
         
print(loss)
loss = -math.log(softmax_output[0])
print(loss)

print(-math.log(0.7)) # the higher the confidence, the lower the lost "0.35667494393873245"
print(-math.log(0.5)) # the lower the confidence, the higher the lost "0.6931471805599453"