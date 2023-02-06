# Softmax regression (or multinomial logistic regression) is a generalization of logistic regression
# to the case where we want to handle multiple classes.
# Z = XW = b
# Z is not a proper probability value but can be considered as a score given to each class label for each observation!
# In order to convert the score matrix Z to probabilities, we use Softmax function. softmax function will do 2 things:
# 1.convert all scores to probabilities.
# 2.sum of all probabilities is 1.

# Since softmax function provides us with a vector of probabilities of each class label for a given observation, we need to convert target vector in the same format to calculate the cost function! 
# Corresponding to each observation, there is a target vector (instead of a target value!) composed of only zeros and ones where only correct label is set as 1. 
# This technique is called one-hot encoding.See the diagram given below for a better understanding:

import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(mnist)