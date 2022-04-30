import pandas as pd
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt

#ex3.m
input_layer_size = 400
num_label = 10

#Loading and Visualizing Data
print('Loading and Visualizing Data ...\n')
mat = io.loadmat('ex3data1.mat')
data = pd.DataFrame(np.hstack((mat['X'], mat['y'])))
m = mat['X'].shape[0]
rand_indices = np.random.permutation(m)
sel = mat['X'][rand_indices[:100]]
sel0 = pd.DataFrame(sel)
sel0.to_csv('random number.csv')
print('Random number in \'random number.csv\'')

#Vectorize Logistic Regression
print('\nTesting lrCostFunction() with regularization')
theta_t = np.array([-2], [-1], [1], [2])
X_t