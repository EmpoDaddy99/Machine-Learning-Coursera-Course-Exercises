import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.optimize as op

def sigmoid(z):
	g = np.ones(z.shape)
	g /= (math.e ** (-1 * z) + 1)
	return g

def costFunction(theta, X, y):
	m = len(y)
	J = 0
	grad = np.zeros(theta.shape)
	for i in range(m):
		J -= (y[i] * math.log(sigmoid(np.dot(np.transpose(theta), np.transpose(X[i, :]))), math.e) + (1 - y[i]) * math.log(1 - sigmoid(np.dot(np.transpose(theta), np.transpose(X[i, :]))), math.e))
		for j in range(len(grad)):
			grad[j] += (sigmoid(np.dot(np.transpose(theta), np.transpose(X[i, :]))) - y[i]) * X[i, j]
	J /= m
	grad /= m
	return J, grad

def mapFeature(X1, X2):
	out = [1]
	for i in range(6):
		for j in range(i):
			out.append((X1 ** (i - j)) * (X2 ** j))
	out = np.array(out)
	return out

#ex2.m
data = pd.read_csv('ex2data1.csv')
X = data[['X1', 'X2']]
y = data['y']

#Plotting
print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n')
plt.plot(data.loc[data['y'] == 1, ['X1']], data.loc[data['y'] == 1, ['X2']], 'g+')
plt.plot(data.loc[data['y'] == 0, ['X1']], data.loc[data['y'] == 0, ['X2']], 'r.')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(['Admitted', 'Not admitted'])
plt.show()

#Compute Cost and Gradient
m = X.shape[0]
n = X.shape[1]
X = pd.concat((pd.DataFrame(np.ones((m,1)), columns = ['X0']), X), axis = 1)
X = X.to_numpy()
y = y.to_numpy()
initial_theta = np.zeros((n + 1, 1))
cost = 0
grad = 0
cost, grad = costFunction(initial_theta, X, y)
print('Cost at initial theta (zeros):', cost)
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros): \n', grad)
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')
test_theta = np.array([[-24], [0.2], [0.2]])
cost, grad = costFunction(test_theta, X, y)
print('Cost at test theta:', cost)
print('Expected cost (approx): 0.218\n')
print('Gradient at test theta: \n', grad)
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')

#Optimizing using fminunc
op_initial_theta = op.minimize(fun = costFunction,
						  x0 = initial_theta,
						  args = (X, y),
						  method='TNC',
						  jac=True,
						  options={'maxiter':400})
theta = op_initial_theta.x
cost = op_initial_theta.fun
print('Cost at theta found by fminunc:', cost)
print('Expected cost (approx): 0.203\n')
print('theta: \n')
print(theta)
print('Expected theta (approx):\n')
print(' -25.161\n 0.206\n 0.201\n')
plt.plot(data.loc[data['y'] == 1, ['X1']], data.loc[data['y'] == 1, ['X2']], 'g+')
plt.plot(data.loc[data['y'] == 0, ['X1']], data.loc[data['y'] == 0, ['X2']], 'r.')
u = np.linspace(-1, 1.5, 50)
z = np.zeros(len(u))
for i in range(len(u)):
	for j in range(len(u)):
		z[i, j] = np.dot(mapFeature(u[i], u[j]), theta)
plt.plot(z, np.linspace(X[1].min(), X[1].max(), X.shape[0]), 'b-')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(['Admitted', 'Not admitted'])
plt.show()