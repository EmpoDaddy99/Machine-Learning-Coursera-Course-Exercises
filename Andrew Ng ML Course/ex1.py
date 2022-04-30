import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

def warmUpExercise():
	A = np.zeros((5, 5))
	for i in range(A.shape[0]):
		A[i, i] = 1
	return A

def computeCost(X, y, theta):
	m = len(y)
	J = 0
	for i in range(m):
		J += (theta[0, 0] * X[i, 0] + theta[1, 0] * X[i, 1] - y[i]) ** 2
	J /= (2 * m)
	return J

def gradientDescent(X, y, theta, alpha, num_iters):
	m = len(y)
	J_history = np.zeros((num_iters, 1))
	for i in range(num_iters):
		x = 0
		x0 = 0
		for j in range(m):
			x += (theta[0, 0] * X[j, 0] + theta[1, 0] * X[j, 1] - y[j]) * X[j, 0]
			x0 += (theta[0, 0] * X[j, 0] + theta[1, 0] * X[j, 1] - y[j]) * X[j, 1]
		theta[0, 0] -= (alpha / m) * x
		theta[1, 0] -= (alpha / m) * x0
		J_history[i] = computeCost(X, y, theta)
	return theta, J_history

def featureNormalize(X):
	X_norm = X
	mu = X.mean(axis = 0)
	sigma = X.std(axis = 0)
	X_norm = (X - mu) / sigma
	return X_norm, mu, sigma

def computeCostMulti(X, y, theta):
	m = len(y)
	J = sum(sum((np.dot(X, theta) - y) * (np.dot(X, theta) - y))) / (2 * m)
	#J = 0
	#for i in range(m):
	#	J += (theta[0, 0] * X[i, 0] + theta[1, 0] * X[i, 2] + theta[2, 0] * X[i, 2] - y[i]) ** 2
	return J

def gradientDescentMulti(X, y, theta, alpha, num_iters):
	m = len(y)
	J_history = np.zeros((num_iters, 1))
	for i in range(num_iters):
		x = 0
		x0 = 0
		x1 = 0
		for j in range(m):
			x += (theta[0, 0] * X[j, 0] + theta[1, 0] * X[j, 1] - y[j]) * X[j, 0]
			x0 += (theta[0, 0] * X[j, 0] + theta[1, 0] * X[j, 1] - y[j]) * X[j, 1]
			x1 += (theta[0, 0] * X[j, 0] + theta[1, 0] * X[j, 1] - y[j]) * X[j, 2]
		theta[0, 0] -= (alpha / m) * x
		theta[1, 0] -= (alpha / m) * x0
		theta[2, 0] -= (alpha / m) * x1
		J_history[i] = computeCost(X, y, theta)
	return theta, J_history

def normalEqn(X, y):
	theta = np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.dot(np.transpose(X), y))
	return theta

#ex1.m
#Basic Function
print('Running warmUpExercise ... \n')
print('5x5 Identity Matrix: \n')
print(warmUpExercise())

#Plotting
print('Plotting Data ...\n')
data = pd.read_csv('ex1data1.csv')
X = data['X']
y = data['y']
m = len(y)
plt.plot(X, y, 'rx')
plt.show()

#Cost and Gradient Descent
X = pd.concat((pd.DataFrame(np.ones((m,1)), columns = ['X0']), X), axis = 1)
X = X.to_numpy()
y = y.to_numpy()
theta = np.zeros((2, 1))
iterations = 1500
alpha = 0.01
print('\nTesting the cost function ...\n')
J = computeCost(X, y, theta)
print('With theta = [0 ; 0]\nCost computed =', J)
print('Expected cost value (approx) 32.07\n')
J = computeCost(X, y, np.array(([-1], [2])))
print('\nWith theta = [-1 ; 2]\nCost computed =', J)
print('Expected cost value (approx) 54.24\n')
print('\nRunning Gradient Descent ...\n')
theta = gradientDescent(X, y, theta, alpha, iterations)[0]
print('Theta found by gradient descent:\n')
print(theta)
print('Expected theta values (approx)\n')
print(' -3.6303\n  1.1664\n\n')
plt.plot(X[:, 1], y, 'rx')
plt.plot(X[:, 1], np.dot(X, theta), 'b-')
plt.legend(['Training Data', 'Linear Regression'])
plt.show()
predict1 = np.dot(np.array((1, 3.5)), theta)[0]
print('For population = 35,000, we predict a profit of', predict1*10000)
predict2 = np.dot(np.array((1, 7)), theta)[0]
print('For population = 70,000, we predict a profit of', predict2*10000)

#Visualizing J(theta_0, theta_1)
print('Visualizing J(theta_0, theta_1) ...\n')
theta0_vals = np.linspace(-10, 10, num = 100)
theta1_vals = np.linspace(-1, 4, num = 100)
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))
for i in range(len(theta0_vals)):
	for j in range(len(theta1_vals)):
		t = np.array(([theta0_vals[i]], [theta1_vals[j]]))
		J_vals[i, j] = computeCost(X, y, t)
J_vals = J_vals.transpose()
fig = plt.figure()
ax = plt.gca(projection='3d')
surf = ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.show()
plt.contour(theta0_vals, theta1_vals, J_vals)
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.show()

#ex1_multi.m
#Feature Normalization
print('Loading data ...\n')
data = pd.read_csv('ex1data2.csv')
X = data[['X1', 'X2']]
y = data['y']
m = len(y)
print('First 10 examples from the dataset: \n')
print(X.head(10))
print(y.head(10))
print('Normalizing Features ...\n')
X = X.to_numpy()
y = y.to_numpy()
X, mu, sigma = featureNormalize(X)
X = np.concatenate((np.ones((max(X.shape), 1)), X), axis = 1)

#Gradient Descent
print('Running gradient descent ...\n')
alpha = 0.001
num_iters = 10000
theta = np.zeros((3, 1))
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)
plt.plot(J_history, '-b')
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()
print('Theta computed from gradient descent: \n')
print(theta)
print('\n')
price = theta[0] + (1650 - mu[0]) / sigma[0] * theta[1] + (3 - mu[1]) / sigma[1] * theta[2]
print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):', price[0])

#Normal Equations
print('Solving with normal equations...\n')
data = pd.read_csv('ex1data2.csv')
X = data[['X1', 'X2']]
y = data['y']
m = len(y)
X = X.to_numpy()
y = y.to_numpy()
X = np.concatenate((np.ones((max(X.shape), 1)), X), axis = 1)
theta = normalEqn(X, y)
print('Theta computed from normal equations: \n')
print(theta)
print('\n')
price = theta[0] + theta[1] * 1650 + theta[2] * 3
print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations):', price)