import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def plotData(positive, negative):
    "Function to plot training data"
    plt.scatter(positive[:, 0], positive[:, 1], marker='+', label='Accept')
    plt.scatter(negative[:, 0], negative[:, 1], label='Rejected')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')

def mapFeature(x1, x2):
    "Function to create more feature by mapping"
    degree = 6
    out = np.ones((len(x1), 1))

    for i in range(1, degree + 1):
        for j in range(i + 1):
            map = ((x1 ** i-j) * (x2 ** j)).reshape(len(x1), 1)
            out = np.hstack((out, map))
    return out

def sigmoid(z):
    "Function to compute sigmoid function"
    return 1 / (1 + np.exp(-z))

def costFunction(theta, x, y, reg_factor):
    "Function to compute logistic regression with regularization cost function"
    m = len(x)
    h = x @ theta # 118x1 matrix
    J = ((-y.T @ np.log(sigmoid(h)) - (1 - y.T) @ np.log(1 - sigmoid(h))) / m) + ((theta.T @ theta) * (reg_factor / (2 * m)))
    return J

def gradient(theta, x, y, reg_factor):
    "Function to compute logistic regression with regularization gradient "
    m = len(x)
    h = x @ theta # 118x1 matrix
    gradient = (x.T @ (sigmoid(h) - y)) / m
    gradient[1:] = gradient[1:] + (reg_factor / m) * theta[1:]
    return gradient

def minimizeTheta(theta, x, y, reg_factor):
    "Function to optimize cost function with lbfgs"
    minimum = optimize.fmin_l_bfgs_b(costFunction, x0=theta, fprime=gradient ,args=(x, y.flatten(), reg_factor), maxiter=400, iprint=0)
    return minimum

def mapFeaturePlot(x1, x2):
    "Function to create more feature by mapping"
    degree = 6
    out = np.ones(1)

    for i in range(1, degree + 1):
        for j in range(i + 1):
            map = ((x1 ** i-j) * (x2 ** j))
            out = np.hstack((out, map))
    return out

def plotDecisionBoundary(minimum, positive, negative, reg_factor):
    "Function to plot training data with decision boundary"
    theta = minimum[0]
    u = np.linspace(-1, 1.5, 50) 
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros((len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            z[i][j] = mapFeaturePlot(u[i], v[j]) @ theta

    plt.figure()
    plotData(positive, negative)
    plt.contour(u, v, z.T, levels=0)
    plt.title(f'Lambda = {reg_factor}')
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':

    data = np.loadtxt('ex2data2.txt', delimiter=',')

    x = data[:, 0:2].reshape(len(data), 2) # 118x2 matrix
    y = data[:, 2].reshape(len(data), 1) # 118x1 matrix

    # Data for graph plotting
    accept    = np.where(y == 1)
    reject    = np.where(y == 0)
    positive  = x[accept[0]]
    negative  = x[reject[0]]

    # Create more features
    x_b = mapFeature(x[:, 0], x[:, 1]).reshape(len(x), 28) # 118x28 matrix
    
    # Initialize parameters
    theta    = np.zeros((x_b.shape[1], 1)) # 28x1 matrix
    lambda_b = 1
    j        = costFunction(theta, x_b, y, lambda_b)
    grad     = gradient(theta, x_b, y, lambda_b)
    minimum  = minimizeTheta(theta, x_b, y, lambda_b)

    # Plot of various graph with different regularization factor
    for lambda_b in [0, 1, 10, 100]:
        min_theta = minimizeTheta(theta, x_b, y, lambda_b)
        plotDecisionBoundary(min_theta, positive, negative, lambda_b)







    
