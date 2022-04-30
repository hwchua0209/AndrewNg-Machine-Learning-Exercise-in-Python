import numpy as np
import matplotlib.pyplot as plt
from regex import E

def featureNormalize(x):
    "Function to Returned Normalized X with Zero Mean and Unit Variance"
    mu    = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    x_norm = (x - mu) / sigma
    return mu, sigma, x_norm

def computeCost(x, y, theta):
    "Function to compute cost function (MSE)"
    m    = len(x)
    h    = x @ theta # Model (47x1 matrix)
    e_sq = np.square((h - y)) # Error^2 (47x1 matrix)
    J    = np.sum(e_sq, axis=0) / (2 * m) # Mean Square Errors
    return J

def gradientDescent(x, y, theta, alpha, iterations):
    "Function to perform gradient descent"
    J_history = []
    m         = len(x)
    for _ in range(iterations): 
        h        = x @ theta
        error    = h - y
        gradient = (x.T @ error) / m # Simulatenous update of theta
        theta -= alpha * gradient
        J = computeCost(x, y, theta)
        J_history.append(J)
    return theta, J_history

def predict(x, theta, mu, sigma):
    "Function to do prediction based on theta computed"
    # Calculate Norm
    x_norm = (x - mu) / sigma
    b   = np.ones((len(x_norm), 1))
    x_b = np.concatenate((b, x_norm), axis=1) # Add column for bias, b
    y = x_b @ theta
    return float(y)

def plotConverge(J_history):
    "Function to plot convergence graph"
    x = np.linspace(1, 50, 50)
    plt.plot(x, J_history[0:50])
    plt.xlabel('iterations')
    plt.ylabel('Cost Func J')
    plt.grid()
    plt.show()

def normalEqn(x, y):
    "Function to solve linear regression with least squares method"
    b   = np.ones((len(x), 1))
    x_b = np.concatenate((b, x), axis=1) # Add column for bias, b
    theta = np.linalg.inv(x_b.T @ x_b) @ x_b.T @ y
    return theta

if __name__ == '__main__':

    data    = np.loadtxt('ex1data2.txt', delimiter=',')
    data_no = len(data)
    x = data[:, 0:2].reshape(data_no, 2)
    y = data[:, 2].reshape(data_no, 1)

    mu, sigma, x_norm = featureNormalize(x)
    b   = np.ones((len(x_norm), 1))
    x_b = np.concatenate((b, x_norm), axis=1) # Add column for bias, b

    # Initialize Parameters
    theta = np.zeros((x_b.shape[1], 1)) # Initialize weights to zeros
    alpha = 0.1 # Learning Rate
    iterations = 400

    # Compute cost function and gradient descent
    J = computeCost(x_b, y, theta)
    theta, J_history = gradientDescent(x_b, y, theta, alpha, iterations)
    plotConverge(J_history)

    # Make prediction for house price
    house_size    = 1650
    no_of_bedroom = 3

    x_pred = np.array([house_size, no_of_bedroom]).reshape(1, 2)
    house_price = predict(x_pred, theta, mu, sigma)

    # Results for gradient descent
    print(70 * '*')
    print(f'Theta computed from gradient descent: {float(theta[0]):.2f}, {float(theta[1]):.2f}, {float(theta[2]):.2f}')
    print(f'The house price from graient descent is ${house_price:.2f}')
    print(70 * '*')

    # Results for LLS
    theta_LLS = normalEqn(x, y)
    house_price_LLS = predict(x_pred, theta_LLS, 0, 1)
    print(f'Theta computed from LLS: {float(theta_LLS[0]):.2f}, {float(theta_LLS[1]):.2f}, {float(theta_LLS[2]):.2f}')
    print(f'The house price from LLS is ${house_price_LLS:.2f}')
    print(70 * '*')
