import numpy as np
import matplotlib.pyplot as plt
from regex import E

def plotData(x, x_b, y, theta):
    plt.scatter(x, y, marker='x', c='orange')
    plt.plot(x, x_b @ theta)
    plt.legend(['Training Data', 'Linear Regression'])
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show()

def computeCost(x, y, theta):
    "Function to compute cost function (MSE)"
    m    = len(x)
    h    = x @ theta # Model (97x1 matrix)
    e_sq = np.square((h - y)) # Error^2 (97x1 matrix)
    J    = np.sum(e_sq, axis=0) / (2 * m) # Mean Square Errors
    return J

def gradientDescent(x, y, theta, alpha, iterations):
    "Function to perform gradient descent"
    J_history = []
    m         = len(x)
    for _ in range(iterations): 
        h        = x @ theta
        error    = h - y
        gradient = (x.T @ error) / m # Simulatenous update of theta 0 and theta 1
        theta -= alpha * gradient
        J = computeCost(x, y, theta)
        J_history.append(J)
    return theta, J_history
    
def visualJ(x, y):
    "Function to visualize cost function J"
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

    for i in range(len(theta0_vals)):
        for j in range(len(theta1_vals)):
            t = np.array([theta0_vals[i], theta1_vals[j]]).reshape(2, 1)
            J_vals[i][j] = computeCost(x, y, t)

    ax = plt.axes(projection='3d')
    ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap='rainbow')
    ax.set_xlabel('theta 0')
    ax.set_ylabel('theta 1')
    ax.set_zlabel('Cost Func J')
    ax.set_title('Cost Function Across Different Theta Value')
    plt.show()

if __name__ == '__main__':

    data = np.loadtxt('ex1data1.txt', delimiter=',')

    x = data[:, 0].reshape(97, 1)
    y = data[:, 1].reshape(97, 1)
    
    # Initialize parameters
    b     = np.ones((len(x), 1))
    x_b   = np.concatenate((b, x), axis=1) # Add column for bias, b
    theta = np.zeros((2, 1)) # Initialize weights to zeros. 2 weights as only 2 features
    alpha = 0.01 # Learning Rate
    iterations = 1500

    # Compute cost function and gradient descent
    J = computeCost(x_b, y, theta)
    theta, J_history = gradientDescent(x_b, y, theta, alpha, iterations)

    # Print results and plot linear fit
    print(f'Theta computed from gradient descent: {theta[0]}, {theta[1]}')
    plotData(x, x_b, y, theta)
    visualJ(x_b, y)



    
