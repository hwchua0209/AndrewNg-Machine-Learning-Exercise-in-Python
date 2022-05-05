from cProfile import label
from cmath import e, exp
from turtle import position
import numpy as np
import matplotlib.pyplot as plt
from regex import E, X
from scipy import optimize
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def plotData(positive, negative):
    "Function to plot training data"
    plt.scatter(positive[:, 0], positive[:, 1], marker='+', label='Admitted')
    plt.scatter(negative[:, 0], negative[:, 1], label='Not Admitted')
    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')

def sigmoid(z):
    "Function to compute sigmoid function"
    return 1 / (1 + np.exp(-z))

def costFunction(theta, x, y):
    "Function to compute logistic regression cost function"
    m = len(x)
    h = x @ theta # 100x1 matrix
    J = (-y.T @ np.log(sigmoid(h)) - (1 - y.T) @ np.log(1 - sigmoid(h))) / m
    return J

def gradient(theta, x, y):
    "Function to compute logistic regression gradient"
    m = len(x)
    h = x @ theta # 100x1 matrix
    gradient = (x.T @ (sigmoid(h) - y)) / m
    return gradient

def minimizeTheta(theta, x, y):
    "Function to optimize cost function with lbfgs"
    minimum = optimize.fmin_l_bfgs_b(costFunction, x0=theta, fprime=gradient ,args=(x, y.flatten()), maxiter=400, iprint=0)
    return minimum

def plotDecisionBoundary(minimum, positive, negative, x):
    "Function to plot training data with decision boundary"
    theta = minimum[0]
    # Calculate the x and y limit of the graph 
    x_boundary = [0, -theta[0] / theta[1]]
    y_boundary = [-theta[0] / theta[2], 0]

    plotData(positive, negative)
    plt.plot(x_boundary, y_boundary, label='Decision Boundary')
    plt.xlim(min(x[:, 0] - 5), max(x[:, 0] + 5))
    plt.ylim(min(x[:, 1] - 5), max(x[:, 1] + 5))
    plt.legend(loc='best')
    plt.show()

def predict(minimum, exam1, exam2):
    "Function to predict with logistic regression"
    x_b   = np.array([1, exam1, exam2]).reshape(1, 3)
    theta = np.array(minimum[0]).reshape(3, 1)
    prob  = sigmoid(x_b @ theta)
    return prob

def accuracy(x, y, minimum):
    "Function to calculate training accuracy"
    theta   = np.array(minimum[0]).reshape(3, 1)
    pred    = sigmoid(x @ theta)
    log_reg = np.where(pred>0.5, 1, 0)
    error   = np.abs(y - log_reg)
    accur   = len(np.where(error == 0)[0]) / len(error)
    return accur

if __name__ == '__main__':

    data = np.loadtxt('ex2data1.txt', delimiter=',')

    x = data[:, 0:2].reshape(len(data), 2) # 100x2 matrix
    y = data[:, 2].reshape(len(data), 1) # 100x1 matrix

    # Data for graph plotting
    admit     = np.where(y == 1)
    not_admit = np.where(y == 0)
    positive  = x[admit[0]]
    negative  = x[not_admit[0]]

    # Initialize parameters
    b     = np.ones((len(x), 1))
    x_b   = np.concatenate((b, x), axis=1) # 100x3 matrix
    theta = np.zeros((x_b.shape[1], 1)) # 3x1 matrix
    
    j = costFunction(theta, x_b, y)
    minimum = minimizeTheta(theta, x_b, y)

    plotDecisionBoundary(minimum, positive, negative, x)

    # Prediction input
    exam1_score = 45
    exam2_score = 85

    # Prediction
    proba = float(predict(minimum, exam1_score, exam2_score))
    print(f'For a student with scores {exam1_score} and {exam2_score}, we predict an admission probability of {(proba * 100):.2f} %')

    metric = accuracy(x_b, y, minimum)
    print(f'The accuracy of the trained model on training set is {metric}')






    
