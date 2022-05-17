from turtle import pos
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.svm import SVC

def loaddata(matfile):
    data = loadmat(matfile)
    X = data['X']
    y = data['y'] 
    return X, y

def loaddata3(matfile):
    data = loadmat(matfile)
    X = data['X']
    y = data['y']
    Xval = data['Xval']
    yval = data['yval']
    return X, y, Xval, yval

def plotData(pos, neg):
    "Function to plot data"
    plt.scatter(pos[:, 0], pos[:, 1], marker='+', label='Positive Sample')
    plt.scatter(neg[:, 0], neg[:, 1], label='Negative Sample')
    plt.legend(loc='best')

def visualBoundary(pos, neg, X, model):
    plotData(pos, neg)
    x0 = np.linspace(X[:, 0].min(), X[:, 0].max(), num=100)
    x1 = np.linspace(X[:, 1].min(), X[:, 1].max(), num=100)
    xx, yy = np.meshgrid(x0, x1)  
    z = model.predict(np.array([xx.flatten(), yy.flatten()]).T)
    z = z.reshape(x0.shape[0], -1)
    plt.contour(xx, yy, z, 0, colors='b')
    plt.legend(loc='best')
    
def train(C, kernel, x, y, gamma_param):
    "sklearn SVM"
    clf = SVC(C=C, kernel=kernel, gamma=gamma_param) #instantiate SVC
    clf.fit(x, y)
    return clf

def gamma_sigma(sigma):
    "Function to calculate gamma from sigma for RBF kernel"
    return 1 / (2 * np.power(sigma, 2))

def cross_validate(x, y, xval, yval):
    cs     = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigmas = cs
    param  = {}
    for c in cs:
        for sigma in sigmas:
            gamma = gamma_sigma(sigma)
            svm   = train(c, 'rbf', x, y.flatten(), gamma)
            model_score = svm.score(xval, yval)
            param[(c, sigma)] = model_score # store all accuracy score as dictionary with c, sigma pair
    best_pair  = max(param, key=param.get)
    best_score = max(param.values())
    return best_pair, best_score

if __name__ == '__main__':
    # ex6data1
    X1, y1 = loaddata('ex6data1.mat') 
    positive_1 = X1[np.where(y1 == 1)[0]]
    negative_1 = X1[np.where(y1 == 0)[0]]
    plotData(positive_1, negative_1)
    plt.show()

    for c in [1, 10, 100]:
        svm_linear = train(c, 'linear', X1, y1.flatten(), 'auto')
        visualBoundary(positive_1, negative_1, X1, svm_linear) # Larger C tries to classify as many points as possible
        plt.title(f'C = {c}')
        plt.show()

    # ex6data2
    X2, y2 = loaddata('ex6data2.mat') 
    positive_2 = X2[np.where(y2 == 1)[0]]
    negative_2 = X2[np.where(y2 == 0)[0]]
    plotData(positive_2, negative_2)
    plt.show() 
    
    for sigma in [0.05, 0.1, 1]: # The higher the gamma, the harder it tries to fit the training data
        gamma = gamma_sigma(sigma)
        svm_rbf = train(1, 'rbf', X2, y2.flatten(), gamma)
        visualBoundary(positive_2, negative_2, X2, svm_rbf)
        plt.title(f'Gamma = {gamma:.2f}, sigma = {sigma:.2f}')
        plt.show()

    # ex6data3
    X3, y3, Xval3, yval3 = loaddata3('ex6data3.mat')
    positive_3 = X3[np.where(y3 == 1)[0]]
    negative_3 = X3[np.where(y3 == 0)[0]]
    plotData(positive_3, negative_3)
    plt.show()

    best_pair, best_score = cross_validate(X3, y3, Xval3, yval3)
    print(f'The best C and sigma value is {best_pair[0]} and {best_pair[1]} with score {best_score}')

    gamma_3 = gamma_sigma(best_pair[1])
    svm_3 = train(best_pair[0], 'rbf', X3, y3.flatten(), gamma_3)
    visualBoundary(positive_3, negative_3, X3, svm_3)
    plt.title(f'C = {best_pair[0]:.2f}, sigma = {best_pair[1]:.2f}')
    plt.show()