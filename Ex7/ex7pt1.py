# Instead of using the fix centriods as per the exercise, this algorithm is capable of performing
# K-means clustering with a random centriods position. 
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
np.random.seed(1)

def plotData(num_centriods, data, centriods_history, label):
    # Scatter plot of training points with different color indicating different clusters
    colors = ['y', 'r', 'g', 'm', 'b', 'w', 'c', 'k']
    for i in range(1, num_centriods+1):
        idx = np.where(label == i)
        x = data[:, 0][idx]
        y = data[:, 1][idx]
        plt.scatter(x, y, color=colors[i-1], label=f'Cluster {i}')
    x_ctr = centriods_history[-1][:, 0]
    y_ctr = centriods_history[-1][:, 1]
    # Plot centriods with star
    plt.plot(x_ctr, y_ctr, '*', markersize='15', color='k', label='centriod')
    plt.legend()
    plt.show()

def kMeansInitCentroids(num_centriods, data):
    "Function to initialize random centriods position according to number of centroids chosen"
    x = data[:, 0]
    y = data[:, 1]
    x_centroids, y_centroids = [], []
    for _ in range(1, num_centriods+1):
        x_cen = np.random.randint(min(x), max(x))
        y_cen = np.random.randint(min(y), max(y))
        x_centroids.append(x_cen)
        y_centroids.append(y_cen)
    return np.array(list(zip(x_centroids, y_centroids)))

def findClosestCentroids(num_centriods, data, centroids):
    "Label training data with centroids that has min eucledian distanace"
    dist = np.zeros((data.shape[0], num_centriods))
    for i in range(num_centriods):
        for j in range(dist.shape[0]):
            dist[j, i] = np.linalg.norm(data[j] - centroids[i])
    label = np.argmin(dist, axis=1) + 1
    return label

def computeMeans(num_centriods, data, centriods):
    "Function to compute new centroids"
    label = findClosestCentroids(num_centriods, data, centriods)
    x_centroids_new, y_centroids_new = [], []
    for i in range(1, num_centriods+1):
        idx   = np.where(label == i)
        x_new = np.mean(data[:, 0][idx])
        y_new = np.mean(data[:, 1][idx])
        x_centroids_new.append(x_new)
        y_centroids_new.append(y_new)
    return np.array(list(zip(x_centroids_new, y_centroids_new))), label

if __name__ == "__main__":
    data = loadmat('ex7data2.mat')['X']

    # K-means clustering
    num_centriods     = 3
    centriods_history = []
    centriods         = kMeansInitCentroids(num_centriods, data)
    centriods_history.append(centriods)
    for iter in range(300):
        new_mean, cluster = computeMeans(num_centriods, data, centriods)
        if (new_mean == centriods).all():
            break
        else:
            centriods = new_mean
            centriods_history.append(centriods)

    plotData(num_centriods, data, centriods_history, cluster)



