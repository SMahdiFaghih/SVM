from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import random


def generateSamples(sampleType, kernel, noise):
    global X, Y
    if not kernel in ["linear", "poly", "rbf", "sigmoid"]:
        print("input kernel is not correct")
        exit()
    if sampleType == "blob":
        numOfSamples = 500
        X, Y = make_blobs(n_samples=numOfSamples, centers=2, cluster_std=noise * 2)
    elif sampleType == "moon":
        numOfSamples = 500
        X, Y = make_moons(n_samples=numOfSamples, noise=noise)
    elif sampleType == "circle":
        numOfSamples = 250
        X, Y = make_circles(numOfSamples, noise=noise)
    else:
        print("input sampleType is not correct")
        exit()

    plt.scatter(X[Y == 0, 0], X[Y == 0, 1], color="r", s=100, label="class1", marker='+', edgecolors='black')
    plt.scatter(X[Y == 1, 0], X[Y == 1, 1], color="b", s=50, label="class2", marker='x', edgecolors='black')
    model = SVC(kernel=kernel)
    model.fit(X, Y)
    return model


def plotMargin(model):
    ax = plt.gca()
    xLim = ax.get_xlim()
    yLim = ax.get_ylim()

    x = np.linspace(xLim[0], xLim[1], 30)
    y = np.linspace(yLim[0], yLim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    ax.contour(X, Y, P, colors='black', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

def startClassification():
    model = generateSamples(sampleType="blob", kernel="rbf", noise=random.uniform(0, 1))
    plotMargin(model)
    plt.legend()
    plt.show()
