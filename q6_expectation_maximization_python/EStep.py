import numpy as np
from getLogLikelihood import getLogLikelihood
from numpy import matrix
import math
def EStep(means, covariances, weights, X):
    # Expectation step of the EM Algorithm
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each Gaussian DxDxK
    # X              : Input data NxD
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.

    X = matrix(X)
    means = matrix(means)
    weights = matrix(weights)
    K = len(means)
    N,D = X.shape

    gamma = np.zeros([N, K])
    Sum = [0]*N
    for i in range(0, N):
        for j in range(0, K):
            mean = means[j, :]
            if weights.ndim == 2:
                weight = weights[0, j]
            else:
                weight = weights[j]
            covariance = matrix(covariances[:, :, j])
            Sum[i] = Sum[i] + weight/(((2 * math.pi)**(D/2))*((np.linalg.det(covariance))**0.5))*math.exp(
                   -0.5*(X[i, :]-mean)*np.linalg.inv(covariance) * (X[i, :]-mean).T)
    for i in range(0, N):
        for j in range(0, K):
            mean = means[j, :]
            if weights.ndim == 2:
                weight = weights[0, j]
            else:
                weight = weights[j]
            covariance = matrix(covariances[:, :, j])
            data = X[i, :]
            inv = np.linalg.inv(covariance)
            t = np.linalg.det(-0.5 * (data - mean) * inv * ((data - mean).T))
            a = weight/(((2 * math.pi)**(D/2))*((np.linalg.det(covariance))**0.5))*np.exp(t)/Sum[i]
            gamma[i, j] = a
    logLikelihood = getLogLikelihood(means, weights, covariances, X)
    return logLikelihood, gamma
