import numpy as np
from getLogLikelihood import getLogLikelihood
import math
from numpy import matrix

def MStep(gamma, X):
    # Maximization step of the EM Algorithm
    #
    # INPUT:
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.
    # X              : Input data (NxD matrix for N datapoints of dimension D).
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # means          : Mean for each gaussian (KxD).
    # weights        : Vector of weights of each gaussian (1xK).
    # covariances    : Covariance matrices for each component(DxDxK).

    gamma = matrix(gamma)
    N, K = gamma.shape
    X = matrix(X)
    _, D = X.shape
    Nj = np.sum(gamma,axis=0)
    weights = Nj / N
    means = np.zeros([K, D])
    for j in range(0, K):
        sum = [0] * D
        for i in range(0, N):
          sum = sum + gamma[i, j] * X[i, :]
        means[j, :] = sum / Nj[0, j]
    means = matrix(means)
    covariances = np.zeros([D, D, K])
    for j in range(0, K):
        sum = np.zeros([D, D])
        for i in range(0, N):
            sum = sum + gamma[i, j] * (((X[i, :]-means[j, :])).T*(X[i, :]-means[j, :]))
        covariances[:, :, j] = sum / Nj[0, j]

    logLikelihood = getLogLikelihood(means, weights, covariances, X)

    return weights, means, covariances, logLikelihood
