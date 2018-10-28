import numpy as np
from numpy import matrix
import math

def getLogLikelihood(means, weights, covariances, X):
    # Log Likelihood estimation
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each gaussian DxDxK
    # X              : Input data NxD
    # where N is number of data points
    # D is the dimension of the data points
    # K is number of gaussians
    #
    # OUTPUT:
    # logLikelihood  : log-likelihood
    K = len(weights)
    means = matrix(means)
    X = matrix(X)
    N,D = X.shape
    logLikelihood = 0
    # initialize the loglikelihood
    for i in range(0, N):
        logLikelihood_temp = 0
        for j in range(0, K):
            mean = means[j, :]
            if len(weights) == 1:
                weight = weights[0, j]
            else:
                weight = weights[j]
            covariance = matrix(covariances[:, :, j])
            logLikelihood_temp = logLikelihood_temp+weight/(((2 * math.pi)**(D/2))*((np.linalg.det(covariance))**0.5))*math.exp(
                   -0.5*(X[i, :]-mean)*np.linalg.inv(covariance) * (X[i, :]-mean).T)
        if logLikelihood_temp != 0:
            logLikelihood = logLikelihood + np.log(logLikelihood_temp)
    return logLikelihood

