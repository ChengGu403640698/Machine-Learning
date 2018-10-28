import numpy as np
from numpy import matrix
import math

def regularize_cov(covariance, epsilon):
    # regularize a covariance matrix, by enforcing a minimum
    # value on its singular values. Explanation see exercise sheet.
    #
    # INPUT:
    #  covariance: matrix
    #  epsilon:    minimum value for singular values
    #
    # OUTPUT:
    # regularized_cov: reconstructed matrix
    D = len(covariance)
    a = [1] * D
    I = matrix(np.diag(a))
    regularized_cov = covariance + epsilon * I
    return regularized_cov
