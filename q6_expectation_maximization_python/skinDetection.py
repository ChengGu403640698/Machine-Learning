import numpy as np
from estGaussMixEM import estGaussMixEM
from getLogLikelihood import getLogLikelihood
from numpy import matrix

def skinDetection(ndata, sdata, K, n_iter, epsilon, theta, img):
    # Skin Color detector
    #
    # INPUT:
    # ndata         : data for non-skin color
    # sdata         : data for skin-color
    # K             : number of modes
    # n_iter        : number of iterations
    # epsilon       : regularization parameter
    # theta         : threshold
    # img           : input image
    #
    # OUTPUT:
    # result        : Result of the detector for every image pixel
    n_weight, n_mean, n_covariance = estGaussMixEM(ndata, K, n_iter, epsilon)
    s_weight, s_mean, s_covariance = estGaussMixEM(sdata, K, n_iter, epsilon)

    result = img.copy()

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            px_given_n = np.exp(getLogLikelihood(n_mean, n_weight, n_covariance, result[i, j]))
            px_given_s = np.exp(getLogLikelihood(s_mean, s_weight, s_covariance, result[i, j]))
            if px_given_s / px_given_n > theta:
                result[i, j] = [1, 1, 1]
            else:
                result[i, j] = [0, 0, 0]


    return result
