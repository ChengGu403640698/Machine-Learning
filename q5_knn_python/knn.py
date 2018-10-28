import numpy as np


def knn(samples, k):
    # compute density estimation from samples with KNN
    # Input
    #  samples    : DxN matrix of data points
    #  k          : number of neighbors
    # Output
    #  estDensity : estimated density in the range of [-5, 5]

    N = len(samples)
    D = 1
    disrance_cal = np.ones((N, N)) * 0
    estDensity = np.ones((N, 2)) * 0
    pos = np.arange(-5, 5.0, 0.1)
    estDensity[:, 0] = pos
    for i in range(0, 100):
        for j in range(0, N):
            disrance_cal[i,j] = abs(samples[j]-pos[i])
    for i in range(0, 100):
        for m in range(1,k+1):
         a = disrance_cal[i,:]
         j = a.argmin(axis=0)
         h = disrance_cal[i, j]
         disrance_cal[i, j]  = 100000
         estDensity[i,1]=k/(2*h*N)
    # Compute the number of the samples created
    return estDensity
