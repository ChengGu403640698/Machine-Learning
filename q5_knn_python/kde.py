import numpy as np
import math


def kde(samples, h):
    # compute density estimation from samples with KDE
    # Input
    #  samples    : DxN matrix of data points
    #  h          : (half) window size/radius of kernel
    # Output
    #  estDensity : estimated density in the range of [-5,5]
    N = len(samples)
    D = 1
    estDensity = np.ones((N,2))*0
    pos = np.arange(-5, 5.0, 0.1)
    estDensity[:, 0] = pos
    for i in range(0, 100):
      for j in range(0, N):
        if(abs(samples[j]- pos[i]) <= h):
            u = abs(samples[j]- pos[i])
            k = 1/((8*math.pi*h*h)**0.5) * math.exp(-u*u/(8*h*h))
            estDensity[i, 1] = estDensity[i, 1] + k/(2*h*N)
    # Compute the number of samples created
    return estDensity
