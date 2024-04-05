import numpy as np
import math


def tf_anchor(x, y, theta, width, height):
    xr = x - math.cos(theta) * width + math.sin(theta) * height/2
    yr = y - math.sin(theta) * width - math.cos(theta) * height/2
    return (xr, yr)


def model_checking(x, z, spec, k):

    L = (1 + z.shape[0]) * z.shape[1]
    xx = np.zeros([L, z.shape[2]])

    for i in range(z.shape[2]):
        xx[:z.shape[1], i] = x[:, i]
        xx[z.shape[1]:, i] = z[:, :, i].reshape(1, -1)[0]

    rho = spec.robustness(xx, k)

    return rho