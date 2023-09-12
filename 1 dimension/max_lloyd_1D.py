

# Max-Lloyd algorithm for finding the optimal quantizer
# in dimension 1

import math
import random
import scipy
import matplotlib.pyplot as plt
from scipy import integrate
import numpy as np



# def uniform(t):
#     if t<=1 and t>=-1:
#         return 0.5
#     else:
#         return 0

# def gaussian(t):
#     return math.exp(-t**2/2)/math.sqrt(2*math.pi)

# # function studied: chose between uniform and gaussian functions
# def f(t):
#     return gaussian(t)

# # distribution studied
# def random_distrib():
#     return random.uniform(-1,1)
#     #return random.gauss(0,1)

# computes MSE between 2 adjacent decision thresholds (on one segment)
def interval_MSE(x, t1, t2, f):
    return integrate.quad(lambda t: ((t - x)**2) * f(t), t1, t2)[0]

# computes mean squared error on R
def MSE(t, x, f):
    # s = interval_MSE(x[0], -float('Inf'), t[0], f) + interval_MSE(x[-1], t[-1], float('Inf'), f)
    s = interval_MSE(x[0], -np.pi, t[0], f) + interval_MSE(x[-1], t[-1], np.pi, f)
    for i in range(1,len(x)-1):
        s = s + interval_MSE(x[i], t[i-1], t[i], f)
    return s

# t1 and t2 are the boundaries of the interval on which the centroid is calculated
def centroid(t1, t2, f):
    if integrate.quad(f, t1, t2)[0] == 0 or t1 == t2:
        # return 0
        return (t1+t2)/2
    else:
        return integrate.quad(lambda t:t*f(t), t1, t2)[0] / integrate.quad(f, t1, t2)[0]

# t is an array containing the initial decision thresholds
# x is an array containing the representation levels
# error_threshold is the threshold to reach for the algorithm to stop
def maxlloyd_orig(input_t, input_x, f, error_threshold):
    x = 1*input_x
    t = 1*input_t
    e = MSE(t, x, f)
    error = [e]
    c = 0
    while e > error_threshold and c < 10:
        c = c+1
        if c%2 == 1:
            # adjust thresholds
            # for i in range(len(t)):
            #     t[i] = 0.5 * ( x[i] + x[i+1] )
            t =  moving_average(x, n=2)
        else:
            # adjust levels
            # x[0] = centroid(-float('Inf'), t[0], f)
            x[0] = centroid(-np.pi, t[0], f)
            # x[-1] = centroid(t[-1], float('Inf'), f)
            x[-1] = centroid(t[-1], np.pi, f)
            for i in range(1,len(x)-1):
                x[i] = centroid(t[i-1], t[i], f)
            e = MSE(t, x, f)
            error.append(e)
            if ((error[-1]/error[-2])>0.8):
                break
        # print(e)
    return x, t, error

def moving_average(a, n=3):
    if (len(a.shape)<2):
        a = a.reshape(-1,1)
    if (np.iscomplexobj(a)):
        ret = np.cumsum(a, dtype=np.complex128, axis=0)
    else:
        ret = np.cumsum(a, dtype=np.float64, axis=0)
    ret[n:,:] = ret[n:,:] - ret[:-n,:]
    return ret[n - 1:,:] / n























