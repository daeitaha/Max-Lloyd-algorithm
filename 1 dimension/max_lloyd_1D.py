
# Max-Lloyd algorithm for finding the optimal quantizer
# in dimension 1

import math
import random
import scipy
import matplotlib.pyplot as plt
from scipy import integrate
import numpy as np
import time
import torch


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

# def trapezoid_fast_integral(f, thresholds, n_int=1000):
#     n_th = thresholds.size
#     grid = np.linspace(thresholds[np.arange(n_th-1)],thresholds[np.arange(n_th-1)+1],n_int)
#     delta =  grid[1,0]-grid[0,0]
#     f_grid = f(grid)
#     f_grid[1:-1,:] = 2*f_grid[1:-1,:]
#     integration = np.sum(f_grid*delta/2,axis=0)
    
#     return integration
    
# f = lambda x: x**2
# thresholds = np.arange(10)
# y = trapezoid_fast_integral(f, thresholds, n_int=1000)

# computes MSE between 2 adjacent decision thresholds (on one segment)
# def interval_MSE(x, t1, t2, f):
#     return integrate.quad(lambda t: ((t - x)**2) * f(t), t1, t2)[0]

# # computes mean squared error on R
# def MSE(t, x, f):
#     # s = interval_MSE(x[0], -float('Inf'), t[0], f) + interval_MSE(x[-1], t[-1], float('Inf'), f)
#     s = interval_MSE(x[0], -np.pi, t[0], f) + interval_MSE(x[-1], t[-1], np.pi, f)
#     for i in range(1,len(x)-1):
#         s = s + interval_MSE(x[i], t[i-1], t[i], f)
#     return s

# # t1 and t2 are the boundaries of the interval on which the centroid is calculated
# def centroid(t1, t2, f):
#     if integrate.quad(f, t1, t2)[0] == 0 or t1 == t2:
#         # return 0
#         return (t1+t2)/2
#     else:
#         return integrate.quad(lambda t:t*f(t), t1, t2)[0] / integrate.quad(f, t1, t2)[0]
    
def quantization_error_calc(f, x , thresholds, n_int=1000):
    n_th           = thresholds.size
    grid           = np.linspace(thresholds[np.arange(n_th-1)],thresholds[np.arange(n_th-1)+1],n_int)
    delta          =  grid[1,:]-grid[0,:]
    f_grid         = f(grid)
    f_grid[1:-1,:] = 2*f_grid[1:-1,:]
    error          = np.sum(f_grid*((grid-x)**2)*delta/2)
    return error


def quantization_error_calc_torch(f, x , thresholds, n_int=1000):
    device         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_th           = thresholds.size
    grid           = torch.linspace(thresholds[np.arange(n_th-1)],thresholds[np.arange(n_th-1)+1],n_int, device=device)
    delta          =  grid[1,:]-grid[0,:]
    f_grid         = f(grid)
    f_grid[1:-1,:] = 2*f_grid[1:-1,:]
    error          = torch.sum(f_grid*((grid-x)**2)*delta/2) 
    return error

def centroid_calc(f, thresholds, n_int=1000):
    n_th           = thresholds.size
    grid           = np.linspace(thresholds[np.arange(n_th-1)],thresholds[np.arange(n_th-1)+1],n_int)
    delta          =  grid[1,:]-grid[0,:]
    f_grid         = f(grid)
    f_grid[1:-1,:] = 2*f_grid[1:-1,:]
    
    num_int     = np.sum(f_grid*grid*delta/2,axis=0)
    denum_int   = np.sum(f_grid*delta/2,axis=0)
    
    x      = num_int/denum_int
    idx    = np.where(denum_int==0)[0]
    x[idx] = (thresholds[idx] + thresholds[idx+1])/2
    
    error   = np.sum(f_grid*((grid-x)**2)*delta/2) 
    
    return x, error


def centroid_calc_torch(f, thresholds, n_int=1000):
    device         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_th           = thresholds.size
    grid           = torch.linspace(thresholds[np.arange(n_th-1)],thresholds[np.arange(n_th-1)+1],n_int, device=device)
    delta          =  grid[1,:]-grid[0,:]
    f_grid         = f(grid)
    f_grid[1:-1,:] = 2*f_grid[1:-1,:]
    
    num_int     = torch.sum(f_grid*grid*delta/2,axis=0)
    denum_int   = torch.sum(f_grid*delta/2,axis=0)
    
    x      = num_int/denum_int
    idx    = torch.where(denum_int==0)[0]
    x[idx] = (thresholds[idx] + thresholds[idx+1])/2
    error  = torch.sum(f_grid*((grid-x)**2)*delta/2) 
    
    return x, error

# def centroid_paralel(f, t):
#     y = f(t)
#     num_int     = np.diff(integrate.cumulative_trapezoid(y*t, t, initial=0))
#     denum_int   = np.diff(integrate.cumulative_trapezoid(y, t, initial=0))
#     output = num_int/denum_int
#     idx1 = np.where(denum_int==0)[0]
#     idx2 = np.unique(np.concatenate((idx1, idx1+1)))
#     output[idx1] = moving_average(t[idx2], n=2).squeeze()
    
#     return output

#     # if integrate.quad(f, t1, t2)[0] == 0 or t1 == t2:
#     #     # return 0
#     #     return (t1+t2)/2
#     # else:
#     #     return integrate.quad(lambda t:t*f(t), t1, t2)[0] / integrate.quad(f, t1, t2)[0]

# t is an array containing the initial decision thresholds
# x is an array containing the representation levels
# error_threshold is the threshold to reach for the algorithm to stop
def maxlloyd_torch(input_t, input_x, f, error_threshold, max_iter=10):
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x       = 1*input_x
    t       = 1*input_t
    e       = quantization_error_calc_torch(f, x , t, n_int=1000)
    error   = [e]
    c       = 0
    while e > error_threshold and c < max_iter:
        c       = c+1
        t       =  moving_average_torch(x, n=2)
        x, e    = centroid_calc_torch(f, torch.concatenate((torch.tensor([-torch.pi]),t.squeeze(),torch.tensor([torch.pi]))),n_int=1000)
        error.append(e)
        if ((error[-1]/error[-2])>0.8):
            break
    return x, t, error


def maxlloyd(input_t, input_x, f, error_threshold, max_iter=10):
    x       = 1*input_x
    t       = 1*input_t
    e       = quantization_error_calc(f, x , t, n_int=1000)
    error   = [e]
    c       = 0
    while e > error_threshold and c < max_iter:
        
        c    = c+1
        t    =  moving_average(x, n=2)
        x, e = centroid_calc(f, np.concatenate((np.array([-np.pi]),t.squeeze(),np.array([np.pi]))),n_int=1000)
        error.append(e)
        if ((error[-1]/error[-2])>0.8):
            break
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


def moving_average_torch(a, n=3):
    if (len(a.shape)<2):
        a = a.reshape(-1,1)
    if (torch.is_complex(a)):
        ret = torch.cumsum(a, dtype=torch.complex128, axis=0)
    else:
        ret = torch.cumsum(a, dtype=torch.float64, axis=0)
    ret[n:,:] = ret[n:,:] - ret[:-n,:]
    return ret[n - 1:,:] / n


# k = 10
# mu = 0
# f = lambda t: k*np.cos(t-mu)

# # Test of maxlloyd function
# def test_maxlloyd():
#     # t = [-0.5,0,0.5]
#     # print(t)
#     # x = [-1,0,1,1.5]
#     x = np.arange(-30,31)/60*2*np.pi
#     t = moving_average(x,n=2)
#     print(t)
#     # mu = 0
#     # sigma = 1
#     # f = lambda t: np.exp(-(t-mu)**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)

    
#     x2,t2,error = maxlloyd(t, x, f, 0.001)
#     # print(x2,t2)
#     plt.plot(error)
#     plt.show()
#     print(t2)
#     return x2,t2

# x2,t2 = test_maxlloyd()
# x = np.arange(-30,31)/60*2*np.pi
# t = moving_average(x,n=2)
# plt.plot(x,f(x)/2/np.pi/np.i0(k),'-ro')
# plt.show()
# plt.plot(x2,f(x2)/2/np.pi/np.i0(k),'-bo')
# plt.show()
# x2 = np.array(x2).squeeze()
# print(np.abs(np.diff(x2)))

# def estimate(x,t,value):
#     for i in range(len(t)):
#         if t[i] > value:
#             return x[i]
#     return x[-1]

# # Plot of average error
# def plot_avg_error(N):
#     x,t = test_maxlloyd()
#     avg_E = []
#     realizations = []
#     square_error = []
#     for i in range(N):
#         realizations.append(random_distrib())
#         square_error.append((realizations[-1] - estimate(x,t,realizations[-1]))**2)
#         avg_E.append(sum(square_error)/len(square_error))
#     plt.figure(2)
#     plt.plot(avg_E)
#     plt.show()

# plot_avg_error(20000)

































