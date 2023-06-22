#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 01:13:09 2023

@author: taha
"""

from max_lloyd_1D import maxlloyd
import numpy as np
from max_lloyd_1D import moving_average

import time



input_x = np.arange(-np.pi,np.pi,0.2)
input_t = moving_average(input_x, n=2)
k = 100
f = lambda x: np.exp(k*(np.cos(x-0.7)-1))
t_seconds = time.time()
x, t, error = maxlloyd(input_t, input_x, f, 0.0001)
print('elapsed %.3f'%(time.time()-t_seconds))

print(x)
print(t)