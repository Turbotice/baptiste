# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 16:53:23 2023

@author: Banquise
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.fft as fft
from baptiste.signal_processing import smooth_signal

def aleaGauss(sigma):
    U1 = random.random()
    U2 = random.random()
    return sigma*np.sqrt(-2*np.log(U1))*np.cos(2*np.pi*U2)




sigma = 1
length = 500

x = np.linspace(0,length,length)


randomm = []
for i in range(length) :
    randomm.append(aleaGauss(sigma))
randomm = np.asarray(randomm)



randomm = randomm / 2 + 15

randomm = smooth_signal.savgol(randomm, 50, 2) * 2

plt.figure()
plt.plot(x, randomm)



# plt.figure()
# plt.plot(np.abs(fft.fft(randomm)) - np.mean(randomm))

bruit_blanc = np.vstack((x, randomm))



np.savetxt('D:\Banquise\\Baptiste\\Resultats\\bruit_blanc.txt', randomm)