# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 17:44:01 2023

@author: Banquise
"""


import numpy as np
import matplotlib.pyplot as plt


uuu = np.zeros((100,80,200))
f = 0.1

for i in range(uuu.shape[0]) :
    print(i)
    for j in range(uuu.shape[1]) :
        for k in  range (uuu.shape[2]):
            for w in range (3):
                uuu[i,j,k] += np.sin((i*0.2 + j + k) / f* w)
#%%
plt.figure()
plt.pcolormesh(uuu[:,:,2])


#%%
import scipy.fft as fft

YY = fft.fft2(uuu - np.nanmean(uuu))

plt.figure()
plt.pcolormesh(np.abs(YY[0,:,:]))

def demodulation(t,s, fexc):
    c = np.nanmean(s*np.exp(1j * 2 * np.pi * t[None,None,:] * fexc),axis=2)
    return c


#%%

facq = 1
fmin = 0.001
fmax = 0.2
nb_f = 100

padding = 10     #puissance de 2 pour le 0 padding
k_xx = []
k_yy = []
kk = []
theta = []
fff = []

nx,ny,nt = uuu.shape

t = np.linspace(0,facq, nt)
plt.figure()
for i in np.linspace(fmin, fmax, nb_f) :
    print(i)
    demod = demodulation(t,uuu,i)
    demod_padding = np.zeros((2**padding,2**padding), dtype = 'complex128')
    for l in range (nx):
        for m in range(ny):
            demod_padding[l,m] = demod_padding[l,m] + demod[l,m]
    Y_FFT = fft.fft2(demod_padding)
    Y_FFT = fft.fftshift(Y_FFT)
    nx_FFT = np.shape(Y_FFT)[0]
    ny_FFT = np.shape(Y_FFT)[1]
    max_fft = np.asarray([(np.where(np.max(np.abs(Y_FFT)) == np.abs(Y_FFT))[0][0] - nx_FFT/2) / nx_FFT,
                          (np.where(np.max(np.abs(Y_FFT)) == np.abs(Y_FFT))[1][0] - ny_FFT/2) / ny_FFT])
    
    k_xx.append(max_fft[0])
    k_yy.append(max_fft[1])
    kk.append(np.sqrt(max_fft[0]**2 + max_fft[1]**2))
    theta.append(max_fft[0]/max_fft[1])
    fff.append(i)

#%%

plt.figure()
plt.plot(fff,theta)
plt.figure()
plt.plot(fff,kk)

plt.figure()
plt.pcolormesh(np.abs(Y_FFT))


plt.figure()
plt.pcolormesh(np.real(demod))

#%%
plt.figure()
for i in range (10) : 
    plt.pcolormesh(np.abs(YY[:,:,i]))
    plt.pause(0.2)


#%%

Y_demod = fft.fft2(demod)
plt.figure()
plt.pcolormesh(np.real(Y_demod))

k_max_fft_x = nx_FFT/2 - np.where(np.max(np.abs(Y_FFT)) == np.abs(Y_FFT))[0][0]
k_max_fft_y = ny_FFT/2 - np.where(np.max(np.abs(Y_FFT)) == np.abs(Y_FFT))[1][0]/ny_FFT


















