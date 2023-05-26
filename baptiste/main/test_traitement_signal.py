# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 15:10:27 2023

@author: Banquise
"""
import numpy as np
import matplotlib.pyplot as plt

fexc = 1
facq = 101

n_T = 12 #nombre periode
t = np.linspace(0,n_T/fexc, n_T * facq)
sinus= (np.sin(t * 2 * np.pi * fexc) + np.sin( 2* t * 2 * np.pi * fexc + np.pi / 12  ) * 0.3 + np.sin( 4* t * 2 * np.pi * fexc  ) * 0.1 )* 0.0035 + 0.0035
sinus = sinus - np.mean(sinus)
length = len(sinus)

plt.figure()
plt.plot(t, sinus)
plt.title("sin(fexc)+ sin( 2fexc + np.pi/12)*0.3+ sin(4fexc)*0.1)*0.0035+0.0035")

# Y1 = np.fft.fft(sinus, norm = 'backward')

# # Y1_0 = np.fft.fft(sinus, norm = 'backward')

# FFT_norm = np.abs(Y1)

# # FFT_norm_0 = np.abs(Y1_0/length)


# f = np.arange(0,facq, facq/length)

# padding = 20

# plt.figure()
# plt.plot(f, FFT_norm)
# # plt.plot(f, FFT_norm_0)

# sinus_0 = np.append(sinus, np.zeros(2**padding - length))

# Y1_0 = np.fft.fft(sinus_0, norm = 'backward')

# FFT_norm_0 = np.abs(Y1_0 / length)

# f_0 = np.arange(0,facq, facq / 2**padding)

# plt.figure()
# plt.plot(f_0, np.abs(Y1_0))

# #On a un signal et sa FFT

# #Parseval

# dt = 1/ facq
# sum_amp_reelle = np.sum ( np.abs(sinus) **2)# / (length) * dt

# df = facq / length
# sum_amp_FFT = np.sum(np.abs(Y1)**2 ) / length# / facq * df * 2 * np.pi


# df_0 = facq/2**padding
# sum_amp_FFT_0 = np.sum(np.abs(Y1_0)**2 ) / 2**padding#/ facq * df * 2 * np.pi

# print(sum_amp_reelle)
# print(sum_amp_FFT)
# print(sum_amp_FFT_0)

# print('maxs : ')

# print(np.max(sinus))
# print(np.max(FFT_norm))
# print(np.max(FFT_norm_0))


