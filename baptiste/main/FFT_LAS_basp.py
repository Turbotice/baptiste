# -*- coding: utf-8 -*-
"""
Created on Wed May 25 11:10:41 2022

@author: Banquise
"""

import numpy as np
import scipy.fft as fft
import matplotlib.pyplot as plt


s = signal_medfilt[:,300:]
s = s-signal_medfilt[:,0][:,None]


TF_t = fft.fft2(s-np.mean(s,axis=1)[:,None],axes=1)



#%%

plt.figure()

TF_t_moy = np.mean(np.abs(TF_t),axis=0)

n = TF_t_moy.shape[0]
f = np.linspace(0,facq,n)
plt.loglog(f,TF_t_moy)



#%%


TF_x = fft.fft2(s-np.mean(s,axis=0)[None,:],axes=0)



#%%

plt.figure()

TF_x_moy = np.mean(np.abs(TF_x),axis=1)
nx = TF_x_moy.shape[0]
k = np.linspace(0,1/mmparpixelz,nx)
plt.plot(k,TF_x_moy)

#%%
#Zero padding

[nx,nt] = s.shape

indices = np.arange(0,nt,1)
ni = len(indices)

FFT_x = []
for i in indices:
    print(i)
    sig = s[:,i]
    
    nxt = int(np.ceil(np.log2(nx))) #nombre pair de pixels necessaire
    size_padding = pow(2,nxt+1)-int(nx/2)

    sr = np.append(sig*np.hanning(nx),np.zeros(size_padding))
    szp = np.append(np.zeros(size_padding),sr)

    FFT_x.append(abs(fft.fft(szp)))
    

FFT_x = np.asarray(FFT_x)
FFT_x_moy = np.mean(FFT_x,axis=0)
print(FFT_x.shape)

#%%

plt.figure()

nk = FFT_x_moy.shape[0]
k = np.linspace(0,1/mmparpixelz,nk)
plt.loglog(k[:nk//2],FFT_x_moy[:nk//2])

#plt.axis([0,0.1,0,2000])


#%%

x0 = 1000
A = np.correlate(s[x0,:],s[x0,:],mode='full')
plt.figure()

plt.plot(A)

#%%

import scipy.signal as signal

C = signal.correlate2d(s,s)



