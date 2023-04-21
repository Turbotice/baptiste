# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 15:29:58 2023

@author: Banquise
"""
import numpy as np
import matplotlib.pyplot as plt
import baptiste.display.display_lib as disp
import baptiste.files.save as sv
import scipy.fft as fft


def add_padding (data, padding) :
    # padding = [2**9,2**9,2**11]
    datasize = data.shape
    data_pad = np.zeros(np.power(2,padding))
    if len(padding) == 1 :
        data_pad[:datasize] = data
    if len(padding)  == 2 :
        data_pad[:datasize[0], :datasize[1]] = data
    if len(padding)  == 3 :
        data_pad[:datasize[0], :datasize[1], :datasize[2]] = data
    return data_pad


def fft_bapt(data, df1, df2 = False, df3 = False, og_shape = False, abso = True):

    dim = len(data.shape)       

    # On va regarder la periode sur signal.
    if dim == 1 :
        n1 = data.shape[0]
        Y1 = fft.fft(data - np.nanmean(data))
        if og_shape != False :
            P2 = abs(Y1/og_shape[0])
        else :
            P2 = abs(Y1/n1)
        P1 = P2 
        
        
        f = np.linspace(0, df1/2, n1)
        
        if abso :
            return P1, f  
        else :
            return Y1, f
    
    if dim == 2 :
        [n1, n2] = data.shape
        Y1 = fft.fft2(data - np.nanmean(data))
        if og_shape == False :
            P2 = np.abs(Y1/(n2 * n1) )
        else :
            P2 = np.abs(Y1 / og_shape[0] * og_shape[1])
        P1 = P2
        # P1[2:-1] = 2*P1[2:-1]
        
        f1 = np.linspace( - df1/2,df1/2,n1)
        f2 = np.linspace( - df2/2,df2/2,n2)
        
        # parceval 2D python : np.sum(np.abs(sig)**2) == np.sum(np.abs(np.fft.fft(sig))**2)/sig.size
        if abso :
            return P1, f1, f2
        else :
            return Y1, f1, f2
        
    if dim == 3 :
        [n1, n2, n3] = data.shape
        Y1 = fft.fft2(data - np.nanmean(data))
        if og_shape == False :
            P2 = np.abs(Y1/(n2 * n1 * n3) )
        else :
            P2 = np.abs(Y1 / og_shape[0] * og_shape[1] * og_shape[2])
        P1 = P2

        
        f1 = np.linspace( - df1/2,df1/2,n1)
        f2 = np.linspace( - df2/2,df2/2,n2)
        f3 = np.linspace( - df3/2,df3/2,n3)
        
        # parceval 2D python : np.sum(np.abs(sig)**2) == np.sum(np.abs(np.fft.fft(sig))**2)/sig.size
        if abso :
            return P1, f1, f2, f3
        else :
            return Y1, f1, f2, f3


    
    
def plot_fft(Y1, f1, f2 = False, f3 = False, log = True, xlabel = r'kx(m$^{-1})$', ylabel = r'f(Hz)'):
    disp.figurejolie()
    dim = len(Y1.shape)
    if log :
        Y1 = np.log(np.abs(Y1))
    else :
        Y1 = np.abs(Y1)
    if dim == 1 :
        disp.joliplot(r'f(Hz)', r'P1', f1, Y1, color = 4, exp = False)
        
    if dim == 2 :
        plt.pcolormesh(f1, f2, fft.fftshift(np.flip(np.rot(Y1), 0)), shading = 'auto')
        cbar = plt.colorbar()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
    if dim == 3 :
        plt.pcolormesh(f1, f2, fft.fftshift(np.flip(np.rot(Y1[:,:,0]), 0)), shading = 'auto')
        cbar = plt.colorbar()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

def demodulation(t,s, fexc):
    c = np.nanmean(s*np.exp(1j * 2 * np.pi * t[None,None,:] * fexc),axis=2)
    return c
    