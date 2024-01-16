# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 14:43:39 2023

@author: Banquise
"""

import numpy as np

import baptiste.display.display_lib as disp



#%%

h = 0.058 #m
omega = np.linspace(6,60, 100)





def Amax (omega, h) :
    return (9.81**2) / np.power(omega,4) / 0.2481 / h 



disp.figurejolie()

disp.joliplot(r'$\omega$ (Hz)', r'$A_{max}$ (mm)', omega, Amax(omega, h) * 1000, exp = False)

#%%

omega = 2 * 2 * np.pi


lamb = 9.81 * 2 * np.pi / omega**2

lamb_2 = lamb**2

h = lamb_2 / (4 * np.pi * np.pi * 0.25 * 0.01)