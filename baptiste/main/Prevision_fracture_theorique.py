# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 18:03:24 2024

@author: Banquise


Theorie fracture

"""
import numpy as np
import matplotlib.pyplot as plt
import baptiste.display.display_lib as disp


#%% Vernis

Gc = 0.3
Ld = 0.005
h = 0.0001
size = 1000
rho = 1000
g = 9.81
boost_rnl = 5

lamb = np.linspace(Ld*10,Ld*100, size)
A = np.linspace(0.0,0.03,size)
k = np.pi * 2 / lamb

L = np.zeros((size, size))
kappa = np.zeros((size, size))
Energie = np.zeros((size, size))
Ak = np.zeros((size, size))

for i in range(len(lamb)) :
    for j in range(len(A)) :
        
        Ak[i,j] = A[j] * k[i]
        
        kappa[i,j] = Ak[i,j]* k[i] * ( 1 + boost_rnl*Ak[i,j])
        
        if Ak[i,j] > 0.05 : #Ak > 0.05 donne L = Ak^-1.1 * lambda
            L[i,j] = (Ak[i,j])**(-1.1) / 54 * lamb[i]
        
        if Ak[i,j] > 0.2 : #Ak > 0.02 donne L = 0.1 * lambda
            L[i,j] = 0.1 * lamb[i]
        
        if  Ak[i,j] <= 0.05: #Ak < 0.05 donne L = 0.5 * lambda
            L[i,j] = 0.5 * lamb [i]
            
        if L[i,j] < Ld : #si L < Ld, L = Ld
            L[i,j] = Ld
            
        Energie[i,j] = kappa[i,j]**2 * L[i,j] * Ld**4 * rho * g / h

#%% Glace

Gc = 2
Ld = 3
h = 0.12
size = 1000
rho = 1000
g = 9.81
boost_rnl = 5

lamb = np.linspace(10,40, size)
A = np.linspace(0.1,0.5,size)
k = np.pi * 2 / lamb

L = np.zeros((size, size))
kappa = np.zeros((size, size))
Energie = np.zeros((size, size))
Ak = np.zeros((size, size))

for i in range(len(lamb)) :
    for j in range(len(A)) :
        
        Ak[i,j] = A[j] * k[i]
        
        kappa[i,j] = Ak[i,j]* k[i] * ( 1 + boost_rnl*Ak[i,j])
        
        if Ak[i,j] > 0.05 : #Ak > 0.05 donne L = Ak^-1.1 * lambda
            L[i,j] = (Ak[i,j])**(-1.1) / 54 * lamb[i]
        
        if Ak[i,j] > 0.2 : #Ak > 0.02 donne L = 0.1 * lambda
            L[i,j] = 0.1 * lamb[i]
        
        if  Ak[i,j] <= 0.05: #Ak < 0.05 donne L = 0.5 * lambda
            L[i,j] = 0.5 * lamb [i]
            
        if L[i,j] < Ld : #si L < Ld, L = Ld
            L[i,j] = Ld
            
        Energie[i,j] = kappa[i,j]**2 * L[i,j] * Ld**4 * rho * g / h

#%% PLOTS


disp.figurejolie()
disp.joliplot(r'$\lambda$ (m)', 'A (m)', lamb, A, table = L)
# plt.colorbar()
# plt.xscale('log')
# plt.yscale('log')
plt.title('L')


# eloise.epaud@espci.fr

# erwan.le-roux@espci.fr

disp.figurejolie()
disp.joliplot(r'$\lambda$ (m)', 'A (m)', lamb, A, table = Ak)
# plt.colorbar()
# plt.xscale('log')
# plt.yscale('log')
plt.title('pente')
plt.clim(0,0.2)


disp.figurejolie()
disp.joliplot(r'$\lambda$ (m)', 'A (m)', lamb, A, table = kappa)
# plt.colorbar()
# plt.xscale('log')
# plt.yscale('log')
plt.title('kappa')
plt.clim(0,40)

disp.figurejolie()
disp.joliplot(r'$\lambda$ (m)', 'A (m)', lamb, A, table = Energie)
# plt.colorbar()
# plt.xscale('log')
# plt.yscale('log')
plt.title('Energie')
plt.clim(0,Gc)
