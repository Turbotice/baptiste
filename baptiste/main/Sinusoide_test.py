# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 18:00:01 2022

@author: Banquise
"""

#mesure longueur sinusoide, et l'allongement relatif en foction de Amp et lambda vague

import numpy as np
import matplotlib.pyplot as plt
%run display_lib_gld.py

#%%
figurejolie()
#bonna range pour ce qu'on observe pr les pentes avec un lambda de 1
Amp0 = 0.005
pas_amp = 0.0001
nb_pas_amp = 1000

lambda0 = 0.1
pas_lambda = 0.1
nb_pas_lambda = 10

res = 1000
allongement = []
theorie = []

for j in range (nb_pas_amp):
    for k in range (nb_pas_lambda):
        if np.mod((j+1)*(k+1),1000)==0:
            print ("boucle " + str ((j+1)*(k + 1)) + " sur " + str(nb_pas_amp * nb_pas_lambda) )
        
        Amp = Amp0 + j * pas_amp
        lambdavague = lambda0 + k * pas_lambda
     
        x = np.linspace(0,1,res)
        
        cosinus = Amp * np.cos (x / lambdavague * 2 * np.pi)
        theorie.append(np.sqrt(1 + (Amp**2/lambdavague**2)) )
        
        
        len_cos = 0
        
        len_droite = 0
        
        for i in range( len (x) - 1):
            len_cos += np.sqrt(   (x[i + 1] - x[i])**2 + (cosinus[i+1] - cosinus[i])**2 )
            len_droite += x[i+1] - x[i]
        
        allongement.append([len_cos / (len_droite), Amp ,lambdavague] )
allongement = np.asarray(allongement)
all_rel = allongement [:,0] # /allongement[:,2]
pente = allongement [:,1] / allongement[:,2]
plt.plot (pente, all_rel, 'k.')
plt.plot (pente, theorie, 'm-')
plt.xlabel('pente')
plt.ylabel('allongement relatif')
# plt.yscale('log')
# plt.xscale('log')

