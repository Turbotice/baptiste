# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 14:32:41 2023

@author: Banquise
"""
"""
LISTE DES MODES PROPRES DE LA CUVE PABLO"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
%run Functions_FSD.py
%run parametres_FSD.py
%run display_lib_gld.py

g = 9.81
l_cuve = 0.64 #en m
n_min = 1
n_max = 8
liste_tot = [] #liste avec n, lambda, f


for i in range (n_min,n_max):
    n = i
    lambda_n = l_cuve/n
    f_n = np.sqrt(9.81/(2 * np.pi * lambda_n))
    
    
    liste_tot.append([n,lambda_n,f_n])
    
liste_tot = np.asarray(liste_tot)
figurejolie()
joliplot(r'numéro du mode',r'f (Hz)', liste_tot [:,0], liste_tot[:,2], color = 2, legend = str(round(n_max - n_min)) + " modes propres pour une cuve de " + str(round(l_cuve * 100)) + " cm")
plt.xticks(np.arange(n_min,n_max,1))
plt.yticks(liste_tot[:,2])

