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
import baptiste.display.display_lib as disp
import baptiste.math.RDD as rdd

g = 9.81
l_cuve = 0.6 #en m
larg_cuve = 0.38
n_min = 1
n_max = 8
liste_tot = [] #liste avec n, lambda, f
liste_large_tot = []


for i in range (n_min,n_max):
    n = i
    lambda_n = l_cuve/n
    f_n = np.sqrt(9.81/(2 * np.pi * lambda_n))
    
    
    liste_tot.append([n,lambda_n,f_n])
    
for i in range (n_min,n_max):
    n = i
    lambda_n = larg_cuve/n
    f_n = np.sqrt(9.81/(2 * np.pi * lambda_n))
    
    
    liste_large_tot.append([n,lambda_n,f_n])
    
liste_tot = np.asarray(liste_tot)
liste_large_tot = np.asarray(liste_large_tot)
disp.figurejolie()
disp.joliplot(r'numéro du mode',r'f (Hz)', liste_tot [:,0], liste_tot[:,2], color = 2, legend = str(round(n_max - n_min)) + " modes propres pour une cuve de " + str(round(l_cuve * 100)) + " cm")
plt.xticks(np.arange(n_min,n_max,1))
plt.yticks(np.append(liste_tot[:,2], liste_large_tot[:,2]))

disp.joliplot(r'numéro du mode',r'f (Hz)', liste_large_tot [:,0], liste_large_tot[:,2], color = 5, legend = str(round(n_max - n_min)) + " modes propres LARGEUR " + str(round(larg_cuve * 100)) + " cm")


#%% F from lambda

l_onde = l_cuve
depth = 0.11

k = 2 * np.pi / l_onde

f_onde = rdd.RDD_depth(k, depth) / 2 / np.pi

print(f_onde)