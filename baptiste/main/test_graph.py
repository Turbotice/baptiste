# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 14:50:03 2022

@author: Banquise
"""

#%% 
import numpy as np 
import matplotlib.pyplot as plt
%run display_lib_gld.py

#%%
folder = 'D:\Banquise\Baptiste\Resultats_video\d220531\\relation_de_dispersion_test2\\'
file = folder + 'longueur_donde_all_test2_tri.txt'
df = np.loadtxt(file)

plt.figure(figsize = set_size(width=350, fraction = 1, subplots = (1,1)))
plt.plot(df[:,0], df[:,1], ' p', color = 'm', mfc = 'None', markeredgewidth = 1.8, ms = 6.5)
plt.xlabel(r'$\lambda$ (mm$^{-1}$)')
plt.ylabel(r'$\omega$ (s$^{-1}$)')
plt.grid()
# plt.axis('equal')
plt.tight_layout()


#%% 

fig, axes = figurejolie(subplot = (1,2))
axes = joliplot(r'$\lambda$ (mm$^{-1}$)', r'$\omega$ (s$^{-1}$)',df[:,0], df[:,1],color = 0, fig = fig, axes = axes, title = r'la fizic', subplot = (1,2), legend = r'ma legend $\mathcal{L}$', log = True, exp = False)

#%%

figurejolie()
joliplot(r'$\lambda$ (mm$^{-1}$)', r'$\omega$ (s$^{-1}$)',df[:,0], df[:,1] + 2, color = 0, fig = fig, axes = axes, title = r'la fizic', legend = r'ma legend $\mathcal{L}$', log = False, exp = True)
joliplot(r'$\lambda$ (mm$^{-1}$)', r'$\omega$ (s$^{-1}$)',df[:,0], df[:,1] + 5, color = 1, fig = fig, axes = axes, title = r'la fizic', legend = r'ma legend $\mathcal{L}$', log = False, exp = True)
joliplot(r'$\lambda$ (mm$^{-1}$)', r'$\omega$ (s$^{-1}$)',df[:,0], df[:,1] + 10, color = 2, fig = fig, axes = axes, title = r'la fizic', legend = r'ma legend $\mathcal{L}$', log = False, exp = True)
joliplot(r'$\lambda$ (mm$^{-1}$)', r'$\omega$ (s$^{-1}$)',df[:,0], df[:,1] * 2,color = 3, fig = fig, axes = axes, title = r'la fizic', legend = r'ma legend $\mathcal{L}$', log = False, exp = True)
joliplot(r'$\lambda$ (mm$^{-1}$)', r'$\omega$ (s$^{-1}$)',df[:,0], df[:,1]- 10,color = 4, fig = fig, axes = axes, title = r'la fizic', legend = r'ma legend $\mathcal{L}$', log = False, exp = True)
joliplot(r'$\lambda$ (mm$^{-1}$)', r'$\omega$ (s$^{-1}$)',df[:,0], df[:,1] / 1.5,color = 5, fig = fig, axes = axes, title = r'la fizic', legend = r'ma legend $\mathcal{L}$', log = False, exp = True)



#%%
ax1.plot(df[:,0], df[:,1], ' o', color = vcolors[0], mfc = 'None', markeredgewidth = 2, label =r'ma legend $\mathcal{L}$')
ax1.plot(df[:,0], df[:,1]*.5, ' P', color = vcolors[4], mfc = 'None', markeredgewidth = 2, label =r'ma legend $\mathcal{C}$')
ax1.set_xlabel(r'$\lambda$ (mm$^{-1}$)')
ax1.set_ylabel(r'$\omega$ (s$^{-1}$)')
ax1.legend()
ax2.plot(df[:,0], df[:,1], ' P', color = vcolors[4], mfc = 'None', markeredgewidth = 2)
ax2.set_xlabel(r'$\lambda$ (mm$^{-1}$)')
ax2.set_ylabel(r'$\omega$ (s$^{-1}$)')
# plt.tight_layout()