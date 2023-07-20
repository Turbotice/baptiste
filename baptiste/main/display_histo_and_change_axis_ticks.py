# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 14:49:35 2023

@author: Banquise

Faire graduations en pi

"""
import baptiste.display.display_lib as disp
import numpy as np
import matplotlib.pyplot as plt

disp.figurejolie( )
nb_morc = np.linspace(0,250,250)
disp.joliplot(r'$\phi$',r'Numéro échantillon', params['phi'][:,0], nb_morc, color = 17)

axes = plt.gca()

axes.xaxis.set_ticks([0, np.pi, 2 * np.pi])

axes.xaxis.set_ticklabels([r'$0$', r'$\pi$', r'$2\pi$'])

#%%




f = 0

for f in range (0,5) :
    disp.figurejolie()

    x = np.linspace(0, 25, 24)
    
    h = plt.hist(params['phi'][:,f], params['n_bins'])
    h = np.asarray(h)
    h[1] = (h[1][1:]+h[1][:-1]) / 2
    
    disp.joliplot(r'$\phi$',r"Nombre d'occurence", x, histogram1[:,f]  * 250 / np.cumsum(histogram1[:,10])[-1], color = 17, legend = 'Histogramme pondéré', exp = False)
    disp.joliplot(r'$\phi$',r"Nombre d'occurence", x, h[0] , color = 2, legend = 'Histogramme non pondéré', exp = False)
    
    
    def set_axe_pi (nb_ticks, x, axxe = 'x') :
        if axxe == 'x'or axxe == 'xy' :
            axes = plt.gca()
            
            [2 * np.pi *u for u in range (0,nb_ticks)]
            axes.xaxis.set_ticks([u * np.max(x) / (nb_ticks - 1) for u in range (0, nb_ticks)])
            
            axes.xaxis.set_ticklabels([r'$0$', r'$\pi$'] + [ str(u) + r'$\pi$' for u in range(2, nb_ticks)])
        if axxe == 'y' or axxe == 'xy' :
            axes = plt.gca()
            
            [2 * np.pi *u for u in range (0,nb_ticks)]
            axes.yaxis.set_ticks([u * np.max(x) / (nb_ticks - 1) for u in range (0, nb_ticks)])
            
            axes.yaxis.set_ticklabels([r'$0$', r'$\pi$'] + [ str(u) + r'$\pi$' for u in range(2, nb_ticks)])
            
    set_axe_pi(3, x, axxe = 'x')

#%%


disp.figurejolie( )

x = np.linspace(0, 25, 24)

plt.bar(x, hist_pond  * 250 / np.cumsum(hist_pond)[-1], align = 'edge', width = 0.45, color = disp.vcolors(4), label = 'Histogramme pondéré')

plt.bar(x, h[0], align = 'edge', width = -0.45, color = '#990000', label = 'Histogramme non pondéré')
    
    
axes = plt.gca()

axes.xaxis.set_ticks([0, 12.5, 25])

axes.xaxis.set_ticklabels([r'$0$', r'$\pi$', r'$2\pi$'])

plt.xlabel(r'$\phi$') 
plt.ylabel(r"Nombre d'occurence")
plt.legend()

# disp.joliplot(r'$\phi$',r"Nombre d'occurence", x, hist_pond  * 250 / np.cumsum(hist_pond)[-1], color = 17, exp = False)
# disp.joliplot(r'$\phi$',r"Nombre d'occurence", x, h[0] , color = 2, exp = False)