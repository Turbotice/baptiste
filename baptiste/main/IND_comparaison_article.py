# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 17:04:46 2022

@author: Banquise
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 17:03:52 2022

@author: Banquise
"""

import cv2 
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from skimage import feature
from scipy.signal import convolve2d
from scipy.signal import savgol_filter, gaussian
from scipy.signal import medfilt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy import stats
import scipy.fft as fft
from scipy.interpolate import interp1d
import os
from PIL import Image
# %run Functions_FSD.py
# %run parametres_FSD.py
# %run display_lib_gld.py



import baptiste.display.display_lib as disp
import baptiste.experiments.import_params as ip
import baptiste.files.file_management as fm
import baptiste.image_processing.image_processing
import baptiste.files.dictionaries as dic
import baptiste.tools.tools as tools


date = '220708'
nom_exp = 'IJSP2'
exp = True
exp_type = 'IND'


dico, params, loc = ip.initialisation(date, nom_exp, exp = True, display = False, exp_type = 'IND')


exp_type = "IND"
date = '221005'
loc = "D:\Banquise\Baptiste\Mesures_autres\d" + date + "\\" #+ "d" + date + "_" + nom_exp
path_mesures, liste_mesures, titre_exp = ip.import_images(loc, nom_exp, exp_type, nom_fich = '\mesure\\')


#%% load data
data_exp = []
epaisseurs = []
rtige = []
for i in range (len (liste_mesures)):
    if exp_type in liste_mesures[i]:
        data_exp.append(np.loadtxt(path_mesures + liste_mesures[i]))
        epaisseurs.append (float ( liste_mesures[i][liste_mesures[i].index("ha") + 2:liste_mesures[i].index("ha") + 5]) / 100)
        rtige.append (float ( liste_mesures[i][liste_mesures[i].index("tige") + 4:liste_mesures[i].index("tige") + 7]) / 100) #en mm
        
epaisseurs = np.asarray(epaisseurs) * 1E-3
rtige = np.asarray(rtige) * 1E-3
path_th_article = "D:\Banquise\Baptiste\Mesures_autres\d220719"
data_articletau0 = np.loadtxt(path_th_article + '\\' + 'ForceLawTau=0.txt')
data_articletau1 = np.loadtxt(path_th_article + '\\' + 'ForceLawTau=1.txt')
data_articletau5 = np.loadtxt(path_th_article + '\\' + 'ForceLawTau=5.txt')
data_articletau10 = np.loadtxt(path_th_article + '\\' + 'ForceLawTau=10.txt')
data_articletau33 = np.loadtxt(path_th_article + '\\' + 'ForceLawTau=33.txt')


#%%
g = 9.81
rho_vernis = 900
rho = 1000
rho_membrane = 1100

E = 1.6E6
nu = 0.25

tension_surface = 70E-3
surface_vernis = (8.384 / 100) * (11.400 / 100)


surface_tige = (rtige)** 2 * np.pi 
D = (E * pow(epaisseurs,3)) / (12 * (1 - nu**2))

tau = tension_surface/ ((rho * g * D)**(0.5))

parametres = ["g, :" + str(g),"rho_vernis, :" + str(rho_vernis),"rho, :" + str(rho),"rho_membrane, :" + str(rho_membrane),"E, :" + str(E),"nu, :" + str(nu),"tension_surface, :" + str(tension_surface),"surface_tige, :" + str(surface_tige)]

    

    
#%%
evaporation = True
save = False
#pour TIPP1  
force_adim_tot = []
ind_adim_tot = []

for j in range (len (data_exp)):
    indentation = data_exp[j][:,0] * 1E-3
    contrainte = data_exp[j][:,1]
    
    
    if evaporation :
        evap = []
        for k in range (len(contrainte)):
            evap.append(0.01 * int (k/10) / surface_tige[j] * g / 1000)
        contrainte = contrainte - evap
    figurejolie(num_fig = 1)
    joliplot( r"Indentation (m)", r"Contrainte (Pa)", indentation, contrainte, color = j+1, legend = r'h = ' + str(round(epaisseurs[j]*1E3,3)) + " (mm)", exp = True, log = False)
    
    force = contrainte * surface_tige[j]
    indentation_adim = indentation/epaisseurs[j]
    force_adim = force / (epaisseurs[j] * (D[j] * rho * g)**(0.5))
    
    figurejolie(num_fig = 2)
    joliplot( r"$\delta$ = $\zeta / h$", r"F = $F /[h(B \rho g)^{1/2}]$", indentation_adim, force_adim, color = j+2, legend = r'h = ' + str(round(epaisseurs[j]*1E3, 3)) + r" (mm), $ \tau $ = " + str(round(tau[j], 3))+r', rtige = ' + str(round(rtige[j]* 1000,1)) + " mm", exp = True, log = True)
    
    figurejolie(num_fig = 3)
    joliplot( r"Indentation (m)", r"Force (N)", indentation, force, color = j+1, legend = r'h = ' + str(round(epaisseurs[j]*1E3,3)) + " (mm)" + r', rtige = ' + str(round(rtige[j]* 1000,1)) + " mm", exp = True, log = False)
    
    xnew = np.linspace(0.03,10,1000)
    indentation_adim = np.concatenate([[0.01],indentation_adim[1:],[30]])
    force_adim = np.concatenate([[np.nan],force_adim[1:],[np.nan]])
    
    force_adim_tot.append(interp1d(indentation_adim, force_adim))
    ind_adim_tot.append(indentation_adim)
    
    
    
tableau_valeurs = []
for w in range (len (ind_adim_tot)):
    tableau_valeurs.append(force_adim_tot[w](xnew))
mean_tot = np.nanmean(tableau_valeurs,0)

figurejolie(num_fig = 4)
joliplot( r"$\delta$ = $\zeta / h$", r"F = $F /[h(B rho g)^{1/2}]$", xnew, mean_tot, color = j+2, legend = r'h = ' + str(round(epaisseurs[j]*1E3, 3)) + r" (mm), $ \tau $ = " + str(round(tau[j], 3))+r', rtige = ' + str(round(rtige[j]* 1000,1)) + " mm", exp = True, log = True)


figurejolie(num_fig = 2)
plt.plot(data_articletau0[:,0],data_articletau0[:,1], 'k', label = r"Article $ \tau $ = 0")
plt.plot(data_articletau1[:,0],data_articletau1[:,1], 'k-', label = r'Article $ \tau $ = 1')
# plt.plot(data_articletau5[:,0],data_articletau5[:,1], 'k-', label = r'Article $ \tau $ = 5')
# plt.plot(data_articletau10[:,0],data_articletau10[:,1], 'k-', label = r'Article $ \tau $ = 10')
# plt.plot(data_articletau33[:,0],data_articletau33[:,1], 'k-', label = r'Article $ \tau $ = 33')
plt.xlim (0.01,100)
plt.ylim(0.1,10000)
plt.legend()

figurejolie(num_fig = 4)
plt.plot(data_articletau0[:,0],data_articletau0[:,1], 'k', label = r"Article $ \tau $ = 0")
plt.plot(data_articletau1[:,0],data_articletau1[:,1], 'k-', label = r'Article $ \tau $ = 1')
# plt.plot(data_articletau5[:,0],data_articletau5[:,1], 'k-', label = r'Article $ \tau $ = 5')
# plt.plot(data_articletau10[:,0],data_articletau10[:,1], 'k-', label = r'Article $ \tau $ = 10')
# plt.plot(data_articletau33[:,0],data_articletau33[:,1], 'k-', label = r'Article $ \tau $ = 33')
# plt.xlim (0.01,100)
# plt.ylim(0.1,10000)
plt.legend()

if save :
    parametres = np.asarray(parametres)
    # path_mesures[:-8] + "\resultats/" + name_fig_IND + 

    np.savetxt(path_mesures[:-8] + "/resultats/" + name_fig_IND + "param_premierjet.txt", parametres,'%s' )
    # path_images[:-15] + "resultats/" + name_fig + "data_histo_distribution.txt"
    plt.savefig(path_mesures[:-8] + "/resultats/" + name_fig_IND + "adimensionne_1erejet.pdf")
    
figurejolie(num_fig = 1)
if save :
    parametres = np.asarray(parametres)
    np.savetxt(path_mesures[:-8] + "/resultats/" + name_fig_IND + "param_1erejet.txt", parametres,"%s" )
    plt.savefig(path_mesures[:-8] + "/resultats/" + name_fig_IND + "pts_bruts_1erejet.pdf" )

#%%
save = False

if save :
    np.savetxt(path_mesures[:-8] + "/resultats/" + name_fig_IND + "param_6exp.txt", parametres,"%s" )


#%% fit E sur la courbe

def fct_fit_E (x,Young) :
    u = interp1d(xnew, mean_tot)
    resultat = u(x)
    return resultat * Young

figurejolie(num_fig = 5)
joliplot( r"$\delta$ = $\zeta / h$", r"F = $F /[h(B \rho g)^{1/2}]$", xnew, mean_tot, color = 3, legend = r'Moyenne des données expérimentale pour E = 1.6Mpa', exp = True, log = True)
plt.plot(data_articletau0[:,0],data_articletau0[:,1], 'k', label = r"Article $ \tau $ = 0")
plt.plot(data_articletau1[:,0],data_articletau1[:,1], 'k-', label = r'Article $ \tau $ = 1')
plt.plot(data_articletau5[:,0],data_articletau5[:,1], 'k-', label = r'Article $ \tau $ = 5')
plt.plot(data_articletau10[:,0],data_articletau10[:,1], 'k-', label = r'Article $ \tau $ = 10')

popt, pcov = curve_fit(fct_fit_E, data_articletau1[11:-11,0],data_articletau1[11:-11,1], p0= [0.7], bounds = (0.5,2))

figurejolie(num_fig = 6)
joliplot( r"$\delta$ = $\zeta / h$", r"F = $F /[h(B \rho g)^{1/2}]$", xnew, mean_tot * popt[0], color = 4, legend = r'Moyenne des données avec fit de E : E = ' + str(round(E / popt[0]**2,0)), exp = True, log = True)
plt.plot(data_articletau0[:,0],data_articletau0[:,1], 'k', label = r"Article $ \tau $ = 0")
plt.plot(data_articletau1[:,0],data_articletau1[:,1], 'k-', label = r'Article $ \tau $ = 1')

if save :
    plt.savefig(path_mesures[:-8] + "/resultats/" + name_fig_IND + "E_fitté_6exp.pdf" )


print(popt)





#%%


tt = [data_imgpds_2,data_imgpds_3,data_imgpds_4,data_imgind_2,data_imgind_3,data_imgind_4]

for i in range (3) :
    
    #force = (tt[i][:,1] - tt[i][0,1]) / 1000 * g - (rho * g * indentation)
    imgfor = tt[i][:,0]
    
    deplacement = tt[i+3][1] / 1000
    imgdpc = tt[i+3][0]
    
    dif_cam = [23,-20,-20] #IJSP2 23 IJSP3 -20 IJSP4 -20
    sync_img = int(imgdpc[0] - imgfor[0] - dif_cam[i])
    imgfin = imgfor - imgfor[0] - sync_img
        
    contrainte = []
    indentation = []
       
    
        

        
    for j in range (2,len(imgfor)) :
        indentation.append(-deplacement[int(imgfin[j])])
    
    indentation = np.asarray(indentation)
    xx = np.linspace (min(indentation),max(indentation), len(indentation))
    correction = (rho * g * indentation* surface_tige)    
    force = (tt[i][:,1] - tt[i][0,1]) / 1000 * g
    force_adim_sous = ( force[2:]  - correction ) / (h_vernis[i+1] * (D[i+1] * rho_vernis * g)**(0.5))
    force_adim = force[2:]  / (h_vernis[i+1] * (D[i+1] * rho_vernis * g)**(0.5))
    contrainte = force[2:] / surface_tige# / h_vernis[i+1]**(5/2)
    indent_adim = indentation/h_vernis[i+1]
    
    figurejolie(num_fig = 1)   
    joliplot( r"Indentation (m)", r"Contrainte (Pa)", indentation, contrainte, color = i +1, legend = r'Experience ' + str(i + 2) + ", h = " + str(round(h_vernis[i+1], 6)) + " (m)", exp = True, log= True)
    
    figurejolie(num_fig = 2)
    joliplot( r"$\delta$ = $\zeta / h$", r"$F$ = F$/[h(B \rho g)^{1/2}]$", indent_adim, force_adim, color = i +1, legend = r'Experience ' + str(i + 2) + r", h = " + str(round(h_vernis[i+1], 6)) + r" (m), $ \tau $ = " + str(round(tau[i + 1], 3)), exp = True, log = False)
    
    if i == 2:
        plt.plot(data_articletau0[:,0],data_articletau0[:,1], 'k', label = r"Article $ \tau $ = 0")
        plt.plot(data_articletau1[:,0],data_articletau1[:,1], 'k-', label = r'Article $ \tau $ = 1')
        plt.plot(data_articletau5[:,0],data_articletau5[:,1], 'k-', label = r'Article $ \tau $ = 5')
        plt.plot(data_articletau10[:,0],data_articletau10[:,1], 'k-', label = r'Article $ \tau $ = 10')
        plt.plot(data_articletau33[:,0],data_articletau33[:,1], 'k-', label = r'Article $ \tau $ = 33')
    
        if TIPP1:
            plt.xlim (0.1,150)
            plt.ylim(2,15000)
        else :
            plt.xlim (0,20)
            plt.ylim(0,2000)
        plt.title(r'E = ' + str(E)+ r" $\nu$ = " + str(nu))
 
    figurejolie(num_fig = 3)
    joliplot( r"$\delta$ = $\zeta / h$", r"$F$ =( F - $\rho g \zeta ) /[h(B \rho g)^{1/2}]$", indent_adim, force_adim_sous, color = i +1, legend = r'Experience ' + str(i + 2) + r", h = " + str(round(h_vernis[i+1], 6)) + r" (m), $ \tau $ = " + str(round(tau[i + 1], 3)), exp = True, log = False)
    
    if i == 2:
        plt.plot(data_articletau0[:,0],data_articletau0[:,1], 'k', label = r"Article $ \tau $ = 0")
        plt.plot(data_articletau1[:,0],data_articletau1[:,1], 'k-', label = r'Article $ \tau $ = 1')
        plt.plot(data_articletau5[:,0],data_articletau5[:,1], 'k-', label = r'Article $ \tau $ = 5')
        plt.plot(data_articletau10[:,0],data_articletau10[:,1], 'k-', label = r'Article $ \tau $ = 10')
        plt.plot(data_articletau33[:,0],data_articletau33[:,1], 'k-', label = r'Article $ \tau $ = 33')
        
        if TIPP1:
            plt.xlim (0.1,150)
            plt.ylim(2,15000)
        else :
            plt.xlim (0,20)
            plt.ylim(0,2000)
        plt.title(r'E = ' + str(E)+ r" $\nu$ = " + str(nu))
        
    if i == 2:
        
        figurejolie(num_fig = 4)
        joliplot( r"$\zeta$", r"$F$", xx ,force[2:], color = i +1, legend = r'Force Experience ' + str(i + 2) , exp = True, log = False)
        joliplot( r"$\zeta$", r"$Correction$", xx ,correction, color = i +1, legend = r' correction Experience ' + str(i + 2), exp = True, log = False)
        # if TIPP1:
        #     plt.xlim (0.1,150)
        #     plt.ylim(2,15000)
        # else :
        #     plt.xlim (0,20)
        #     plt.ylim(0,2000)
    
    p = np.polyfit(indentation, contrainte, 1)
    figurejolie(num_fig = 5)
    joliplot( r"$\zeta$", r"$F$", xx ,force[2:], color = i +1, legend = r'Force Experience ' + str(i + 2) , exp = True, log = False)
    joliplot( r"$\zeta$", r"$Correction$", xx ,correction, color = i +1, legend = r' correction Experience ' + str(i + 2), exp = True, log = False)
    
   
plt.legend()

#%% fit pente en fct de h






#%% Fit E et nu sur les données de tau = 1


def fct (force, E, nu):
    D = E * h_vernis**3 / (12 * (1 - nu**2))
    return force / (h_vernis[0] * (D * rho_vernis * g)**(0.5))

popt, pcov = curve_fit(fct, data_articletau0[:,0], data_articletau0[:,1])

