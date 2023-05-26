# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 17:06:42 2022

@author: Banquise
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 31 17:23:26 2022

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
import os
from PIL import Image
%run Functions_FSD.py
%run parametres_FSD.py
%run display_lib_gld.py

#Va chercher les images et stock leur chemin dasn liste images

path_images, liste_images, titre_exp = import_images(loc,nom_exp,"LAS")

#Importe les paramètres de l'experience

facq, texp, Tmot, Vmot, Hw, Larg_ice, Long_ice, tacq, type_exp = import_param (titre_exp,date)   

mmparpixelx, mmparpixely, mmparpixelz, angle_cam_LAS, mmparpixel = import_calibration(titre_exp,date)          
              
# Creates a file txt with all the parameters of the experience and of the analysis
    
param_complets = ["Paramètres d'adimensionnement :",  "lambda_vague = " + str(lambda_vague) , "Ampvague = " + str(Ampvague) ,  "Paramètres d'analyse : ", "debut = " + str(debut) ,"kernel_size = " + str(k_size_crack), "kernel_iteration = " + str(k_iteration_crack) ,"nbframe = " + str(nbframe) , "minsize_crack = " + str(crack_lenght_min) , "sepwb = " + str(sepwb_cracks) , "size_crop = " + str(size_crop), "mmparpixely = " + str(mmparpixely), "Paramètres experimentaux : ", "facq = " + str(facq) , "texp = " + str(texp) , "Tmot = " + str(Tmot) , "Vmot = " + str(Vmot), "Hw = " + str(Hw), "Larg_ice = " + str(Larg_ice), "Long_ice = " + str(Long_ice), "tacq = " + str(tacq), "type_exp = " + str(type_exp)]

#%%points expérimentaux

RDDA_YYY = True
g = 9.81
tension_surface = 50E-3
rho = 900


m = (226.93-188.24)/3.1

S = 0.4*.26
h = m / (S * rho)
# r_d_d = np.load(loc + "relation_de_dispersion\longueur_donde_all.txt")



# doc =  open(fichiers[i],"r")
# nb_line = 0
# liste_doc.append(doc)
# for line in liste_doc[i]:
#     nb_line += 1
#     coords1 = line.split()
#     x1[i].extend([float(coords1[0])])
#     y1[i].extend([float(coords1[1])])

# RDDA = np.loadtxt("D:\Banquise\Baptiste\Resultats_video\d221012\RDDA_lambda_erreur_kappa_erreur_F.txt")
RDDA = np.loadtxt("D:\Banquise\Baptiste\Resultats_video\d221024\lambda_kappa_DPA09.txt")
# r_d_d_rdmf2 = np.loadtxt("D:\Banquise\Baptiste\Resultats_video\d220531/relation_de_dispersion_test2/longueur_donde220523_RDMF2_test2_.txt")
# r_d_d_rdmf3 = np.loadtxt("D:\Banquise\Baptiste\Resultats_video\d220531/relation_de_dispersion_test2/longueur_donde220523_RDMF3_test2_.txt")
# r_d_d_rdmf4 = np.loadtxt("D:\Banquise\Baptiste\Resultats_video\d220531/relation_de_dispersion_test2/longueur_donde220523_RDMF4_test2_.txt")
# r_d_d_rdmf5 = np.loadtxt("D:\Banquise\Baptiste\Resultats_video\d220531/relation_de_dispersion_test2/longueur_donde220523_RDMF5_test2_.txt")
# r_d_d_rdmf6 = np.loadtxt("D:\Banquise\Baptiste\Resultats_video\d220531/relation_de_dispersion_test2/longueur_donde220523_RDMF6_test2_.txt")
# r_d_d.sort(key=lambda x:x[0])
# r_d_d[:,0] = 1/ r_d_d[:,0]



lambda_RDDA = RDDA[:,0] #en metres
Err_lambda = RDDA[:,1]
kappa_RDDA_1 = RDDA[:,2]
Err_kappa_1 = RDDA[:,3]
omega_RDDA = 2 * np.pi * RDDA[:,4] #en Hz

k_RDDA = 2 * np.pi / lambda_RDDA
Err_k = np.zeros(len(Err_lambda))
for i in range( len(Err_lambda)):
    Err_k[i] = ((Err_lambda[i] / lambda_RDDA[i]) + 0.02) * k_RDDA[i]

kappa_RDDA = []
Err_kappa = []
omega_kappa_RDDA = []
for j in range (len(kappa_RDDA_1)):
    if kappa_RDDA_1[j] >= 0.06 :
        kappa_RDDA.append(kappa_RDDA_1[j])
        Err_kappa.append(Err_kappa_1[j])
        omega_kappa_RDDA.append(omega_RDDA[j])
 
        
 
figurejolie()
joliplot( r'$\omega$ (s$^{-1}$)', r'$\kappa$ (cm$^{-1}$)',omega_kappa_RDDA, kappa_RDDA, color = 3, legend = r'Width = ' + str(round(h,4)) + ' mm', log = True, exp = True)

plt.errorbar(omega_kappa_RDDA, kappa_RDDA, yerr = Err_kappa, fmt = 'none', capsize = 5, ecolor = vcolors[2], zorder = 1) 



figurejolie()
joliplot(r'k (m$^{-1}$)', r'$\omega$ (s$^{-1}$)', k_RDDA, omega_RDDA, color = 7, legend = r'Width = ' + str(round(h,6)) + ' mm', log = True, exp = True)

plt.errorbar(k_RDDA, omega_RDDA, xerr = Err_k, fmt = 'none', capsize = 5, ecolor = vcolors[4],elinewidth = 1,capthick = 2)



# plt.grid("off")

#%% fit et courbe RDD

k_tot = []
omega_tot =[]


# if pdp :
#     k_tot = k_pdp
#     omega_tot = omega_pdp
# if pdp and poco8:
#     k_tot = np.append(k_pdp ,k_poco8 )
#     omega_tot = np.append(omega_pdp, omega_poco8)
# if epiv and pdp :
#     k_tot = np.append(k_pdp ,k_epiv )
#     omega_tot = np.append(omega_pdp, omega_epiv)
# if rsbp and rdmf :
#     k_tot = np.append(k_rdmf ,k_rsbp )
#     omega_tot = np.append(omega_rdmf, omega_rsbp)

if RDDA_YYY :
    k_tot = k_RDDA #+ np.sqrt(Err_k)
    omega_tot = omega_RDDA
    



def RDD_fl (k, dsurrho):
    return np.sqrt(g * k + dsurrho * k**5)

def RDD_Tr (k, Tsurrho):
    return np.sqrt(g * k + Tsurrho * k**3)

def RDD_comp (k, dsurrho):
    return np.sqrt(g * k + tension_surface/rho * k**3 + dsurrho * k**5)

# popt, pcov = curve_fit(RDD_fl, k_tot, omega_tot)
# poptrdmf, pcovrdmf = curve_fit(RDD_fl, k_rdmf, omega_rdmf)
# poptpdp, pcovpdp = curve_fit(RDD_fl, k_pdp, omega_pdp)
# poptpdp_cap, pcovpdp_cap = curve_fit(RDD_Tr, k_pdp, omega_pdp)

# if epiv :
#     poptepiv, pcovepiv = curve_fit(RDD_fl, k_epiv, omega_epiv)

popt_comp, pcov_comp = curve_fit(RDD_comp, k_tot, omega_tot)
# popt_comp, pcov_comp = curve_fit(RDD_comp, k_rdmf, omega_rdmf)




    
k_ligne = np.linspace(min(k_tot),max(k_tot),100)
# k_ligne = np.linspace(min(k_rdmf),max(k_rdmf),100)


# if epiv :
#     joliplot(r'k (m$^{-1}$)', r'$\omega$ (s$^{-1}$)', k_ligne, RDD_fl(k_ligne, poptepiv[0]),color = 3,  title = False, legend = r'Fit h = 0.31mm, D = ' + str(round( Depiv, 8)) +" Nm", log = True, exp = False) #
# joliplot(r'k (m$^{-1}$)', r'$\omega$ (s$^{-1}$)', k_ligne, RDD_fl(k_ligne, popt[0]),color = 6,  title = False, legend = r'Fit with gk + Dk$^{5}$/rho, D = ' + str(round( (popt[0] * rho), 8)) +" Nm", log = True, exp = False) #
# if pdp :
#     joliplot(r'k (m$^{-1}$)', r'$\omega$ (s$^{-1}$)', k_ligne, RDD_fl(k_ligne, poptpdp[0]),color = 1,  title = False, legend = r'Fit width = 0.22mm, D = ' + str(round( Dpdp, 8)) +" Nm", log = True, exp = False) #
# plt.plot(k_ligne, RDD_fl(k_ligne, popt[0]), 'm-', label = 'Fit en gk + Dk5/rho, D = ' + str(round( (popt[0] * rho), 8)))
# joliplot(r'k (m$^{-1}$)', r'$\omega$ (s$^{-1}$)', k_ligne, rel_grav, color = 2, title = False, legend = r'Gravitary regime', log = True, exp = False)
# plt.plot(k_ligne, rel_comp, 'r-', label = 'relation de dispersion avec capilarité (Y = 50mNm-1)')
# joliplot(r'k (m$^{-1}$)', r'$\omega$ (s$^{-1}$)', k_ligne, RDD_fl(k_ligne, poptrdmf[0]),color = 4,  title = False, legend = r'Flexion fit h = 0.40mm, D = ' + str(round( Drdmf, 8)) +" Nm", log = True, exp = False) #
# joliplot(r'k (m$^{-1}$)', r'$\omega$ (s$^{-1}$)', k_ligne, RDD_Tr(k_ligne, poptpdp_cap[0]),color = 2,  title = False, legend = 'Capilary fit, h = 0.22mm, Y = ' + str(round(poptpdp_cap[0] * rho,4)) + " Nm$^{-1}$", log = True, exp = False) #
# plt.plot(k_ligne, RDD_Tr(k_ligne, poptpdp_cap[0]), 'g-', label = 'Capilary fit, Y = ' + str(round(poptpdp_cap[0] * rho,5)))




rel_comp = np.sqrt(g * k_ligne + popt_comp[0] * k_ligne**5 + tension_surface/rho * k_ligne**3 )
rel_cap = np.sqrt(g * k_ligne + tension_surface/rho * k_ligne**3 )
rel_grav = np.sqrt(k_ligne * g)



joliplot(r'k (m$^{-1}$)', r'$\omega$ (s$^{-1}$)', k_ligne, RDD_comp(k_ligne, popt_comp[0]),color = 2, legend = r'Fit en gk + $\gamma$k$^{3}$ / $\rho$ + D$k^{5}$/ $\rho$, D = ' + str(round( popt_comp[0] * rho , 8) ) + " Nm", exp = False )


   
plt.plot(k_ligne, rel_cap, 'k-', label = r'Régime capillaire, $\gamma$ = ' + str(tension_surface) + " Nm$^{-1}$")


# plt.axis('equal')
plt.grid()
plt.legend()


#%% Calcul E et Ld
rho = 900
g = 9.81
nu = 0.4

hpdp = 0.059E-3
hepiv = 0.089E-3
hrdmf = 0.106E-3

Dpdp = popt_comp[0] * rho
# Drdmf = popt_comp[0] * rho

Epdp = (Dpdp * 12 * ( 1 - nu**2) )/ (h/1000)**3
# Erdmf = (Drdmf * 12 * ( 1 - nu**2) )/ hrdmf**3
# Eepiv = (Depiv * 12 * ( 1 - nu**2) )/ hepiv**3

Lpdp = pow(Dpdp/(rho * g),0.25)
# Lrdmf = pow(Drdmf/(rho * g),0.25)
# Lepiv = pow(Depiv/(rho * g),0.25)

#%%kappa(k)



# k_ligne, RDD_comp(k_ligne, popt_comp[0])


from scipy.optimize import minimize

x = np.arange(10,1000,0.2)

y = np.zeros(x.shape)

def diff(x,a):
    yt = RDD_comp(x,popt_comp[0])
    return (yt - a )**2

for idx,x_value in enumerate(x):
    res = minimize(diff, 1.0, args=(x_value), method='Nelder-Mead', tol=1e-6)
    y[idx] = res.x[0]

figurejolie()
joliplot(r'k (m$^{-1})$', r'$\kappa (cm^{-1})$', np.interp(omega_kappa_RDDA,x,y), kappa_RDDA, color = 3,legend = r'Width = ' + str(round(h,4)) + ' mm', log = True)
plt.errorbar(np.interp(omega_kappa_RDDA,x,y), kappa_RDDA, yerr = Err_kappa, fmt = 'none', capsize = 5, ecolor = vcolors[2], zorder = 1) 


# figurejolie()
# joliplot(r'k (m$^{-1}$)', r'$\omega$ (s$^{-1}$)', k_ligne, RDD_comp(k_ligne, popt_comp[0]),color = 2, legend = r'Fit en gk + $\gamma$k$^{3}$ / $\rho$ + D$k^{5}$/ $\rho$, D = ' + str(round( popt_comp[0] * rho , 8) ) + " Nm", exp = False )


#%%

ppp = np.polyfit(np.interp(omega_kappa_RDDA,x,y), kappa_RDDA,1)
joliplot(r'k (m$^{-1})$', r'$\kappa (cm^{-1})$', np.interp(omega_kappa_RDDA,x,y), np.interp(omega_kappa_RDDA,x,y) * ppp[0] + ppp[1], color = 2,legend = r'Width = ' + str(round(h,4)) + ' mm', log = True, exp = False)
plt.axis("equal")

#%% kappa/Ks

viscosite = 1E-6
kS = np.sqrt(viscosite/(2 * np.asarray(omega_kappa_RDDA)))* np.interp(omega_kappa_RDDA,x,y)**2
kB = 2 * viscosite / (np.asarray(omega_kappa_RDDA)) * (np.interp(omega_kappa_RDDA,x,y)**3)

figurejolie()
joliplot(r'$\omega$ (s${-1}$)', r'$\kappa$ / Ks', omega_kappa_RDDA, np.asarray(kappa_RDDA)*100/kS, color = 12, log = True)
plt.errorbar(omega_kappa_RDDA, np.asarray(kappa_RDDA)*100/kS, yerr = np.asarray(Err_kappa)*100/kS, fmt = 'none', capsize = 5, ecolor = vcolors[5], zorder = 1) 



figurejolie()
joliplot(r'$\omega$ (s${-1}$)', r'$\kappa$ / Kb', omega_kappa_RDDA, np.asarray(kappa_RDDA)*100/kB, color = 3, log = True)
plt.errorbar(omega_kappa_RDDA, np.asarray(kappa_RDDA)*100/kB, yerr = np.asarray(Err_kappa)*100/kB, fmt = 'none', capsize = 5, ecolor = vcolors[2], zorder = 1) 


# figurejolie()
# joliplot(r'$\omega$ (s${-1}$)', r'$\kappa$ / Kb / Ks', omega_kappa_RDDA, np.asarray(kappa_RDDA)*100/kB / kS)




