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

facq, texp, Tmot, Vmot, Hw, Larg_ice, Long_ice, tacq, type_exp = import_param (titre_exp)   

mmparpixelx, mmparpixely, mmparpixelz, angle_cam_LAS = import_calibration(titre_exp,date)          
              
# Creates a file txt with all the parameters of the experience and of the analysis
    
param_complets = ["Paramètres d'adimensionnement :",  "lambda_vague = " + str(lambda_vague) , "Ampvague = " + str(Ampvague) ,  "Paramètres d'analyse : ", "debut = " + str(debut) ,"kernel_size = " + str(k_size_crack), "kernel_iteration = " + str(k_iteration_crack) ,"nbframe = " + str(nbframe) , "minsize_crack = " + str(crack_lenght_min) , "sepwb = " + str(sepwb_cracks) , "size_crop = " + str(size_crop), "mmparpixely = " + str(mmparpixely), "Paramètres experimentaux : ", "facq = " + str(facq) , "texp = " + str(texp) , "Tmot = " + str(Tmot) , "Vmot = " + str(Vmot), "Hw = " + str(Hw), "Larg_ice = " + str(Larg_ice), "Long_ice = " + str(Long_ice), "tacq = " + str(tacq), "type_exp = " + str(type_exp)]

#%%points expérimentaux, RDMF (220523)

exp_separees = False

pdp = False
epiv = False
rsbp = False
poco8 = False
rdmf = True

g = 9.81
tension_surface = 55E-3
rho = 900
figurejolie()
# r_d_d = np.load(loc + "relation_de_dispersion\longueur_donde_all.txt")


# doc =  open(fichiers[i],"r")
# nb_line = 0
# liste_doc.append(doc)
# for line in liste_doc[i]:
#     nb_line += 1
#     coords1 = line.split()
#     x1[i].extend([float(coords1[0])])
#     y1[i].extend([float(coords1[1])])

# r_d_d_rdmf1 = np.loadtxt("D:\Banquise\Baptiste\Resultats_video\d220531/relation_de_dispersion_test2/longueur_donde220523_RDMF1_test2_.txt")
# r_d_d_rdmf2 = np.loadtxt("D:\Banquise\Baptiste\Resultats_video\d220531/relation_de_dispersion_test2/longueur_donde220523_RDMF2_test2_.txt")
# r_d_d_rdmf3 = np.loadtxt("D:\Banquise\Baptiste\Resultats_video\d220531/relation_de_dispersion_test2/longueur_donde220523_RDMF3_test2_.txt")
# r_d_d_rdmf4 = np.loadtxt("D:\Banquise\Baptiste\Resultats_video\d220531/relation_de_dispersion_test2/longueur_donde220523_RDMF4_test2_.txt")
# r_d_d_rdmf5 = np.loadtxt("D:\Banquise\Baptiste\Resultats_video\d220531/relation_de_dispersion_test2/longueur_donde220523_RDMF5_test2_.txt")
# r_d_d_rdmf6 = np.loadtxt("D:\Banquise\Baptiste\Resultats_video\d220531/relation_de_dispersion_test2/longueur_donde220523_RDMF6_test2_.txt")
# r_d_d.sort(key=lambda x:x[0])
# r_d_d[:,0] = 1/ r_d_d[:,0]


if rdmf :   
    r_d_d_rdmf = np.loadtxt("D:\Banquise\Baptiste\Resultats_video\d220531/relation_de_dispersion_test2/longueur_donde_all_test2_RDMF.txt")
if rsbp :
    r_d_d_rsbp = np.loadtxt("D:\Banquise\Baptiste\Resultats_video\d220531/relation_de_dispersion_test2/longueur_donde220516_RSBP1_test2_.txt")
if pdp :
    r_d_d_pdp = np.loadtxt("D:\Banquise\Baptiste\Resultats_video\d220531/relation_de_dispersion_+depts/longueur_donde_220608_+depts_tri.txt")
if epiv :
    r_d_d_epiv = np.loadtxt("D:\Banquise\Baptiste\Resultats_video\d220531/relation_de_dispersion_+depts/longueur_donde_220607_+depts_tri.txt")
if poco8 :
    r_d_d_poco8 = np.loadtxt("D:\Banquise\Baptiste\Resultats_video\d220531/relation_de_dispersion_+depts/longueur_donde220608_POCO8_+depts_.txt")




if exp_separees :
    
    k_rdmf1 = (2 * np.pi) / r_d_d_rdmf1[:,0] * 1000
    omega_rdmf1 = 2 * np.pi * r_d_d_rdmf1[:,1]
    k_rdmf2 = (2 * np.pi) / r_d_d_rdmf2[:,0] * 1000
    omega_rdmf2 = 2 * np.pi * r_d_d_rdmf2[:,1]
    k_rdmf3 = (2 * np.pi) / r_d_d_rdmf3[:,0] * 1000
    omega_rdmf3 = 2 * np.pi * r_d_d_rdmf3[:,1]
    k_rdmf4 = (2 * np.pi) / r_d_d_rdmf4[:,0] * 1000
    omega_rdmf4 = 2 * np.pi * r_d_d_rdmf4[:,1]
    k_rdmf5 = (2 * np.pi) / r_d_d_rdmf5[:,0] * 1000
    omega_rdmf5 = 2 * np.pi * r_d_d_rdmf5[:,1]
    k_rdmf6 = (2 * np.pi) / r_d_d_rdmf6[:,0] * 1000
    omega_rdmf6 = 2 * np.pi * r_d_d_rdmf6[:,1]
    
    plt.plot(k_rdmf1,omega_rdmf1, marker = 'x', label = 'RDMF1', ls = '')
    plt.plot(k_rdmf2,omega_rdmf2, marker = 'x', label = 'RDMF2', ls = '')
    plt.plot(k_rdmf3,omega_rdmf3, marker = 'x', label = 'RDMF3', ls = '')
    plt.plot(k_rdmf4,omega_rdmf4, marker = 'x', label = 'RDMF4', ls = '')
    plt.plot(k_rdmf5,omega_rdmf5, marker = 'x', label = 'RDMF5', ls = '')
    plt.plot(k_rdmf6,omega_rdmf6, marker = 'x', label = 'RDMF6', ls = '')

title = r'Relation de dispersion'
title = False



if pdp :   
    k_pdp = (2 * np.pi) / r_d_d_pdp[:,0] * 1000
    omega_pdp = 2 * np.pi * r_d_d_pdp[:,1]
    joliplot(r'k (m$^{-1}$)', r'$\omega$ (s$^{-1}$)', k_pdp, omega_pdp, color = 3, title = title, legend = r'Width = 0.059 mm', log = True, exp = True)
     
if epiv :
    k_epiv = (2 * np.pi) / r_d_d_epiv[:,0] * 1000
    omega_epiv = 2 * np.pi * r_d_d_epiv[:,1]
    joliplot(r'k (m$^{-1})$', r'$\omega$ (s$^{-1}$)', k_epiv, omega_epiv, color = 2, title = r'Relation de dispersion', legend = r'Width = 0.083 mm', log = True, exp = True)
# Width = 0.16mm
    
if rsbp :
    k_rsbp = (2 * np.pi) / r_d_d_rsbp[:,0] * 1000
    omega_rsbp = 2 * np.pi * r_d_d_rsbp[:,1]
    joliplot(r'k (m$^{-1})$', r'$\omega$ (s$^{-1}$)', k_rsbp, omega_rsbp, color = 4, title = title, legend = r'Width = 0.041 mm', log = True, exp = True)

if poco8 :
    k_poco8 = (2 * np.pi) / r_d_d_poco8[:,0] * 1000
    omega_poco8 = 2 * np.pi * r_d_d_poco8[:,1]
    joliplot(r'k (m$^{-1})$', r'$\omega$ (s$^{-1}$)', k_poco8, omega_poco8, color = 5, title = r'Relation de dispersion', legend = r'Relation de dispersion expérimentale POCO8 (e = 0.105 mm)', log = True, exp = True)

if rdmf :
    k_rdmf = (2 * np.pi) / r_d_d_rdmf[:,0] * 1000
    omega_rdmf = 2 * np.pi * r_d_d_rdmf[:,1]
    joliplot(r'k (m$^{-1}$)', r'$\omega$ (s$^{-1}$)', k_rdmf, omega_rdmf, color = 5, title =title, legend = r'Width = 0.106 mm', log = True, exp = True)



# plt.grid("off")

#%% fit et courbe RDD
tension_surface = 50E-3
k_tot = []
omega_tot =[]


if pdp :
    k_tot = k_pdp
    omega_tot = omega_pdp
if pdp and poco8:
    k_tot = np.append(k_pdp ,k_poco8 )
    omega_tot = np.append(omega_pdp, omega_poco8)
if epiv and pdp :
    k_tot = np.append(k_pdp ,k_epiv )
    omega_tot = np.append(omega_pdp, omega_epiv)
if rsbp and rdmf :
    k_tot = np.append(k_rdmf ,k_rsbp )
    omega_tot = np.append(omega_rdmf, omega_rsbp)


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

if epiv :
    poptepiv, pcovepiv = curve_fit(RDD_fl, k_epiv, omega_epiv)

# popt_comp, pcov_comp = curve_fit(RDD_comp, k_tot, omega_tot)
popt_comp, pcov_comp = curve_fit(RDD_comp, k_rdmf, omega_rdmf)




    
# k_ligne = np.linspace(min(k_tot),max(k_tot),100)
k_ligne = np.linspace(min(k_rdmf),max(k_rdmf),100)


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



joliplot(r'k (m$^{-1}$)', r'$\omega$ (s$^{-1}$)', k_ligne, RDD_comp(k_ligne, popt_comp[0]),color = 3, legend = r'Fit en gk + $\gamma$k$^{3}$ / $\rho$ + D$k^{5}$/ $\rho$, D = ' + str(round( popt_comp[0] * rho , 8) ) + " Nm", exp = False )


   
plt.plot(k_ligne, rel_cap, 'k-', label = r'Régime capillaire, $\gamma$ = ' + str(tension_surface) + " Nm$^{-1}$")


plt.axis('equal')
# plt.grid()
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

Epdp = (Dpdp * 12 * ( 1 - nu**2) )/ hpdp**3
# Erdmf = (Drdmf * 12 * ( 1 - nu**2) )/ hrdmf**3
# Eepiv = (Depiv * 12 * ( 1 - nu**2) )/ hepiv**3

Lpdp = pow(Dpdp/(rho * g),0.25)
# Lrdmf = pow(Drdmf/(rho * g),0.25)
# Lepiv = pow(Depiv/(rho * g),0.25)


