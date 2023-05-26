# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 10:55:00 2022

@author: Banquise
"""

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

# mmparpixelx, mmparpixely, mmparpixelz, angle_cam_LAS, mmparpixel = import_calibration(titre_exp,date)          
              
# Creates a file txt with all the parameters of the experience and of the analysis
    
param_complets = ["Paramètres d'adimensionnement :",  "lambda_vague = " + str(lambda_vague) , "Ampvague = " + str(Ampvague) ,  "Paramètres d'analyse : ", "debut = " + str(debut) ,"kernel_size = " + str(k_size_crack), "kernel_iteration = " + str(k_iteration_crack) ,"nbframe = " + str(nbframe) , "minsize_crack = " + str(crack_lenght_min) , "sepwb = " + str(sepwb_cracks) , "size_crop = " + str(size_crop), "mmparpixely = " + str(mmparpixely), "Paramètres experimentaux : ", "facq = " + str(facq) , "texp = " + str(texp) , "Tmot = " + str(Tmot) , "Vmot = " + str(Vmot), "Hw = " + str(Hw), "Larg_ice = " + str(Larg_ice), "Long_ice = " + str(Long_ice), "tacq = " + str(tacq), "type_exp = " + str(type_exp)]


openn_dico = True
if openn_dico :
    dico = open_dico()



#%%points expérimentaux

exp_à_traiter = ["DAP07","DAP08","DAP09","DAP10"] #[ "DAP10","DAP09","DAP07","PIVA6"]#["DAP01","DAP02","DAP03","DAP04","DAP05","DAP06","DAP07","DAP08","DAP09","DAP10"] #["all","DAP"] 
exp_séparées = False
ATT = False
RDD = True


g = 9.81
tension_surface = 50E-3
rho = 900


m = (226.93-188.24)/3.1

S = 0.4*.26
h = m / (S * rho) #importer h depuis les paramètres


RDD_exp = np.asarray([[0,0,0]])
ATT_exp = np.asarray([[0,0,0]])


if exp_à_traiter[0] == 'all' :
    for i in exp_à_traiter[1:]:
        if RDD :
            fichiers_RDD = os.listdir(loc_resultats + "\\" + date + "_" + i + "/RDD\\")
            RDD_exp = np.loadtxt(loc_resultats + "\\" + date + "_" + i + "/RDD\\" + fichiers_RDD[0])
        if ATT :
            fichiers_ATT = os.listdir(loc_resultats + "\\" + date + "_" + i + "/ATT\\")
            ATT_exp = np.loadtxt(loc_resultats + "\\" + date + "_" + i + "/ATT\\" + fichiers_ATT[0])
        
else :
    len_exp_RDD = [0]
    len_exp_ATT_1 = [0]
    RDD_exp = np.asarray([[0,0,0]])
    ATT_exp = np.asarray([[0,0,0]])
    for i in exp_à_traiter:
        if RDD :
            path_images, liste_images, titre_exp = import_images(loc,i,"LAS")
            RDD_temp = np.loadtxt(path_images[:-15] + "\\resultats\lambda_err_fexc" + date + "_" + i + ".txt")
            len_exp_RDD.append(len(RDD_temp) + len_exp_RDD[-1])
            RDD_exp = np.concatenate((RDD_exp, RDD_temp), axis = 0)
        if ATT :
            ATT_temp = np.loadtxt(path_images[:-15] + "\\resultats\kappa_err_fexc" + date + "_" + i + ".txt")
            len_exp_ATT_1.append(len(ATT_temp) + len_exp_ATT_1[-1])
            ATT_exp = np.concatenate((ATT_exp, ATT_temp), axis = 0)


if RDD :
    lambda_RDD = RDD_exp[:,0] #en metres
    Err_lambda = RDD_exp[:,1]
    omega_RDD = 2 * np.pi * RDD_exp[:,2] #en Hz
    omega_RDD = omega_RDD[1:]
    k_RDD = 2 * np.pi / lambda_RDD[1:]
    Err_k = ((Err_lambda[1:] / lambda_RDD[1:]) + 0.02) * k_RDD
    #Si on veut enlever des pts absurdes
    k_RDD_2 = []
    omega_RDD_2 = []
    for i in range(len( k_RDD)) :
        if omega_RDD[i] > 100 or k_RDD[i] < 400:
            if k_RDD[i] < 2000  :
                k_RDD_2.append(k_RDD[i]) #+ np.sqrt(Err_k)
                omega_RDD_2.append(omega_RDD[i])

if ATT :
    kappa_ATT_1 = ATT_exp[:,0] # en cm-1
    Err_kappa_1 = ATT_exp[:,1]
    omega_ATT_1 = 2 * np.pi * ATT_exp[:,2] #en Hz   
    #Pour enlever les valeurs de kappa qui sont très faibles (donc mal fitées)
    kappa_ATT = []
    Err_kappa = []
    omega_ATT = []
    len_exp_ATT = [0 for u in range (len(len_exp_ATT_1))]
    for j in range (1,len(exp_à_traiter)+1):
        len_exp_ATT[j] += len_exp_ATT[j-1]
        for i in range(len_exp_ATT_1[j] - len_exp_ATT_1[j-1]):
            if kappa_ATT_1[(len_exp_ATT_1[j-1]) + i ] >= 0.06 :
                kappa_ATT.append(kappa_ATT_1[(len_exp_ATT_1[j-1]) + i ])
                Err_kappa.append(Err_kappa_1[(len_exp_ATT_1[j-1]) + i ])
                omega_ATT.append(omega_ATT_1[(len_exp_ATT_1[j-1]) + i ])
                len_exp_ATT[j] += 1

    

          
if ATT :
    figurejolie()
    if exp_séparées :
        color = [2,6,9,3,1,4,7,5,12,8,11]
        for i in range (len(exp_à_traiter)) :
            joliplot( r'$\omega$ (s$^{-1}$)', r'$\kappa$ (cm$^{-1}$)',omega_ATT[len_exp_ATT[i]:len_exp_ATT[i+1]], kappa_ATT[len_exp_ATT[i]:len_exp_ATT[i+1]],color = color[i], legend = r'Width = ' + str(round(h,5)) + ' mm'+ ' exp :' + exp_à_traiter[i], log = True, exp = True)
            plt.errorbar(omega_ATT, kappa_ATT, yerr = Err_kappa, fmt = 'none', capsize = 5, ecolor = vcolors[2], zorder = 1) 
    
        
    else :
        joliplot( r'$\omega$ (s$^{-1}$)', r'$\kappa$ (cm$^{-1}$)',omega_ATT, kappa_ATT, color = 3, legend = r'Width = ' + str(round(h,5)) + ' mm', log = True, exp = True)
        plt.errorbar(omega_ATT, kappa_ATT, yerr = Err_kappa, fmt = 'none', capsize = 5, ecolor = vcolors[2], zorder = 1) 


if RDD :
    figurejolie()
    if exp_séparées :
        color = [8,6,2,3,1,4,7,5,12,9,11]
        for i in range(len(exp_à_traiter)):
            joliplot(r'k (m$^{-1}$)', r'$\omega$ (s$^{-1}$)', k_RDD_2[len_exp_RDD[i]:len_exp_RDD[i+1]], omega_RDD_2[len_exp_RDD[i]:len_exp_RDD[i+1]],color = color[i],  legend = r'Width = ' + str(round(h,5)) + ' mm' + ' exp :' + exp_à_traiter[i], log = True, exp = True)
            plt.errorbar(k_RDD, omega_RDD, xerr = Err_k, fmt = 'none', capsize = 5, ecolor = vcolors[4],elinewidth = 1,capthick = 2)
        
    else :
        joliplot(r'k (m$^{-1}$)', r'$\omega$ (s$^{-1}$)', k_RDD_2, omega_RDD_2, color = 4, legend = r'Donées expérimentales, h = ' + str(round(120)) + r'$\mu$m', log = True, exp = True)
        # plt.errorbar(k_RDD, omega_RDD, xerr = Err_k, fmt = 'none', capsize = 5, ecolor = vcolors[4],elinewidth = 1,capthick = 2)



#%% fit et courbe RDD

"""BUT : plotter tt les exp ou seumement une (ou plusieurs à la fois) pour regarder les points apres mesure automatisée 

A noter : Il faudrait mettre une frequence seuil à partir de la quelle on ne regarde lus les points """


k_tot = []
omega_tot =[]

for i in range(len( k_RDD_2)) :
    if omega_RDD_2[i] > 2000 or k_RDD_2[i] < 6500:
        if k_RDD_2[i] < 100000 :
            k_tot.append(k_RDD_2[i]) #+ np.sqrt(Err_k)
            omega_tot.append(omega_RDD_2[i])
    
# k_tot = k_RDD
# omega_tot = omega_RDD


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



joliplot(r'k (m$^{-1}$)', r'$\omega$ (s$^{-1}$)', k_ligne, RDD_comp(k_ligne, popt_comp[0]),color = 2, legend = r'Ajustement: gk + $\gamma$k$^{3}$ / $\rho$ + D$k^{5}$/ $\rho$. On mesure D = ' + str(round( popt_comp[0] * rho , 6) ) + " Nm", exp = False )


   
plt.plot(k_ligne, rel_cap, 'k-', label = r'Régime capillaire, $\gamma$ = ' + str(tension_surface) + " Nm$^{-1}$")


# plt.axis('equal')
plt.grid()
plt.legend()



#%% SAVE
save = False


rho = rho
g = g
nu = 0.5

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

if save :
    param_complets.extend([ "rho = " + str(rho), "nom_exp = " + str(nom_exp),  "D_mesure = " + str(Dpdp),"tension_surface = " + str(tension_surface), "rho = " + str(rho), "tension_surface = " + str(tension_surface)])
    dico = add_dico(dico,date,nom_exp,'Ld', Lpdp)
    dico = add_dico(dico,date,nom_exp,'rho_utilise', rho)
    dico = add_dico(dico,date,nom_exp,'D', Dpdp)
    param_complets = np.asarray(param_complets)
    np.savetxt(path_images[:-15] + "resultats" + "/Paramètres_RDD_" + nom_exp + ".txt", param_complets, "%s")
    plt.savefig(path_images[:-15] + "resultats" + "/" + "RDD_" + nom_exp + ".pdf") 
    save_dico(dico)
 
    

#%%kappa(k)

popt_comp[0]

# k_ligne, RDD_comp(k_ligne, popt_comp[0])


from scipy.optimize import minimize

x = np.arange(10,1000,1)

y = np.zeros(x.shape)

def diff(x,a):
    yt = RDD_comp(x,popt_comp[0])
    return (yt - a )**2

for idx,x_value in enumerate(x):
    res = minimize(diff, 1.0, args=(x_value), method='Nelder-Mead', tol=1e-6)
    y[idx] = res.x[0]

figurejolie()
joliplot(r'k (m$^{-1})$', r'$\kappa (cm^{-1})$', np.interp(omega_ATT,x,y), kappa_ATT, color = 3,legend = r'Width = ' + str(round(h,4)) + ' mm', log = True)
plt.errorbar(np.interp(omega_ATT,x,y), kappa_ATT, yerr = Err_kappa, fmt = 'none', capsize = 5, ecolor = vcolors[2], zorder = 1) 


# figurejolie()
# joliplot(r'k (m$^{-1}$)', r'$\omega$ (s$^{-1}$)', k_ligne, RDD_comp(k_ligne, popt_comp[0]),color = 2, legend = r'Fit en gk + $\gamma$k$^{3}$ / $\rho$ + D$k^{5}$/ $\rho$, D = ' + str(round( popt_comp[0] * rho , 8) ) + " Nm", exp = False )


#%%

ppp = np.polyfit(np.interp(omega_ATT,x,y), kappa_ATT,1)
joliplot(r'k (m$^{-1})$', r'$\kappa (cm^{-1})$', np.interp(omega_ATT,x,y), np.interp(omega_ATT,x,y) * ppp[0] + ppp[1], color = 2,legend = r'Width = ' + str(round(h,4)) + ' mm', log = True, exp = True)
plt.axis("equal")

#%% kappa/Ks

viscosite = 1E-6
kS = np.sqrt(viscosite/(2 * np.asarray(omega_ATT)))* np.interp(omega_ATT,x,y)**2
kB = 2 * viscosite / (np.asarray(omega_ATT)) * (np.interp(omega_ATT,x,y)**3)

figurejolie()
joliplot(r'$\omega$ (s${-1}$)', r'$\kappa$ / Ks', omega_ATT, np.asarray(kappa_ATT)*100/kS, color = 12, log = True)
plt.errorbar(omega_ATT, np.asarray(kappa_ATT)*100/kS, yerr = np.asarray(Err_kappa)*100/kS, fmt = 'none', capsize = 5, ecolor = vcolors[5], zorder = 1) 



figurejolie()
joliplot(r'$\omega$ (s${-1}$)', r'$\kappa$ / Kb', omega_ATT, np.asarray(kappa_ATT)*100/kB, color = 3, log = True)
plt.errorbar(omega_ATT, np.asarray(kappa_ATT)*100/kB, yerr = np.asarray(Err_kappa)*100/kB, fmt = 'none', capsize = 5, ecolor = vcolors[2], zorder = 1) 


# figurejolie()
# joliplot(r'$\omega$ (s${-1}$)', r'$\kappa$ / Kb / Ks', omega_ATT, np.asarray(kappa_ATT)*100/kB / kS)




