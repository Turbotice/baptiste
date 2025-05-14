# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 18:06:43 2023

@author: Banquise

Description du fonctionnement dans cahier de Manip Mars 2024
"""

import cv2 
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from skimage import feature
from scipy.signal import convolve2d
from scipy.signal import savgol_filter, gaussian
from scipy.signal import medfilt
from scipy.signal import medfilt2d
from scipy.signal import find_peaks
from scipy.signal import detrend
from scipy.optimize import curve_fit
from scipy import ndimage
from scipy import stats
import scipy.fft as fft
import os
from PIL import Image
from datetime import datetime
from scipy.optimize import minimize
import matplotlib.cm as cm
import pandas

import baptiste.display.display_lib as disp
import baptiste.experiments.import_params as ip
import baptiste.files.file_management as fm
import baptiste.image_processing.image_processing
import baptiste.files.dictionaries as dic
import baptiste.tools.tools as tools
import baptiste.math.RDD as rdd
import baptiste.math.fits as fits

dico = dic.open_dico()


                
#%% Liste top refaite mieux

# nb_exp = 12
nb_exp = 10
# nb_exp = 8

# exps = ['TMFR', 'ECTD', 'ETHY', 'EDTH', 'NPDP', 'RLPY', 'MLO', 'EJCJ', 'DML', 'QSC0', 'TNB0', 'CCMO']
exps = ['ECTD', 'EDTH', 'NPDP', 'RLPY', 'MLO', 'EJCJ', 'DML', 'QSC0', 'TNB0', 'CCMO']
# exps = ['ECTD', 'EDTH', 'NPDP', 'RLPY', 'DML', 'QSC0', 'TNB0', 'CCMO']


date_min = 231115
date_max = 240117
save = False

long_onde = np.array([])
amp_max = np.array([])
amp_moy = np.array([])
l_cracks = np.array([])

# liste_top = np.zeros((7,160), dtype = object)
liste_top = np.zeros((6,156), dtype = object)
# liste_top = np.zeros((7,133), dtype = object)

tableau_1 = pandas.read_csv('E:\\Baptiste\\Resultats_exp\\Tableau_params\\Tableau1_Params_231117_240116\\tableau_1.txt', sep = '\t', header = 0)
tableau_1 = np.asarray(tableau_1)

# courbure
best_a = True
if best_a :
    path = ['E:\\Baptiste\\Resultats_exp\\Courbure\\231120_ECTD\\' + '20240422_180550_params_courbure.pkl' ,
            'E:\\Baptiste\\Resultats_exp\\Courbure\\231121_EDTH_a30\\' + '20240424_172527_params_courbure.pkl',
            'E:\\Baptiste\\Resultats_exp\\Courbure\\231122_NPDP\\' + '20240419_171920_params_courbure.pkl' , 
            'E:\\Baptiste\\Resultats_exp\\Courbure\\231124_RLPY\\' + '20240422_153853_params_courbure.pkl' ,
            'E:\\Baptiste\\Resultats_exp\\Courbure\\231129_MLO_a50\\' + '20240422_182428_params_courbure.pkl',
            'E:\\Baptiste\\Resultats_exp\\Courbure\\231129_EJCJ_a30\\' + '20240423_123023_params_courbure.pkl' ,
            'E:\\Baptiste\\Resultats_exp\\Courbure\\231130_DML_a50\\' + '20240424_190359_params_courbure.pkl',
            'E:\\Baptiste\\Resultats_exp\\Courbure\\240109_QSC_a50\\' + '20240422_171834_params_courbure.pkl',
            'E:\\Baptiste\\Resultats_exp\\Courbure\\240115_TNB_a30\\' + '20240424_123149_params_courbure.pkl',
            'E:\\Baptiste\\Resultats_exp\\Courbure\\240116_CCM_ttpts\\' + 'CCM_nomexp_A_kappamax_kappath_a30.txt']

u = 0
for date in dico.keys() :
    if date.isdigit() :
        if float(date) >= date_min and float(date) < date_max :
            if float(date) != 231123 :
                for nom_exp in dico[date].keys() :
                    for exp in exps :
                        if nom_exp[:3] in exp :
                            liste_top[0,u] = nom_exp
                            u += 1

mmparpixel = []
gr = []

aa = 0
for date in dico.keys() :
    if date.isdigit() :
        if float(date) >= date_min and float(date) < date_max :
            if float(date) != 231123 :
                for nom_exp in dico[date].keys() :
                    if nom_exp in liste_top[0,:]:
                        print(nom_exp)
                        for i in range(len(exps)) :
                            if nom_exp[:3] in exps[i] :
                                aa = i
                        print(exps[aa])
                        indice = np.where(nom_exp == liste_top[0,:])[0][0]
                        mmparpixel.append(dico[date][nom_exp]['mmparpixel'])
                        gr.append(dico[date][nom_exp]['grossissement'])
                        
                        if 'lambda' in dico[date][nom_exp].keys() :
                            liste_top[1,indice] = float(dico[date][nom_exp]['lambda'])
                            
                        if 'Amp_moy' in dico[date][nom_exp].keys() :
                            liste_top[2,indice] = dico[date][nom_exp]['Amp_moy']
                            
                        if 'Amp_max' in dico[date][nom_exp].keys() :
                            liste_top[3,indice] = dico[date][nom_exp]['Amp_max']
                            
                        if 'l_cracks' in dico[date][nom_exp].keys() :
                            liste_top[4,indice] = np.nansum(dico[date][nom_exp]['l_cracks']) / 1000
                        


                        if 'CCM' in nom_exp :  #CCM  
                            params_k = pandas.read_csv(path[-1], header = None, sep = '\t')
                            params_k = np.asarray(params_k)
                            if nom_exp in params_k[:,0] :
                                blbl = np.where(nom_exp == params_k[:,0])[0][0]
                                k_max = params_k[blbl,2]
                            else :
                                k_max = 0
                            
                            liste_top[5,indice] = k_max

                        
                                

                        elif 'EDTH' in nom_exp : #ECTD et EDTH
                        
                            params_k = dic.open_dico(path[aa])
                            
                            if nom_exp[-2:].isdigit() :
                                k_max = params_k['courbure']['k_minmin'][int(nom_exp[-2:])]
                            elif nom_exp[-1:].isdigit() :
                                k_max = params_k['courbure']['k_minmin'][int(nom_exp[-1:])]
                            else :
                                k_max = 0
                                
                            liste_top[5,indice] = k_max
                        
                        elif 'ECT' in nom_exp :   
                            params_k = dic.open_dico(path[aa])
                            
                            if nom_exp[-2:].isdigit() :
                                if int(nom_exp[-2:]) != 12 :
                                    k_max = params_k['courbure']['k_maxmax'][int(nom_exp[-2:])]
                                else :
                                    k_max = 0
                            elif nom_exp[-1:].isdigit() :
                                k_max = params_k['courbure']['k_maxmax'][int(nom_exp[-1:])]
                            else :
                                k_max = 0
                  
                            liste_top[5,indice] = k_max
                            
                        
                        elif 'RLP' in nom_exp :   
                            params_k = dic.open_dico(path[aa])
                            if nom_exp[-1:].isdigit() :
                                if not nom_exp[-2:].isdigit() :
                                    k_max = params_k['courbure']['k_minmin'][int(nom_exp[-1:])]
                                else :
                                    k_max = 0
                            else :
                                k_max = 0
                  
                            liste_top[5,indice] = k_max
                        
                        elif 'DML' in nom_exp :   
                            params_k = dic.open_dico(path[aa])
                            
                            if nom_exp[-2:].isdigit() :
                                if int(nom_exp[-2:]) < 12 :
                                    k_max = params_k['courbure']['k_minmin'][int(nom_exp[-2:])]
                                else :
                                    k_max = 0
                            elif nom_exp[-1:].isdigit() :
                                k_max = params_k['courbure']['k_minmin'][int(nom_exp[-1:])]
                            else :
                                k_max = 0
                  
                            liste_top[5,indice] = k_max
                        
                        elif 'NPDP' in nom_exp :   
                            params_k = dic.open_dico(path[aa])
                            
                            if nom_exp[-1:].isdigit() :
                                if int(nom_exp[-1:]) < 8 :
                                    k_max = params_k['courbure']['k_minmin'][int(nom_exp[-1:])]
                                else :
                                    k_max = 0
                            else :
                                k_max = 0
                  
                            liste_top[5,indice] = k_max
                            
                        elif 'MLO2' in nom_exp :   
                            params_k = dic.open_dico(path[aa])
                            
                            if nom_exp[-2:].isdigit() :
                                if int(nom_exp[-2:]) < 25 :
                                    k_max = params_k['courbure']['k_minmin'][int(nom_exp[-2:])]
                                else :
                                    k_max = 0
                            elif nom_exp[-1:].isdigit() :
                                k_max = params_k['courbure']['k_minmin'][int(nom_exp[-1:])]
                            else :
                                k_max = 0
                            
                            liste_top[5,indice] = k_max
                            
                        elif 'EJCJ' in nom_exp :   
                            params_k = dic.open_dico(path[aa])
                            
                            if nom_exp[-2:].isdigit() :
                                if int(nom_exp[-2:]) < 25 :
                                    k_max = params_k['courbure']['k_minmin'][int(nom_exp[-2:])]
                                else :
                                    k_max = 0
                            elif nom_exp[-1:].isdigit() :
                                k_max = params_k['courbure']['k_minmin'][int(nom_exp[-1:])]
                            else :
                                k_max = 0
                            
                            liste_top[5,indice] = k_max
                            
                        elif 'QSC' in nom_exp :   
                            params_k = dic.open_dico(path[aa])
                            
                            if nom_exp[-2:].isdigit() :
                                if int(nom_exp[-2:]) < 22 :
                                    k_max = params_k['courbure']['k_minmin'][int(nom_exp[-2:])]
                                else :
                                    k_max = 0
                            elif nom_exp[-1:].isdigit() :
                                k_max = params_k['courbure']['k_minmin'][int(nom_exp[-1:])]
                            else :
                                k_max = 0
                            
                            liste_top[5,indice] = k_max
                            
                        elif 'TNB' in nom_exp :   
                            params_k = dic.open_dico(path[aa])
                            
                            if nom_exp[-2:].isdigit() :
                                if int(nom_exp[-2:]) < 31 :
                                    k_max = params_k['courbure']['k_minmin'][int(nom_exp[-2:])]
                                else :
                                    k_max = 0
                            elif nom_exp[-1:].isdigit() :
                                k_max = params_k['courbure']['k_minmin'][int(nom_exp[-1:])]
                            else :
                                k_max = 0
                            
                            liste_top[5,indice] = k_max
                            
                        
                        else :
                            
                            params_k = dic.open_dico(path[aa])
                            
                            if nom_exp[-2:].isdigit() :
                                k_max = params_k['courbure']['k_minmin'][int(nom_exp[-2:]) - 1]
                            elif nom_exp[-1:].isdigit() :
                                k_max = params_k['courbure']['k_minmin'][int(nom_exp[-1:]) - 1]
                            else :
                                k_max = 0
                  
                            liste_top[5,indice] = k_max

mmparpixel = np.array([mmparpixel])
gr = np.array([gr])                       
# df = pandas.DataFrame(liste_top)
# df.to_csv('E:\\Baptiste\\Resultats_exp\\Tableau_params\\Tableau3_Params_231117_240116\\tableau_3.txt', index=False, header=False, sep = '\t')   


#%% Plot Amp_max (nom_exp)



for i in exps :
    expp = np.array([])
    ampp = np.array([])
    for j in liste_top[0,:] :
        if i[:3] in j :
            indice = np.where(j == liste_top[0,:])[0][0]
            expp = np.append(expp, j)
            ampp = np.append(ampp, liste_top[3, indice])
            
    # disp.figurejolie()
    # disp.joliplot("Nom exp", "Amp max", expp, ampp , color = 2)


#%% Plot Lcrack tot (nom_exp)



for i in exps :
    expp = np.array([])
    lcracktot = np.array([])
    for j in liste_top[0,:] :
        if i[:3] in j :
            indice = np.where(j == liste_top[0,:])[0][0]
            expp = np.append(expp, j)
            lcracktot = np.append(lcracktot, liste_top[4, indice])
            
    # disp.figurejolie()
    # disp.joliplot(r"Nom exp", r"$L_{crack}$ tot", expp, lcracktot , color = 4)

#%% Plot Lcracktot (Amp_max) + fit

display = True
save = False

path_save = 'E:\\Baptiste\\Resultats_exp\\Amp seuil en fonction de lambda\\Resultats_20240528\\'

a = np.zeros(nb_exp)
b = np.zeros(nb_exp)
amp_s = np.zeros(nb_exp)
long_onde = np.zeros(nb_exp)
erreur = np.zeros(nb_exp, dtype = object)
lcracktot = [np.array([]) for i in range (nb_exp)]
lcrack_lc0 = [np.array([]) for i in range (nb_exp)]
ampp = [np.array([]) for i in range (nb_exp)]
ampp_lc0 = [np.array([]) for i in range (nb_exp)]
long_onde_lc0 = [np.array([]) for i in range (nb_exp)]

u = 0
for i in exps :
    for j in liste_top[0,:] :
        if i[:3] == j[:3] :
            indice = np.where(j == liste_top[0,:])[0][0]
            if liste_top[3, indice] != 0.0 :
                if i == 'MLO' :
                    ampp[u] = np.append(ampp[u], liste_top[3, indice])
                    long_onde[u] = liste_top[1,indice]
                    lcracktot[u] = np.append(lcracktot[u], liste_top[4, indice])
                    
                    lcrack_lc0[u] = np.append(lcrack_lc0[u], liste_top[4, indice])
                    long_onde_lc0[u] = liste_top[1,indice]
                    ampp_lc0[u] = np.append(ampp_lc0[u], liste_top[3, indice])
                    
                else :    
                    if liste_top[4, indice] != 0.0 :
                        ampp[u] = np.append(ampp[u], liste_top[3, indice])
                        lcracktot[u] = np.append(lcracktot[u], liste_top[4, indice])
                        long_onde[u] = liste_top[1,indice]
                        
                        ampp_lc0[u] = np.append(ampp_lc0[u], liste_top[3, indice])
                        lcrack_lc0[u] = np.append(lcrack_lc0[u], liste_top[4, indice])
                        long_onde_lc0[u] = liste_top[1,indice]
                        
                    else :
                        ampp_lc0[u] = np.append(ampp_lc0[u], liste_top[3, indice])
                        lcrack_lc0[u] = np.append(lcrack_lc0[u], liste_top[4, indice])
                        long_onde_lc0[u] = liste_top[1,indice]
                        
    x_amp = np.linspace(np.min(ampp[u]), np.max(ampp[u]) , 100)
    # plt.plot(x_amp, lambdasur2)
    
    if len (ampp[u]) > 2 :
        if i != 'ECTD' and i != 'EJCJ' and i != 'EDTH' :
            ampp[u] = ampp[u][:-1]
            lcracktot[u] = lcracktot[u][:-1]
            ampp_lc0[u] = ampp_lc0[u][:-1]
            lcrack_lc0[u] = lcrack_lc0[u][:-1]
    
    if display :
        disp.figurejolie()
        disp.joliplot(r"Amp (cm)", r"$L_{crack}$ (cm)", ampp_lc0[u] , lcrack_lc0[u] , color = 8, zeros = True)
    
    # popt = np.polyfit(ampp[u], lcracktot[u], 1, full = True)
    def fit_1(x, a, b) :
        return a * x + b
    
    popt, pcov = curve_fit(fit_1, ampp[u], lcracktot[u])#, p0 = [10000000, 10000], bounds = [[0,0], [100000000, 10000000]]) #np.polyfit(hmin_QSC, kmin_QSC,2)
    erreur[u] = pcov
    # popt_kappa[i] = popt
    # a[u] = popt[0][0]
    # b[u] = popt[0][1]
    a[u] = popt[0]
    b[u] = popt[1]
    amp_s[u] = (0 - b[u]) / a[u] #(long_onde[u] - b[u]) / a[u] 
    # if len (ampp[u]) > 2 :
    #     erreur[u] = popt[1:][0][0]
    # elif len (ampp[u]) == 2 :
    #     erreur[u] = 0#(ampp[1] - amp_s[u])/ampp[1] / 2

    x_amp = np.linspace(-200, 200, 1000)
    if display :
        disp.joliplot(r"Amplitude (m)", r"$L_{crack}$ (m)", x_amp, x_amp * a[u] + b[u] , color = 5, exp = False, width = 8.6/2)
        
    zeroo = np.linspace(0, 0, 1000)
    plt.plot(x_amp, zeroo)
    disp.joliplot(r"Amplitude (cm)", r"$L_{crack}$ (cm)", x_amp, zeroo, color = 5, exp = False)
    # plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    # plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    
    plt.xlim([0, np.max(ampp_lc0[u]) * 1.05  ])
    plt.ylim([-np.max(lcrack_lc0[u]) / 10 , np.max(lcrack_lc0[u]) * 1.1  ]) 
    
    if i == 'QSC0' :
        print(ampp)
    
    if save :
        plt.savefig(path_save + "l_crack_amplitude+fit_" + i + '_' + str(tools.datetimenow()) + '_' + i  + '.png', dpi = 200)
        plt.savefig(path_save + "l_crack_amplitude+fit_" + i + '_pdf' + str(tools.datetimenow()) + '_' + i  + '.pdf')
    
    

    u += 1
    
#%% Plot Lcracktot (kappa) + fit

display = True
save = False


a = np.zeros(nb_exp)
b = np.zeros(nb_exp)
k_s = np.zeros(nb_exp)
long_onde = np.zeros(nb_exp)
erreur = np.zeros(nb_exp, dtype = object)


u = 0
for i in exps :
    kappa = np.array([])
    lcracktot = np.array([])
    lambdasur2 = 0
    for j in liste_top[0,:] :
        
        if i[:3] == j[:3] :
            indice = np.where(j == liste_top[0,:])[0][0]
            
            if liste_top[5, indice] != 0.0 :
                # if i == 'EDTH' or i == 'EJCJ' or i == 'TBN0':
                if liste_top[4, indice] != 0.0 :
                    kappa = np.append(kappa, liste_top[5, indice])
                    lcracktot = np.append(lcracktot, liste_top[4, indice])
                    lambdasur2 = liste_top[1,indice] / 2
                    # kappa = np.append(kappa, liste_top[5, indice])
                    # lcracktot = np.append(lcracktot, liste_top[4, indice])
                    # lambdasur2 = liste_top[1,indice] / 2
            
    if display :
        disp.figurejolie()
        disp.joliplot(r"$\kappa$ (m$^{-1}$)", r"$L_{crack}$ (m)", kappa, lcracktot , color = 8, legend = i)
    
    long_onde[u] = lambdasur2 * 2
    lambdasur2 = np.linspace(lambdasur2, lambdasur2, 100)
    x_amp = np.linspace(np.min(kappa), np.max(kappa), 100)
    # plt.plot(x_amp, lambdasur2)
    
    # if len (kappa) > 2 :
    #     kappa = kappa[:-1]
    #     lcracktot = lcracktot[:-1]
    
    popt = np.polyfit(kappa, lcracktot, 1, full = True)
    a[u] = popt[0][0]
    b[u] = popt[0][1]
    k_s[u] = (long_onde[u] - b[u]) / a[u] #(0 - b[u]) / a[u] 
    if len (kappa) > 2 :
        erreur[u] = popt[1:][0][0]
    elif len (kappa) == 2 :
        erreur[u] = 0#(ampp[1] - k_s[u])/ampp[1] / 2

    x_amp = np.linspace(np.min(kappa), np.max(kappa), 100)
    if display :
        disp.joliplot(r"$\kappa$ (m$^{-1}$)", r"$L_{crack}$ (m)", x_amp, x_amp * a[u] + b[u] , color = 5, legend = "fit", exp = False, zeros= True)
        if save :
            plt.savefig('E:\\Baptiste\\Resultats_exp\\Amp seuil en fonction de lambda\\240506_lcrack_Aetkappa\\' + "l_crack_amplitude+fit_" + str(tools.datetimenow()) + '_' + i  + '.png', dpi = 200)
    
    

    u += 1
#%% kappa seuil (lambda) avec erreur

disp.figurejolie()
disp.joliplot( r"$\lambda$ (m)", r"$\kappa_s$ (m$^{-1}$)", long_onde, k_s, color = 5, zeros=True)
plt.errorbar(long_onde, k_s, yerr = erreur * k_s * 2, fmt = 'none',capsize = 5, ecolor = 'red',elinewidth = 3)
for i in range (nb_exp):
    plt.annotate(exps[i], (long_onde[i], k_s[i]))
    
# plt.xlim(0, 0.6)
# plt.ylim(0, 0.025)

if save :
    plt.savefig('E:\\Baptiste\\Resultats_exp\\Amp seuil en fonction de lambda\\Test_fit_2024_04_16\\' + "ampseuil_lambda_lcrack_lambda" + str(tools.datetimenow()) + '.pdf', dpi = 1)

#%% amp seuil (lambda) avec erreur

disp.figurejolie()
disp.joliplot( r"$\lambda$ (m)", r"A$_c$ (m)", long_onde, amp_s, color = 5, zeros=True)
plt.errorbar(long_onde, amp_s, yerr = erreur * amp_s * 2, fmt = 'none',capsize = 5, ecolor = 'red',elinewidth = 3)
for i in range (nb_exp):
    plt.annotate(exps[i], (long_onde[i], amp_s[i]))
    
# plt.xlim(0, 0.6)
# plt.ylim(0, 0.025)

if save :
    plt.savefig('E:\\Baptiste\\Resultats_exp\\Amp seuil en fonction de lambda\\Test_fit_2024_04_16\\' + "ampseuil_lambda_lcrack_lambda" + str(tools.datetimenow()) + '.pdf', dpi = 1)



#%% a (lambda)

exps_filt = [exps[1]] + exps [3:5] + exps[6:-1]
long_onde_filt = np.append( np.append([long_onde[1]], long_onde [3:5]), long_onde[6:-1])
a_filt = np.append( np.append([a[1]], a[3:5]), a[6:-1])

disp.figurejolie()
disp.joliplot( r"$\lambda$ (m)", "a", long_onde_filt, a_filt, color = 5, zeros=False)
# for i in range (nb_exp):
#     plt.annotate(exps[i], (long_onde[i], a[i]))
# plt.savefig('E:\\Baptiste\\Resultats_exp\\Amp seuil en fonction de lambda\\Test_fit_2024_04_16\\' + "a_lambda" + str(tools.datetimenow()) + '.pdf', dpi = 1)


popt = np.polyfit(long_onde_filt, a_filt, 1)

disp.joliplot( r"$\lambda$ (m)", "a", long_onde_filt, long_onde_filt * popt[0] + popt[1], color = 2, zeros=True, exp = False)


disp.figurejolie()
disp.joliplot( r"$\lambda$ (m)", "a", long_onde, a, color = 5, zeros=False)
for i in range (nb_exp):
    plt.annotate(exps[i], (long_onde[i], a[i]))

#%% b (lambda)

disp.figurejolie()
disp.joliplot( r"$\lambda$ (m)", "b", long_onde, b, color = 5, zeros=False)
# plt.savefig('E:\\Baptiste\\Resultats_exp\\Amp seuil en fonction de lambda\\Test_fit_2024_04_16\\' + "b_lambda" + str(tools.datetimenow()) + '.pdf', dpi = 1)


#%%Fit : scaling Amplitude


# fits.fit_powerlaw(long_onde, amp_s, display = True, xlabel =r"$\lambda$ (m)", ylabel = r"A$_c$ (m)" )
disp.figurejolie()
disp.joliplot(r"$\lambda$ (m)", r"A$_c$ (m)", long_onde, amp_s, color = 2, log = True)#, legend = 'Threshold'  )

plt.xlim(0.04, 1)
plt.ylim(0.001, 0.025)

pente = amp_s/long_onde * 2 * np.pi
lambda_sort, pente_sort = tools.sort_listes(long_onde,pente)
H = 0.11
Ursell_number = amp_s * long_onde**2 / H**3
lambda_sur_A = long_onde/amp_s

if save :
    plt.savefig(path_save + "ampseuil_lambda_lcrack_lambda_powerlaw" + str(tools.datetimenow()) + '.pdf', dpi = 1)


"""ISO COURBURE"""
lamlam = np.linspace(0.04, 1, 500)
AA = np.linspace(0.001, 0.025, 500)

# kappa_th = np.zeros((500,500))

# for i in  range (len (AA)) :
#     for j in range (len(lamlam)) :
#         kappa_th[i,j] = AA[i] * 4 * np.pi**2/ lamlam[j]**2
    
# blbl = np.logspace( np.log10(np.min(kappa_th)), np.log10(np.max(kappa_th)), 10)        
# plt.contour(lamlam, AA, np.flip(np.rot90(kappa_th),0), blbl, linestyles = 'dashed', linewidths = 0.5, colors = 'k')
# plt.colorbar()


"""VISCOUS STRESS"""


visc = 10**-3
g = 9.81
omega = np.sqrt(g * 2 * np.pi / lamlam)
h = 1e-4
rho = 680
stress = np.zeros((500,500))

for i in  range (len (AA)) :
    for j in range (len(omega)) :
        stress[i,j] = np.sqrt(visc / omega[j])/ h * rho * g * AA[i]
        
blbl = np.logspace( np.log10(np.min(stress)), np.log10(np.max(stress)), 10)        
plt.contour(lamlam, AA, np.flip(np.rot90(stress),0), blbl, linestyles = 'dashed', linewidths = 0.5)
plt.colorbar()







#%%Fit : scaling courbure


fits.fit_powerlaw(long_onde, k_s, display = True, legend = 'Threshold', xlabel =r"$\lambda$ (m)", ylabel = "courbure (m)" )

pente = k_s/long_onde * 2 * np.pi
lambda_sort, pente_sort = tools.sort_listes(long_onde,pente)
H = 0.11
Ursell_number = k_s * long_onde**2 / H**3
lambda_sur_A = long_onde/k_s

if save :
    plt.savefig('E:\\Baptiste\\Resultats_exp\\Amp seuil en fonction de lambda\\Test_fit_2024_04_16\\' + "ampseuil_lambda_lcrack_lambda_powerlaw" + str(tools.datetimenow()) + '.pdf', dpi = 1)


#%%ADD POINTS NICOLAS

liste_top_nico = np.zeros((8,66), dtype = object)

#dedans on met : lambda, amp (on moyenne si file las le meme qu'avant), lcrack (0 si casse pas, 1 si casse), bruit (on peut faire errorbar avec), lambda gk, lambda  gk yk3 

# for date in ['230725', '230726','230727','230728'] :
#     params = {}
#     params['date'] = date
#     params['loc'] = 'E:\\Nicolas\\d' + date + '\\'
#     exp = os.listdir(params['loc'])
    
#     for data in exp :
#         params['loc_data'] = os.listdir(params['loc'] + data + '\\Data\\')
        
        
#         for file_las in params['loc_data'] :
            
#             if file_las[-4:] == '.pkl' :
#                 print(data[-6:])
#                 print(file_las)
                
#                 params = dic.open_dico(params['loc'] + data + '\\Data\\' + file_las)
                
#                 params['loc_las'] = params['loc'] + data + '\\Data\\' + file_las
#                 params['facq_las'] = float(file_las[4:8])
#                 params['file_las'] = file_las
#                 params['fexc_1'] = float ( data[data.index("g_") + 2:data.index("g_") + 5]) / 2
#                 params['fexc_2'] = params['fexc_1'] * 2
#                 params['d_f_1'] = 2
#                 params['d_f_2'] = 2
                
#                 params['path_las'] = params['loc'] + data + '\\Data\\'


files = fm.get_files('E:\\Nicolas', 'params', '.pkl')
u = 0
lambda_f = {'40.0':0.02, '10.0': 0.0385, '60.0':0.0095,'80.0':0.0093,'100.0':0.007,'20.0' : 0.031}
for path in files : 
    params = dic.open_dico(path)
    liste_top_nico[0, u] = 0 #lcrack

    liste_top_nico[1, u] = params['amp_FFT_10Hz']/1000#amp_moy
    liste_top_nico[2, u] = 0#std
    
    liste_top_nico[3, u] = params['bruit']                                     #bruit
    liste_top_nico[4, u] = lambda_f[str(params['fexc_2'])]                     #lambda mesuré
    liste_top_nico[5, u] = 2 * np.pi * 9.81 / (params['fexc_1'] * 2 * np.pi)**2#lambda avec omega 
    liste_top_nico[6, u] = np.power(0.05 * (2 * np.pi)**3  / 1000 / (params['fexc_1'] * 2 * np.pi)**2 , 0.33)   #lambda avec gamma (flemme)
    liste_top_nico[7, u] = params['path_las'][-12:-7] #Volt                                                
    u += 1
    



# liste_top[0,86] = 'NICO2'
# liste_top[1,86] = 0.031
# liste_top[2,86] = 0.00031
# liste_top[3,86] = 0.00031
# liste_top[4,86] = 1
# liste_top[5,86] = 1


# liste_top[0,87] = 'NICO8'
# liste_top[1,87] = 0.013
# liste_top[2,87] = 0.000036
# liste_top[3,87] = 0.000036
# liste_top[4,87] = 1
# liste_top[5,87] = 1
disp.figurejolie()


colors_nico = np.zeros((66,4))
for i in range (66): 
    colors_nico[i,:] = disp.vcolors(int(liste_top_nico[3,i] * 9))
     
plt.scatter(liste_top_nico[4,:],liste_top_nico[1,:],color=colors_nico)

plt.xscale('log')
plt.yscale('log')

#%%Plot DDP NUL


# disp.figurejolie()

# plt.plot(liste_top[1,:], liste_top[2,:], 'kx')

disp.figurejolie()

colors = cm.rainbow(np.linspace(0, 1, 86))

for i in range (liste_top.shape[1]) :
    if liste_top[2,i] > 0 :
        plt.scatter(liste_top[1,i], liste_top[2,i], color=colors[int(liste_top[4,i] / np.max(liste_top[4,:]) * 85)])
        
        
disp.figurejolie()

colors = cm.rainbow(np.linspace(0, 1, 86))


for i in range (liste_top.shape[1]) :
    if liste_top[2,i] > 0 :
        plt.plot(liste_top[1,i], liste_top[2,i], 'x', color=disp.vcolors(int(liste_top[5,i] * 9))) 
plt.xlabel(r'$\lambda$ (m)') 
plt.ylabel('Amplitude (m)')


# for i in range (liste_top.shape[1]) :
#     if liste_top[3,i] > 0 :
#         if liste_top[4,i] > 0 :
#             plt.plot(liste_top[1,i], liste_top[3,i], color=colors[])
#         else:
#             plt.plot(liste_top[1,i], liste_top[3,i], color=colors[0])
            
            
#%% Plot bien
disp.figurejolie()
colors = np.zeros((169,4))
for i in range (169): 
    colors[i,:] = disp.vcolors(int(liste_top[5,i] * 9))
     
plt.scatter(liste_top[1,:],liste_top[3,:],color=colors)
# plt.scatter(liste_top_nico[6,:],liste_top_nico[1,:],color=colors_nico)
# plt.ylim([0.001,0.024])

#%%L_cracks (Amp)

disp.figurejolie()

plt.plot(liste_top[2,:], liste_top[4,:], 'kx')

plt.scatter(liste_top[1,i], liste_top[3,i], color=colors[int(liste_top[4,i] / np.max(liste_top[4,:(i+8)]) * 85)])   

#%% Ancien DDP 
date_min = 221003
date_max = 230120 #AJD ?

casse = []
l_onde = []
amplitude = []
exp = []
l_d = []
ldk = []





for date in dico :
    if date.isdigit() : 
        if float(date) > date_min and float(date) < date_max :
            print(date)
    
            for nom_exp in dico[str(date)] :
    
                if 'amp_fracs_fft' in dico[date][nom_exp] and nom_exp != 'MPPF2' and nom_exp != 'MPPF3' :
                    # if "non" in dico[date][nom_exp]['casse'] :
                    print(nom_exp)
                    for j in range (dico[date][nom_exp]['amp_fracs_fft'].shape[0]):
                        if np.shape(dico[date][nom_exp]['amp_fracs_fft'])[1] < 7 :
                            if 'oui' == dico[date][nom_exp]['casse'] :
                                casse.append(True)
                            else :
                                casse.append(False)
                        else :
                            if dico[date][nom_exp]['amp_fracs_fft'][j,6]:
                                casse.append(True)
                            else :
                                casse.append(False)
                            
                       
                        amplitude.append(dico[date][nom_exp]['amp_fracs_fft'][j,4] * np.sqrt(2))# * np.pi /dico[date][nom_exp]['lambda'] )
                        exp.append(nom_exp)
                        l_d.append(dico[date][nom_exp]['Ld'])
                        ldk.append(2 * np.pi * dico[date][nom_exp]['Ld'] /dico[date][nom_exp]['lambda'])
                        l_onde.append(dico[date][nom_exp]['lambda'])
  
    
#%% Graph
annotated_name = False
annotated_ld = False

# figurejolie()
for i in range (len(amplitude)) :
    if casse[i] :
        disp.joliplot(r"$\lambda$ (m)", r"Amplitude (m)", l_onde[i], amplitude[i], color = 8)
        if annotated_name :
            plt.annotate (exp[i],(l_onde[i], amplitude[i]))
        if annotated_ld :
            plt.annotate (round(l_d[i], 4),(l_onde[i], amplitude[i]))
    else :
        disp.joliplot(r"$\lambda$ (m)", r"Amplitude (m)", l_onde[i], amplitude[i], color = 7)
        if annotated_name :
            plt.annotate (exp[i],(l_onde[i], amplitude[i]))
        if annotated_ld :
            plt.annotate (round(l_d[i], 4),(l_onde[i], amplitude[i]))

# plt.xlim(0,0.45)
# plt.ylim(0,0.016)
# disp.joliplot(r"$\lambda$ (m)", r"Amplitude (m)", 0, 0, color = 7, legend = r'Intact')
# disp.joliplot(r"$\lambda$ (m)", r"Amplitude (m)", 0, 0, color = 18, legend = r'Fracturé')


# disp.joliplot(r"$\lambda$ (m)", r"Amplitude (m)", 0, 0, color = 13, legend = 'Casse stationnaire')
# disp.joliplot(r"$\lambda$ (m)", r"Amplitude (m)", 0, 0, color = 7, legend = 'Casse pas stationnaire')
#%% Stage
data = np.loadtxt("D:\Banquise\Baptiste\Resultats\\220628_diagramme_de_phase\\diagramme_de_phase_trié.txt")

omega = data[:,0]
lambdames = data[:,1]
lambdaest = data[:,2]
amp = data[:,3]
cassage = data[:,4]
hpese = data[:,5]
hbonbonne = data[:,6]

# disp.figurejolie()

lambdacomplet = np.zeros(len( lambdames))

data_traitées0 = np.zeros((len(lambdaest), 7)) # casse pas
data_traitées1 = np.zeros((len(lambdaest), 7)) # fissure
data_traitées2 = np.zeros((len(lambdaest), 7)) # casse

#data_traitées avec omega, lambda, Amp, h, Ld, Ld/lambda, pente

for i in range( len(omega)):
    if lambdaest[i] == -1 :
        lambdacomplet[i] = lambdames[i] / 1000
    if lambdames[i] == -1 :
        lambdacomplet [i] = lambdaest[i] / 1000
    # if lambdaest[i] ==-1 and lambdaest[i] == -1 :
    #     lambdacomplet[i] = (lambdames[i] + lambdaest[i])/2 / 1000
     
        
        
    if cassage[i] == 0 :
        
        
        if hbonbonne[i] !=-100 : #-1 si on veut mettre h dans les DDP
            if lambdacomplet[i] > 0.1 :
                data_traitées0[i,3] = hbonbonne[i] / 1000
                data_traitées0[i,0] = omega[i]
                data_traitées0[i,1] = lambdacomplet[i]
                data_traitées0[i,2] = amp[i] / 1000
                # data_traitées0[i,4] = ( (Eyoung * data_traitées0[i,3]**3) / (12 * (1 - nu**2) * rho * g) )**(1/4) # Ld
                data_traitées0[i,5] = data_traitées0[i,4] / data_traitées0[i,1] # Ld/lambda
                data_traitées0[i,6] = data_traitées0[i,2] / data_traitées0[i,1] # pente
        
        
        
    if cassage[i] == 1 or cassage [i] == 2 :
        
        
        if hbonbonne[i] !=-100 :
            data_traitées1[i,3] = hbonbonne[i] / 1000
            data_traitées1[i,0] = omega[i]
            data_traitées1[i,1] = lambdacomplet[i]
            data_traitées1[i,2] = amp[i] / 1000
            # data_traitées1[i,4] = ( (Eyoung * data_traitées1[i,3]**3) / (12 * (1 - nu**2) * rho * g) )**(1/4)
            data_traitées1[i,5] = data_traitées1[i,4] / data_traitées1[i,1] # Ld/lambda
            data_traitées1[i,6] = data_traitées1[i,2] / data_traitées1[i,1] # pente

        
        
        
        
    # if cassage [i] == 2 :
        
    #     if hbonbonne[i] !=-100 : 
    #         data_traitées2[i,3] = hbonbonne[i] / 1000
    #         data_traitées2[i,0] = omega[i]
    #         data_traitées2[i,1] = lambdacomplet[i]
    #         data_traitées2[i,2] = amp[i] / 1000
    #         data_traitées2[i,4] = ( (Eyoung * data_traitées2[i,3]**3) / (12 * (1 - nu**2) * rho * g) )**(1/4)
    #         data_traitées2[i,5] = data_traitées2[i,4] / data_traitées2[i,1] # Ld/lambda
    #         data_traitées2[i,6] = data_traitées2[i,2] / data_traitées2[i,1] # pente
            
        
       
#ajoute les points de MPBV6 et MPPF3-4
# lambda_cassepas = np.append(data_traitées0[:,1], 0.38)
# amp_cassepas = np.append(data_traitées0[:,2], 0.0103)

# lambda_casse = np.append(data_traitées2[:,1], 0.265)
# amp_casse = np.append(data_traitées2[:,2], 0.0068)

# lambda_cassepas = np.append(lambda_cassepas, 0.265)
# amp_cassepas = np.append(amp_cassepas, 0.0063)

 # 1.03 cm casse pas lambda 40 cm
 # 0.63 casse pas
 # 0.68 casse lambda 26.5 cm
# Tout est en m !



# figurejolie()
disp.joliplot(r'$\lambda$ (m)', r'Amplitude (m)', data_traitées0[:,1], data_traitées0[:,2], color = 7, title = False, legend = r'Unbroken', exp = True)
disp.joliplot(r'$\lambda$ (m)', r'Amplitude (m)', data_traitées1[:,1], data_traitées1[:,2], color = 8, title = False, legend = r'Fractured', exp = True)
# joliplot(r'$\lambda$ (m)', r'Amplitude (m)', data_traitées2[:,1], data_traitées2[:,2], color = 2, title = r'Recherche de seuil, $\lambda$(amp)', legend = r'Casse', exp = True)
#%% find scaling
x = np.linspace(0.0068,0.55,100)
y = 0.006/0.25 * x 
y_2 = x**(1.5) * 0.05
y_3 = x**2 * 0.08
y_4 = x ** 0.5 * 0.008
plt.plot(x,y, label = 'Modèle linéaire')

plt.plot(x,y_2, label = 'Modèle 3/2')
plt.plot(x,y_4, label = 'Modèle racine')



# plt.plot(x,y_3, label = 'Modèle 2')

# # plt.plot(x,y_4, label = 'Modèle x3/2')

plt.legend()
plt.xlabel(r'$\lambda$ (m)') 
plt.ylabel('Amplitude (m)')
plt.xscale('log')
plt.yscale('log')
plt.axis('equal')

