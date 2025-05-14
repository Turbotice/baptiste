# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 17:18:32 2024

@author: Banquise
"""

import pandas
import pickle 
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

import baptiste.display.display_lib as disp
import baptiste.experiments.import_params as ip
import baptiste.signal_processing.fft_tools as ft
import baptiste.image_processing.image_processing as imp
import baptiste.math.fits as fits
import baptiste.math.RDD as rdd
import baptiste.files.save as sv
import baptiste.files.dictionaries as dic
import baptiste.tools.tools as tools



dico = dic.open_dico()

#%% Importation params utiles
save = False
save_path = 'E:\\Baptiste\\Resultats_exp\\All_RDD\\Resultats\\'

tableau_1 = pandas.read_csv('E:\\Baptiste\\Resultats_exp\\Tableau_params\\Tableau1_Params_231117_240116\\tableau_1.txt', sep = '\t', header = 0)
tableau_1 = np.asarray(tableau_1)

g = 9.81
A = np.array(tableau_1[:,9], dtype = float)
long_onde = np.array(tableau_1[:,6], dtype = float)
k = 2 * np.pi / long_onde
H = np.array(tableau_1[:,10], dtype = float)
omega = np.sqrt(g * k + np.tanh(k * H))
x = np.zeros(len(omega), dtype = object)
t = np.zeros(len(omega), dtype = object)
for i in range (len(omega)) :
    x[i] = np.linspace(0, long_onde[i], 500)
    t[i] = np.pi / omega[i] * 2 * 3 / 4





#verification du domaine de validité :
    
    
# disp.figurejolie()
# disp.joliplot('long_onde', 'Ordre2 / Ordre 1', long_onde, 3/8*A * k / k**3 / H **3)


#%% test 1 poin
#%% Charge donées exp

date = '240116'
nom_exp = 'CCM03'
exp = True
exp_type = 'LAS'

save = False
display = True

dico, params, loc = ip.initialisation(date, nom_exp, exp = True, display = False)

folder_results = params['path_images'][:-15] + "resultats"
name_file = "positionLAS.npy"
data_originale = np.load(folder_results + "\\" + name_file)

if True : #cam_dessus :
    data_originale = np.rot90(data_originale)
    data_originale = np.flip(data_originale, 0)
   

    
# data_originale = data_originale - np.nanmean(data_originale, axis = 0)    


params['debut_las'] = 1
params['fin_las'] = np.shape(data_originale)[0] - 1720
#100 / 400 Pour DAP
#650 / 300 Pour CCCS1
#1 / 200 pr CCCS2

params['t0'] = 1
params['tf'] = np.shape(data_originale)[1] - 1


[nx,nt] = data_originale[params['debut_las']:params['fin_las'],params['t0']:params['tf']].shape


data = data_originale[params['debut_las']:params['fin_las'],params['t0']:params['tf']]


#mise à l'échelle en m
data_m = data *  params['mmparpixely'] / 1000

data_m = data_m / params['grossissement']

 

t = np.arange(0,nt)/params['facq']
x = np.arange(0,nx)*params['mmparpixelz'] / 1000

data = data_m.copy()
# data = data - np.nanmean(data, axis = 0)  #enleve la moyenne spatiale pr chaque tps
data = data - np.nanmean(data)


if display:
    disp.figurejolie()
    # plt.pcolormesh(t, x, data,shading='auto')
    plt.pcolormesh(data,shading='auto')
    plt.xlabel("Temps (frames)")
    plt.ylabel("X (pixel)")
    cbar = plt.colorbar()
    cbar.set_label('Amplitude (m)')
    plt.clim(-0.01,0.01)


#%% Comparaison exp : ESPACE


#trace forme du laser superposé à forme de la théorie. On prend les params de l'exp. EN ESPACE

#FORME EXP (x)

lambda_exp = float(dico[date][nom_exp]['lambda'])
k_exp = 2 * np.pi / lambda_exp
omega_exp = 2 * np.pi * float(dico[date][nom_exp]['fexc'])
H_exp = float(dico[date][nom_exp]['Hw'])
facq = float(dico[date][nom_exp]['facq'])

# x_l2 = 737
# x0 = x_l2 - lambda_exp/2 * params['mmparpixel'] / 1000
# xf = x_l2 + lambda_exp/2 * params['mmparpixel'] / 1000
# t0 = 86
# tf = t0 + 13


x_l2 = 737
x0 = 1    #x_l2 - int(lambda_exp/4 / params['mmparpixel'] * 1000)
xf = 230 # x_l2 + int(lambda_exp/4 / params['mmparpixel'] * 1000)
t0 = 71

#ATTENTION, ON MET MOINS (-) CAR LE HAUT EST EN BAS POUR LE LASER

forme = -data[x0:xf, t0]

x_plotexp = np.linspace(0, (xf-x0) * params['mmparpixel'] / 1000, (xf-x0))

#FORME THEORIQUE (x)

Amp_exp_eta3 = np.max(forme) * 0.5
Amp_exp_eta2 = np.max(forme) * 1.1
Amp_exp_eta1 = np.max(forme)


t = (71 - t0) / facq + 2*np.pi / omega_exp / 4

x = np.linspace(0, 0 + lambda_exp * (xf-x0) / (lambda_exp / params['mmparpixel'] * 1000), (xf-x0))



eta1 = Amp_exp_eta1 * np.sin(k_exp * x - omega_exp * t)

eta1_2 = Amp_exp_eta2 * np.sin(k_exp * x - omega_exp * t)
terme1_2 = Amp_exp_eta2**2 * k_exp / 4
terme2_2 = 3 * (1 / np.tanh(k_exp * H_exp))**3 - 1/ np.tanh(k_exp * H_exp)
eta2 = terme1_2 * terme2_2 * np.cos(2 * (k_exp * x - omega_exp * t)) #- A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i]) #+ A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i])


eta1_3 = Amp_exp_eta3 * np.sin(k_exp * x - omega_exp * t)
terme1_3 = Amp_exp_eta3**2 * k_exp / 4
terme2_3 = 3 * (1 / np.tanh(k_exp * H_exp))**3 - 1/ np.tanh(k_exp * H_exp)
eta2_3 = terme1_3 * terme2_3 * np.cos(2 * (k_exp * x - omega_exp * t)) #- A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i]) #+ A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i])
k_exp_prime = k_exp * (1 - Amp_exp_eta3 ** 2 * k_exp ** 2)
eta3 = (1 - 3* Amp_exp_eta3**2 * k_exp**2 / 8) * Amp_exp_eta3 * np.cos(k_exp_prime * x - omega_exp * t) + Amp_exp_eta3 **2 * k_exp * np.cos(2 * (k_exp_prime * x - omega_exp * t)) + 3 * Amp_exp_eta3**3 * k_exp ** 2 * np.cos(3 * (k_exp_prime * x - omega_exp * t)) / 8


disp.figurejolie()
disp.joliplot(r'x (0 à $\frac{\lambda}{2}$)', 'x (m)', x_plotexp, eta1, exp = False, color = 4, legend = '$\eta_1$')
disp.joliplot(r'x (0 à $\frac{\lambda}{2}$)', 'x (m)', x_plotexp, eta1_2 + eta2, exp = False, color = 2, legend = '$\eta_1 + \eta_2$')
# disp.joliplot('t (0 à T)', 'x (m)', t_exp, eta1_3 + eta2_3 + eta3 - np.mean(eta1_3 + eta2_3 + eta3), exp = False, color = 12, legend = '$\eta_1 + \eta_2 + \eta_3$')
disp.joliplot(r'x (0 à $\frac{\lambda}{2}$)', 'x (m)', x_plotexp, forme, exp = False, color = 14, legend = 'Expérience')

#%% POUR UNE VIDEO
lambda_exp = float(dico[date][nom_exp]['lambda'])
k_exp = 2 * np.pi / lambda_exp
omega_exp = 2 * np.pi * float(dico[date][nom_exp]['fexc'])
H_exp = float(dico[date][nom_exp]['Hw'])
facq = float(dico[date][nom_exp]['facq'])

boost_nonlineaire = 2
t00 = 250
tff = 350
x0 = 1    #x_l2 - int(lambda_exp/4 / params['mmparpixel'] * 1000)
xf = 500 # x_l2 + int(lambda_exp/4 / params['mmparpixel'] * 1000)

t00 = 470
tff = 570
x0 = 1
xf = 199


nb_periodes = 1
liste_t = np.linspace(t00,tff,tff-t00, dtype = int)

x = np.linspace(1 *lambda_exp / 16, 1*lambda_exp / 16 + nb_periodes * lambda_exp * (xf-x0) / (lambda_exp / params['mmparpixel'] * 1000), (xf-x0) * 2)

x_plotexp = np.linspace(0, (xf-x0) * params['mmparpixel'] / 1000, (xf-x0) * 2)

err = 0.1

a = 30
sig = 0.001
stddd = np.std(data[x0:xf,:], axis = 0)
meann = np.mean(data[x0:xf, stddd < sig])

hmin = np.zeros(np.shape(liste_t))
hmax = np.zeros(np.shape(liste_t))
popt_min = np.zeros(np.shape(liste_t), dtype = object)
popt_max = np.zeros(np.shape(liste_t), dtype = object)
err_min = np.zeros(np.shape(liste_t))
err_max = np.zeros(np.shape(liste_t))
kmin = np.zeros(np.shape(liste_t))
kmax = np.zeros(np.shape(liste_t))
imin = np.zeros(np.shape(liste_t), dtype = int)
imax = np.zeros(np.shape(liste_t), dtype = int)

imin0 = np.zeros(np.shape(liste_t), dtype = int)
imax0 = np.zeros(np.shape(liste_t), dtype = int)

formes_max = np.zeros(np.shape(liste_t), dtype = object)
formes_min = np.zeros(np.shape(liste_t), dtype = object)

xplotexp_max = np.zeros(np.shape(liste_t), dtype = object)
xplotexp_min = np.zeros(np.shape(liste_t), dtype = object)


forme_verif = np.zeros(np.shape(liste_t), dtype = object) 

for i in range(len( liste_t)) :
    t0 = liste_t[i]
    forme = data[x0:xf, t0] - meann
    # forme = np.append(np.flip(forme), forme)
    n = forme.shape[0]
    t = t0 / facq
    forme_verif[i] = forme
    
    
    
    '''         THEORIE
    
    Amp_exp_eta2 = np.max(data[x0:xf, liste_t]) * boost_nonlineaire
    
    k_exp_prime = k_exp * (1 - Amp_exp_eta2 ** 2 * k_exp ** 2)
    terme1 = Amp_exp_eta2**2 * k_exp / 4
    terme2 = 3 * (1 / np.tanh(k_exp * H_exp))**3 - 1/ np.tanh(k_exp * H_exp)
    terme_1 = Amp_exp_eta2**2 * -k_exp / 4
    terme_2 = 3 * (1 / np.tanh(-k_exp * H_exp))**3 - 1/ np.tanh(-k_exp * H_exp)
    eta1 = Amp_exp_eta2 * np.sin(k_exp * x - omega_exp * t)
    eta_1 = Amp_exp_eta2 * np.sin(-k_exp * x - omega_exp * t)
    eta2 = terme1 * terme2 * np.cos(2 * (k_exp * x - omega_exp * t)) #- A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i]) #+ A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i])
    eta_2 = terme_1 * terme_2 * np.cos(2 * (-k_exp * x - omega_exp * t))
    eta3 = (1 - 3* Amp_exp_eta2**2 * k_exp**2 / 8) * Amp_exp_eta2 * np.cos(k_exp_prime * x - omega_exp * t) + Amp_exp_eta2 **2 * k_exp * np.cos(2 * (k_exp_prime * x - omega_exp * t)) + 3 * Amp_exp_eta2**3 * k_exp ** 2 * np.cos(3 * (k_exp_prime * x - omega_exp * t)) / 8
    eta_3 = (1 - 3* Amp_exp_eta2**2 * -k_exp**2 / 8) * Amp_exp_eta2 * np.cos(-k_exp_prime * x - omega_exp * t) + Amp_exp_eta2 **2 * -k_exp * np.cos(2 * (-k_exp_prime * x - omega_exp * t)) + 3 * Amp_exp_eta2**3 * -k_exp ** 2 * np.cos(3 * (-k_exp_prime * x - omega_exp * t)) / 8

    eta1_2_3 = Amp_exp_eta2 * np.sin(k_exp * x - omega_exp * t)
    terme1_2 = Amp_exp_eta2**2 * k_exp / 4
    terme2_2 = 3 * (1 / np.tanh(k_exp * H_exp))**3 - 1/ np.tanh(k_exp * H_exp)
    eta2 = terme1_2 * terme2_2 * np.cos(2 * (k_exp * x - omega_exp * t)) #- A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i]) #+ A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i])

    t = 6*np.pi / omega_exp / 4 - 0.03
    eta1_2 = Amp_exp_eta2 * np.sin(k_exp * x - omega_exp * t)
    terme1_2 = Amp_exp_eta2**2 * k_exp / 4
    terme2_2 = 3 * (1 / np.tanh(k_exp * H_exp))**3 - 1/ np.tanh(k_exp * H_exp)
    eta2_1 = terme1_2 * terme2_2 * np.cos(2 * (k_exp * x - omega_exp * t)) #- A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i]) #+ A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i])
    '''


    imax0[i] = np.argmax(forme[0 : -a]) + 0
    
    formes_max[i] = np.append(np.flip(forme)[:-imax0[i]], forme[imax0[i]:])
    imax[i] = np.argmax(formes_max[i][a : -a]) + a
    xplotexp_max[i] = np.linspace(0, len(formes_max[i]) * params['mmparpixel'] / 1000, len(formes_max[i]))
    if False :#np.nanmax(formes_max[i]) > np.nanmax(forme) :
        kmax[i] = None
        err_max[i] = None
        hmax[i] = None
    else :
        hmax[i] = formes_max[i][imax[i]]
        
       
        
        yfit = formes_max[i][imax[i]-a:imax[i]+a]
        xfit = xplotexp_max[i][imax[i]-a:imax[i]+a]
        popt_max[i] = np.polyfit(xfit,yfit, 2, full = True)
        yth = np.polyval(popt_max[i][0], xfit)
        err_max[i] = np.sqrt(popt_max[i][1][0])/ np.abs(np.max(hmax[i]))
        if err_max[i] > err :
            kmax[i] = None
            err_max[i] = None
            hmax[i] = None
        else :
            kmax[i] = popt_max[i][0][0]*2
        

    


    imin0[i] = np.argmin(forme[0 : -a]) + 0
    formes_min[i] = np.append(np.flip(forme)[:-imin0[i]], forme[imin0[i]:])
    imin[i] = np.argmin(formes_min[i][a : -a]) + a
    xplotexp_min[i] = np.linspace(0, len(formes_min[i]) * params['mmparpixel'] / 1000, len(formes_min[i]))
    if False :#np.nanmin(formes_min[i]) < np.nanmin(forme) :
        kmin[i] = None
        err_min[i] = None
        hmin[i] = None
    else :
        hmin[i] = formes_min[i][imin[i]]
        
        
        
        yfit = formes_min[i][imin[i]-a:imin[i]+a]
        xfit = xplotexp_min[i][imin[i]-a:imin[i]+a]
        popt_min[i] = np.polyfit(xfit,yfit, 2, full = True)
        yth = np.polyval(popt_min[i][0], xfit)
        err_min[i] = np.sqrt(popt_min[i][1][0])/ np.abs(np.max(hmin[i]))
        if err_min[i] > err :
            kmin[i] = None
            err_min[i] = None
            hmin[i] = None
        else :
            kmin[i] = popt_min[i][0][0]*2
            
            
            
        
        
    
    
    

    
    # disp.joliplot(r'x (0 à $\frac{\lambda}{2}$)', 'x (m)', x_plotexp, eta1, exp = False, color = 4)#, legend = '$\eta_1$')
    # disp.joliplot(r'x (0 à $\frac{\lambda}{2}$)', 'x (m)', x_plotexp, (eta1_2_3 + eta2)/boost_nonlineaire, exp = False, color = 2)#, legend = '$\eta_1 + \eta_2$')
    # disp.joliplot(r'x (0 à $\frac{\lambda}{2}$)', 'x (m)', x_plotexp, (eta1_2 + eta2_1)/boost_nonlineaire, exp = False, color = 4)
    # disp.joliplot(r'x (0 à $\frac{\lambda}{2}$)', 'x (m)', x_plotexp, (eta1 + eta_1 + eta2 + eta_2) / boost_nonlineaire, exp = False, color = 4)
    # disp.joliplot(r'x (0 à $\frac{\lambda}{2}$)', 'x (m)', x_plotexp, (eta1 + eta2) / boost_nonlineaire, exp = False, color = 2)
    # disp.joliplot(r'x (0 à $\frac{\lambda}{2}$)', 'x (m)', x_plotexp, (eta_1 + eta_2) / boost_nonlineaire, exp = False, color = 8)
    # disp.joliplot(r'x (0 à $\frac{\lambda}{2}$)', 'x (m)', x_plotexp, -(eta1 + eta_1 + eta2 + eta_2 + eta3 + eta_3) / boost_nonlineaire/ 4, exp = False, color = 8)
    # disp.joliplot(r'x (0 à $\frac{\lambda}{2}$)', 'x (m)', x_plotexp, forme, exp = False, color = 14)#, legend = 'Expérience')
    # plt.grid('off')
    # plt.title(str(t0))
    # plt.pause(0.3)
    
#%% Test pour a

date = '240109'
nom_exp = 'QSC17'
exp = True
exp_type = 'LAS'

save = False
display = True

dico, params, loc = ip.initialisation(date, nom_exp, exp = True, display = False)

folder_results = params['path_images'][:-15] + "resultats"
name_file = "positionLAS.npy"
data_originale = np.load(folder_results + "\\" + name_file)

if True : #cam_dessus :
    data_originale = np.rot90(data_originale)
    data_originale = np.flip(data_originale, 0)
   

    
data_originale = data_originale - np.nanmean(data_originale, axis = 0)    


params['debut_las'] = 150
params['fin_las'] = np.shape(data_originale)[0] - 250
#100 / 400 Pour DAP
#650 / 300 Pour CCCS1
#1 / 200 pr CCCS2

params['t0'] = 111
params['tf'] = np.shape(data_originale)[1] - 1800


[nx,nt] = data_originale[params['debut_las']:params['fin_las'],params['t0']:params['tf']].shape


data = data_originale[params['debut_las']:params['fin_las'],params['t0']:params['tf']]

params['ordre_savgol'] = 2
params['taille_savgol'] = 11
params['size_medfilt'] = 51


#mise à l'échelle en m
data_m = data *  params['mmparpixely'] / 1000

data_m = data_m / params['grossissement']


t = np.arange(0,nt)/params['facq']
x = np.arange(0,nx)*params['mmparpixelz'] / 1000

data = data_m.copy()


lambda_exp = float(dico[date][nom_exp]['lambda'])
k_exp = 2 * np.pi / lambda_exp
omega_exp = 2 * np.pi * float(dico[date][nom_exp]['fexc'])
H_exp = float(dico[date][nom_exp]['Hw'])
facq = float(dico[date][nom_exp]['facq'])

t00 = 1
tff = 150
x0 = 20    #x_l2 - int(lambda_exp/4 / params['mmparpixel'] * 1000)
xf = 280 # x_l2 + int(lambda_exp/4 / params['mmparpixel'] * 1000)

t00 = 65
tff = 90
x0 = 20
xf = 1350



nb_periodes = 1
liste_t = np.linspace(t00,tff,tff-t00, dtype = int)
x = np.linspace(1 *lambda_exp / 16, 1*lambda_exp / 16 + nb_periodes * lambda_exp * (xf-x0) / (lambda_exp / params['mmparpixel'] * 1000), (xf-x0))
x_plotexp = np.linspace(0, (xf-x0) * params['mmparpixel'] / 1000, (xf-x0))

err = 0.2


sig = 0.001
stddd = np.std(data[x0:xf,:], axis = 0)
meann = np.mean(data[x0:xf, stddd < sig])




liste_a = np.linspace(5,95,10, dtype = int)
somme_err = np.zeros(len(liste_a))
kappa_max = np.zeros(len(liste_a))
kappa_min = np.zeros(len(liste_a))

for j in range(len(liste_a)) :
    a = liste_a[j]
    hmin = np.zeros(np.shape(liste_t))
    hmax = np.zeros(np.shape(liste_t))
    popt_min = np.zeros(np.shape(liste_t), dtype = object)
    popt_max = np.zeros(np.shape(liste_t), dtype = object)
    err_min = np.zeros(np.shape(liste_t))
    err_max = np.zeros(np.shape(liste_t))
    kmin = np.zeros(np.shape(liste_t))
    kmax = np.zeros(np.shape(liste_t))
    imin = np.zeros(np.shape(liste_t), dtype = int)
    imax = np.zeros(np.shape(liste_t), dtype = int)
    for i in range(len( liste_t)) :
        t0 = liste_t[i]
        forme = -data[x0:xf, t0] + meann
        n = forme.shape[0]
        t = t0 / facq
        
        #QCS
        imin[i] = np.argmin(forme[int(1.5*n/6):int(7*n/8)]) + int(1.5*n/6)
        imax[i] = np.argmax(forme[int(1.5*n/6):int(7*n/8)]) + int(1.5*n/6)
        #CCM
        # imin[i] = np.argmin(forme[a:int(1*n/2)]) + a
        # imax[i] = np.argmax(forme[a:int(1*n/2)]) + a
        # #NPDP
        # imin[i] = np.argmin(forme[a + int(0 * n / 5):int(8 * n / 9)]) + a + int(0 * n / 5)
        # imax[i] = np.argmax(forme[a + int(0 * n / 5):int(8 * n / 9)]) + a + int(0 * n / 5)
        
        hmin[i] = forme[imin[i]]
        hmax[i] = forme[imax[i]]
        
        yfit = forme[imin[i]-a:imin[i]+a]
        xfit = x_plotexp[imin[i]-a:imin[i]+a]
        popt_min[i] = np.polyfit(xfit,yfit, 2, full = True)
        yth = np.polyval(popt_min[i][0], xfit)
        err_min[i] = np.sqrt(popt_min[i][1][0])/ np.abs(np.max(hmin[i]))
        if err_min[i] > err :
            kmin[i] = None
            err_min[i] = None
            hmin[i] = None
        else :
            kmin[i] = popt_min[i][0][0]*2
        
        yfit = forme[imax[i]-a:imax[i]+a]
        xfit = x_plotexp[imax[i]-a:imax[i]+a]
        popt_max[i] = np.polyfit(xfit,yfit, 2, full = True)
        yth = np.polyval(popt_max[i][0], xfit)
        err_max[i] = np.sqrt(popt_max[i][1][0]) / np.abs(np.max(hmax[i]))
        if err_max[i] > err :
            kmax[i] = None
            err_max[i] = None
            hmax[i] = None
        else :
            kmax[i] = popt_max[i][0][0]*2
        
    disp.figurejolie()
    plt.plot(x_plotexp[imin], hmin, 'rv')
    plt.plot(x_plotexp[imax], hmax, 'r^')
    for i in range(len(liste_t)) :
        if not np.isnan(hmin[i]) :
            xfit = x_plotexp[imin[i]-a:imin[i]+a]
            yth = np.polyval(popt_min[i][0], xfit)
            plt.plot(xfit, yth, 'r-') 
        if not np.isnan(hmax[i]) :
            xfit = x_plotexp[imax[i]-a:imax[i]+a]
            yth = np.polyval(popt_max[i][0], xfit)
            plt.plot(xfit, yth, 'r-')
    disp.joliplot(r'x (0 à $\frac{\lambda}{2}$)', 'x (m)', x_plotexp, -data[x0:xf, liste_t] + meann, exp = False, color = 14)
    if save :
        plt.savefig('E:\\Baptiste\\Resultats_exp\\Courbure\\' + params['date']  + '_' + params['nom_exp'] + '_' + 'exp+fitcourbure_' + tools.datetimenow() + '.pdf')


    disp.figurejolie()
    disp.joliplot(r'$\frac{\eta - \eta_{min}}{\eta_{max}}$' ,r'$\kappa$ (m$^{-1}$)', (-hmin - np.nanmin(np.abs(-hmin))) / np.nanmax(-hmin), kmin, exp = True, color = 8, legend = 'Curvature at minimum')
    plt.errorbar((-hmin - np.nanmin(np.abs(-hmin))) / np.nanmax(-hmin), kmin, yerr = err_min, fmt = 'none',capsize = 5, ecolor = 'black',elinewidth = 3)
    disp.joliplot(r'$\frac{\eta - \eta_{min}}{\eta_{max}}$' ,r'$\kappa$ (m$^{-1}$)', (hmax - np.nanmin(np.abs(hmax))) / np.nanmax(hmax), -kmax, exp = True, color = 2, legend = 'Curvature at maximum')
    plt.errorbar((hmax - np.nanmin(np.abs(hmax))) / np.nanmax(hmax), -kmax, yerr = err_max, fmt = 'none',capsize = 5, ecolor = 'red',elinewidth = 3)
    disp.joliplot(r'$\frac{\eta - \eta_{min}}{\eta_{max}}$' ,r'$\kappa$ (m$^{-1}$)', 0.02, np.nanmax(np.abs(hmin)) * k_exp**2, color = 10)
    plt.title(str(a))
    if save :
        plt.savefig('E:\\Baptiste\\Resultats_exp\\Courbure\\' + params['date']  + '_' + params['nom_exp'] + '_' + 'exp+fitcourbure_' + tools.datetimenow() + '.pdf')
        
    somme_err[j] = (np.nansum(err_max) + np.nansum(err_min))/a
    kappa_max[j] = np.nanstd(kmax) #np.nanmax(np.abs(kmax))
    kappa_min[j] = np.nanstd(kmin) #np.nanmax(np.abs(kmin)) 
    # kappa_moins[j] = np.nansum(k)
    
disp.figurejolie()
disp.joliplot('a', r'$\sigma_{err}$', liste_a, somme_err, exp = False, color = 2)
disp.figurejolie()
disp.joliplot('a', r'$std(\kappa_{max})$', liste_a, kappa_max, exp = True, color = 2)
disp.figurejolie()
disp.joliplot('a', r'$std(\kappa_{min})$', liste_a, kappa_min, exp = True, color = 2)




#%% PLOT courbure
save = True

u = 0
for i in xplotexp_max :
    if u <= len(i):
        u = len(i)    
plotmax = np.linspace(0, u * params['mmparpixel'] / 1000, u)

u = 0
for i in xplotexp_min :
    if u <= len(i):
        u = len(i)    
plotmin = np.linspace(0, u * params['mmparpixel'] / 1000, u)




disp.figurejolie()

# plt.plot(x_plotexp[imin], hmin, 'rv')
# plt.plot(x_plotexp[imax], hmax, 'r^')

# for i in range(len(liste_t)) :
#     if not np.isnan(hmin[i]) :
#         xfit = x_plotexp[imin[i]-a:imin[i]+a]
#         yth = np.polyval(popt_min[i][0], xfit)
#         plt.plot(xfit, yth, 'r-') 
#     if not np.isnan(hmax[i]) :
#         xfit = x_plotexp[imax[i]-a:imax[i]+a]
#         yth = np.polyval(popt_max[i][0], xfit)
#         plt.plot(xfit, yth, 'r-') 

for i in range (len(formes_min)) :
    disp.joliplot(r'x (0 à $\frac{\lambda}{2}$)', 'x (m)', xplotexp_min[i], formes_min[i], exp = False, color = 14)
    plt.plot(xplotexp_min[i][imin[i]], hmin[i], 'rv')
    if not np.isnan(hmin[i]) :
            xfit = plotmin[imin[i]-a:imin[i]+a]
            yth = np.polyval(popt_min[i][0], xfit)
            plt.plot(xfit, yth, 'r-') 
            
if save :
    plt.savefig('E:\\Baptiste\\Resultats_exp\\Courbure\\' + params['date']  + '_' + params['nom_exp'] + '_' + 'exp+fitcourbure_MIN_' + tools.datetimenow() + '.pdf')


disp.figurejolie()

for i in range (len(formes_max)) :
    disp.joliplot(r'x (0 à $\frac{\lambda}{2}$)', 'x (m)', xplotexp_max[i], formes_max[i], exp = False, color = 14)
    plt.plot(xplotexp_max[i][imax[i]], hmax[i], 'r^')
    if not np.isnan(hmax[i]) :
            xfit = plotmax[imax[i]-a:imax[i]+a]
            yth = np.polyval(popt_max[i][0], xfit)
            plt.plot(xfit, yth, 'r-') 
    
if save :
    plt.savefig('E:\\Baptiste\\Resultats_exp\\Courbure\\' + params['date']  + '_' + params['nom_exp'] + '_' + 'exp+fitcourbure_MAX_' + tools.datetimenow() + '.pdf')


disp.figurejolie()


disp.joliplot(r'$\frac{\eta - \eta_{min}}{\eta_{max} - \eta_{min}}$' ,r'$\kappa$ (m$^{-1}$)', (-hmin - np.nanmin(-hmin)) / (np.nanmax(-hmin)- np.nanmin(-hmin)), kmin, exp = True, color = 8, legend = 'Curvature at minimum')
plt.errorbar( (-hmin - np.nanmin(-hmin)) / (np.nanmax(-hmin)- np.nanmin(-hmin)), kmin, yerr = err_min, fmt = 'none',capsize = 5, ecolor = 'black',elinewidth = 3)


disp.joliplot(r'$\frac{\eta - \eta_{min}}{\eta_{max} - \eta_{min}}$' ,r'$\kappa$ (m$^{-1}$)', (hmax - np.nanmin(hmax)) / (np.nanmax(hmax)- np.nanmin(hmax)), -kmax, exp = True, color = 2, legend = 'Curvature at maximum')
plt.errorbar((hmax - np.nanmin(hmax)) / (np.nanmax(hmax)- np.nanmin(hmax)), -kmax, yerr = err_max, fmt = 'none',capsize = 5, ecolor = 'red',elinewidth = 3)

h_plot = np.linspace(0,1,100)
amp_exp = float(dico[date][nom_exp]['Amp_max'])
kappa_th = np.zeros(100) + amp_exp * k_exp**2

disp.joliplot(r'$\frac{\eta - \eta_{min}}{\eta_{max} - \eta_{min}}$' ,r'$\kappa$ (m$^{-1}$)', h_plot, kappa_th, color = 14, legend = r'$\kappa_{th}$', exp = False)

if save :
    plt.savefig('E:\\Baptiste\\Resultats_exp\\Courbure\\' + params['date']  + '_' + params['nom_exp'] + '_' + 'Courburemax_courburemin_courburetherique_0a1_' + tools.datetimenow() + '.pdf')



# disp.figurejolie()

# for i in range (len(formes_max)) :
#     disp.joliplot(r'x (0 à $\frac{\lambda}{2}$)', 'x (m)', xplotexp_max[i], formes_max[i], exp = False, color = 14)
#     plt.plot(xplotexp_max[i][imax[i]], hmax[i], 'r^')
#     nn = len(forme_verif[i])
    
#     plot_verif0 = np.linspace( 0 * params['mmparpixel'] / 1000 , nn* params['mmparpixel'] / 1000, len(forme_verif[i]))
#     plot_verif1 = np.linspace( (nn - 2 * imax0[i]) * params['mmparpixel'] / 1000,  ( 2 * nn - 2 *imax0[i])* params['mmparpixel'] / 1000, len(forme_verif[i]))
    
#     disp.joliplot(r'x (0 à $\frac{\lambda}{2}$)', 'x (m)', plot_verif0, np.flip(forme_verif[i]), exp = False, color = 2)
#     disp.joliplot(r'x (0 à $\frac{\lambda}{2}$)', 'x (m)', plot_verif1, forme_verif[i], exp = False, color = 2)
    
#     if not np.isnan(hmax[i]) :
#             xfit = plotmax[imax[i]-a:imax[i]+a]
#             yth = np.polyval(popt_max[i][0], xfit)
#             plt.plot(xfit, yth, 'r-') 
            
# disp.figurejolie()

# for i in range (len(formes_min)) :
#     disp.joliplot(r'x (0 à $\frac{\lambda}{2}$)', 'x (m)', xplotexp_min[i], formes_min[i], exp = False, color = 14)
#     plt.plot(xplotexp_min[i][imin[i]], hmin[i], 'rv')
#     nn = len(forme_verif[i])
    
#     plot_verif0 = np.linspace( 0 * params['mmparpixel'] / 1000 , nn* params['mmparpixel'] / 1000, len(forme_verif[i]))
#     plot_verif1 = np.linspace( (nn - 2 * imin0[i]) * params['mmparpixel'] / 1000,  ( 2 * nn - 2 *imin0[i])* params['mmparpixel'] / 1000, len(forme_verif[i]))
    
#     disp.joliplot(r'x (0 à $\frac{\lambda}{2}$)', 'x (m)', plot_verif0, np.flip(forme_verif[i]), exp = False, color = 2)
#     disp.joliplot(r'x (0 à $\frac{\lambda}{2}$)', 'x (m)', plot_verif1, forme_verif[i], exp = False, color = 2)
    
#     if not np.isnan(hmin[i]) :
#             xfit = plotmin[imin[i]-a:imin[i]+a]
#             yth = np.polyval(popt_min[i][0], xfit)
#             plt.plot(xfit, yth, 'r-') 



#%% Save courbure.

if save :
    params['courbure'] = {}
    params['courbure']['hmin'] = hmin
    params['courbure']['hmax'] = hmax
    params['courbure']['imin'] = imin
    params['courbure']['imax'] = imax
    params['courbure']['popt_min'] = popt_min
    params['courbure']['popt_max'] = popt_max
    params['courbure']['kmin'] = kmin
    params['courbure']['kmax'] = kmax
    params['courbure']['err_min'] = err_min
    params['courbure']['err_max'] = err_max
    params['courbure']['x_plotexp'] = x_plotexp
    params['courbure']['x0'] = x0
    params['courbure']['xf'] = xf
    params['courbure']['t00'] = t00
    params['courbure']['tff'] = tff
    params['courbure']['a'] = a
    params['courbure']['sig'] = sig
    params['courbure']['err'] = err

    dic.save_dico(params, 'E:\\Baptiste\\Resultats_exp\\Courbure\\params_courbure_' + tools.datetimenow() + '.pkl')


#%% Courbure sur tout une exp 

save = False
display = True
date = '240116'
exp = True
exp_type = 'LAS'

nb_exps = 6

err = 0.4
sig = 0.002 #QSC
sig = 0.00075 #NPDP
sig = 0.016 #RLPY

#CCM
t00 =254
tff = 255
x0 = 2
xf = 80


aa = 0.005135 #Lkappa

k_maxmax = np.zeros(nb_exps)
k_minmin = np.zeros(nb_exps)
h_maxmax = np.zeros(nb_exps)
h_mimmin = np.zeros(nb_exps)
amplitude_exp = np.zeros(nb_exps)
k_exp_th = np.zeros(nb_exps)

for j in range (3,nb_exps) :
    
    '''CHARGE LES DATA'''
    
    if j < 10 :
        nom_exp = 'CCM0' + str(j)
    else : 
        nom_exp = 'CCM' + str(j)
    
    print(nom_exp)        
    
    # if j == 5:
    #     # t00 = 100
    #     # tff = 200
    #     # x0 = 1
    #     err = 0.06
    # elif j == 1 :
    #     # t00 = 60
    #     # tff = 150
    #     # x0 = 1
    #     err = 0.08
    # elif j == 2 :
    #     err = 0.07
    # else :
    #     # t00 = 20
    #     # tff = 120
    #     # x0 = 1
    #     # xf = 300

    path_save = 'E:\\Baptiste\\Resultats_exp\\Courbure\\'+ '240116_CCM_Lkappa\\' + date + '_' + nom_exp + '\\'
    
    dico, params, loc = ip.initialisation(date, nom_exp, exp = True, display = False)
    if save :
        os.mkdir(path_save)
    folder_results = params['path_images'][:-15] + "resultats"
    name_file = "positionLAS.npy"
    data_originale = np.load(folder_results + "\\" + name_file)
    data_originale = np.rot90(data_originale)
    data_originale = np.flip(data_originale, 0)
    # enleve moyenne moyenne en espace pour chaque temps (pas ouf)
    # data_originale = data_originale - np.nanmean(data_originale, axis = 0)    

    params['debut_las'] = 1
    params['fin_las'] = np.shape(data_originale)[0] - 700

    params['t0'] = 1
    params['tf'] = np.shape(data_originale)[1] - 1
    
    lambda_exp = float(dico[date][nom_exp]['lambda'])
    k_exp = 2 * np.pi / lambda_exp
    omega_exp = 2 * np.pi * float(dico[date][nom_exp]['fexc'])
    H_exp = float(dico[date][nom_exp]['Hw'])
    facq = float(dico[date][nom_exp]['facq'])
    amp_exp = float(dico[date][nom_exp]['Amp_max'])
    
    
    
    [nx,nt] = data_originale[params['debut_las']:params['fin_las'],params['t0']:params['tf']].shape

    data = data_originale[params['debut_las']:params['fin_las'],params['t0']:params['tf']]
    
    params['im_ref'] = False

    #enlever moyenne pr chaque pixel

    if params['im_ref'] :
        mean_pixel = np.nanmean(data,axis = 1) #liste avec moyenne en temps de chaque pixel 
        for i in range (0,nt):
            data[:,i] = data[:,i] - mean_pixel #pour chaque pixel, on enleve la moyenne temporelle de chaque pixel

    #mise à l'échelle en m
    data_m = data *  params['mmparpixely'] / 1000
    data_m = data_m / params['grossissement']
    a = int(aa / params['mmparpixely'] * 1000/2)
    
    #Filtre savgol
    params['savgol'] = False
    params['ordre_savgol'] = 2
    params['taille_savgol'] = int(a) * 2 + 1
    signalsv = np.zeros(data.shape)
    # for i in range(0,nt):  
    #     signalsv[:,i] = savgol_filter(data_m[:,i], params['taille_savgol'],params['ordre_savgol'], mode = 'nearest')
    #     if np.mod(i,1000)==0:
    #         print('On processe l image numero: ' + str(i) + ' sur ' + str(nt))
    # print('Done !')


    if params['savgol'] :
        data = signalsv.copy()
    else :
        data = data_m.copy()
        
        
    data = data - np.nanmean(data)
    
    t = np.arange(0,nt)/params['facq']
    x = np.arange(0,nx)*params['mmparpixelz'] / 1000
    
    
    
    '''INITIE PARAMS'''
    
    

    liste_t = np.linspace(t00,tff,tff-t00, dtype = int)
    
    x = np.linspace(-(xf-x0), (xf-x0), (xf-x0)*2)
    
    x_plotexp = np.linspace(-(xf-x0) * params['mmparpixel'] / 1000, (xf-x0) * params['mmparpixel'] / 1000, (xf-x0)*2)

    stddd = np.std(data[x0:xf,:], axis = 0)
    meann = np.mean(data[x0:xf, stddd < sig])

    hmin = np.zeros(np.shape(liste_t))
    hmax = np.zeros(np.shape(liste_t))
    popt_min = np.zeros(np.shape(liste_t), dtype = object)
    popt_max = np.zeros(np.shape(liste_t), dtype = object)
    err_min = np.zeros(np.shape(liste_t))
    err_max = np.zeros(np.shape(liste_t))
    kmin = np.zeros(np.shape(liste_t))
    kmax = np.zeros(np.shape(liste_t))
    imin = np.zeros(np.shape(liste_t), dtype = int)
    imax = np.zeros(np.shape(liste_t), dtype = int)
   

    for i in range(len( liste_t)) :
        t0 = liste_t[i]
        forme = -data[x0:xf, t0] + meann
        
        forme = np.append(np.flip(forme), forme)
        
        n = forme.shape[0]
        t = t0 / facq
   
        imin[i] = np.argmin(forme[a :-a]) + a
        imax[i] = np.argmax(forme[a :-a]) + a
        
        hmin[i] = forme[imin[i]]
        hmax[i] = forme[imax[i]]
        
        yfit = forme[imin[i]-a:imin[i]+a]
        xfit = x_plotexp[imin[i]-a:imin[i]+a]
        popt_min[i] = np.polyfit(xfit,yfit, 2, full = True)
        yth = np.polyval(popt_min[i][0], xfit)
        err_min[i] = np.sqrt(popt_min[i][1][0])/ np.abs(np.max(hmin[i]))
        if err_min[i] > err :
            kmin[i] = None
            err_min[i] = None
            hmin[i] = None
        else :
            kmin[i] = popt_min[i][0][0]*2
        
        yfit = forme[imax[i]-a:imax[i]+a]
        xfit = x_plotexp[imax[i]-a:imax[i]+a]
        
        popt_max[i] = np.polyfit(xfit,yfit, 2, full = True)
        popt_max[i]
        
        yth = np.polyval(popt_max[i][0], xfit)
        err_max[i] = np.sqrt(popt_max[i][1][0]) / np.abs(np.max(hmax[i]))
        if err_max[i] > err :
            kmax[i] = None
            err_max[i] = None
            hmax[i] = None
        else :
            kmax[i] = popt_max[i][0][0]*2
            
        
    
    if display :
        disp.figurejolie()    
        plt.plot(x_plotexp[imin], hmin, 'rv')
        plt.plot(x_plotexp[imax], hmax, 'r^')    
        for i in range(len(liste_t)) :
            t0 = liste_t[i]
            forme = -data[x0:xf, t0] + meann
            forme = np.append(np.flip(forme), forme)
            colors = disp.vcolors( int(i / len(liste_t) * 9) )
            plt.plot(x_plotexp,forme,color=colors)
                
            if not np.isnan(hmin[i]) :
                xfit = x_plotexp[imin[i]-a:imin[i]+a]
                yth = np.polyval(popt_min[i][0], xfit)
                plt.plot(xfit, yth, 'r-') 
            if not np.isnan(hmax[i]) :
                xfit = x_plotexp[imax[i]-a:imax[i]+a]
                yth = np.polyval(popt_max[i][0], xfit)
                plt.plot(xfit, yth, 'r-') 
            
        
        if save :
            plt.savefig(path_save + 'exp+fitcourbure_' + tools.datetimenow() + '.pdf')
    
        disp.figurejolie()    
        disp.joliplot(r'$\frac{\eta - \eta_{min}}{\eta_{max}}$' ,r'$\kappa$ (m$^{-1}$)', (-hmin - np.nanmin(-hmin)) / (np.nanmax(-hmin)- np.nanmin(-hmin)), kmin, exp = True, color = 8, legend = 'Curvature at minimum')
        plt.errorbar((-hmin - np.nanmin(-hmin)) / (np.nanmax(-hmin)- np.nanmin(-hmin)), kmin, yerr = err_min, fmt = 'none',capsize = 5, ecolor = 'black',elinewidth = 3)
        disp.joliplot(r'$\frac{\eta - \eta_{min}}{\eta_{max}}$' ,r'$\kappa$ (m$^{-1}$)', (hmax - np.nanmin(hmax)) / (np.nanmax(hmax)- np.nanmin(hmax)), -kmax, exp = True, color = 2, legend = 'Curvature at maximum')
        plt.errorbar((hmax - np.nanmin(hmax)) / (np.nanmax(hmax)- np.nanmin(hmax)), -kmax, yerr = err_max, fmt = 'none',capsize = 5, ecolor = 'red',elinewidth = 3)
        
        h_plot = np.linspace(0,1,100)
        kappa_th = np.zeros(100) + amp_exp * k_exp**2
        disp.joliplot(r'$\frac{\eta - \eta_{min}}{\eta_{max} - \eta_{min}}$' ,r'$\kappa$ (m$^{-1}$)', h_plot, kappa_th, color = 14, legend = r'$\kappa_{th}$', exp = False)
        
        if save :
            plt.savefig(path_save + 'Courburemax_courburemin_courburetherique_0a1' + tools.datetimenow() + '.pdf')


    k_maxmax[j] = np.nanmax(np.abs(kmax))
    k_minmin[j] = np.nanmax(np.abs(kmin))
    h_maxmax[j] = np.nanmax(np.abs(hmax))
    h_mimmin[j] = np.nanmax(np.abs(hmin))
    amplitude_exp[j] = amp_exp
    k_exp_th[j] = k_exp

courbure_th = amplitude_exp * k_exp_th**2
disp.figurejolie()
disp.joliplot('A (m)', r'$\kappa$ (m$^{-1}$)', amplitude_exp, k_maxmax, legend = 'Courbure au max', color = 2)
disp.joliplot('A (m)', r'$\kappa$ (m$^{-1}$)', amplitude_exp, k_minmin, legend = 'Courbure au min', color = 8)
disp.joliplot('A (m)', r'$\kappa$ (m$^{-1}$)', amplitude_exp, amplitude_exp * k_exp_th**2, legend = 'Courbure théorique', color = 14)
if save :
    plt.savefig(path_save[:-13] + date + '_' + nom_exp[:3] + '_couburemax_courburemin_courburetheorique_A_' + tools.datetimenow() + '.pdf')
#%% Save params

save = True
k_maxmax = [44.27773927225495, 55.40932769409098, 70.57614854066883, 74.7173706837188]
h_maxmax = [0.00264763, 0.00306854, 0.00349946, 0.0035007931898662255]
amplitude_exp = [0.00264763, 0.00306854, 0.00349946, 0.0035007931898662255]


if save :
    params['courbure'] = {}
    params['courbure']['k_maxmax'] = k_maxmax
    params['courbure']['k_minmin'] = k_minmin
    params['courbure']['h_maxmax'] = h_maxmax
    params['courbure']['h_mimmin'] = h_mimmin
    params['courbure']['k_exp_th'] = k_exp_th
    params['courbure']['courbure_th'] = courbure_th
    params['courbure']['amplitude_exp'] = amplitude_exp
    params['courbure']['x_plotexp'] = x_plotexp
    params['courbure']['x0'] = x0
    params['courbure']['xf'] = xf
    params['courbure']['t00'] = t00
    params['courbure']['tff'] = tff
    params['courbure']['a'] = a
    params['courbure']['sig'] = sig
    params['courbure']['err'] = err

    dic.save_dico(params, 'E:\\Baptiste\\Resultats_exp\\Courbure\\231124_RLPY\\' + tools.datetimenow() + '_params_courbure' + '.pkl')



#%% Comparaison exp : TEMPS


#trace forme du laser superposé à forme de la théorie. On prend les params de l'exp. EN TEMPS


#FORME EXP (t)

x_l2 = 737
x0 = x_l2 - lambda_exp/2 * params['mmparpixel'] / 1000
xf = x_l2 + lambda_exp/2 * params['mmparpixel'] / 1000
t0 = 86
tf = t0 + 13


forme = -data[x_l2, t0:tf]

t_plotexp = np.linspace(0, (tf-t0) / facq,tf-t0)

#FORME THEORIQUE (t)
Amp_exp_eta3 = np.max(forme) * 0.5
Amp_exp_eta2 = np.max(forme) * 1.1
Amp_exp_eta1 = np.max(forme)

nb_periodes = 1
x = lambda_exp / 2
t_exp = np.linspace(0, 2 * nb_periodes * np.pi / omega_exp, 500)
t_exp_eta_3 = np.linspace(0 + 0.142, 2 * nb_periodes * np.pi / omega_exp + 0.142, 500)


eta1 = Amp_exp_eta1 * np.sin(k_exp * x - omega_exp * t_exp)

eta1_2 = Amp_exp_eta2 * np.sin(k_exp * x - omega_exp * t_exp)
terme1_2 = Amp_exp_eta2**2 * k_exp / 4
terme2_2 = 3 * (1 / np.tanh(k_exp * H_exp))**3 - 1/ np.tanh(k_exp * H_exp)
eta2 = terme1_2 * terme2_2 * np.cos(2 * (k_exp * x - omega_exp * t_exp)) #- A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i]) #+ A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i])


eta1_3 = Amp_exp_eta3 * np.sin(k_exp * x - omega_exp * t_exp)
terme1_3 = Amp_exp_eta3**2 * k_exp / 4
terme2_3 = 3 * (1 / np.tanh(k_exp * H_exp))**3 - 1/ np.tanh(k_exp * H_exp)
eta2_3 = terme1_3 * terme2_3 * np.cos(2 * (k_exp * x - omega_exp * t_exp)) #- A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i]) #+ A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i])
k_exp_prime = k_exp * (1 - Amp_exp_eta3 ** 2 * k_exp ** 2)
eta3 = (1 - 3* Amp_exp_eta3**2 * k_exp**2 / 8) * Amp_exp_eta3 * np.cos(k_exp_prime * x - omega_exp * t_exp_eta_3) + Amp_exp_eta3 **2 * k_exp * np.cos(2 * (k_exp_prime * x - omega_exp * t_exp_eta_3)) + 3 * Amp_exp_eta3**3 * k_exp ** 2 * np.cos(3 * (k_exp_prime * x - omega_exp * t_exp_eta_3)) / 8


disp.figurejolie()
disp.joliplot('t (0 à T)', 'x (m)', t_exp, eta1, exp = False, color = 4, legend = '$\eta_1$')
disp.joliplot('t (0 à T)', 'x (m)', t_exp, eta1_2 + eta2 - np.mean(eta1_2 + eta2), exp = False, color = 2, legend = '$\eta_1 + \eta_2$')
# disp.joliplot('t (0 à T)', 'x (m)', t_exp, eta1_3 + eta2_3 + eta3 - np.mean(eta1_3 + eta2_3 + eta3), exp = False, color = 12, legend = '$\eta_1 + \eta_2 + \eta_3$')
disp.joliplot('t (0 à T)', 'x (m)', t_plotexp, forme, exp = True, color = 8, legend = 'Expérience')

#%% COURBURE SEUIL

save = False
save_path = 'E:\\Baptiste\\Resultats_exp\\All_RDD\\Resultats\\'

tableau_1 = pandas.read_csv('E:\\Baptiste\\Resultats_exp\\Tableau_params\\Tableau1_Params_231117_240116\\tableau_1.txt', sep = '\t', header = 0)
tableau_1 = np.asarray(tableau_1)


path = ['E:\\Baptiste\\Resultats_exp\\Courbure\\240109_QSC_a50\\' + '20240422_171834_params_courbure.pkl',
        'E:\\Baptiste\\Resultats_exp\\Courbure\\231120_ECTD\\' + '20240422_180550_params_courbure.pkl' ,
        'E:\\Baptiste\\Resultats_exp\\Courbure\\231122_NPDP\\' + '20240419_171920_params_courbure.pkl' , 
        'E:\\Baptiste\\Resultats_exp\\Courbure\\231124_RLPY\\' + '20240422_153853_params_courbure.pkl' ,
        'E:\\Baptiste\\Resultats_exp\\Courbure\\231129_EJCJ_a30\\' + '20240423_123023_params_courbure.pkl' ,
        'E:\\Baptiste\\Resultats_exp\\Courbure\\231129_MLO_a50\\' + '20240422_182428_params_courbure.pkl']

index = [7, 0, 2, 3, 4, 5]
a_s = np.zeros(len(path))
k_s = np.zeros(len(path))
l_s = np.zeros(len(path))

for i in range (len(path)) :
    params = dic.open_dico(path[i])
    kmin_QSC = params['courbure']['k_minmin']
    hmin_QSC = params['courbure']['h_mimmin']
    
    
    def fit_2(x, a, b) :
        return a * x**2 + b * x
    
    popt, pcov = curve_fit(fit_2, hmin_QSC, kmin_QSC, p0 = [10000000, 10000], bounds = [[0,0], [100000000, 10000000]]) #np.polyfit(hmin_QSC, kmin_QSC,2)
    
    h_tot = np.linspace(np.min(hmin_QSC), np.max(hmin_QSC), 100)
    
    disp.figurejolie()
    disp.joliplot('A (m)', r'$\kappa$ (m$^{-1}$)', hmin_QSC, kmin_QSC, color = 5)
    disp.joliplot('A (m)', r'$\kappa$ (m$^{-1}$)', h_tot, h_tot**2 * popt[0] + h_tot * popt[1], color = 2, exp = False)
    a_s[i] = tableau_1[index[i], 9]
    k_s[i] = a_s[i]**2 * popt[0] + a_s[i] * popt[1]
    l_s[i] = tableau_1[index[i], 6]
    plt.plot(a_s[i], k_s[i], 'ko')
    plt.title(str(tableau_1[i,0]))
    if save :    
        plt.savefig('E:\\Baptiste\\Resultats_exp\\Courbure\\Resultats\\' + params['date']  + '_' + params['nom_exp'][:3] + '_' + 'fitcourbure_seuil' + tools.datetimenow() + '.pdf')#, dpi = 500)



disp.figurejolie()
disp.joliplot('$\lambda$ (m)', r'$\kappa$ (m$^{-1}$)', l_s, k_s, zeros = True, color = 8)
if save : 
    plt.savefig('E:\\Baptiste\\Resultats_exp\\Courbure\\Resultats\\' + 'Courbure_seuil_' + tools.datetimenow() + '.pdf')#, dpi = 500)


fits.fit_powerlaw(l_s, k_s, display = True, xlabel = '$\lambda$ (m)', ylabel = r'$\kappa$ (m$^{-1}$)')
if save :
    plt.savefig('E:\\Baptiste\\Resultats_exp\\Courbure\\Resultats\\' + 'Courbure_seuil_lambda_powerlaw_' + tools.datetimenow() + '.png', dpi = 500)





