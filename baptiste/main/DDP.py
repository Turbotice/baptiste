# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 18:06:43 2023

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


import baptiste.display.display_lib as disp
import baptiste.experiments.import_params as ip
import baptiste.files.file_management as fm
import baptiste.image_processing.image_processing
import baptiste.files.dictionaries as dic
import baptiste.tools.tools as tools
import baptiste.math.RDD as rdd
import baptiste.math.fits as fits

dico = dic.open_dico()

#%% liste_tot

date_min = 231115
date_max = 231231
save = False

long_onde = np.array([])
amp_max = np.array([])
amp_moy = np.array([])
l_cracks = np.array([])

liste_top = np.zeros((7,86), dtype = object)



u = 0
v = 1
for date in dico.keys() :
    if date.isdigit() :
        if float(date) >= date_min and float(date) < date_max :
            for nom_exp in dico[date].keys() :
                
                liste_top[0,u] = nom_exp
                u += 1

    

u = 0
for date in dico.keys() :
    if date.isdigit() :
        if float(date) >= date_min and float(date) < date_max :
            print(date)                
            for nom_exp in dico[date].keys() :
                print(nom_exp)
                if 'lambda' in dico[date][nom_exp].keys() :
                    liste_top[1,u] = dico[date][nom_exp]['lambda']
                    
                if 'l_cracks' in dico[date][nom_exp].keys() :
                    liste_top[4,u] = np.nansum(dico[date][nom_exp]['l_cracks']) / 1000
                    
                if 'Amp_moy' in dico[date][nom_exp].keys() :
                    liste_top[2,u] = dico[date][nom_exp]['Amp_moy']
                    
                if 'Amp_max' in dico[date][nom_exp].keys() :
                    liste_top[3,u] = dico[date][nom_exp]['Amp_max']
                
                
                
                
                u += 1
                if u > 85 :
                    for k in range (v) :
                        liste_top[5, u - k-1] = liste_top[4, u-k-1] / np.max(liste_top[4, u-v:u-1])
                    plt.figure()
                    plt.plot(liste_top[3, u-v:u], liste_top[4, u-v:u],'kx')
                    if save :
                        plt.savefig('E:\Baptiste\Resultats_exp\\' + nom_exp + "crack_amp.pdf")
                    v = 1
                    print(v)
                    amp_lcrack = np.stack((liste_top[3, u-v:u], liste_top[5, u-v:u]))
                    np.savetxt('E:\\Baptiste\\Resultats_exp\\amp_crack\\amp_lcrack_' + liste_top[0, u-v] + ".txt", amp_lcrack)
                    
                else :
                    if liste_top[0,u-1][:3] == liste_top[0,u][:3] :
                        v +=1
                    else :
                        for k in range (v) :
                            liste_top[5, u - k] = liste_top[4, u-k] / np.max(liste_top[4, u-v:u])
                        plt.figure()
                        plt.plot(liste_top[3, u-v:u], liste_top[4, u-v:u], 'kx')
                        if save :
                            plt.savefig('E:\Baptiste\Resultats_exp\\' + nom_exp + "crack_amp.pdf")
                        v = 1
                        print(v)
                        amp_lcrack = np.stack((liste_top[3, u-v:u], liste_top[5, u-v:u]))
                        np.savetxt('E:\\Baptiste\\Resultats_exp\\amp_crack\\amp_lcrack_' + liste_top[0, u-v] + ".txt", amp_lcrack)
                

# for nom_exp in liste_top[0,:] :
#     if nom_
#%% Intersection Lambda Crack :
u = 0
v = 1 
for date in dico.keys() :
    if date.isdigit() :
        if float(date) >= date_min and float(date) < date_max :
            print(date)                
            for nom_exp in dico[date].keys() :
                print(nom_exp)            
                u += 1
                if u > 85 :
                    for k in range (v) :
                        liste_top[5, u - k-1] = liste_top[4, u-k-1] / np.max(liste_top[4, u-v:u-1])
                #     plt.figure()
                #     plt.plot(liste_top[3, u-v:u], liste_top[4, u-v:u],'kx')
                #     if save :
                #         plt.savefig('E:\Baptiste\Resultats_exp\\' + nom_exp + "crack_amp.pdf")
                    amp_lcrack = np.stack((liste_top[3, u-v:u], liste_top[5, u-v:u]))
                    np.savetxt('E:\\Baptiste\\Resultats_exp\\amp_crack\\amp_lcrack_' + liste_top[0, u-v] + ".txt", amp_lcrack)
                    
                    print(v)
                    v = 1

                    
                else :
                    
                    if u == 46 :
                        for k in range (v) :
                            liste_top[5, u - k] = liste_top[4, u-k] / np.max(liste_top[4, u-v:u])
                #         plt.figure()
                #         plt.plot(liste_top[3, u-v:u], liste_top[4, u-v:u], 'kx')
                #         if save :
                #             plt.savefig('E:\Baptiste\Resultats_exp\\' + nom_exp + "crack_amp.pdf")
                        print(v)
                        print('odfjgqjgokqjrt')
                        v = 1
                        
                    
                    elif liste_top[0,u-1][:3] == liste_top[0,u][:3] :
                        v +=1
                        
                    
                        
                    else :
                        for k in range (v) :
                            liste_top[5, u - k] = liste_top[4, u-k] / np.max(liste_top[4, u-v:u])
                #         plt.figure()
                #         plt.plot(liste_top[3, u-v:u], liste_top[4, u-v:u], 'kx')
                #         if save :
                #             plt.savefig('E:\Baptiste\Resultats_exp\\' + nom_exp + "crack_amp.pdf")
                        print(v)
                        v = 1



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
colors = np.zeros((86,4))
for i in range (86): 
    colors[i,:] = disp.vcolors(int(liste_top[5,i] * 9))
     
plt.scatter(liste_top[1,:],liste_top[3,:],color=colors)
plt.scatter(liste_top_nico[6,:],liste_top_nico[1,:],color=colors_nico)
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
        disp.joliplot(r"$\lambda$ (m)", r"Amplitude (m)", l_onde[i], amplitude[i], color = 13)
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
disp.joliplot(r'$\lambda$ (m)', r'Amplitude (m)', data_traitées0[:,1], data_traitées0[:,2], color = 16, title = False, legend = r'Intact', exp = True)
disp.joliplot(r'$\lambda$ (m)', r'Amplitude (m)', data_traitées1[:,1], data_traitées1[:,2], color = 15, title = False, legend = r'Fracturé', exp = True)
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

