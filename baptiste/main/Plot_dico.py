# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 11:39:31 2023

@author: Banquise


Pour plot les paramètes de dictionnaires entre eux
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


#%% plot var1 (var2) depuis le dico
date_min = 230103
date_max = 230120 #AJD ?

var_1 = []
name_1 = 'h'
var_2 = []
name_2 = 'Ld'


for date in dico :
    if date != 'variables_globales' :
        if float(date) >= date_min and float(date) <= date_max :
            print(date)
            for nom_exp in dico[np.str(date)] :
                if name_1 in dico[date][nom_exp] and name_2 in dico[date][nom_exp] :
                    var_1.append(float(dico[date][nom_exp][name_1]))
                    var_2.append(float(dico[date][nom_exp][name_2]))
                    # var_1.append(nom_exp)
                    # var_2.append(dico[date][nom_exp][name_2])


# figurejolie()
# # joliplot(name_1, name_2, var_1, var_2)

# joliplot('', name_2, var_1, var_2)


# casse = np.vstack((var_1, var_2))


#%% Diagramme de phase

dico = dic.open_dico()

date_min = 230103
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
        disp.joliplot(r"$\lambda$ (m)", r"Amplitude (m)", l_onde[i], amplitude[i], color = 14)
        if annotated_name :
            plt.annotate (exp[i],(l_onde[i], amplitude[i]))
        if annotated_ld :
            plt.annotate (round(l_d[i], 4),(l_onde[i], amplitude[i]))

plt.xlim(0,0.45)
plt.ylim(0,0.016)



disp.joliplot(r"$\lambda$ (m)", r"Amplitude (m)", 0, 0, color = 13, legend = 'Casse stationnaire')
disp.joliplot(r"$\lambda$ (m)", r"Amplitude (m)", 0, 0, color = 14, legend = 'Casse pas stationnaire')

# x = np.linspace(0,0.42,100)
# y = 0.006/0.25 * x 
# y_2 = x**(1.5) * 0.056
# y_3 = x**2 * 0.1
# # y_4 = x ** 1.5 * 0.055

# plt.plot(x,y, label = 'Modèle linéaire')
# plt.plot(x,y_2, label = 'Modèle 1.5')

# plt.plot(x,y_3, label = 'Modèle 2')

# # plt.plot(x,y_4, label = 'Modèle x3/2')

# plt.legend()
#%% Ajoute points de stage
data = np.loadtxt("D:\Banquise\Baptiste\Resultats\\220628_diagramme_de_phase\\diagramme_de_phase_trié.txt")

omega = data[:,0]
lambdames = data[:,1]
lambdaest = data[:,2]
amp = data[:,3]
cassage = data[:,4]
hpese = data[:,5]
hbonbonne = data[:,6]

disp.figurejolie()

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
   
    
#%% Plus joli

disp.figurejolie()
#Données de stage
disp.joliplot(r'$\lambda$ (m)', r'Amplitude (m)', data_traitées0[:,1], data_traitées0[:,2], color = 16, title = False, legend = r'Intact', exp = True)
disp.joliplot(r'$\lambda$ (m)', r'Amplitude (m)', data_traitées1[:,1], data_traitées1[:,2], color = 15, title = False, legend = r'Fracturé', exp = True)

#Données stationnaire

for i in range (len(amplitude)) :
    if casse[i] :
        disp.joliplot(r"$\lambda$ (m)", r"Amplitude (m)", l_onde[i], amplitude[i], color = 15)
        if annotated_name :
            plt.annotate (exp[i],(l_onde[i], amplitude[i]))
        if annotated_ld :
            plt.annotate (round(l_d[i], 4),(l_onde[i], amplitude[i]))
    else :
        disp.joliplot(r"$\lambda$ (m)", r"Amplitude (m)", l_onde[i], amplitude[i], color = 16)
        if annotated_name :
            plt.annotate (exp[i],(l_onde[i], amplitude[i]))
        if annotated_ld :
            plt.annotate (round(l_d[i], 4),(l_onde[i], amplitude[i]))

plt.xlim(0,0.45)
plt.ylim(0,0.016)


#FIT
x = np.linspace(0,0.42,100)
y = 0.006/0.25 * x**(0.75) * 0.6
y_2 = x**(1.5) * 0.056
y_3 = x**2 * 0.1
# y_4 = x ** 1.5 * 0.055

# joliplot(r"$\lambda$ (m)", r"Amplitude (m)", x,y, color = 3, legend = r'Modèle contrainte visqueuse ($\frac{3}{4}$)', exp = False)
# joliplot(r"$\lambda$ (m)", r"Amplitude (m)",x,y_2,color = 8, legend = r'Modèle flexion ($\frac{3}{2}$)', exp = False)
# joliplot(r"$\lambda$ (m)", r"Amplitude (m)",x,y_3,color = 6, legend = r'Modèle $^{2}$', exp = False)

# plt.plot(x,y_4, label = 'Modèle x3/2')
plt.grid()
plt.legend()
#%% RDD theorique

disp.figurejolie()

k = np.logspace(-7, 3)

def RDD_full(k,drhoh, Dsurrho, Tsurrho, H, g):
    return np.sqrt( np.tanh(H * k) * ((g * k + Tsurrho * k**3 + Dsurrho * k **5) * (( 1 + drhoh * k)**(-1))) )

disp.joliplot(r'$\omega$', r'k',k, RDD_full(k,0.001,1,10000,100, 10), exp = False, log = True, color = 8)

plt.axis('equal')

#%% RDD theorique

disp.figurejolie()

k = np.logspace(-8, 3)

def RDD_full(k, Dsurrho, Tsurrho, g):
    return np.sqrt( (g * k + Tsurrho * k**3 + Dsurrho * k **5) )

disp.joliplot(r'$\omega$', r'k',k, RDD_full(k,1e6,10000000,10), exp = False, color = 8, log = True)

# plt.axis('equal')

#%% RDD fictive FULLLL

x1 = np.logspace(-1,0,100)
x2 = np.logspace(0,1,100)
x3 = np.logspace(1,2,100)
x4 = np.logspace(2,3,100)
x5 = np.logspace(3,4,100)

disp.figurejolie()

disp.joliplot( r'k $(m^{-1})$',r'$\omega$ (Hz)',x1, x1, exp = False, color = 8, log = True)
disp.joliplot( r'k $(m^{-1})$',r'$\omega$ (Hz)',x2, x2**(0.5), exp = False, color = 8, log = True)
disp.joliplot( r'k $(m^{-1})$',r'$\omega$ (Hz)',x3, x3**(1.5)/10, exp = False, color = 8, log = True)
disp.joliplot( r'k $(m^{-1})$',r'$\omega$ (Hz)',x4, x4**(2.5)/1000, exp = False, color = 8, log = True)
disp.joliplot( r'k $(m^{-1})$',r'$\omega$ (Hz)',x5, x5**2/np.sqrt(1000), exp = False, color = 8, log = True)

plt.grid()

# plt.axis('equal')

#%% RDD fictive g et D

x1 = np.logspace(-1,0,100)
x2 = np.logspace(0,1,100)

disp.figurejolie()

disp.joliplot(r'k $(m^{-1})$',r'$\omega$ (Hz)',x1, x1**(0.5), exp = False, color = 8, log = True)
disp.joliplot(r'k $(m^{-1})$',r'$\omega$ (Hz)',x2, x2**(2.5), exp = False, color = 8, log = True)

# plt.axis('equal')

#%% Coeff magique
#%% D et h : tableau 2

save = False
save_path = 'E:\\Baptiste\\Resultats_exp\\All_RDD\\Resultats\\'

tableau_2 = pandas.read_csv('E:\\Baptiste\\Resultats_exp\\Tableau_params\\Tableau2_Params_220101_240116\\tableau_2_v2.txt', sep = '\t', header = 1)
tableau_2 = np.asarray(tableau_2)

D = np.array(tableau_2[:,2][np.where(tableau_2[:,2] != 0)], dtype = float)
h = np.array(tableau_2[:,4][np.where(tableau_2[:,2] != 0)], dtype = float)

"""
ATTENTION
h RECTIFIE CAR ON A MESURE RHO A POSTERIORI, H DANS LE DICO EST CELUI SI RHO EST EGAL A 900 kg.m-3, OR EN VRAI C'EST 680 kg.m-3
"""
h = h * 900 / 680
"""
FIN DE ATTENTION
"""

dates = np.array(tableau_2[:,1][np.where(tableau_2[:,2] != 0)], dtype = int)

disp.figurejolie()
disp.joliplot('date', 'All D', tableau_2[:,1], tableau_2[:,2])
if save : 
    plt.savefig(save_path + "D_date" + tools.datetimenow() + ".pdf", dpi = 1)


disp.figurejolie()
disp.joliplot('date', 'All h', tableau_2[:,1], tableau_2[:,4])
if save : 
    plt.savefig(save_path + "h_date" + tools.datetimenow() + ".pdf", dpi = 1)


disp.figurejolie()
disp.joliplot('D', 'h', D, h, log = False)
for i in range (len(dates)): 
    plt.annotate(dates[i], (D[i], h[i]))
if save : 
    plt.savefig(save_path + "h_D" + tools.datetimenow() + ".pdf", dpi = 1)


model_robust, inliers, outliers = fits.fit_powerlaw(D, h, xlabel = 'D (Nm)', ylabel = 'h (m)', display = True, fit = 'ransac')    
for i in range (len(dates)): 
    plt.annotate(dates[i], (np.log(D)[i], np.log(h)[i]))
if save : 
    plt.savefig(save_path + "h_D_plus_fit" + tools.datetimenow() + ".pdf", dpi = 1)
    
    
model_robust, inliers, outliers = fits.fit_powerlaw(D, h, fit = 'ransac')    
for i in range (len(dates)): 
    plt.annotate(dates[i], (D[i], h[i]))
if save : 
    plt.savefig(save_path + "h_D_plus_fit" + tools.datetimenow() + ".pdf", dpi = 1)


E = D / h**3 * 10
disp.figurejolie()
disp.joliplot('date', 'E', dates, E)
# for i in range (len(D)): 
#     plt.annotate(dates[i], (D[i], E[i]))
if save : 
    plt.savefig(save_path + "E_date" + tools.datetimenow() + ".pdf", dpi = 1)
    
    
    
def lineaire_3(x, a):
    return x ** (1/3) * a

popt, pcov = fits.fit(lineaire_3, D, h, display = False)

D_scale = np.linspace(np.min(D), np.max(D), 200)

disp.figurejolie()
disp.joliplot('D (Nm)', 'h (m)', D_scale, lineaire_3(D_scale, popt[0]),color = 2, exp = False, log = True, legend = 'E = ' + str(round(10 / popt[0]**3 / 1e6)) + ' MPa')
disp.joliplot('D (Nm)', 'h (m)', D, h,color = 4, exp = True, log = True)


plt.savefig(save_path + "D_h_fit_E_hpuissance3" + tools.datetimenow() + ".pdf", dpi = 1)


#%% D et h : tableau 1

save = True
save_path = 'E:\\Baptiste\\Resultats_exp\\All_RDD\\Resultats\\'

tableau_1 = pandas.read_csv('E:\\Baptiste\\Resultats_exp\\Tableau_params\\Tableau1_Params_231117_240116\\tableau_1.txt', sep = '\t', header = None)
tableau_1 = np.asarray(tableau_1)





#%% plot tanh(kh) des exps

lam = np.linspace(0.062, 0.541, 1000)
k = 2 * np.pi / lam
H = np.linspace(1,100,1000)

disp.figurejolie()
disp.joliplot(r'$\lambda$', r'tanh(kH)', lam, np.tanh(k * 0.11), exp = False, color = 5)



#%% Coeff magique

data_coeff = np.loadtxt("D:\Banquise\Baptiste\\Resultats\\Coeff_magique\\coeff_magique.txt", skiprows= 1)

pds_in = data_coeff[:,0]
pds_fi = data_coeff[:,1]
CM = data_coeff[:,2] 

pour_remp_bonbonne = (pds_in) / np.max(pds_in) * 100
x = np.linspace(0, np.max(pour_remp_bonbonne), 100)

std_cm = np.linspace(np.std(CM), np.std(CM), 100)
mean_cm = np.linspace(np.mean(CM), np.mean(CM), 100)

disp.figurejolie(width = 6)
disp.joliplot(r'Remplissage bonbonne (\%)', r'$C_m$', pour_remp_bonbonne, CM, color = 2)






plt.plot(x,mean_cm, 'k-', linewidth = 1)
plt.plot(x,mean_cm - std_cm, 'k--', linewidth = 1)
plt.plot(x,mean_cm + std_cm, 'k--', linewidth = 1)
plt.xlim(20,100)
plt.ylim(2.5,5)





#%% Homogeneite

from scipy.ndimage import gaussian_filter

path = 'Y:\Banquise\\Baptiste\\Resultats_video\\d221104_PIVA6_PIV_44sur026_facq151Hz_texp5000us_Tmot010_Vmot410_Hw12cm_tacq020s\\references\\transqparence_ref.jpg' 

im1 = cv2.imread(path, cv2.IMREAD_UNCHANGED)

im = np.array(im1[:,:,1])

mmparpixel = 0.19434

I0 = im[111,1340]

X = np.linspace(0, im.shape[0], im.shape[0]) * mmparpixel / 10
Y = np.linspace(0, im.shape[1], im.shape[1]) * mmparpixel / 10

# disp.figurejolie(width = 12)
# disp.joliplot('x','y',X,Y,table = im )
# plt.grid('off')
# plt.axis('equal')
# plt.grid('off')

# plt.plot(110 * mmparpixel,1340 * mmparpixel, 'ko')

hmoy = 133 * 900 / 680

Imoy = np.mean(im)
alpha = 1/hmoy*np.log(Imoy/I0)

im_gauss = gaussian_filter(im, 50)

h = 1/alpha*np.log(im_gauss/I0)


disp.figurejolie(width = 4)
disp.joliplot('X (cm)','Y (cm)',X,Y,table = im_gauss )
plt.grid()
plt.axis('equal')

disp.figurejolie(width = 10)
disp.joliplot('X (cm)','Y (cm)',X,Y,table = h / 1000 )
plt.colorbar(label = '$h$ (mm)')
plt.grid()
plt.axis('equal')

plt.savefig('Y:\Banquise\\Baptiste\\Manuscrit_these\\Figures\\Laboratoire\\Le vernis\\TEST.png', dpi = 500)


disp.figurejolie(width = 4)
# plt.hist(im_gauss, range = (np.min(im_gauss), np.max(im_gauss)), bins = 10)


from skimage import exposure

hist = exposure.histogram(h, nbins=40)
h_x = np.linspace(np.min(h), np.max(h), len(hist[0]))


h_hist = hist[0] / np.size(im) * 100

disp.joliplot(r'$h$ ($\mu$m)','\%',h_x, h_hist , exp = False, color = 5)

def gauss_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


popt, pcov = curve_fit(gauss_function, h_x, h_hist, p0 = [np.max(h_hist), np.mean(h), np.std(h)])

disp.joliplot(r'$h$ ($\mu$m)','\%',h_x, gauss_function(h_x, popt[0], popt[1], popt[2]), exp = False, color = 8, linewidth= 0.7)

# plt.plot(h_x, gauss_function(h_x, popt[0], popt[1], popt[2]))
def LMH(x,y) :
    half_max = np.nanmax(y) * 1/ 2
    d = np.sign(half_max - y)
    k = np.array(np.where(d == -1))
    
    return x[k[0,-1]] - x[k[0,0]]

u = LMH(h_x, gauss_function(h_x, popt[0], popt[1], popt[2]))

plt.vlines(x=popt[1] - np.std(h),ymin = 0,ymax = 1.67, color = 'k', ls = '--', linewidth = 0.4)
plt.vlines(x=popt[1],ymin = 0, ymax = 12.8, color = 'k', ls = '-', linewidth = 0.6)
plt.vlines(x=popt[1] + np.std(h),ymin = 0,ymax = 1.67, ls = '--', color = 'k', linewidth = 0.4)



#%% Taille de grain
from sklearn.preprocessing import binarize

path = 'D:\Banquise\\Baptiste\\Resultats_video\\d221024\\d221024_DAP08_LAS_44sur026_facq143Hz_texp6655us_Tmot020_Vmot071_Hw12cm_tacq020s\\image_sequence\\Basler_a2A1920-160ucBAS__40232066__20221024_184243445_0014.tiff'
im1 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
im1 = np.array(im1)

loop = 1
tr = np.zeros(loop)
display = False
if loop < 5 :
    display = True
d_grain_mean = np.zeros(loop)

for i in range (loop):
    tr[i] = 10000 + i * 10000
    im_b = np.array(binarize(im1, threshold=tr[i]), dtype = bool)
    # plt.figure()
    # plt.pcolormesh(im_b)
    
    im_top = np.zeros(im1.shape)
    im_top[np.where(im_b == True)[0],np.where(im_b == True)[1] ] = im1[np.where(im_b == True)[0],np.where(im_b == True)[1] ]
    
    mmparpixel = 0.19434
    
    i_zero_pad = np.zeros((4096,4096))
    
    i_zero_pad[:im1.shape[0], :im1.shape[1]] = im1
    # i_zero_pad[:im_top.shape[0], :im_top.shape[1]] = im_top
    
    Y = fft.fft2(i_zero_pad - np.mean(i_zero_pad))
    # Y = fft.fft2(im1 - np.mean(im1))
    if display :
        disp.figurejolie(width = 12)
        plt.pcolormesh(im_top)
    
    # disp.figurejolie()
    # plt.pcolormesh(np.log(np.abs(Y)))
    
    
    
    gr = 5.8
    d_max = Y.shape[0]
    dy = int(d_max/3)
    
    kx = np.linspace(0, 2 * np.pi * 1000 / mmparpixel * Y.shape[0] / im1.shape[0],Y.shape[0]-1)
    Y_x = np.sum(np.abs(Y[1:d_max,:dy]), axis = 1)
    # Y_y = np.sum(np.abs(Y[:dy,10:d_max]), axis = 0)
    
    if display :
        disp.figurejolie(width = 12)
        disp.joliplot('$k_x$ (m$^{-1}$)', 'A(FFT)', kx, Y_x, exp = False, color = 5)
        
    # if display :
    #     disp.figurejolie(width = 12)
    #     disp.joliplot('$k_y$ (m$^{-1}$)', 'A(FFT)', kx, Y_y, exp = False, color = 5)
    
    d_grain_mean[i] = 2 * np.pi / kx[np.argmax(Y_x[:1000])]
    # d_grain_mean[i] = 2 * np.pi / kx[np.argmax(Y_y[:1000])]
    
    if display :
        print('Taille moyenne de grain : ' + str(d_grain_mean * 1000) + ' mm')
        
    if np.mod(i,7)==0:
        print(str(int(i / loop * 100)) + ' %')

disp.figurejolie(width = 12)
disp.joliplot('Threshold', r'$d_{grain}$ (mm)', tr[20:], d_grain_mean[20:] * 1000, color = 5, exp = False)


# from skimage import exposure
# hist = exposure.histogram(im1)
# h_x = np.linspace(np.min(im1), np.max(im1), len(hist[0]))
# # plt.plot(hist[0])
# disp.figurejolie()
# disp.joliplot(r'$h$ ($\mu$m)','nb',h_x, hist[0] / np.size(im1), exp = False, color = 5)















