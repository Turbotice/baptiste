# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 16:02:21 2022

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
from skimage.measure import profile_line
from scipy import ndimage
from scipy import stats
import scipy.fft as fft
import os
import time
from PIL import Image
from datetime import datetime
from scipy.optimize import minimize
%run Functions_FSD.py
%run parametres_FSD.py
%run display_lib_gld.py


loc_h = 'D:\Banquise\Baptiste\Resultats_video\d221104\d221104_PIVA6_PIV_44sur026_facq151Hz_texp5000us_Tmot010_Vmot410_Hw12cm_tacq020s/'


import scipy.io as io

nom_exp = "CCCS1"
path_mat, list_mat, titre_exp = import_images(loc, nom_exp, "LAS", nom_fich = "\\fichiers_mat\\")
data_brut = io.loadmat(path_mat + 'u_original.mat')

parametres = []
list_lambda = []
list_kappa =[]

mmparpixel = import_calibration(titre_exp, date)
facq = 151


#%%
u = os.listdir(loc_h)

h_pixel212 = np.loadtxt(loc_h + "references\\a.txt", delimiter = ',')
x= np.linspace(0,len(h_pixel212),len(h_pixel212))

figurejolie()
joliplot(r'x (mm)', r'h ($\mu$m)', x,h_pixel212, exp = False)

#%%LECTURE DONNEES PIV JUAN

loc_JUAN_PIV = "Y:\Banquise\Baptiste\Juan_PIV\Test_Baptiste_Banquise\Champ_u"
fichiers = os.listdir(loc_JUAN_PIV)

import pandas as pd

all_values = []

for i in range (0,300) : # (len(fichiers)/2):
    df = pd.read_csv(loc_JUAN_PIV + "//" + fichiers[i], sep = ' ', header=None)
    all_values.append(df.values)
    if np.mod(i,25)==0:
        print('On processe l image numero: ' + str(i) + ' sur ' + str(300))
print('Done !')
 

data= all_values[5]   
plt.figure()
plt.imshow(data, cmap = 'turbo')
plt.ylabel("Y (pixel)")
plt.xlabel("X (pixel)")
plt.title("image numero " + str(i))
cbar = plt.colorbar()
cbar.set_label('Amplitude (m)')
    
data = np.asarray(all_values)

#%%

"""
HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
                PARTIE 2
     Traitement de signal 3D, s(t,y,x)
HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
"""


#%% DONNEES PIVLAB
path_mat = 'H:\Canada\Data\Stereo\d230310\\haha_10HZ_21minutes_2\\'

data_brut = io.loadmat(path_mat + 'PIVlab_1_101_PIV.mat')

#%%PARAMETRES

mmparpixel = 1

display = True
save_param = False
save = False
  
facq = 10 #151
ratio_PIV = 16
mmparpixel_PIV = mmparpixel * ratio_PIV


if save_param :
    parametres.extend(["facq = " + str(facq), "ratio_PIV = " + str(ratio_PIV), "mmparpixel = " + str(mmparpixel)])

#%%MISE EN FORME

champ_u = data_brut["u_original"]

u0 = champ_u[:,0][0]
data = []

#On enleve la moyenne spatiale à chaque temps
for i in range (np.shape(champ_u)[0]):
    data.append(champ_u[i][0] - np.mean(champ_u[i][0]))
data = np.asarray(data)

#On enleve la moyenne temporelle pour chaque pixel
for j in range(data.shape[1]):
    for k in range (data.shape[2]):
        data[:,j,k] = data[:,j,k] - np.mean(data[:,j,k])

[nt,ny,nx] = data.shape

x = np.linspace(0, nx, nx)  #Taille X en m
y = np.linspace(0, ny, ny)  #Taille Y en m
t = np.linspace(0, nt /facq, nt)                    #Vecteur temps en secondes

data = np.transpose(data) #Pour passer de s(t,y,x) à s(x,y,t)

# Petit film du champ de vitesse

display_video = False
if display_video :
    figurejolie()
    plt.pcolormesh(y, x, data[:,:,0], shading='auto')
    plt.xlabel("Y (cm)")
    plt.ylabel("X (cm)")
    cbar = plt.colorbar()
    cbar.set_label("Champ u")
    plt.axis("equal")
    for mmm in range (1,10):
        figurejolie()
        plt.pcolormesh(y*100, x*100, data[:,:,mmm], shading='auto')
        plt.pause(0.01)
        cbar = plt.colorbar()
        # plt.clim(-2,2)

# Affichage du champ de vitesse de la première image
if display :
    figurejolie()
    plt.pcolormesh(y*100, x*100, data[:,:,80], shading='auto')
    plt.xlabel("Y (cm)")
    plt.ylabel("X (cm)")
    cbar = plt.colorbar()
    cbar.set_label("Champ u")
    plt.axis("equal")
    # plt.clim(-1,1)      

#%%MOUVEMENT D'UN PIXEL

figurejolie()
plt.plot(t, data[20,33,:])
plt.title('pos pixel en fct du tps')
fft_1 = fft.fft( data[20,33,:])
figurejolie()
plt.plot(np.abs(fft_1))
plt.title('fft de ce mvt')
#%%DEMODULATION

def demodulation(t,s, fexc):
    c = np.nanmean(s*np.exp(1j * 2 * np.pi * t[None,None,:] * fexc),axis=2)
    return c


f_exc = 10

demod = demodulation(t,data,f_exc)

demod[np.isnan(demod)] = 0 * demod[np.isnan(demod)] #on met les points nan à 0

if display :
    figurejolie()
    plt.pcolormesh(y * 100, x * 100, (np.real(demod)), shading = 'auto')
    plt.axis('equal')
    cbar = plt.colorbar()
    plt.xlabel("Y (cm)")
    cbar.set_label("Champ u démodulé à " + str(f_exc) + "Hz")
    plt.ylabel("X (cm)")
    # plt.clim(vmin = -0.03, vmax = 0.03)


#%% FFT

padpad = False   #Est ce qu'on applique du 0 padding
padding = 10     #puissance de 2 pour le 0 padding

zone_dinteret_x = np.asarray([int(5/(x[-1]*100) * nx),int(25/(x[-1]*100) * nx)])
zone_dinteret_y = np.asarray([int(1/(x[-1]*100) * ny),int(21/(x[-1]*100) * ny)])

X_interet = np.linspace(zone_dinteret_x[0],zone_dinteret_x[1],zone_dinteret_x[1]-zone_dinteret_x[0] ) * mmparpixel_PIV / 10
Y_interet = np.linspace(zone_dinteret_y[0],zone_dinteret_y[1],zone_dinteret_y[1]-zone_dinteret_y[0] ) * mmparpixel_PIV / 10

if display :
    figurejolie()
    plt.pcolormesh(Y_interet, X_interet, (np.real(demod[zone_dinteret_x[0]:zone_dinteret_x[1],zone_dinteret_y[0]:zone_dinteret_y[1]])), shading = 'auto')
    plt.axis('equal')
    cbar = plt.colorbar()
    plt.xlabel("Y (cm)")
    cbar.set_label("Champ u démodulé à " + str(f_exc) + "Hz COUPE")
    plt.ylabel("X (cm)")

if padpad :
    demod_padding = np.zeros((2**padding,2**padding), dtype = 'complex128')
    for i in range (nx):
        for j in range(ny):
            demod_padding[i,j] = demod_padding[i,j] + demod[i,j]
    Y_FFT = fft.fft2(demod_padding)

else :
    Y_FFT = fft.fft2(demod[zone_dinteret_x[0]:zone_dinteret_x[1],zone_dinteret_y[0]:zone_dinteret_y[1]])
    

nx_FFT = np.shape(Y_FFT)[0]
ny_FFT = np.shape(Y_FFT)[1]

if padpad :    
    title_FFT = "FFT 2D spatiale 0 padding (2**" + str(padding) + "), démodulé à " + str(f_exc) + "Hz"
else :
    title_FFT = "FFT 2D spatiale, démodulé à " + str(f_exc) + "Hz"
    
k_x = np.linspace(2 * np.pi / mmparpixel_PIV * 1000, 0, nx_FFT)
k_y = np.linspace(0, 2 * np.pi / mmparpixel_PIV * 1000, ny_FFT)
    
if display :
    figurejolie()
    plt.pcolormesh(k_y , k_x, np.abs(Y_FFT), shading = 'auto')
    plt.xlabel(r"$k_y (m^{-1})$")
    plt.ylabel(r"$k_x (m^{-1})$")
    cbar = plt.colorbar()
    cbar.set_label(title_FFT)

#max de la FFT en cooedonnées kx ky
max_fft = np.asarray([(np.transpose(np.where(np.max(np.abs(Y_FFT)) == np.abs(Y_FFT)))[0][1])/mmparpixel_PIV * 1000 * 2 * np.pi/nx_FFT, 2 * np.pi *(nx_FFT-np.transpose(np.where(np.max(np.abs(Y_FFT)) == np.abs(Y_FFT)))[0][0]) * 1000 / mmparpixel_PIV/nx_FFT])

if True : #f_exc > facq/2 :
    lambda_demod = nx_FFT * mmparpixel_PIV / (np.transpose(np.where(np.max(np.abs(Y_FFT)) == np.abs(Y_FFT)))[0][0]) / 1000
else :
    lambda_demod = nx_FFT * mmparpixel_PIV / (nx_FFT-np.transpose(np.where(np.max(np.abs(Y_FFT)) == np.abs(Y_FFT)))[0][0]) / 1000

err_lambda = nx * mmparpixel_PIV / 1000 / nx_FFT
           
        
if display :
    plt.plot(max_fft[0],max_fft[1], 'ro', label = 'MAX FFT')
    plt.legend()

print ("lambda = ", lambda_demod)
print ("k = ", 2 * np.pi / lambda_demod)

k = 160
lambda_demod = 2 * np.pi / k


#%%SAUVEGARDE LAMBDA
save = True
if save :
    list_lambda.append([lambda_demod, err_lambda, f_exc])
    

#%%
parametres.extend(["PARAMETRES DE TRAITEMENT :","padpad = " + str(padpad), "padding = " + str(padding)])
np.savetxt(path_mat[:-13] + "resultats" + "/Paramètres_" + nom_exp + "_" + date + ".txt", parametres, "%s")
np.savetxt(path_mat[:-13] + "resultats" + "/lambda_err_fexc_PIV_" + nom_exp + "_" + date + ".txt", list_lambda, "%s")



#%%Profil moyen pour attenuation

#max de la FFT en indices
x_max_FFT = np.transpose(np.where(np.max(np.abs(Y_FFT)) == np.abs(Y_FFT)))[0][0]
y_max_FFT = np.transpose(np.where(np.max(np.abs(Y_FFT)) == np.abs(Y_FFT)))[0][1]


y_new_x = np.zeros(np.shape(Y_FFT), dtype = 'complex128') #FFT dans le sens de propagation
y_new_moinsx = np.zeros(np.shape(Y_FFT), dtype = 'complex128') #FFT dans le sens reflechis

#On choisi le quart de FFT à sélectionner en fonction de son max. ax/bx/ay/by vont déterminer les coordonnées du carré

if x_max_FFT < nx_FFT/2 :
    bx = int(nx_FFT/2)
    ax = 0
else :
    bx = nx_FFT
    ax = int(nx_FFT/2) + 1 
    
if y_max_FFT < ny_FFT/2 :
    by = int(ny_FFT/2)
    ay = 0 
else :
    by = ny_FFT
    ay = int(ny_FFT/2) + 1
    
ay = 0
by = ny_FFT

y_new_x[ax:bx,ay:by] = Y_FFT[ax:bx,ay:by]

if x_max_FFT < nx_FFT/2 :
    bx = nx_FFT
    ax = int(nx_FFT/2) + 1 
else :
    bx = int(nx_FFT/2)
    ax = 0
    
y_new_moinsx[ax:bx,ay:by] = Y_FFT[ax:bx,ay:by]

if display :
    figurejolie()
    plt.pcolormesh(np.abs(y_new_x))
    figurejolie()
    plt.pcolormesh(np.abs(y_new_moinsx))



demod_stat_x = fft.ifft2(y_new_x)
demod_stat_moinsx = fft.ifft2(y_new_moinsx)

if display :
    figurejolie()
    plt.pcolormesh(y , x ,np.real(demod_stat_x)[])
    plt.title("FFT inverse sens de propagation")
    
    figurejolie()
    plt.pcolormesh(y , x, np.abs(demod_stat_moinsx))
    plt.title("FFT inverse onde réflechie")

profil_amp_x = []
profil_amp_moinsx = []
x_ATT= np.linspace(25 * mmparpixel_PIV / 10, (nx-25) * mmparpixel_PIV / 10, nx-50)


for j in x_ATT:
    uuu = profile_line((np.abs(demod_stat_x)), (j, 0), (j, 66), mode = 'reflect')
    uuu.sort()
    uuu = uuu[10:-10]
    profil_amp_x.append(np.mean(uuu))

figurejolie()
plt.semilogy(profil_amp_x)
profil_amp_x = np.asarray(profil_amp_x)
profil_amp_x = profil_amp_x**2

for j in x_ATT:
    uuu = profile_line((np.abs(demod_stat_x)), (j, 0), (j, 66), mode = 'reflect')
    uuu.sort()
    uuu = uuu[10:-10]
    profil_amp_moinsx.append(np.mean(uuu))

figurejolie()
plt.semilogy(profil_amp_moinsx)
profil_amp_moinsx = np.asarray(profil_amp_moinsx)
profil_amp_moinsx = profil_amp_x**2



def exppp(x, a, b):
    return a * np.exp(-b * x)

attenuation_x = np.polyfit(x_ATT, np.log(profil_amp_x), 1)
attenuation_moinsx = np.polyfit(x_ATT, np.log(profil_amp_moinsx), 1)

figurejolie()
joliplot(r"x (cm)", r"I", x_ATT, profil_amp_x, color = 3, exp = False, log = True, legend = r"f = " + str(int(f_exc) ) + " Hz")
joliplot(r"x (cm)", r"I", x_ATT, x_ATT * attenuation_x[0] + attenuation_x[1], color = 5, exp = False, log = True, legend = r"fit, on trouve $\kappa = $" + str(round(attenuation[0][1],4)))
plt.xscale('linear')
plt.yscale('log')


figurejolie()
joliplot(r"x (cm)", r"I", x_ATT, profil_amp_moinsx, color = 3, exp = False, log = True, legend = r"f = " + str(int(f_exc) ) + " Hz")
joliplot(r"x (cm)", r"I", x_ATT, x_ATT * attenuation_moinsx[0] + attenuation_moinsx[1], color = 5, exp = False, log = True, legend = r"fit, on trouve $\kappa = $" + str(round(attenuation[0][1],4)))
plt.xscale('linear')
plt.yscale('log')





# plt.figure()
# for i in range (0,66) :
#     p = profile_line(np.log(np.abs(demod_stat)), (55, i), (0, i))
#     plt.plot(p, 'x')
#%% k_extraction

import numpy as np
from radialavg2 import radialavg2

def kextraction(data, fitlength, step_ana):
    """
    Extracts the wavenum oneach point of the wavefield.
    It calls for the function radialavg2 to reconstruct the Bessel function of first order on each ooint of the 2D matrix data.
    :param: 
   
        * data : complex demodulated field;
        * fitelength : resolution of the reconstructed bessel function;
        * step_ana : step of analysis.
   
    :return:
   
        Return k the wavenum field
    
    Example
    -------
    >>> step_ana = 1
    >>> fitlength = 30
    >>> kfield  = kextraction(c, fitlength, step_ana)
    """

    [nx,ny] = data.shape
    cx = 0 
    k2 = np.zeros((int((ny-fitlength)/step_ana), int((nx-fitlength)/step_ana)-1))
    phase_locale = np.ones((2*fitlength,2*fitlength))
    signal_local = np.zeros(phase_locale.shape)
    for x0 in range(fitlength, nx-fitlength+1, step_ana):
        if np.mod(x0,60)==0:
            print(str(np.round(x0*100/(nx-fitlength),0))+ ' % ')
        cy = 0
        for y0 in range(fitlength, ny-fitlength+1, step_ana):
            phase_locale = np.ones((2*fitlength,2*fitlength))*np.exp(1j*np.angle(data[x0,y0]))
            signal_local = np.real(data[x0-fitlength:x0+fitlength, y0-fitlength:y0+fitlength]*phase_locale)
            [r2,zr2] = radialavg2(signal_local, 1, fitlength+1, fitlength+1)
            xx = r2[0:fitlength]
            xx2 = np.concatenate((np.flipud(-xx),xx))
            test = np.abs(zr2[0:fitlength])
            test2 = np.concatenate((np.flipud(test),test))
            pp = np.polyfit(xx2,test2,deg=2)
            pp[0]=np.abs(pp[0])
            pp[2]=np.abs(pp[2])
            k2[cy,cx]=np.sqrt(4*pp[0]/pp[2])
            cy+=1
        cx += 1
    return k2




fit_length = 5
champ_k = kextraction(demod_stat, fit_length, 1)
#%%
k_tot = np.transpose(champ_k[:-fit_length,:-fit_length])
#AFFICHAGE CHAMP K
k_tot = medfilt2d(k_tot, kernel_size=5)
figurejolie()
plt.pcolormesh(y, x * 100, k_tot**(5/3))
plt.title('champ k, démodulé à ' + str(f_exc) + " Hz")
cbar = plt.colorbar()
cbar.set_label('k (m?)')
# plt.clim(vmin = 0, vmax = 20)
plt.axis("equal")

