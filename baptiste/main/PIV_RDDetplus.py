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
loc_h = 'W:\Banquise\\Data_DD_UQAR\\'

import scipy.io as io
titre_exp = 'CCCS1'
mmparpixel = import_calibration(titre_exp, date)
mmparpixel = 1
facq = 1

"""
                                                    HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
                                                                    PARTIE 1
                                                         Traitement de signal 3D, s(t,y,x)
                                                    HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
"""
parametres = []

#%% DONNEES PIVLAB
# data_brut = io.loadmat(loc_h + 'PIVlab_3000.mat')
data_brut = io.loadmat(loc_h + 'PIVlab_6m30_7m10_3couches_64_32_16_traitement230412')

#%%PARAMETRES


display = True
save_param = False
save = False
  
facq = 30 #151
ratio_PIV = 8
mmparpixel_PIV = mmparpixel * ratio_PIV


if save_param :
    parametres.extend(["facq = " + str(facq), "ratio_PIV = " + str(ratio_PIV), "mmparpixel = " + str(mmparpixel)])

#%%MISE EN FORME
champ_u = data_brut["u_original"]
champ_v = data_brut["v_original"]

u0 = champ_u[:,0][0]
data_u = []
data_v = []


for i in range (np.shape(champ_u)[0]):
    data_u.append(champ_u[i][0]- np.nanmean(champ_u[i][0]))
data_u = np.asarray(data_u)

for i in range (np.shape(champ_v)[0]):
    data_v.append(champ_v[i][0]- np.nanmean(champ_v[i][0]))
data_v = np.asarray(data_v)


data = np.sqrt(np.power(data_v,2) + np.power(data_u,2))
data = data_u

data = np.transpose(data, (2,1,0)) #Pour passer de s(t,y,x) à s(x,y,t)
data = np.flip(data, 1)

[nx,ny,nt] = data.shape

x = np.linspace(0, nx * mmparpixel_PIV , nx)  #Taille X en pixel
y = np.linspace(0, ny * mmparpixel_PIV , ny)  #Taille Y en pixel
t = np.linspace(0, nt /facq, nt)       


#On enleve la moyenne temporelle pour chaque pixel
for j in range(data.shape[1]):
    for k in range (data.shape[2]):
        data[:,j,k] = data[:,j,k] - np.mean(data[:,j,k])

#%% Interpolation et affichage
        
        
                                                            """INTERPOLATION"""

interp = False 
"""
Methode 1 : skimage
"""
# if interp :  
#     from skimage.restoration import inpaint
#     for i in range(len(t)):
#         missing_data = np.isnan(data[:,:,i])
#         data[:,:,i] = inpaint.inpaint_biharmonic(data[:,:,i], missing_data)
#         print(i)

"""
Methode 2 : interpolate
"""
        
if interp :        
    from scipy.interpolate import griddata
    
    grid_x, grid_y = np.meshgrid(np.linspace(0, nx, nx, dtype = 'int'), np.linspace(0, ny, ny, dtype = 'int'), indexing='ij')     
    for i in range (len(t)):
        data_tofit =np.array( np.where(np.isfinite(data[:,:,i])))
        data_index = np.transpose(data_tofit)
        data[:,:,i] = griddata(data_index, data[data_index[:,0],data_index[:,1],i], (grid_x, grid_y), method = 'linear')
        print(i)

        
# data[np.isnan(data)] = 0 #met les nan à 0
# data[np.isnan(data)] = 0 #met les nan à 0

# Petit film du champ de vitesse

display_video = False
if display_video :
    figurejolie()
    plt.pcolormesh(y*100, x*100, data[:,:,0], shading='auto')
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
        plt.clim(-2,2)

# Affichage du champ de vitesse de la première image
if display :
    figurejolie()
    plt.pcolormesh(y*100, x*100, data[:,:,80], shading='auto')
    plt.xlabel("Y (cm)")
    plt.ylabel("X (cm)")
    cbar = plt.colorbar()
    cbar.set_label("Champ u")
    plt.axis("equal")
    plt.clim(-1,1)      

#%%MOUVEMENT D'UN PIXEL
x_pixel = 20
y_pixel = 33

plt.plot(x_pixel* mmparpixel_PIV *100 ,y_pixel* mmparpixel_PIV *100, 'mo')
figurejolie()
data_pixel = data[x_pixel,y_pixel,:]
plt.plot(t, data_pixel)
plt.title('pos pixel en fct du tps')
fft_1 = fft.fft(data_pixel)

figurejolie()
f = np.linspace(0,facq,nt)
plt.plot(f,np.abs(fft_1))
plt.title('fft de ce mvt')
#%%DEMODULATION

def demodulation(t,s, fexc):
    c = np.nanmean(s*np.exp(1j * 2 * np.pi * t[None,None,:] * fexc),axis=2)
    return c


f_exc = 0.28

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

#%%RDD
data[np.isnan(data)] = 0 #met les nan à 0
# 1) FFT 2 
ff = np.linspace(0,facq,nt)
kky = np.linspace(0,mmparpixel_PIV,ny)
kkx = np.linspace(0,mmparpixel_PIV,nx)
YY = fft.fft2(data - np.nanmean(data))
figurejolie()
plt.pcolormesh( np.log(np.abs(fft.fftshift(YY[0,:,:]))))
cbar = plt.colorbar()
#%%
# 2) Demoduler pour angle
fmin = 0.01
fmax = 1.4
nb_f = 500

padding = 9    #puissance de 2 pour le 0 padding
k_xx = []
k_yy = []
kk = []
theta = []
fff = []

mparpixel = 0.07988064791133845 * ratio_PIV #mesuré avec la taille du bateau...


plotplot = False

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

for i in np.linspace(fmin, fmax, nb_f) :
    print(i)
    demod = demodulation(t,data,i)
    demod_padding = np.zeros((2**padding,2**padding), dtype = 'complex128')
    for l in range (nx):
        for m in range(ny):
            demod_padding[l,m] = demod_padding[l,m] + demod[l,m]
    Y_FFT = fft.fft2(demod_padding)
    Y_FFT = fft.fftshift(Y_FFT)
    nx_FFT = np.shape(Y_FFT)[0]
    ny_FFT = np.shape(Y_FFT)[1]
    
    #zone kx>0 et ky <0 :
    # [int(nx_FFT/2):,:int(nx_FFT/2)]
    max_fft = np.asarray([(np.where(np.max(np.abs(Y_FFT[int(nx_FFT/2):,:int(ny_FFT/2)])) == np.abs(Y_FFT[int(nx_FFT/2):,:int(ny_FFT/2)]))[0][0])/ nx_FFT,
                          (np.where(np.max(np.abs(Y_FFT[int(nx_FFT/2):,:int(ny_FFT/2)])) == np.abs(Y_FFT[int(nx_FFT/2):,:int(ny_FFT/2)]))[1][0] - ny_FFT/2) / ny_FFT]) /mparpixel * 2 * np.pi
    
    nx_FFT = np.shape(Y_FFT)[0]
    ny_FFT = np.shape(Y_FFT)[1]
    k_x = np.linspace(-2*np.pi/(2* mparpixel), 2*np.pi/(mparpixel* 2), nx_FFT)
    k_y = np.linspace(-2*np.pi/(2* mparpixel), 2*np.pi/(mparpixel* 2), ny_FFT)
    
    k_xx.append(max_fft[0])
    k_yy.append(max_fft[1])
    kk.append(cart2pol(max_fft[0],max_fft[1])[0])
    theta.append(cart2pol(max_fft[0],max_fft[1])[1])
    fff.append(i)
    
    if plotplot :
        figurejolie()
        plt.pcolormesh(k_y,k_x, np.abs(Y_FFT),shading='auto')
        plt.plot(k_yy[-1],k_xx[-1], 'ro', label = 'MAX FFT')
        plt.legend()
 
k_xx = np.asarray(k_xx)
k_yy = np.asarray(k_yy)
kk = np.asarray(kk)
theta = np.asarray(theta)
fff = np.asarray(fff)


#%% Affichage :
figurejolie()
# joliplot('f(Hz)', 'angle (radian)',fff,theta, exp = False)
joliplot('f(Hz)', 'angle (degrés)',fff,theta * 180 / np.pi, exp = False, color = 5)
figurejolie()
# joliplot('f(Hz)', r'K (m$^{-1}$)', fff*2, kk, color = 4)
joliplot('f(Hz)', r'K (m$^{-1}$)', fff, kk, color = 6)
# joliplot('f(Hz)', r'K (m$^{-1}$)', fff*2/3, kk, color = 9)

# figurejolie()
# plt.pcolormesh(np.abs(Y_FFT),shading='auto')


# figurejolie()
# plt.pcolormesh(np.real(demod),shading='auto')
#%%Fit RDD :
    
    
def RDD(k, a, b):
    return a * k + b
    
logomega = np.log(2 * np.pi * fff)
logk = np.log(kk)

# part1 = [:59]
# part2 = [78:160]
# part3 = [161:275]


popt, pcov = curve_fit(RDD, logk, logomega)

figurejolie()
joliplot(r'log(k)', r'log($\omega$)', logk, logomega, color= 13, exp = True)

popt1, pcov1 = curve_fit(RDD, logk[:59], logomega[:59])
popt2, pcov2 = curve_fit(RDD, logk[78:160], logomega[78:160])
popt3, pcov3 = curve_fit(RDD, logk[165:230], logomega[165:230])

joliplot(r'log(k)', r'log($\omega$)', logk[:59], RDD(logk[:59], popt1[0], popt1[1]), color= 3, exp = False, legend = r'coeff = ' + str(round(popt1[0],2)))

joliplot(r'log(k)', r'log($\omega$)', logk[78:160], RDD(logk[78:160], popt2[0], popt2[1]), color= 5, exp = False, legend = r'coeff = ' + str(round(popt2[0],2)))

joliplot(r'log(k)', r'log($\omega$)', logk[165:230], RDD(logk[165:230], popt3[0], popt3[1]), color= 8, exp = False, legend = r'coeff = ' + str(round(popt3[0],2)))

#%% RANSAC pour le fit :
    
from skimage import io
from skimage.measure import ransac, LineModelND

no_use_data = np.zeros(len(logk), dtype = 'bool')

data1 = np.stack((logk[15:59],logomega[15:59]), axis = -1)
data2 = np.stack((logk[78:160],logomega[78:160]), axis = -1)
data3 = np.stack((logk[161:275],logomega[161:275]), axis = -1)

model_robust, inliers = ransac(data1, LineModelND, min_samples=2,
                               residual_threshold=0.1, max_trials=2000)
outliers = (inliers == False)

no_use_data[15:59] = True
no_use_data[78:160] = True
no_use_data[161:275] = True

xx = logk[15:59]

yy = model_robust.predict_y(xx)

figurejolie()

joliplot(r'log(k)', r'log($\omega$)',data1[inliers, 0], data1[inliers, 1], legend='Inlier data 1', color = 3)
joliplot(r'log(k)', r'log($\omega$)',data1[outliers, 0], data1[outliers, 1], color = 7)
joliplot(r'log(k)', r'log($\omega$)', xx, yy, exp = False, color = 2)

plt.annotate('Pente : ' + str(round(model_robust.params[1][1],2)),(logk[25],logomega[17]))

plt.legend(loc='lower left')

g = np.exp(2 * data2[inliers2,1] - data2[inliers2,0])
   
    
model_robust, inliers2 = ransac(data2, LineModelND, min_samples=2,
                               residual_threshold=0.05, max_trials=2000)
outliers = (inliers2 == False)

xx = logk[78:160]

yy = model_robust.predict_y(xx)

joliplot(r'log(k)', r'log($\omega$)',data2[inliers2, 0], data2[inliers2, 1], legend='Inlier data 2', color = 16)
joliplot(r'log(k)', r'log($\omega$)',data2[outliers, 0], data2[outliers, 1], color = 7)
joliplot(r'log(k)', r'log($\omega$)', xx, yy, exp = False, color = 2)

plt.annotate('Pente : ' + str(round(model_robust.params[1][1],2)),(logk[105],logomega[78]))

g = np.mean(np.exp(2 * data2[inliers2,1] - data2[inliers2,0]))
print('g = ' + str(round(g,3)))

plt.legend(loc='lower right')
   
model_robust, inliers = ransac(data3, LineModelND, min_samples=2,
                               residual_threshold=0.05, max_trials=2000)
outliers = (inliers == False)

xx = logk[161:275]

yy = model_robust.predict_y(xx)

joliplot(r'log(k)', r'log($\omega$)',data3[inliers, 0], data3[inliers, 1], legend='Inlier data 3', color = 12)
joliplot(r'log(k)', r'log($\omega$)',data3[outliers, 0], data3[outliers, 1], legend='Outlier', color = 7)
joliplot(r'log(k)', r'log($\omega$)', xx, yy, legend ='Robust line model', exp = False, color = 2)

plt.annotate('Pente : ' + str(round(model_robust.params[1][1],2)),(logk[241],logomega[111]))

no_use_data = (no_use_data == False)
joliplot(r'log(k)', r'log($\omega$)', logk[no_use_data], logomega[no_use_data], color= 11, exp = True, legend = 'Data without fit')

plt.legend(loc='lower right')
plt.show()    

#%% SAVE
save = False
if save :
    parametres = []
    figurejolie()
    # joliplot('f(Hz)', 'angle (radian)',fff,theta, exp = False)
    joliplot('f(Hz)', 'angle (degrés)',fff,theta * 180 / np.pi, exp = False, color = 5)
    plt.savefig()
    figurejolie()
    joliplot('f(Hz)', r'K (m$^{-1}$)', fff, kk, color = 4)
    plt.savefig()
    parametres.extend(["facq = " + str(facq), "ratio_PIV = " + str(ratio_PIV), "mparpixel = " + str(mparpixel)])
    parametres.extend(["facq = " + str(facq), "ratio_PIV = " + str(ratio_PIV), "mparpixel = " + str(mparpixel)])
    parametres.extend(["facq = " + str(facq), "ratio_PIV = " + str(ratio_PIV), "mparpixel = " + str(mparpixel)])



#%% FFT

padpad = True   #Est ce qu'on applique du 0 padding
padding = 10     #puissance de 2 pour le 0 padding

zone_dinteret_x = np.asarray([0, nx])
zone_dinteret_y = np.asarray([0,ny])

X_interet = np.linspace(zone_dinteret_x[0],zone_dinteret_x[1],zone_dinteret_x[1]-zone_dinteret_x[0] ) 
Y_interet = np.linspace(zone_dinteret_y[0],zone_dinteret_y[1],zone_dinteret_y[1]-zone_dinteret_y[0] ) 

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
Y_FFT = fft.fftshift(Y_FFT)
if padpad :    
    title_FFT = "FFT 2D spatiale 0 padding (2**" + str(padding) + "), démodulé à " + str(f_exc) + "Hz"
else :
    title_FFT = "FFT 2D spatiale, démodulé à " + str(f_exc) + "Hz"
    
k_x = np.linspace(-.5, .5, nx_FFT)
k_y = np.linspace(-.5, .5, ny_FFT)
    
if display :
    figurejolie()
    plt.pcolormesh(k_y , k_x, np.abs(Y_FFT), shading = 'auto')
    plt.xlabel(r"$k_y (m^{-1})$")
    plt.ylabel(r"$k_x (m^{-1})$")
    cbar = plt.colorbar()
    cbar.set_label(title_FFT)

#max de la FFT en cooedonnées kx ky
k_max_fft_x = (np.where(np.max(np.abs(Y_FFT)) == np.abs(Y_FFT))[0][0] - nx_FFT/2) / nx_FFT
k_max_fft_y = (np.where(np.max(np.abs(Y_FFT)) == np.abs(Y_FFT))[1][0] - ny_FFT/2) / ny_FFT
k_max_fft = np.sqrt(k_max_fft_x**2 + k_max_fft_y**2)

if True : #f_exc > facq/2 :
    lambda_demod = nx_FFT/(np.transpose(np.where(np.max(np.abs(Y_FFT)) == np.abs(Y_FFT)))[0][0])
else :
    lambda_demod = nx_FFT / (nx_FFT-np.transpose(np.where(np.max(np.abs(Y_FFT)) == np.abs(Y_FFT)))[0][0])

err_lambda = nx / nx_FFT
           
        
if display :
    plt.plot(k_max_fft_y,k_max_fft_x, 'ro', label = 'MAX FFT')
    # plt.plot(0,0, 'ro', label = 'MAX FFT')
    plt.legend()

print ("lambda = ", lambda_demod)
print ("k = ", k_max_fft)

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
    plt.pcolormesh(y , x ,np.real(demod_stat_x))
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
joliplot(r"x (cm)", r"I", x_ATT, x_ATT * attenuation_x[0] + attenuation_x[1], color = 5, exp = False, log = True, legend = r"fit, on trouve $\kappa = $" + str(round(attenuation_x[1],4)))
plt.xscale('linear')
plt.yscale('log')


figurejolie()
joliplot(r"x (cm)", r"I", x_ATT, profil_amp_moinsx, color = 3, exp = False, log = True, legend = r"f = " + str(int(f_exc) ) + " Hz")
joliplot(r"x (cm)", r"I", x_ATT, x_ATT * attenuation_moinsx[0] + attenuation_moinsx[1], color = 5, exp = False, log = True, legend = r"fit, on trouve $\kappa = $" + str(round(attenuation_x[1],4)))
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
champ_k = kextraction(demod_stat_x, fit_length, 1)
#%%
k_tot = np.transpose(champ_k[:-fit_length,:-fit_length])
#AFFICHAGE CHAMP K
k_tot = medfilt2d(k_tot, kernel_size=5)
figurejolie()
plt.pcolormesh(k_tot**(5/3))
plt.title('champ k, démodulé à ' + str(f_exc) + " Hz")
cbar = plt.colorbar()
cbar.set_label('k (m?)')
# plt.clim(vmin = 0, vmax = 20)
plt.axis("equal")

