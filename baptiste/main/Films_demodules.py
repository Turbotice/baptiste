# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 18:29:27 2022

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
%run Functions_FSD.py
%run parametres_FSD.py
%run display_lib_gld.py


#Va chercher les images et stock leur chemin dasn liste images

path_images, liste_images, titre_exp = import_images(loc,nom_exp, "LAS")

#Importe les paramètres de l'experience

facq, texp, Tmot, Vmot, Hw, Larg_ice, Long_ice, tacq, type_exp = import_param (titre_exp, date)   

mmparpixelx, mmparpixely, mmparpixelz, angle_cam_LAS, mmparpixel = import_calibration(titre_exp,date)          
              
# Creates a file txt with all the parameters of the experience and of the analysis
    
param_complets = ["Paramètres d'adimensionnement :",  "lambda_vague = " + str(lambda_vague) , "Ampvague = " + str(Ampvague) ,  "Paramètres d'analyse : ", "debut = " + str(debut) ,"kernel_size = " + str(k_size_crack), "kernel_iteration = " + str(k_iteration_crack) ,"nbframe = " + str(nbframe) , "minsize_crack = " + str(crack_lenght_min) , "sepwb = " + str(sepwb_cracks) , "size_crop = " + str(size_crop), "mmparpixely = " + str(mmparpixely), "Paramètres experimentaux : ", "facq = " + str(facq) , "texp = " + str(texp) , "Tmot = " + str(Tmot) , "Vmot = " + str(Vmot), "Hw = " + str(Hw), "Larg_ice = " + str(Larg_ice), "Long_ice = " + str(Long_ice), "tacq = " + str(tacq), "type_exp = " + str(type_exp), "nom_exp = " + str(nom_exp)]



# import_angle (date, nom_exp, loc, display = True)

#%%Paramètre de traitement



save = False
display = True



f_exc = round(Tmot)

grossissement = 2.5#CCCS 5.797850883410025DAP #import_angle(date, nom_exp, loc)

param_complets.extend([ "grossissement = " + str(grossissement) ,"f_exc = " + str(f_exc) ])

#%%Charge les données (data) et en fait l'histogramme




folder_results = path_images[:-15] + "resultats"
name_file = "positionLAS.npy"
data_originale = np.load(folder_results + "\\" + name_file)

data_originale = np.rot90(data_originale)

debut_las = 1
fin_las = np.shape(data_originale)[0] - 1


t0 = 1
tf = np.shape(data_originale)[1] - 1

if False:
    figurejolie()
    [y,x] = np.histogram((data_originale[debut_las:fin_las,t0:tf]),10000)
    xc= (x[1:]+x[:-1]) / 2
    joliplot("x (pixel)", "Position du laser (pixel)", xc,y, exp = False)
    plt.yscale('log')

# signal contient la detection de la ligne laser. C'est une matrice, avec
# dim1=position (en pixel), dim2=temps (en frame) et valeur=position
# verticale du laser en pixel.

#%% Pre-traitement

savgol = True
im_ref = False

ordre_savgol = 3
taille_savgol = 21
size_medfilt = 31

[nx,nt] = data_originale[debut_las:fin_las,t0:tf].shape


data = data_originale[debut_las:fin_las,t0:tf]


#enlever moyenne pr chaque pixel

if im_ref :
    mean_pixel = np.mean(data,axis = 1) #liste avec moyenne en temps de chaque pixel 
    for i in range (0,nt):
        data[:,i] = data[:,i] - mean_pixel #pour chaque temps, on enleve la moyenne temporelle de chaque pixel


#mise à l'échelle en m
data_m = data #*  mmparpixely / 1000

#data_m = data_m / grossissement


t = np.arange(0,nt)/facq
x = np.arange(0,nx)*mmparpixelz / 1000

signalsv = np.zeros(data.shape)

#filtre savgol

for i in range(0,nt):  
    signalsv[:,i] = savgol_filter(data_m[:,i], taille_savgol,ordre_savgol, mode = 'nearest')
    if np.mod(i,500)==0:
        print('On processe l image numero: ' + str(i) + ' sur ' + str(nt))
print('Done !')


if savgol :
    data = signalsv
else :
    data = data_m
    
if display:
    figurejolie()
    # plt.pcolormesh(t, x, data,shading='auto')
    plt.pcolormesh(data,shading='auto')
    plt.xlabel("Temps (s)")
    plt.ylabel("X (m)")
    cbar = plt.colorbar()
    cbar.set_label('Amplitude (m)')
    plt.clim(-100,100)

data_tot = data  
#%%AFFICHAGE FFT2

debut_las = 0
fin_las = 1920 - 130
t0 = 2486
tf = 4500



[nx,nt] = data_originale[debut_las:fin_las,t0:tf].shape

data = data_tot[debut_las:fin_las,t0:tf]

k_x = np.linspace(-1 / mmparpixel * 1000 / 2,1 / mmparpixel * 1000 / 2, nx)
f = np.linspace(-facq/2, facq/2, nt)

if display :
    figurejolie()
    Y_fft2 = fft.fft2(data)
    Y_fft2_shift = fft.fftshift(Y_fft2)
    plt.pcolormesh(f, k_x, np.abs(Y_fft2_shift),shading='auto')
    cbar = plt.colorbar()
    cbar.set_label('Amplitude (m)')
    plt.xlabel('f (Hz)')
    plt.ylabel(r'k $(m^{-1})$')
    plt.clim(-1,1)

if display:
    figurejolie()
    # plt.pcolormesh(t, x, data,shading='auto')
    plt.pcolormesh(data,shading='auto')
    plt.xlabel("Temps (s)")
    plt.ylabel("X (m)")
    cbar = plt.colorbar()
    cbar.set_label('Amplitude (m)')
    plt.clim(-100,100)


#%% Demodulation en TEMPS

save = False
# param_complets = param_complets.tolist()
f_exc = 5
ecart_fq = int(f_exc/10)
ecart_fexc = 1 #int(ecart_fq / facq * nt)#en points pas fréquence (1 pt = facq/nt Hz) (0.02Hz pr nt = 8000 facq = 175Hz)
amp_demod = []


fft_data_demod_tps = np.zeros(data.shape, dtype = 'complex128')
    
X = np.linspace(0, nx * mmparpixel/10, nx) #echelle des x en cm

fft_data_demod_tps[:,int(f_exc/facq*nt)-ecart_fexc:int(f_exc/facq*nt)+ecart_fexc] = Y_fft2[:,int(f_exc/facq*nt)-ecart_fexc:int(f_exc/facq*nt)+ecart_fexc]
# fft_data_demod_tps[:,nt-int(f_exc/facq*nt)-ecart_fexc:nt-int(f_exc/facq*nt)+ecart_fexc] = Y_fft2[:,nt-int(f_exc/facq*nt)-ecart_fexc:nt-int(f_exc/facq*nt)+ecart_fexc]

# fft_data_demod_tps = Y_fft2

data_dmod_tps = fft.ifft2(fft_data_demod_tps)

figurejolie()
plt.pcolormesh(np.real(data_dmod_tps), shading = 'auto')
plt.title("data démodulée à " + str(f_exc) + " Hz")

# res_10Hz = demod_sensmoinsx
# res_10Hz_moinsx = demod_sensx

# res_20Hz = demod_sensmoinsx
# res_20Hz_moinsx = demod_sensx

#%%Demodulation en ESPACE

"""On veut les ondes dans 1 sens"""

Y_FFT_tpsespace = fft.fft2(data_dmod_tps)

fft_demod_sensx = np.zeros(Y_FFT_tpsespace.shape, dtype = "complex128")

fft_demod_sensx[0:int(nx/2), :] = Y_FFT_tpsespace[0:int(nx/2), :]

demod_sensx = fft.ifft2(fft_demod_sensx)

plt.figure()
plt.pcolormesh(np.real(demod_sensx), shading ='auto')


fft_demod_sensmoinsx = np.zeros(Y_FFT_tpsespace.shape, dtype = "complex128")

fft_demod_sensmoinsx[int(nx/2):nx, :] = Y_FFT_tpsespace[int(nx/2):nx, :]

demod_sensmoinsx = fft.ifft2(fft_demod_sensmoinsx)

plt.figure()
plt.pcolormesh(np.real(demod_sensmoinsx), shading ='auto')

# y_new_x = np.zeros(np.shape(Y_FFT), dtype = 'complex128') #FFT dans le sens de propagation
# y_new_moinsx = np.zeros(np.shape(Y_FFT), dtype = 'complex128') #FFT dans le sens reflechis

# if f_0 * u == f_exc :
#     bx = int(nx/2)
#     ax = 0
# else :
#     bx = nx
#     ax = int(nx/2) + 1 
    
    
# y_new_x[ax:bx] = Y_FFT[ax:bx]

# if f_0 * u == f_exc :
#     bx = nx
#     ax = int(nx/2) + 1 
# else :
#     bx = int(nx/2)
#     ax = 0
    
# y_new_moinsx[ax:bx] = Y_FFT[ax:bx]

# demod_stat_x = fft.ifft(y_new_x)
# demod_stat_moinsx = fft.ifft(y_new_moinsx)

# def exppp(x, a, b):
#     return a * np.exp(-b * x)


# I_x = (np.abs(demod_stat_x))**2
# I_moinsx = (np.abs(demod_stat_moinsx))**2

# attenuation = curve_fit (exppp, X, I, p0 = [1,0])
# attenuationA = curve_fit (exppp, X, np.abs(amp_demod), p0 = [1,0])

# attenuation_x = curve_fit (exppp, X, I_x, p0 = [1,0])
# attenuation_moinsx = curve_fit (exppp, X, I_moinsx, p0 = [1,0])

# figurejolie()
# joliplot(r"x (cm)", r"I x", X, I_x, color = 3, exp = False, log = True, legend = r"f = " + str(int(f_exc) ) + " Hz")
# joliplot(r"x (cm)", r"I x", X, exppp(X, attenuation_x[0][0], attenuation_x[0][1]), color = 5, exp = False, log = True, legend = r"fit, on trouve $\kappa = $" + str(round(attenuation_x[0][1],4)))
# plt.xscale('linear')
# plt.yscale('log')

# figurejolie()
# joliplot(r"x (cm)", r"I moins x", X, I_moinsx, color = 3, exp = False, log = True, legend = r"f = " + str(int(f_exc) ) + " Hz")
# joliplot(r"x (cm)", r"I moins x", X, exppp(X, attenuation_moinsx[0][0], attenuation_moinsx[0][1]), color = 5, exp = False, log = True, legend = r"fit, on trouve $\kappa = $" + str(round(attenuation_moinsx[0][1],4)))
# plt.xscale('linear')
# plt.yscale('log')


# figurejolie()
# joliplot(r"x (cm)", r"I", X, I, color = 3, exp = False, log = True, legend = r"f = " + str(int(f_exc) ) + " Hz")
# joliplot(r"x (cm)", r"I", X, exppp(X, attenuation[0][0], attenuation[0][1]), color = 5, exp = False, log = True, legend = r"fit, on trouve $\kappa = $" + str(round(attenuation[0][1],4)))
# plt.xscale('linear')
# plt.yscale('log')

#%% FILM BRUT

data = np.real(demod_sensmoinsx)

nb_img_film = 300
h_film = 300 #hauteur de l'image en pixel
pourcent_img = 1.2
multiplicateur =  h_film / (np.max(data) - np.min(data)) / pourcent_img # pour que le laser bouge sur 1/pourcent_img de l'image

y = np.linspace(-h_film*mmparpixel/grossissement/multiplicateur/2,h_film*mmparpixel/grossissement/multiplicateur/2,h_film)
y = y * 1000
data_plot = data * multiplicateur
im_brut = np.zeros( (h_film,nx,nb_img_film) )
for i in range (nb_img_film) :
    for j in range (nx) :
        im_brut[int(data_plot[j,i]) + int(h_film/2) - 2: int(data_plot[j,i]) + int(h_film/2) + 2,j,i] = 255 

#%% FILM 2 COULEURs

data_1 = np.real(demod_sensmoinsx)
data_2 = np.real(demod_sensx)

nb_img_film = 300
h_film = 300 #hauteur de l'image en pixel
pourcent_img = 1.1
multiplicateur =  h_film / (np.max((data_1,data_2)) - np.min((data_1,data_2)) ) / pourcent_img # pour que le laser bouge sur 1/pourcent_img de l'image

data_plot_1 = data_1 * multiplicateur
data_plot_2 = data_2 * multiplicateur

im_brut = np.zeros( (h_film,nx,nb_img_film,3) )
for i in range (nb_img_film) :
    for j in range (nx) :
        im_brut[int(data_plot_1[j,i]) + int(h_film/2) - 2: int(data_plot_1[j,i]) + int(h_film/2) + 2,j,i,0] = 255 
        im_brut[int(data_plot_2[j,i]) + int(h_film/2) - 2: int(data_plot_2[j,i]) + int(h_film/2) + 2,j,i,1] = 255 

#%% FILM 3 COULEURS Vrai amplitude relative

data_1 = np.real(demod_sensmoinsx)
data_2 = np.real(demod_sensx)
data_3 = data_m

nb_img_film = 300
h_film = 300 #hauteur de l'image en pixel
pourcent_img = 1.1
multiplicateur =  h_film / (np.max((data_1,data_2,data_3)) - np.min((data_1,data_2,data_3)) ) / pourcent_img # pour que le laser bouge sur 1/pourcent_img de l'image

data_plot_1 = data_1 * multiplicateur
data_plot_2 = data_2 * multiplicateur
data_plot_3 = data_3 * multiplicateur

im_brut = np.zeros( (h_film,nx,nb_img_film,3) )
for i in range (nb_img_film) :
    for j in range (nx) :
        im_brut[int(data_plot_1[j,i]) + int(h_film/2) - 2: int(data_plot_1[j,i]) + int(h_film/2) + 2,j,i,0] = 255 
        im_brut[int(data_plot_2[j,i]) + int(h_film/2) - 2: int(data_plot_2[j,i]) + int(h_film/2) + 2,j,i,1] = 255 
        im_brut[int(data_plot_3[j,i]) + int(h_film/2) - 2: int(data_plot_3[j,i]) + int(h_film/2) + 2,j,i,2] = 255 

im_brut = np.int32(im_brut)

#%% FILM 3 COULEURS pas d'amp

data_1 = np.real(demod_sensmoinsx)
data_2 = np.real(demod_sensx)
data_3 = data_m

nb_img_film = 300
h_film = 300 #hauteur de l'image en pixel
pourcent_img = 1.1

multiplicateur_1 =  h_film / (np.max(data_1) - np.min(data_1) ) / pourcent_img # pour que le laser bouge sur 1/pourcent_img de l'image
multiplicateur_2 =  h_film / (np.max(data_2) - np.min(data_2) ) / pourcent_img # pour que le laser bouge sur 1/pourcent_img de l'image
multiplicateur_3 =  h_film / (np.max(data_3) - np.min(data_3) ) / pourcent_img # pour que le laser bouge sur 1/pourcent_img de l'image



data_plot_1 = data_1 * multiplicateur_1
data_plot_2 = data_2 * multiplicateur_2
data_plot_3 = data_3 * multiplicateur_3

im_brut = np.zeros( (h_film,nx,nb_img_film,3) )
for i in range (nb_img_film) :
    for j in range (nx) :
        im_brut[int(data_plot_1[j,i]) + int(h_film/2) - 2: int(data_plot_1[j,i]) + int(h_film/2) + 2,j,i,0] = 255 
        im_brut[int(data_plot_2[j,i]) + int(h_film/2) - 2: int(data_plot_2[j,i]) + int(h_film/2) + 2,j,i,1] = 255 
        im_brut[int(data_plot_3[j,i]) + int(h_film/2) - 2: int(data_plot_3[j,i]) + int(h_film/2) + 2,j,i,2] = 255 

im_brut = np.int32(im_brut)
#%%affichage film avec matplotlib grayscale
y = np.linspace(-h_film*mmparpixel/grossissement/multiplicateur/2,h_film*mmparpixel/grossissement/multiplicateur/2,h_film)
y = y * 1000


figurejolie()

plt.pcolormesh( x * 100, y, im_brut[:,:,0], shading='auto')
plt.xlabel("X (cm)")
plt.ylabel(r"Y ($\mu$m)")

for i in range (1,10): 
    figurejolie()
    plt.pcolormesh( x * 100, y, im_brut[:,:,i], shading='auto')
    plt.pause(0.1)

#%%affichage film avec matplotlib couleur

#axes nuls, on met tout à la même echelle (donc amplitud caca) pour voir

plt.imshow(im_brut[:,:,0,:])
# plt.xlabel("X (cm)")
# plt.ylabel(r"Y ($\mu$m)")

for i in range (1,50): 
    # figurejolie()
    plt.imshow(im_brut[:,:,i,:])
    plt.pause(0.1)

#%%pr save film avec axes
plot_name = "blbl.png"

plt.plot( x * 100, data_plot_1[:,0] * mmparpixel * 1000 / grossissement / multiplicateur_1)
plt.xlabel("X (cm)")
plt.ylabel(r"Y ($\mu$m)")
plt.savefig(path_images[:-15] + plot_name, dpi = 200)
plt.close()
# for i in range (1,10):
#     plt.plot( x * 100, data_plot_1[:,i]* mmparpixel * 1000 / grossissement / multiplicateur_1)
#     plt.xlabel("X (cm)")
#     plt.ylabel(r"Y ($\mu$m)")
#     plt.savefig(path_images[:-15] + plot_name, dpi = 200)
#     plt.close()
    

"""ATTENTION METTRE TOOL GRPHIC INLINE POUR QUE CA MARCHE"""

#%%Save la video avec array

frame = im_brut

fps = 5

image_temp_name = 'image.png'

# video_name = 'video_demod_' + str(f_exc) + 'Hz_' + str(fps) + 'fps_'+ str(nb_img_film) + 'images_moinsx.avi'
# video_name = 'video_demod_' + str(f_exc) + 'Hz_' + str(fps) + 'fps_'+ str(nb_img_film) + 'images.avi'
video_name = "test" + str(f_exc) + 'Hz_' + str(fps) + 'fps_'+ str(nb_img_film) + ".avi"

folder_video = path_images[:-15] + video_name

height, width, layers = frame.shape[:3]

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(folder_video, 0, fps, (width,height), isColor = True)# isColor = False)

for i in range (nb_img_film):
    cv2.imwrite(path_images[:-15] + image_temp_name, im_brut[:,:,i])
    im = cv2.imread(path_images[:-15]+ image_temp_name)#, cv2.IMREAD_GRAYSCALE)
    video.write(im)

cv2.destroyAllWindows()
video.release()

#%%Save la video avec plot 1 axe

nb_img_film = 100

frame = im_brut
plot_name = "blbl.png"

fps = 5

image_temp_name = 'image.png'

# video_name = 'video_demod_' + str(f_exc) + 'Hz_' + str(fps) + 'fps_'+ str(nb_img_film) + 'images_moinsx.avi'
# video_name = 'video_demod_' + str(f_exc) + 'Hz_' + str(fps) + 'fps_'+ str(nb_img_film) + 'images.avi'
video_name = "test_3plot1axes_" + str(f_exc) + 'Hz_' + str(fps) + 'fps_'+ str(nb_img_film) + ".avi"

folder_plot = path_images[:-15] + plot_name
folder_video = path_images[:-15] + video_name

height, width, layers = cv2.imread(folder_plot).shape[:3]

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(folder_video, 0, fps, (width,height), isColor = True)# isColor = False)

for i in range (nb_img_film):
    plt.plot( x * 100, data_plot_1[:,i]* mmparpixel * 1000 / grossissement / multiplicateur_1, label = 'Sens x démodulé')
    # plt.plot( x * 100, data_plot_2[:,i]* mmparpixel * 1000 / grossissement / multiplicateur_2, label = 'Sens -x démodulé')
    # plt.plot( x * 100, data_plot_3[:,i]* mmparpixel * 1000 / grossissement / multiplicateur_3, label = 'Signal original')
    plt.xlabel(r"X (cm)")
    plt.ylabel(r"Y ($\mu$m)")
    plt.ylim(-500,500)
    plt.legend(loc="upper left")
    plt.savefig(path_images[:-15] + plot_name, dpi = 200)
    plt.close()
    im = cv2.imread(path_images[:-15] + plot_name)#, cv2.IMREAD_GRAYSCALE)
    video.write(im)

cv2.destroyAllWindows()
video.release()

#%%Save la video avec plot 2 axes

frame = im_brut
plot_name = "blbl.png"
fps = 5

nb_img_film = 200

image_temp_name = 'image.png'

# video_name = 'video_demod_' + str(f_exc) + 'Hz_' + str(fps) + 'fps_'+ str(nb_img_film) + 'images_moinsx.avi'
# video_name = 'video_demod_' + str(f_exc) + 'Hz_' + str(fps) + 'fps_'+ str(nb_img_film) + 'images.avi'
video_name = "video_2plot2axes_" + str(f_exc) + 'Hz' + str(fps) + 'fps_'+ str(nb_img_film) + "frames.avi"

folder_plot = path_images[:-15] + plot_name
folder_video = path_images[:-15] + video_name

height, width, layers = cv2.imread(folder_plot).shape[:3]

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(folder_video, 0, fps, (width,height), isColor = True)# isColor = False)

for i in range (nb_img_film):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel(r"X (cm)")
    ax1.set_ylabel(r"Y ($\mu$m)", color='black')
    ax1.plot( x * 100, data_plot_1[:,i]* mmparpixel * 1000 / grossissement / multiplicateur_1, color=color, label = 'Sens x démodulé 40Hz')
    # ax1.plot( x * 100, data_plot_2[:,i]* mmparpixel * 1000 / grossissement / multiplicateur_2, color='green',label = 'Sens -x démodulé')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(-5, 5)
    plt.legend(loc="upper left")

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(r"Y ($\mu$m)", color=color)  
    ax2.plot( x * 100, data_plot_2[:,i]* mmparpixel * 1000 / grossissement / multiplicateur_2, color = color, label = 'Sens -x démodulé 40Hz')
    # ax2.plot( x * 100, data_plot_3[:,i]* mmparpixel * 1000 / grossissement / multiplicateur_3, color = color, label = "Signal original")
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(-2.5, 2.5)
    plt.legend(loc="upper right")

    # plt.plot( x * 100, data_plot_1[:,i]* mmparpixel * 1000 / grossissement / multiplicateur_1)
    # plt.plot( x * 100, data_plot_2[:,i]* mmparpixel * 1000 / grossissement / multiplicateur_2)
    # plt.plot( x * 100, data_plot_3[:,i]* mmparpixel * 1000 / grossissement / multiplicateur_3)
    # plt.xlabel(r"X (cm)")
    # plt.ylabel(r"Y ($\mu$m)")
    plt.savefig(path_images[:-15] + plot_name, dpi = 200)
    plt.close()
    im = cv2.imread(path_images[:-15] + plot_name)#, cv2.IMREAD_GRAYSCALE)
    video.write(im)

cv2.destroyAllWindows()
video.release()