# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 10:57:26 2022

@author: Banquise
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 16:00:24 2022

@author: Banquise
"""

"""
PROGRAMME QUI DEMODULE ET TROUVE L'AMPLITUDE ET LA LONGUEURE D4ONDE AVEC CA
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
import pickle 
import os
from PIL import Image
%run Functions_FSD.py
%run parametres_FSD.py
%run display_lib_gld.py


#Va chercher les images et stock leur chemin dasn liste images

path_images, liste_images, titre_exp = import_images(loc,nom_exp, "LAS")

#Importe les paramètres de l'experience

facq, texp, Tmot, Vmot, Hw, Larg_ice, Long_ice, tacq, type_exp = import_param (titre_exp, date)   

#Pour traiteer les expériences où la caméra était sur le coté (pendant le stage) et celle où la caméra était au dessus (Thèse)
if float(date) >= 221006 :
    cam_dessus = True     
else :
    cam_dessus = False    

mmparpixelx, mmparpixely, mmparpixelz, angle_cam_LAS, mmparpixel = import_calibration(titre_exp,date) 
              
# Creates a file txt with all the parameters of the experience and of the analysis
    
param_complets = ["Paramètres d'adimensionnement :",  "lambda_vague = " + str(lambda_vague) , "Ampvague = " + str(Ampvague) ,  "Paramètres d'analyse : ", "debut = " + str(debut) ,"kernel_size = " + str(k_size_crack), "kernel_iteration = " + str(k_iteration_crack) ,"nbframe = " + str(nbframe) , "minsize_crack = " + str(crack_lenght_min) , "sepwb = " + str(sepwb_cracks) , "size_crop = " + str(size_crop), "mmparpixely = " + str(mmparpixely), "Paramètres experimentaux : ", "facq = " + str(facq) , "texp = " + str(texp) , "Tmot = " + str(Tmot) , "Vmot = " + str(Vmot), "Hw = " + str(Hw), "Larg_ice = " + str(Larg_ice), "Long_ice = " + str(Long_ice), "tacq = " + str(tacq), "type_exp = " + str(type_exp), "nom_exp = " + str(nom_exp)]

openn_dico = True
if openn_dico :
    dico = open_dico()
    


# import_angle (date, nom_exp, loc, display = True)



#%%Paramètre de traitement



save = False
display = True



f_exc = round(Tmot)

if cam_dessus :
    grossissement = dico[date][nom_exp]["grossissement"]#CCCS2 5.63#CCCS1 5.63 #TRB 5.63#DAP 5.7978508834100255 #import_angle(date, nom_exp, loc,display = True)[0]
else :
    grossissement = 1 #car data à m'échelle est multiplié par mmparpixely qui est deja l'échelle verticale
    mmparpixel = mmparpixelz #horizontal



#%%Charge les données (data) et en fait l'histogramme




folder_results = path_images[:-15] + "resultats"
name_file = "positionLAS.npy"
data_originale = np.load(folder_results + "\\" + name_file)

if True : #cam_dessus :
    data_originale = np.rot90(data_originale)
    
else :
    data_originale_X = np.zeros(data_originale.shape)
    for i in range (data_originale.shape[0]):
        data_originale_X[i,:] = data_originale[-i,:]
    data_originale = data_originale_X

debut_las = 100
fin_las = np.shape(data_originale)[0] - 200
#100 / 400 Pour DAP
#650 / 300 Pour CCCS1

#1 / 200 pr CCCS2

t0 = 1
tf = np.shape(data_originale)[1] - 1

if display:
    figurejolie()
    [y,x] = np.histogram((data_originale[debut_las:fin_las,t0:tf]),10000)
    xc= (x[1:]+x[:-1]) / 2
    joliplot("y (pixel)", "Position du laser (pixel)", xc,y, exp = False)
    plt.yscale('log')

#%%TRAITEMENT DU SIGNAL
# signal contient la detection de la ligne laser. C'est une matrice, avec
# dim1=position (en pixel), dim2=temps (en frame) et valeur=position
# verticale du laser en pixel.50#%% Pre-traitement

savgol = True
im_ref = False
minus_mean = True

ordre_savgol = 2
taille_savgol = 11
size_medfilt = 51

[nx,nt] = data_originale[debut_las:fin_las,t0:tf].shape


data = data_originale[debut_las:fin_las,t0:tf]


#enlever moyenne pr chaque pixel

if im_ref :
    mean_pixel = np.mean(data,axis = 1) #liste avec moyenne en temps de chaque pixel 
    for i in range (0,nt):
        data[:,i] = data[:,i] - mean_pixel #pour chaque temps, on enleve la moyenne temporelle de chaque pixel

if minus_mean :
    data = data - np.mean(data)
#mise à l'échelle en m
data_m = data *  mmparpixely / 1000

data_m = data_m / grossissement


t = np.arange(0,nt)/facq
x = np.arange(0,nx)*mmparpixelz / 1000

signalsv = np.zeros(data.shape)

#filtre savgol

for i in range(0,nt):  
    signalsv[:,i] = savgol_filter(data_m[:,i], taille_savgol,ordre_savgol, mode = 'nearest')
    if np.mod(i,1000)==0:
        print('On processe l image numero: ' + str(i) + ' sur ' + str(nt))
print('Done !')


if savgol :
    data = signalsv.copy()
else :
    data = data_m.copy()
    
if display:
    figurejolie()
    # plt.pcolormesh(t, x, data,shading='auto')
    plt.pcolormesh(data,shading='auto')
    plt.xlabel("Temps (frames)")
    plt.ylabel("X (pixel)")
    cbar = plt.colorbar()
    cbar.set_label('Amplitude (m)')
    plt.clim(-0.008,0.008)

    
#%% Analyse d'un signal temporel
if True:
    
    #On prend un point qcq en x, et on regarde le signal au cours du temps.
    figurejolie()
    i = 200
    
    # On va regarder la periode sur signal.
    
    Y1 = fft.fft(data[i,:]- np.mean(data[i,:]))
    
    P2 = abs(Y1/nt)
    P1 = P2[1:int(nt/2+1)]
    P1[2:-1] = 2*P1[2:-1]
    
    
    f = facq * np.arange(0,int(nt/2)) / nt 
    joliplot('f (Hz)', '|P1(f)|', f, P1, title = 'Single-Sided Amplitude Spectrum of X(t)', exp = False)
    


#%% Analyse avec les harmoniques
if True:
    longueure_tps = 1000 #temps pour découper la FFT temporelle
    
    #taux distortion harmonique https://fr.wikipedia.org/wiki/Taux_de_distorsion_harmonique
    THD_espace = []
    THD_tps = []
    tps= []
    
    for w in range(int(nt/longueure_tps)):#(nx)
    #On prend un point qcq en x, et on regarde le signal au cours du temps.
        n0 = 300 #t0
        # On va regarder la periode sur signal.
        Y1 = np.mean(fft.fft2(data[n0:,w * longueure_tps:(w+1) * longueure_tps]- np.mean(data[n0:,w * longueure_tps:(w+1) * longueure_tps])), axis = 0)
        
        P2 = abs(Y1/longueure_tps)
        P1 = P2[1:int(longueure_tps/2+1)]
        P1[2:-1] = 2*P1[2:-1]
        
        f = facq * np.arange(0,int(longueure_tps/2)) / longueure_tps 
        prominence = np.max(P1)
        qtt_pics = 20
        amp_FFT , _ = find_peaks(abs(P1[0:]), prominence = [prominence/qtt_pics,None], distance = longueure_tps/qtt_pics, width=(1,15))
        if len(amp_FFT) > 1 :
            # tps.append(w * longueure_tps)
            THD_tps.append( np.sqrt(P1[int(f_exc / facq * longueure_tps)]**2/(np.sum(P1[amp_FFT[:5]]**2))))
        else :
            THD_tps.append(1)
            
        # figurejolie()
        # joliplot('','',amp_FFT * facq/longueure_tps, P1[amp_FFT], color = 2)
        # joliplot('f (Hz)', '|P1(f)|', f, P1, title = 'Single-Sided Amplitude Spectrum of X(t) ' + str(i), exp = False)
        
    nx_new = nx
    for i in range (nx_new):
    #On prend un point qcq en x, et on regarde le signal au cours du temps.

        
        # On va regarder la periode sur signal.
        nt_max = nt
        f = facq * np.arange(0,int(nt_max/2)) / nt_max 
        
        Y1_x = fft.fft(data[i,:nt_max]- np.mean(data[i,:nt_max]))
        
        P2 = abs(Y1_x/nt_max)
        P1 = P2[1:int(nt_max/2+1)]
        P1[2:-1] = 2*P1[2:-1]
        
        f = facq * np.arange(0,int(nt_max/2)) / nt_max 
        prominence = np.max(P1)
        qtt_pics = 20
        amp_FFT_x , _ = find_peaks(abs(P1[:]), prominence = [prominence/qtt_pics,None], distance = nt_max/qtt_pics, width=(1,15))
        THD_espace.append(np.sqrt(P1[int(f_exc / facq * nt_max)]**2/np.sum(P1[amp_FFT_x[:5]]**2)))
        
        # figurejolie()
        # joliplot('','',amp_FFT_x * facq/nt_max, P1[amp_FFT_x], color = 2)
        # joliplot('f (Hz)', '|P1(f)|', f, P1, title = 'Single-Sided Amplitude Spectrum of X(t) ' + str(i), exp = False)
        

    figurejolie()
    joliplot('','',amp_FFT_x * facq/nt_max, P1[amp_FFT_x], color = 2)
    joliplot('f (Hz)', '|P1(f)|', f, P1, title = 'Single-Sided Amplitude Spectrum of X(t)', exp = False)
    
    #Part d'harmonique en fct du tps
    tps = np.linspace(0,int(nt/longueure_tps),int(nt/longueure_tps))
    figurejolie()
    joliplot('t','Principal sur harmonique, x de ' + str(n0) + " à nx", tps, THD_tps, exp = False )
    
    #Part d'harmonique en fct de l'espace
    xxx = np.linspace(0,nx_new,nx_new)
    figurejolie()
    joliplot('x','Principal sur harmonique, t de 0 à ' + str(nt_max), xxx, THD_espace,legend = "THD(espace)", exp = False)
    
    #part d'harmonique total
    nt_max = 6000
    Y_tot = np.mean(fft.fft2(data[:,:nt_max]- np.mean(data[:,:nt_max])), axis = 0)
    # Y_tot = fft.fft(data[10,:nt_max]- np.mean(data[10,:nt_max]))
    P2 = abs(Y_tot/nt_max)
    P1 = P2[1:int(nt_max/2+1)]
    P1[2:-1] = 2*P1[2:-1]
    
    f = facq * np.arange(0,int(nt_max/2)) / nt_max 
    prominence = np.max(P1)
    qtt_pics = 20
    amp_FFT_tot , _ = find_peaks(abs(P1[:]), prominence = [prominence/qtt_pics,None], distance = nt_max/qtt_pics, width=(1,15))
    THD_tot = np.sqrt(P1[int(f_exc / facq * nt_max)]**2/np.sum(P1[amp_FFT_tot[:5]]**2))
    
    print ("THD sur nx entre t = 0  et t = " + str(nt_max) + " : ", THD_tot  )
    
    figurejolie()
    joliplot('','',amp_FFT_tot * facq/nt_max, P1[amp_FFT_tot], color = 2)
    joliplot('f (Hz)', '|P1(f)|', f, P1, title = 'FFT X total, THD = '+ str(round(THD_tot,3)) + ", t de 0 à " + str(nt_max), exp = False)
    
                                                      
                        

#%%AFFICHAGE FFT2

[nx,nt] = data_originale[debut_las:fin_las,t0:tf].shape

k_x = np.linspace(-1 / mmparpixel * 1000 / 2,1 / mmparpixel * 1000 / 2, nx)
f = np.linspace(-facq/2, facq/2, nt)

figurejolie()
Y_fft2 = fft.fft2(data)
Y_fft2_shift = fft.fftshift(Y_fft2)
plt.pcolormesh(f, k_x, np.abs(Y_fft2_shift),shading='auto')
cbar = plt.colorbar()
cbar.set_label('Amplitude (mm)')
plt.xlabel('f (Hz)')
plt.ylabel(r'k $(m^{-1})$')
plt.clim(-1,1)

#%%AMPLITUDE AVCE LE TEMPS

#Moyenne temporelle amp aux moments interessants, et avec le temps
longueur_donde = int(dico[date][nom_exp]['lambda'] / mmparpixel * 1000 / 2)
temps = 6400
posx = 420
plage_x = longueur_donde
fexc = dico[date][nom_exp]['fexc']
strobo = int(facq/fexc) + 1

figurejolie()
for i in range (20):
    joliplot("X","A",x, data[:,temps + i * strobo * 5], exp = False, title = "A(t), t = " + str(round(temps)) + " frame, 10 périodes")

amp_1periode = []
t_amp = []
periodes = 5 #nb de periodes par pt de mesure
img_par_periode = (int((facq/fexc) * periodes) + 1)
nb_periode = int(nt/img_par_periode) #nb de pts où on moyenne l'amplitude
for j in range (nb_periode):
    t_amp.append(j * img_par_periode)
    amp_1periode.append(np.mean(np.amax(data[posx:posx + plage_x, j * img_par_periode:(j+1) * img_par_periode], axis = 1) - np.amin(data[posx:posx + plage_x, j * img_par_periode:(j+1) * img_par_periode], axis = 1)))
figurejolie()
Amp_t = amp_1periode
joliplot('t (frame)','Amplitude (m)',t_amp,Amp_t,exp = False , color = 3, legend = "Amp(t) entre " + str(posx) + " et " + str(posx + plage_x))

#%%Amp(t) avec un pixel
pixel = 526
tps_0 = 0
plage_tps = 10000
t_pixel = np.linspace(tps_0, tps_0 + plage_tps, plage_tps)
figurejolie()
joliplot("T",'Y',t_pixel, data[pixel, tps_0:tps_0 + plage_tps], exp = False, color = 3, legend = "Y(t) au temps " + str(tps_0) + " et x = " + str(pixel))
amp_pixel = (np.max(data[pixel, tps_0:tps_0 + plage_tps]) - np.min(data[pixel, tps_0:tps_0 + plage_tps]))/2


#%%Amp signal stationnaire
fexc = dico[date][nom_exp]['fexc']
x_0sta = 1000
x_fsta = 1200
t_0sta = 6000
t_fsta = 6100
amp_moy_x = np.mean(np.max(data[x_0sta:x_fsta, t_0sta:t_fsta], axis = 1) - np.min(data[x_0sta:x_fsta, t_0sta:t_fsta], axis = 1))/2
amp_moy_t = np.mean(np.max(data[x_0sta:x_fsta, t_0sta:t_fsta], axis = 0) - np.min(data[x_0sta:x_fsta, t_0sta:t_fsta], axis = 0))/2
uiui = np.where(np.max(data[x_0sta:x_fsta, t_0sta:t_fsta]) == data)
x_amp_max = uiui[0][0]
t_amp_max = uiui[1][0]
minimum_amp_autour_max = np.min(data[x_amp_max,t_amp_max-int(facq/fexc+1):t_amp_max+int(facq/fexc+1)])
x_amp_min = np.where(data == minimum_amp_autour_max)[0][0]
t_amp_min = np.where(data == minimum_amp_autour_max)[1][0]
amp_max_amp = (data[x_amp_max, t_amp_max] - data[x_amp_min, t_amp_min])/2
amp_moy_tot = (np.max(data[x_0sta:x_fsta, t_0sta:t_fsta]) - np.min(data[x_0sta:x_fsta, t_0sta:t_fsta]))/2
print('amplitude moyenne en espace = ', amp_moy_x)
print('amplitude moyenne en temps = ', amp_moy_t)
print("amplitude max calée = ", amp_max_amp)
print('amplitude max = ', amp_moy_tot)

if display :
    figurejolie()
    joliplot('x','amp',np.arange(x_0sta,x_fsta,1), (np.max(data[x_0sta:x_fsta, t_0sta:t_fsta], axis = 1) - np.min(data[x_0sta:x_fsta, t_0sta:t_fsta], axis = 1))/2, exp = False, legend = 'amp(x)')
    figurejolie()
    joliplot('t','amp',np.arange(t_0sta,t_fsta,1), (np.max(data[x_0sta:x_fsta, t_0sta:t_fsta], axis = 0) - np.min(data[x_0sta:x_fsta, t_0sta:t_fsta], axis = 0))/2, exp = False, legend = "amp(t)")


if False : 
    amplitudes = np.asarray( ["amp_moy_x = " + str(amp_moy_x), "x_0sta = " + str(x_0sta),"x_fsta = " + str(x_fsta),"t0 = " + str(t0),"debut_las = " + str(debut_las),"t_0sta = " + str(t_0sta),"t_fsta = " + str(t_fsta),"amp_moy_tot = " + str(amp_moy_tot),"amp_moy_t = " + str(amp_moy_t)])
    np.savetxt(path_images[:-15] + "resultats" + "/Amp_max_et_moyenne_4" + name_fig_FFT + ".txt", amplitudes, "%s")

#%%Detection t fractures (lent et pas précis)

from skimage import measure
from skimage import morphology
from skimage import feature

fexc = dico[date][nom_exp]['fexc']
nnn = 3
taille_sin = int(facq/fexc * nnn)
zone = nx

x_sin = np.linspace(0,2 * np.pi * nnn, taille_sin)
sinsin = np.sin(x_sin)
sin_kernel = np.zeros((zone, taille_sin))

# figurejolie()
# joliplot('x','sin', x_sin, sinsin, exp = False, legend = 'signal sinus pour ' + str(nnn) + 'periodes')


for i in range (zone) :
    sin_kernel[i] = sinsin / (zone * taille_sin)
    
# sin_kernel = data[140:220,220:240]
sin_kernel = np.ones((taille_sin,taille_sin))/(taille_sin*taille_sin)
# sin_kernel = sinsin / taille_sin
# figurejolie()
# plt.imshow(sin_kernel)

tii = 0
tff = nt#nttii + 40000

data_conv = np.abs(convolve2d(data[:zone,tii:tff], sin_kernel, mode='same', boundary='fill'))

figurejolie()
plt.pcolormesh(data_conv,shading='auto')
plt.xlabel("Temps (frames)")
plt.ylabel("X (pixel)")
cbar = plt.colorbar()
cbar.set_label('Amplitude (m)')
plt.title("resultat conv")
# plt.clim(-0.008,0.008)

figurejolie()
plt.pcolormesh(data[:zone,tii:tff],shading='auto')
plt.xlabel("Temps (frames)")
plt.ylabel("X (pixel)")
cbar = plt.colorbar()
cbar.set_label('Amplitude (m)')
plt.clim(-0.008,0.008)
plt.title('data')

binary = data_conv > 0.0006

figurejolie()
plt.pcolormesh(binary)
plt.title("binary")

contours = measure.find_contours(binary)


figurejolie()
plt.pcolormesh(binary,shading='auto')
for contour in contours :
    plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
plt.title('contour binary')
    
binary_label = measure.label(binary)
props = measure.regionprops(binary_label)

t_debut_frac = np.min(contours[0][:,1]) #attention pas sur que contour[0] soit le contour interessant

#%% Derivée diagramme a(x,t)
zone = nx
tii = 0
tff = nt#nttii + 40000

from scipy.ndimage.filters import convolve,gaussian_filter

sobelX = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
sobelY = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
derivX = convolve(data[:zone,tii:tff],sobelX)
derivY = convolve(data[:zone,tii:tff],sobelY)
gradient = derivX+derivY*1j
G = np.abs(gradient)


# figurejolie()
# plt.pcolormesh(derivX,shading='auto')
# plt.title('derivX')


# figurejolie()
# plt.pcolormesh(derivY,shading='auto')
# plt.title('derivY')


# figurejolie()
# plt.pcolormesh(G,shading='auto')
# plt.title('G')


#%% Erod-dilate et binarise

#Binarise
seuil = 0.015#0.015MPBF4

bin_G = G < seuil
bin_G = np.array(bin_G, dtype = "uint8")
bin_G = bin_G * 255

#Double erod dilate pour garder les gros bouts

kernel_iteration = 5
kernel_size = 8

erod_G = erodedilate(bin_G, kernel_iteration, kernel_size)

erod_G = cv2.bitwise_not(erod_G)

erod_G_2 = erodedilate(erod_G, kernel_iteration, kernel_size)

figurejolie()
plt.pcolormesh(erod_G_2,shading='auto')
plt.title('erod-G-2')

#%% detecton contours
frac_auto = True

if frac_auto :
    n_fracs = 5 #nombre de fractures à detecter
    
    fracs = np.zeros((n_fracs,2))
    
    #trouve les contours et garde les n_fracs plus gros (en aire)
    
    contours, hierarchy = cv2.findContours(erod_G_2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_0 = contours # if imutils.is_cv2() else contours[1]  
    cntsSorted = sorted(contours_0, key=lambda x: cv2.contourArea(x))
    
    big_contours = cntsSorted[-n_fracs:]
    
    img_contours_binaire = np.zeros(np.append(data.shape, 3), np.uint8)
    
    cv2.drawContours(img_contours_binaire,big_contours,-1, color = (255,255,255), thickness = 4)
    
    figurejolie()
    plt.pcolormesh(img_contours_binaire[:,:,0],shading='auto')
    
    #trouve le debut en t et le milieu en x des ces contours ce qui donne t_0 et x_0 frac
    for i in range (len(big_contours)):
        # dans fracs on a les t_0 et x_0 des fractures
        fracs[i,0] = np.min(big_contours[i][:,:,0])
        fracs[i,1] = np.mean(big_contours[i][:,:,1])
else :
    n_fracs = 6
    fracs = np.zeros((n_fracs,2))
    fracs[0,0] = 3270 # t_0_frac 
    fracs[0,1] = 526 # x_0_frac !!!!!!! ATTENTION x imageJ = nx- x pour python !!!!!!!
    fracs[1,0] = 3750
    fracs[1,1] = 640
    fracs[2,0] = 3730
    fracs[2,1] = 848
    fracs[3,0] = 3700
    fracs[3,1] = 950
    fracs[4,0] = 5700
    fracs[4,1] = 1271
    fracs[5,0] = 5450
    fracs[5,1] = 1574
    
    

#%% Amp moyenne avant frac
n_periodes = 10
padding = 12
padpad = True
n_periodesavantfracture = 1

save_fig = False

liste_noeuds = [0,2]
liste_frac = [0,1,2,3,4,5]


fexc = dico[date][nom_exp]['fexc'] # Hz
facq = dico[date][nom_exp]['facq'] # Hz
longueur_donde = int(dico[date][nom_exp]['lambda'] / mmparpixel * 1000 / 2) # pixel
t_nperiodes = (int(facq/fexc) + 1) * n_periodes #durée regardée

amp_fracs_complet = np.zeros((n_fracs,2,longueur_donde,2 ), dtype = np.ndarray)

amp_fracs_fft = np.zeros((n_fracs,8)) #pour sauvegarder amp(f_0), idem avec onde autre sens, t_0, et x(amp_max), et amp_RMS


for j in range(n_fracs) :#fracs.shape[0]) :
    
    #AMP AVEC FFT
    if padpad :
        t_nperiodes = (int(facq/fexc) + 1) * n_periodes
    
    x_0_frac = int(fracs[j,1] - longueur_donde/2)
    x_f_frac = int(fracs[j,1] + longueur_donde/2)
    t_n_frac = int(fracs[j,0] - t_nperiodes - int(t_nperiodes/n_periodes * n_periodesavantfracture))
    t_0_frac = int(fracs[j,0] - int(t_nperiodes/n_periodes * n_periodesavantfracture) )
    
    amp_fracs_complet = np.zeros((2,x_f_frac - x_0_frac,2 ), dtype = np.ndarray)
    
    figurejolie()
    sum_real = []
    sum_FFT = []
    
    
    
    for i in range (x_0_frac, x_f_frac):
        meann = np.mean(data[i,t_n_frac:t_0_frac])
        
        data_used = data[i,t_n_frac:t_0_frac] - meann #data qui nous interesse
        
        Y1 = fft.fft(data_used)
        # Y_12 = Y1.copy()
        if padpad :
            data_padding = np.append(data_used, np.zeros(2**padding - (t_0_frac-t_n_frac)))
            Y1_padding = fft.fft(data_padding)
            Y1 = Y1_padding.copy()
    
            
        Y_t = Y1[:int(len(Y1)/2)] #sens t
        Y_moinst = Y1[int(len(Y1)/2):] #sens moins t
        
        #normalisation
        P2_t = np.abs(Y_t)#/ (t_0_frac-t_n_frac))  
        P2_moinst = np.abs(Y_moinst)
        
        P2 = np.append(P2_t,P2_moinst) #abs(FFT)
        if padpad :
            t_nperiodes = 2**padding
                
        f = facq * np.arange(0,int(t_nperiodes)) / t_nperiodes    
        
        peaks, _ = find_peaks(P2,prominence = np.max(P2)/4, width=(1,50))
    
        peaks_select = np.append(peaks[:1], peaks[-1:]) #que le fondamental et le retour
        
        # peaks_select = np.append(np.argmax(P2_t), np.argmax(P2_moinst))
        
        amp_fracs_complet[0,i - x_0_frac, 0] = peaks_select[0]
        amp_fracs_complet[0,i - x_0_frac, 1] = peaks_select[1] #+ int(len(Y1)/2)
        amp_fracs_complet[1,i - x_0_frac, 0] = P2[peaks_select][0] #P2_t[peaks_select][0]
        amp_fracs_complet[1,i - x_0_frac, 1] = P2[peaks_select][0] #P2_moinst[peaks_select][1]  

        dt = 1/ facq
        sum_amp_reelle = np.sum ( np.abs(data_used) **2) / (t_0_frac-t_n_frac)
        sum_real.append(sum_amp_reelle)
        
        # df = facq / t_nperiodes
        sum_amp_FFT = np.sum(np.abs(P2 /np.sqrt(t_0_frac-t_n_frac) )**2 ) / t_nperiodes
        sum_FFT.append(sum_amp_FFT)
            
                
                
        if np.mod(i,20)==0:
            joliplot("f (Hz)", "Amplitude (m)", f, P2 / (t_0_frac-t_n_frac), exp = False)
            plt.plot(amp_fracs_complet[0,i - x_0_frac] * facq /t_nperiodes, amp_fracs_complet[1,i - x_0_frac] / (t_0_frac-t_n_frac), 'ro', label = "pics trouvés")
    
    amp_fracs_fft[j,0] = t_0_frac #t_0_frac
    amp_fracs_fft[j,1] = np.argmax(amp_fracs_complet[1,:,0]) + x_0_frac #là où on trouve le max, donc x_0 frac (en supposant que ca casse sur le max)
    amp_fracs_fft[j,2] = np.max(amp_fracs_complet[1,:,0]) / (t_0_frac-t_n_frac) #amp(f_0)
    amp_fracs_fft[j,3] = np.max(amp_fracs_complet[1,:,1]) / (t_0_frac-t_n_frac) #amp(f_i)
    amp_fracs_fft[j,4] = np.sqrt(sum_real[int(amp_fracs_fft[j,1] - x_0_frac)]) #amp RMS là où le pic FFT est le plus fort
    amp_fracs_fft[j,5] = amp_fracs_fft[j,2]/ amp_fracs_fft[j,4] #part d'energie dans le pic 
    amp_fracs_fft[j,6] = False #si ca casse sur cette fracture
    amp_fracs_fft[j,7] = False #si c'est peut être sur un noeud
    if j in liste_frac :
        amp_fracs_fft[j,6] = True #si ca casse sur cette fracture
    if j in liste_noeuds :
        amp_fracs_fft[j,7] = True #si c'est peut être sur un noeud


    plt.plot(amp_fracs_complet[0,i - x_0_frac] * facq /t_nperiodes, np.append(amp_fracs_fft[j,2], amp_fracs_fft[j,3]), 'mx', label = "maxs")
    if save_fig :
        plt.savefig(path_images[:-15] + "resultats" + "/" + "FFT_avec_pics_trouves_fracture_" + str(j) + ".pdf", dpi = 1)
        
    figurejolie()
        
    if padpad :
        t_nperiodes = (int(facq/fexc) + 1) * n_periodes
    
    x_amp = np.linspace(0, mmparpixel * (x_f_frac - x_0_frac) / 1000, x_f_frac - x_0_frac)

    joliplot('X (m)','Amplitude RMS (m)',x_amp, np.sqrt(sum_real), legend = 'amp RMS(x) frac numero ' + str(j), exp = False, color = 3 )
        
    # plt.plot( (amp_fracs_fft[j,1] - x_0_frac) * mmparpixel / 1000, amp_fracs_fft[j,2], 'ko')
    plt.plot( (amp_fracs_fft[j,1] - x_0_frac) * mmparpixel / 1000, amp_fracs_fft[j,3], 'ro', label = "pic max FFT")
    plt.plot( (amp_fracs_fft[j,1] - x_0_frac) * mmparpixel / 1000, amp_fracs_fft[j,4], 'go', label = "pic max RMS")
    plt.legend()
     
    if save_fig :
        plt.savefig(path_images[:-15] + "resultats" + "/" + "Amplitude RMS (x)_fracture_" + str(j) + ".pdf", dpi = 1)
    figurejolie()
    t_pixel_max = np.linspace(amp_fracs_fft[j,0] - t_nperiodes, amp_fracs_fft[j,0], t_nperiodes)
    x_pixel_max = int(amp_fracs_fft[j,1])
    meann = np.mean(data[x_pixel_max,t_n_frac:t_0_frac])
    data_used = data[x_pixel_max,t_n_frac:t_0_frac] - meann
    joliplot("t","x", t_pixel_max, data_used, exp = False, color = 3, legend = 'Y(t) sur la crete, de t = ' + str(round(t_pixel_max[0])) + " à t = " + str(round(t_pixel_max[-1])) + " pour x = " + str(x_pixel_max))
    if save_fig :
        plt.savefig(path_images[:-15] + "resultats" + "/" + "amp(t)_x" + str(int(amp_fracs_fft[j,1])) + "_fracture_" + str(j) + ".pdf", dpi = 1)
    print("Parseval = ",np.mean(np.array(sum_FFT)/np.array(sum_real)))
    # figurejolie()
    # plt.plot(sum_FFT, label = 'sum FFT')
    # plt.plot(sum_real, label = 'sum reelle')
    # plt.legend()
    
    
print(amp_fracs_fft) #t_0, x_0, pic 1, pic 2, amp RMS pour chaque fracture

#%% SAVE AMP
save = False
commentaire = "amp max trouvée pour 6 modes visibles, dont 2 noeuds (0 et 2), tt casse"

zone  = nx
tii = 0 
tff = nt
seuil = 0.015

if save :
    dico = open_dico()
    param_amp = []
    param_amp.extend([ "Traitement derivation : "])    
    param_amp.extend([ "zone = " + str(zone),"seuil = " + str(seuil),"kernel_iteration = " + str(kernel_iteration),"kernel_size = " + str(kernel_size),"tii = " + str(tii),"tff = " + str(tff), "frac_auto = " + str(frac_auto) ])
    param_amp.extend([ "Traitement amplitude : "]) 
    param_amp.extend([ 'n_fracs' + str(n_fracs), "padpad = " + str(padpad) ,"padding = " + str(padding),"n_periodes = " + str(n_periodes),"n_periodesavantfracture = " + str(n_periodesavantfracture), 'commentaire = ' + str(commentaire) ])
    param_amp = np.asarray(param_amp)
    np.savetxt(path_images[:-15] + "resultats" + "/Paramètres_FFT_amplitude" + name_fig_FFT + ".txt", param_amp, "%s")
    dico = add_dico(dico, date, nom_exp, 'n_fracs', n_fracs)
    dico = add_dico(dico, date, nom_exp, 'amp_fracs_fft', amp_fracs_fft) #t_0, x_0, pic 1, pic 2, amp RMS, rapport de pic max sur RMS pour chaque fracture
    dico = add_dico(dico, date, nom_exp, 'Parametres_FFT_amplitude', param_amp)
    save_dico(dico)
if save :
    param_complets = ["Paramètres d'adimensionnement :",  "lambda_vague = " + str(lambda_vague) , "Ampvague = " + str(Ampvague) ,  "Paramètres d'analyse : ", "debut = " + str(debut) ,"kernel_size = " + str(k_size_crack), "kernel_iteration = " + str(k_iteration_crack) ,"nbframe = " + str(nbframe) , "minsize_crack = " + str(crack_lenght_min) , "sepwb = " + str(sepwb_cracks) , "size_crop = " + str(size_crop), "mmparpixely = " + str(mmparpixely), "Paramètres experimentaux : ", "facq = " + str(facq) , "texp = " + str(texp) , "Tmot = " + str(Tmot) , "Vmot = " + str(Vmot), "Hw = " + str(Hw), "Larg_ice = " + str(Larg_ice), "Long_ice = " + str(Long_ice), "tacq = " + str(tacq), "type_exp = " + str(type_exp), "nom_exp = " + str(nom_exp)]
    param_complets.extend([ "grossissement = " + str(grossissement) ,"f_exc = " + str(f_exc) ])
    param_complets.extend(["Paramètres de traitement :",  "debut_las = " + str(debut_las) ,"fin_las = " + str(fin_las),"t0 = " + str(t0) ,"tf = " + str(tf) ])
    param_complets.extend(["savgol = " + str(savgol) , "taille_savgol" + str(taille_savgol), "im_ref = " + str(im_ref)])
    param_complets = np.asarray(param_complets)
    np.savetxt(path_images[:-15] + "resultats" + "/Paramètres_FFT_globaux_amplitude" + name_fig_FFT + ".txt", param_complets, "%s")

#%% Demodulation et amplitude

f_0 = fexc
u = 1
t = np.arange(0,nt)/facq

save = False
# param_complets = param_complets.tolist()
f_exc = f_0 * u
amp_demod = []
cut_las = 0
if f_0 * u > 80 :
    cut_las = 500
if f_0 * u > 120 :
    cut_las = 700
    
nx = nx - cut_las
X = np.linspace(0, nx * mmparpixel/10, nx) #echelle des x en cm

#demodulation
for i in range (nx):
    a = data[i,:]
    amp_demod.append(np.sum(a * np.exp(2*np.pi*1j*f_exc*t))*2/nx)
    
if False : #display : 
    figurejolie()
    joliplot("temps(s)", "Amplitude (m)", t,a, exp = False)
 

amp_demod = np.asarray(amp_demod)
I = (np.abs(amp_demod))**2 #tableau intensite (avec amp en m)
 
if True : #display :
    figurejolie()
    joliplot(r"x (cm)",r"amplitude (m)", X, np.abs(amp_demod), exp = False, log = False, legend = r"f = " + str(int(f_exc) ) + " Hz")

Y_FFT = fft.fft(amp_demod)

y_new_x = np.zeros(np.shape(Y_FFT), dtype = 'complex128') #FFT dans le sens de propagation
y_new_moinsx = np.zeros(np.shape(Y_FFT), dtype = 'complex128') #FFT dans le sens reflechis

if f_0 * u == f_exc :
    bx = int(nx/2)
    ax = 0
else :
    bx = nx
    ax = int(nx/2) + 1 
    
    
y_new_x[ax:bx] = Y_FFT[ax:bx]

if f_0 * u == f_exc :
    bx = nx
    ax = int(nx/2) + 1 
else :
    bx = int(nx/2)
    ax = 0
    
y_new_moinsx[ax:bx] = Y_FFT[ax:bx]

if True :
    figurejolie()
    plt.plot(np.abs(y_new_x))
    figurejolie()
    plt.plot(np.abs(y_new_moinsx))



demod_stat_x = fft.ifft(y_new_x)
demod_stat_moinsx = fft.ifft(y_new_moinsx)

if True :
    figurejolie()
    plt.plot(X, np.abs(demod_stat_x))
    plt.title("FFT inverse sens de propagation")
    
    figurejolie()
    plt.plot(X, np.abs(demod_stat_moinsx))
    plt.title("FFT inverse onde réflechie")    


def exppp(x, a, b):
    return a * np.exp(-b * x)


I_x = (np.abs(demod_stat_x))**2
I_moinsx = (np.abs(demod_stat_moinsx))**2

attenuation = curve_fit (exppp, X, I, p0 = [1,0])
attenuationA = curve_fit (exppp, X, np.abs(amp_demod), p0 = [1,0.02])

attenuation_x = curve_fit (exppp, X, I_x, p0 = [1,0])
# attenuation_moinsx = curve_fit (exppp, X, I_moinsx, p0 = [0,0.02])

figurejolie()
joliplot(r"x (cm)", r"I x", X, I_x, color = 3, exp = False, log = True, legend = r"f = " + str(int(f_exc) ) + " Hz")
joliplot(r"x (cm)", r"I x", X, exppp(X, attenuation_x[0][0], attenuation_x[0][1]), color = 5, exp = False, log = True, legend = r"fit, on trouve $\kappa = $" + str(round(attenuation_x[0][1],4)))
plt.xscale('linear')
plt.yscale('log')

# figurejolie()
# joliplot(r"x (cm)", r"I moins x", X, I_moinsx, color = 3, exp = False, log = True, legend = r"f = " + str(int(f_exc) ) + " Hz")
# joliplot(r"x (cm)", r"I moins x", X, exppp(X, attenuation_moinsx[0][0], attenuation_moinsx[0][1]), color = 5, exp = False, log = True, legend = r"fit, on trouve $\kappa = $" + str(round(attenuation_moinsx[0][1],4)))
# plt.xscale('linear')
# plt.yscale('log')

if display :
    figurejolie()
    joliplot(r"x (cm)", r"I", X, I, color = 3, exp = False, log = True, legend = r"f = " + str(int(f_exc) ) + " Hz")
    joliplot(r"x (cm)", r"I", X, exppp(X, attenuation[0][0], attenuation[0][1]), color = 5, exp = False, log = True, legend = r"fit, on trouve $\kappa = $" + str(round(attenuation[0][1],4)))
    plt.xscale('linear')
    plt.yscale('log')
    if save :
        plt.savefig(path_images[:-15] + "resultats" + "/" + "I(x)_fitkappa_" + str(round(f_exc)) + "Hz" + ".pdf", dpi = 1)
if save :  
    param_complets.extend(["Resultats attenuation :","f_exc = " + str(f_exc) ,"attenuation = " + str(attenuation[0][1])])

#%% Longueure d'onde

save = False

padding = 12

ampfoispente = np.append( amp_demod * np.exp(attenuationA[0][1] * X), np.zeros(2**padding - nx))
ampfoispente_0 = np.append( (amp_demod) , np.zeros(2**padding - nx))

figurejolie()
joliplot("X (cm)", "Signal", X, np.real(ampfoispente[:nx]),color =2, legend = r'Signal démodulé * atténuation (m)', exp = False)
joliplot("X (cm)", "Signal", X, np.real(ampfoispente_0[:nx]),color = 10, legend = r'Signal démodulé', exp = False)
if save :
    plt.savefig(path_images[:-15] + "resultats" + "/" + "Partie réelle_Signal_démodulé_" + str(round(f_exc)) + "Hz" + ".pdf", dpi = 1)
    


FFT_demod_padding = fft.fft((ampfoispente)-np.mean((ampfoispente)))
FFT_demod_padding_0 = fft.fft((ampfoispente_0)-np.mean((ampfoispente_0)))
FFT_demod = fft.fft((ampfoispente_0[:nx])-np.mean((ampfoispente_0[:nx])))

# P2 = abs(FFT_demod/nt)
# P1 = P2[1:int(nt/2+1)]
# P1[2:-1] = 2*P1[2:-1]
k_padding = np.linspace(0, nx ,2**padding) 
k = np.linspace(0,nx,nx)

figurejolie()
joliplot(r'k ($mm^{-1}$)', r'|P1(k)|',k, np.abs(FFT_demod), exp = False, title = "FFT")

figurejolie()
joliplot(r'k ($mm^{-1}$)', r'|P1(k)|',k_padding, np.abs(FFT_demod_padding),color = 5, exp = False, legend = "FFT zero padding 2**" + str(padding))
joliplot(r'k ($mm^{-1}$)', r'|P1(k)|',k_padding, np.abs(FFT_demod_padding_0),color = 7, exp = False, legend = "FFT zero padding 2**" + str(padding) + " sans atténuation corrigée")
if save :
    plt.savefig(path_images[:-15] + "resultats" + "/" + "FFT_spatiale_0padding_" + str(round(f_0 * u)) + "Hz" + ".pdf", dpi = 1)
    

#%%find_peaks

D_environ = 0.4E-5
tension_surface = 0.05
g = 9.81
rho = 900
dsurrho = D_environ / rho
padding_0 = True

def RDD_comp (k, dsurrho):
    return np.sqrt(g * k + tension_surface/rho * k**3 + dsurrho * k**5)

from scipy.optimize import minimize

x = np.arange(10,1000,2)

y = np.zeros(x.shape)

def diff(x,a):
    yt = RDD_comp(x,dsurrho)
    return (yt - a )**2

for idx,x_value in enumerate(x):
    res = minimize(diff, 1.0, args=(x_value), method='Nelder-Mead', tol=1e-6)
    y[idx] = res.x[0]


k_theorique = np.interp(f_exc*2*np.pi,x,y)

if padding_0 :
    if f_exc != f_0 * u :
        k_theorique = np.interp((facq - f_exc)*2*np.pi,x,y)
    peaks_theorique =  k_theorique * 2**padding * mmparpixel / (2*np.pi*1000)
    
    if f_exc != f_0 * u :
        peaks_theorique = 2**padding - peaks_theorique
        
        
else :
    if f_exc != f_0 * u :
        k_theorique = np.interp((facq - f_exc)*2*np.pi,x,y)
    peaks_theorique =  k_theorique * nx * mmparpixel / (2*np.pi*1000)
    
    if f_exc != f_0 * u :
        peaks_theorique = nx - peaks_theorique
    
        


plt.figure()
if padding_0 :
    mean_fft = FFT_demod_padding
else :
    mean_fft = FFT_demod


LL = len (mean_fft)
plt.vlines(x=peaks_theorique, ymin=0 , ymax = max(np.abs(mean_fft)), color = "C1", label = "k théorique" )

dist = int(peaks_theorique/ 10)
prominence = max(np.abs(mean_fft))

qtt_pics = 8
if f_0*u > 100 :
    qtt_pics = 10
if f_0*u >150:
    qtt_pics = 20
if f_0*u >200:
    qtt_pics = 30
if f_0*u >250:
    qtt_pics = 50


peaks1, _ = find_peaks(abs(mean_fft), prominence = [prominence/qtt_pics,None], distance = 1, width=(2,50))



pics_lala = []
        
tolerance_k = 1/4
# if f_0*u > 80 :
#     tolerance_k = 1/7
# if f_0*u >140:
#     tolerance_k = 1/5
# if f_0*u >200:
#     tolerance_k = 1/4
# if f_0*u >250:
#     tolerance_k = 1/3


if padding_0:            
    for uuu in peaks1 : 
        if f_exc == f_0 * u :
            if np.abs(uuu - peaks_theorique) < peaks_theorique * tolerance_k :
                pics_lala.append(uuu)
        else :
            if np.abs(uuu - peaks_theorique) < (2**padding - peaks_theorique) * tolerance_k :
                pics_lala.append(uuu)
else :
    for uuu in peaks1 : 
        if f_exc == f_0 * u :
            if np.abs(uuu - peaks_theorique) < peaks_theorique * tolerance_k :
                pics_lala.append(uuu)
        else :
            if np.abs(uuu - peaks_theorique) < (nx - peaks_theorique) * tolerance_k :
                pics_lala.append(uuu)
            
print("pics dans l'ecart : ",pics_lala )           


seuil_arbitraire = 1.2 #ATTENTION, ici on fixe un seuil arbitraire qui peut être modifié. On considere 
                        #qu'a partir du moment où il y a un pic 1.2 fois plus haut que tt les autres
                        #alors il correspond à la longueure d'onde du signal
moymoy = False
for uuu in pics_lala :
    if max(np.abs(mean_fft[pics_lala])) != np.abs(mean_fft[uuu]):
        if max(np.abs(mean_fft[pics_lala])) > np.abs(mean_fft[uuu]) * seuil_arbitraire :
            peaks = np.where(max(np.abs(mean_fft[pics_lala])) == np.abs(mean_fft))[0]
        else :
            moymoy = True
    else :
        peaks = [uuu]
    # if max(np.abs(mean_fft)) == np.abs(mean_fft[uuu]) :
    #     pics_lala = np.where(max(np.abs(mean_fft)) == np.abs(mean_fft))[0]
if moymoy :
    peaks = pics_lala
    if len(peaks) > 2 :
        a = np.abs(mean_fft[peaks])
        a = a.tolist()
        a.sort()
        if a[-2] > a[-3] * seuil_arbitraire :
            peaks = np.concatenate( (np.where(a[-2] == np.abs(mean_fft))[0], np.where(a[-1] == np.abs(mean_fft))[0]), axis = 0)
        
plt.plot(abs(mean_fft), label = 'fft' )
plt.plot(peaks1, abs(mean_fft[peaks1]), 'mx', label = "pics détéctés")
plt.plot(peaks, abs(mean_fft[peaks]), 'ro', label = "pics gardés")
plt.legend()
# plt.xlim((0,100))
# if f_exc != f_0 * u :
    # plt.xlim((2**padding-100, 2**padding))

if save :
    plt.savefig(path_images[:-15] + "resultats" + "/" + "FFT_spatiale_0padding_" + str(round(f_exc)) + "Hz" + ".pdf", dpi = 1)
     
    

#%% Fit second degré
save = False

indice_tot = []

for maxs in peaks :
    
    y0 = maxs 
    
    a = 3
    
    z = np.abs(mean_fft)[y0-a:y0+a]
    
    x = [u for u in range (y0-a,y0+a) - y0 ]
    
    p = np.polyfit(x,z,2)    #on peut utiliser fminsearch pour fitter par une fonction quelconque
    
    imax = -p[1]/(2*p[0])
    
    indice_tot.append(y0 + imax)
    
    
indice = np.mean(indice_tot)

if padding_0 :
    longueur_donde = 2**padding * mmparpixel / (indice) / 1000

    if f_exc != f_0 * u :
        longueur_donde = 2**padding * mmparpixel / (2**padding - indice) / 1000
else :
    longueur_donde = nx * mmparpixel / (indice) / 1000

    if f_exc != f_0 * u :
        longueur_donde = nx * mmparpixel / (nx - indice) / 1000
    

if len(peaks) > 1 :
    error_lambda = (np.max(2**padding * mmparpixel / np.asarray(indice_tot) / 1000) - np.min(2**padding * mmparpixel / np.asarray(indice_tot) / 1000))/2
    if f_exc != f_0 * u :
        error_lambda = (np.max(2**padding * mmparpixel / (2**padding -np.asarray(indice_tot)) / 1000) - np.min(2**padding * mmparpixel / (2**padding -np.asarray(indice_tot)) / 1000))/2  
    
else : 
    error_lambda = 0

print("indice_tot = ", indice_tot)
print("lambda = ", longueur_donde)
    
if save :
    param_complets.extend(["Resultats lambda :","f_exc = " + str(f_exc),"padding = " + str(padding) ,"lambda (m) = " + str(longueur_donde),"error_lambda (m) = " + str(error_lambda)])

dico = add_dico(dico,date,nom_exp,'lambda',longueur_donde)

#%%PLOT longueure d'onde avec le signal

indice = 5 # int(np.round(indice))
ecart_indice = 5

save = False
# param_complets = param_complets.tolist()
f_exc = 5
amp_demod = []
cut_las = 0

if f_0 * u > 80 :
    cut_las = 500
if f_0 * u > 120 :
    cut_las = 700
    
nx = nx - cut_las
X = np.linspace(0, nx * mmparpixel/10, nx) #echelle des x en cm

#demodulation
for i in range (nx):
    a = data[i,:]
    amp_demod.append(np.sum(a * np.exp(2*np.pi*1j*f_exc*t))*2/nx)
    
if False : #display : 
    figurejolie()
    joliplot("temps(s)", "Amplitude (m)", t,a, exp = False)
 

amp_demod = np.asarray(amp_demod)
I = (np.abs(amp_demod))**2 #tableau intensite (avec amp en m)
 
if False : #display :
    figurejolie()
    joliplot(r"x (cm)",r"amplitude (m)", X, np.abs(amp_demod), exp = False, log = False, legend = r"f = " + str(int(f_exc) ) + " Hz")


Y_FFT = fft.fft(amp_demod)

#%%Attenuation précise


y_new_x = np.zeros(np.shape(Y_FFT), dtype = 'complex128') #FFT dans le sens de propagation
y_new_moinsx = np.zeros(np.shape(Y_FFT), dtype = 'complex128') #FFT dans le sens reflechis

if f_0 * u == f_exc :
    bx = int(nx/2)
    ax = 0
else :
    bx = nx
    ax = int(nx/2) + 1 
    
    
y_new_x[ax:bx] = Y_FFT[ax:bx]


if f_0 * u == f_exc :
    bx = nx
    ax = int(nx/2) + 1 
else :
    bx = int(nx/2)
    ax = 0
    
y_new_moinsx[ax:bx] = Y_FFT[ax:bx]

if False :
    figurejolie()
    plt.plot(np.abs(y_new_x))
    figurejolie()
    plt.plot(np.abs(y_new_moinsx))



demod_stat_x = fft.ifft(y_new_x)
demod_stat_moinsx = fft.ifft(y_new_moinsx)

if False :
    figurejolie()
    plt.plot(X, np.abs(demod_stat_x))
    plt.title("FFT inverse sens de propagation")
    
    figurejolie()
    plt.plot(X, np.abs(demod_stat_moinsx))
    
    plt.title("FFT inverse onde réflechie")    


def exppp(x, a, b):
    return a * np.exp(-b * x)


I_x = (np.abs(demod_stat_x))**2
I_moinsx = (np.abs(demod_stat_moinsx))**2

attenuation = curve_fit (exppp, X, I, p0 = [1,0])
attenuationA = curve_fit (exppp, X, np.abs(amp_demod), p0 = [1,0])

attenuation_x = curve_fit (exppp, X, I_x, p0 = [1,0])
# attenuation_moinsx = curve_fit (exppp, X, I_moinsx, p0 = [1,0])

figurejolie()
joliplot(r"x (cm)", r"I x", X, I_x, color = 3, exp = False, log = True, legend = r"f = " + str(int(f_exc) ) + " Hz")
joliplot(r"x (cm)", r"I x", X, exppp(X, attenuation_x[0][0], attenuation_x[0][1]), color = 5, exp = False, log = True, legend = r"fit, on trouve $\kappa = $" + str(round(attenuation_x[0][1],4)))
plt.xscale('linear')
plt.yscale('log')

# figurejolie()
# joliplot(r"x (cm)", r"I moins x", X, I_moinsx, color = 3, exp = False, log = True, legend = r"f = " + str(int(f_exc) ) + " Hz")
# joliplot(r"x (cm)", r"I moins x", X, exppp(X, attenuation_moinsx[0][0], attenuation_moinsx[0][1]), color = 5, exp = False, log = True, legend = r"fit, on trouve $\kappa = $" + str(round(attenuation_moinsx[0][1],4)))
# plt.xscale('linear')
# plt.yscale('log')

if display :
    figurejolie()
    joliplot(r"x (cm)", r"I", X, I, color = 3, exp = False, log = True, legend = r"f = " + str(int(f_exc) ) + " Hz")
    joliplot(r"x (cm)", r"I", X, exppp(X, attenuation[0][0], attenuation[0][1]), color = 5, exp = False, log = True, legend = r"fit, on trouve $\kappa = $" + str(round(attenuation[0][1],4)))
    plt.xscale('linear')
    plt.yscale('log')
    if save :
        plt.savefig(path_images[:-15] + "resultats" + "/" + "I(x)_fitkappa_" + str(round(f_exc)) + "Hz" + ".pdf", dpi = 1)

X = np.linspace(0, nx * mmparpixel/10, nx) #en cm


    
double_demod = []
FFT_double_demod = np.zeros(np.shape(Y_FFT), dtype = 'complex128')
FFT_double_demod[indice-ecart_indice:indice+ecart_indice] = Y_FFT[indice-ecart_indice:indice+ecart_indice]
double_demod = fft.ifft(FFT_double_demod)
if False :
    figurejolie()
    # joliplot("X (cm)", "Amplitude (m)", X, (max(np.abs(ampfoispente_0[:nx])) - min(np.abs(ampfoispente[:nx])) ) * np.cos(2 * np.pi * X / longueur_donde ) / 2, exp = False, legend = "Cosinus fait avec la longueure d'onde mesurée", color = 2)
    joliplot("X (cm)", "Amplitude (m)", X, np.abs(amp_demod), exp = False, legend = 'Signal démodulé en temps', color = 9)
    
    joliplot("X (cm)", "Amplitude (m)", X, np.abs(double_demod), exp = False, legend = 'Signal démodulé en temps et espace', color = 1)

"""RECHERCHE DE LA MEILLEURE ZONE LINEAIRE Signal total"""

range_taille_1 = 3
pas_taille_1 = 50
p, max_r, best_taille_1, best_pos_1 = find_best_lin(np.log(I), X = X, range_taille = range_taille_1, pas_taille = pas_taille_1)
X_top = X[int(nx/ pas_taille_1 * best_pos_1) : int(nx *  (1/best_taille_1 + best_pos_1/pas_taille_1) ) ]
figurejolie()
plt.plot(X, np.log(I), label = 'log(I)')
plt.plot(X_top, p[0] * X_top + p[1], label = "Meilleure zone linéaire, kappa = " + str(round(p[0], 3)))
print("attenuation I :", p[0])
plt.legend()

I_top = I[int(nx/ pas_taille_1 * best_pos_1) : int(nx *  (1/best_taille_1 + best_pos_1/pas_taille_1) ) ]

"""Parametres"""

cm_debut = X[0]
cm_fin = X[-1]

taille_X_cut = int(nx * (cm_fin)/X[-1]) - int(nx * (cm_debut)/X[-1])
X_cut = np.linspace(cm_debut, cm_fin, taille_X_cut) #en cm
                
I_demod = (np.abs(double_demod[int(nx*cm_debut/X[-1]):int(nx*cm_fin/X[-1])]))**2

I_cut = (np.abs(amp_demod[int(nx*cm_debut/X[-1]):int(nx*cm_fin/X[-1])]))**2

"""FIT EXP SUR ZONE D'INTERET AUTOUR DE LAMBDA"""

figurejolie()
attenuation_demod = curve_fit (exppp, X_cut, I_demod, p0 = [1,0])
joliplot("X (cm)", "I (m)", X_cut, I_demod, exp = False, legend = 'I démodulé en temps et espace', color = 1)
# joliplot(r"x (cm)", r"I", X_cut, I_cut, color = 3, exp = False, log = True, legend = r"f = " + str(int(f_exc) ) + " Hz")
joliplot(r"x (cm)", r"I", X_cut, exppp(X_cut, attenuation_demod[0][0], attenuation_demod[0][1]), color = 5, exp = False, log = True, legend = r"fit, on trouve $\kappa = $" + str(round(attenuation_demod[0][1],4)))
plt.xscale('linear')
plt.yscale('log')

"""FIT SUR PICS AUTOUR LAMBDA"""

nb_pics_voulus = 4
peaks_demod, _ = find_peaks(abs(I_demod), prominence = 1E-15, distance = 10, width=(5,50))
peaks_demod, _ = find_peaks(abs(I_demod), distance = len(I_demod)/(nb_pics_voulus+1) )
# peaks_demod = np.asarray([ 38, 209, 358, 500])
figurejolie()
plt.plot(X_cut[peaks_demod], np.log(I_demod[peaks_demod]), label = 'PICS')
plt.plot(X_cut, np.log(I_demod), label = 'I démodulé tps et x')
attenuation_demod_pics = np.polyfit( X_cut[peaks_demod], np.log(I_demod[peaks_demod]), 1)
plt.plot(X_cut, attenuation_demod_pics[0]* X_cut + attenuation_demod_pics[1], label = "kappa avec fit dur pics = " + str(round(attenuation_demod_pics[0],3)))
plt.legend()


"""FIT LINEAIRE SUR ZONE D'INTERET AUTOUR DE LAMBDA"""

range_taille = 1
pas_taille = 20
p, max_r, best_taille, best_pos = find_best_lin(np.log(I_demod), X = X_cut, range_taille = range_taille, pas_taille = pas_taille)
# taille_X_cut = nx
figurejolie()
plt.plot(X_cut, np.log(I_demod), label = 'log(I demod)')
plt.plot(X_cut[int(taille_X_cut/ pas_taille * best_pos) : int(taille_X_cut *  (1/best_taille + best_pos/pas_taille) ) ], p[0] * X_cut[int(taille_X_cut/ pas_taille * best_pos) : int(taille_X_cut *  (1/best_taille + best_pos/pas_taille) ) ] + p[1], label = "Meilleure zone linéaire, kappa = " + str(round(p[0], 3)))
print("attenuation I demod :", p[0])
plt.legend()

if save :
    param_complets.extend(["Resultats lambda :","ecart_indice = " + str(ecart_indice),"pas_taille_1 = " + str(pas_taille_1) ,"range_taille_1 = " + str(range_taille_1),"nb_pics_voulus = " + str(nb_pics_voulus),"cm_fin = " + str(cm_fin),"cm_debut = " + str(cm_debut)])


"""AUTOUR DE ITOP"""

I_demod_top = (np.abs(double_demod[int(nx*X_top[0]/X[-1]):int(nx*X_top[-1]/X[-1])]))**2
range_taille = 1
pas_taille = 20
p, max_r, best_taille, best_pos = find_best_lin(np.log(I_demod_top), X = X_top, range_taille = range_taille, pas_taille = pas_taille)
X_top = np.linspace(X_top[0],X_top[-1], len(I_demod_top))
figurejolie()
plt.plot(X_top, np.log(I_demod_top), label = 'log(I demod)')
plt.plot(X_top, p[0] * X_top + p[1], label = r"Fit lineaire demodule temps espace zone d'interet, $\kappa$ = " + str(round(p[0], 3)))
print("attenuation I demod zone dinteret :", p[0])
plt.legend()

#%% Sauvegarde des parametres et resultats
# param_complets = param_complets.tolist()
if save :
    param_complets.extend([ "grossissement = " + str(grossissement) ,"f_exc = " + str(f_exc) ])
    param_complets.extend(["Paramètres de traitement :",  "debut_las = " + str(debut_las) ,"fin_las = " + str(fin_las),"t0 = " + str(t0) ,"tf = " + str(tf) ])
    param_complets.extend(["savgol = " + str(savgol) ,"im_ref = " + str(im_ref)])
    param_complets.extend(["padding = " + str(padding) ])
    param_complets = np.asarray(param_complets)
    np.savetxt(path_images[:-15] + "resultats" + "/Paramètres_FFT_lambda_kappa" + name_fig_FFT + ".txt", param_complets, "%s")
    

