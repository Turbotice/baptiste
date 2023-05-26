# -*- coding: utf-8 -*-
"""
Created on Wed May 25 16:17:26 2022

@author: Banquise
"""

"""
Programme matlab antonin
"""
#%%



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

#%%Paramètre de traitement
# mmparpixel = 0.28
# mmparpixelx = 0.28
# mmparpixely = 0.28

save_histo = False
save = False
save_fig = False
display = True

num_fig = 1
f_excitation = round(1000 / (2 * Tmot))

find_amp_gauss = False
diff_maxmin = True
taille_grains = False
coeff_attenuation = True

moyenne = False
medianne = True

if coeff_attenuation :
    diff_maxmin = True
    grossissement = 7.316 # import_angle(date, nom_exp, loc)

#%%Charge les données (data) et en fait l'histogramme
num_fig += 1
figurejolie()



folder_results = path_images[:-15] + "resultats"
name_file = "positionLAS.npy"
data_originale = np.load(folder_results + "\\" + name_file)

data_originale = np.rot90(data_originale)

debut_las = 100
fin_las = np.shape(data_originale)[0] - 100


regime_permanent = 1
fin_regime = np.shape(data_originale)[1] - 1

if display:
    [y,x] = np.histogram((data_originale[debut_las:fin_las,regime_permanent:fin_regime]),10000)
    xc= (x[1:]+x[:-1]) / 2
    plt.plot(xc,y)
    plt.axis([0,800,0,max(y)])
    plt.grid()
if save_histo :
    plt.savefig(path_images[:-15] + "resultats" + "/" + name_fig_FFT + "histo_LAS.tiff", dpi = 300)

# signal contient la detection de la ligne laser. C'est une matrice, avec
# dim1=position (en pixel), dim2=temps (en frame) et valeur=position
# verticale du laser en pixel.


    
    
#%% Pre-traitement

savgol = False
medsavgol = False
med = True
im_ref =True

ordre_savgol = 2
taille_savgol = 21
size_medfilt = 51

# plt.figure()
# plt.pcolormesh(data)
# plt.xlabel("Temps (frame)")
# plt.ylabel("X (pixel)")
# cbar = plt.colorbar()
# cbar.set_label('Amplitude (m)')



[nx,nt] = data_originale[debut_las:fin_las,regime_permanent:fin_regime].shape


data = data_originale[debut_las:fin_las,regime_permanent:fin_regime]


#enlever l'image de ref ou moyenne pr chaque pixel (= ombroscopie)
if im_ref :
    mean_pixel = np.mean(data,axis = 1) #liste avec moyenne en temps de chaque pixel 
    for i in range (0,nt):
        data[:,i] = data[:,i] - mean_pixel
    #     data[:,i] = data[:,i] - data[:,0]
    # data[:,0] = 0
        #ou moins la moyenne de chaque pixel ?
        # - np.mean(data,axis = 1)
        
# num_fig += 1
# plt.figure(num_fig)
# plt.pcolormesh(data)
# plt.colorbar()


#mise à l'échelle en m
data_m = data *  mmparpixely / 1000
if coeff_attenuation :
    data_m = data_m / grossissement


t = np.arange(0,nt)/facq
x = np.arange(0,nx)*mmparpixelz / 1000

signalsv = np.zeros(data.shape)
signal_medfilt = np.zeros(data.shape)
medfilt_sv = np.zeros(data.shape)

#filtre sv et médian
for i in range(0,nt):
    
    signalsv[:,i] = savgol_filter(data_m[:,i], taille_savgol,ordre_savgol, mode = 'nearest')
    signal_medfilt[:,i] = medfilt(data_m[:,i], size_medfilt)
    medfilt_sv[:,i] = savgol_filter(signal_medfilt[:,i], taille_savgol, ordre_savgol, mode = 'nearest')
    if np.mod(i,500)==0:
        print('On processe l image numero: ' + str(i) + ' sur ' + str(nt))
print('Done !')
data = data_m

if medsavgol :
    data = medfilt_sv
if med :
    data = signal_medfilt
if savgol :
    data = signalsv
# if display:
#     plt.figure()
#     plt.pcolormesh(data)
#     plt.xlabel("Temps (frame)")
#     plt.ylabel("X (pixel)")
#     cbar = plt.colorbar()
#     cbar.set_label('Amplitude (m)')
    
#%% Trouver l'amplitude en fittant l'histogramme


amplitude = []

if find_amp_gauss :
    # data_filt2 = medfilt2d(data_originale, (201,201))
    plt.figure()
    nb_boites = 10000
    [y,x] = np.histogram((data_m),nb_boites)
    xc= (x[1:]+x[:-1]) / 2
    #centre en 0
    delete = np.argmax(y)
    xc_new = xc - xc[delete]
    #mise à l'échelle
    xc_new = xc_new * mmparpixely / ((xc[-1]- xc[0])/len( xc)) / nb_boites * 1000
    
    y_new = y / max(y)
    
    def gaussian(x, mu, sig):
        return np.exp(- np.power(x - mu, 2) / (2 * sig** 2) ) 
    
    popt, pcov = curve_fit(gaussian, xc_new, y_new)
    plt.plot(xc_new,y_new)
    plt.plot(xc_new, gaussian(xc_new, popt[0], popt[1]))
    
    LMH = 2 * np.sqrt(2 * np.log(2)) * popt[1]
    print (LMH)
    amplitude.extend(["amplitude par gaussienne = " + str(LMH) + " mm "])
    param_complets.extend (["Paramètres pour amp gauss :",  "nb_boites = " + str(nb_boites) ,"sigma = " + str(popt[1]) ,"mu = " + str(popt[0]),"moyenne = " + str(moyenne) ,"medianne = " + str(medianne) ])


if diff_maxmin :
    data_filt2 = data_m
    # data_filt2 = data_originale[250:400]
    # data_filt2 = medfilt2d(data_originale, (51,51))
    tab_amp = []
    lenmedfilt = 501
    # for a in range (data_filt2.shape[1]):
    #     data_filt2[:,a] = detrend(data_filt2[:,a])
        
    for k in range (data_filt2.shape[0]):
        min_las = min(data_filt2[k,:])
        max_las = max(data_filt2[k,:])
        tab_amp.append(max_las-min_las)
    
    tab_amp = np.asarray(tab_amp)
    # tab_amp = medfilt(tab_amp, lenmedfilt)
    tab_amp = tab_amp
    # tab_amp = np.asarray(list(reversed(tab_amp)))

    if medianne :
        amp = np.median(tab_amp)
    if moyenne :
        amp = np.mean(tab_amp)
    
    # if coeff_attenuation :
    #     tab_amp = tab_amp / grossissement
    #     amp = amp / grossissement
    
    print (amp)
    
    amplitude.extend(["amplitude par minmax = " + str(amp) + " m "])
    amplitude.extend(tab_amp)
    param_complets.extend (["Paramètres pour amp minmax :",  "lenmedfilt = " + str(lenmedfilt)])
    
if coeff_attenuation :
    
    XXXt = np.linspace(0,(fin_las - debut_las) * mmparpixel/10, (fin_las - debut_las)) #echelle des x en cm
    
    I = (np.asarray(amplitude[1:])/ grossissement)**2 #tableau intensite (avec amp en m)
    
    figurejolie()
    joliplot(r"x (cm)",r"amplitude (m)",XXXt,np.asarray(amplitude[1:])/ grossissement,exp = False, log = False, legend = r"f = " + str(int(f_excitation) ) + " Hz")
    figurejolie()
    joliplot(r"x (cm)",r"I",XXXt,I,color = 3,exp = False, log = True, legend = r"f = " + str(int(f_excitation) ) + " Hz")
    # plt.plot(XXXt,(np.asarray(amplitude[1:])/1000 / grossissement)**2)
    def exppp(x, a, b):
        return a * np.exp(-b * x)
    
    attenuation = curve_fit (exppp, XXXt,I, p0 = [1,0])
    joliplot(r"x (cm)",r"I",XXXt,exppp(XXXt,attenuation[0][0],attenuation[0][1]),color = 5,exp = False, log = True, legend = r"fit, on trouve $\kappa = $" + str(round(attenuation[0][1],3)))
    plt.xscale('linear')
    plt.yscale('log')
    plt.grid()
    # = $\frac{1}{T} \int_{0}^T |a(x)|^{2}dt$

if taille_grains :
    data_grains = data_originale[:,0:10]
    grains = []
    for i in range (data_grains.shape[0]):
        grains.append(np.mean(data_grains[i,:]))
        grains[-1] = grains[-1] * mmparpixely
    size_grains = np.std(grains)
    plt.plot(grains)
        
if save :
    amplitude = np.asarray(amplitude)
    np.savetxt(path_images[:-15] + "resultats" + "/amplitude" + name_fig_FFT + ".txt", amplitude, "%s")
    param_complets = np.asarray(param_complets)
    np.savetxt(path_images[:-15] + "resultats" + "/param_amplitude" + name_fig_FFT + ".txt", param_complets, "%s")



#%% Analyse d'un signal temporel

#On prend un point qcq en x, et on regarde le signal au cours du temps.
num_fig += 1
plt.figure(12)
i = 200

pos_fixe_t = data[i,:]
pos_fixe_t = pos_fixe_t - np.mean(pos_fixe_t)
# On va regarder la periode sur signal.

Y1 = fft.fft(pos_fixe_t)

T = 1/facq            # Sampling period
L = len(pos_fixe_t)             # Length of data
t = np.arange(0,L)*T        # Time vector

P2 = abs(Y1/L)
P1 = P2[1:int(L/2+1)]
P1[2:-1] = 2*P1[2:-1]


f = facq * np.arange(0,int(L/2)) / L
plt.plot(f, P1)
plt.title('Single-Sided Amplitude Spectrum of X(t)')
plt.xlabel('f (Hz)')
plt.ylabel('|P1(f)|')

# On constate qu'il y a un pic "principal" vers 6.245 Hz, un
# sous-harmonique et des harmoniques (4 ou 5 assez intenses pour être vus).
# On va essayer d'extraire la structure spatiale pour chacun de ces pics.
# Il faut donc travailler sur tous les points spatiaux.

#%% Soustraction de la moyenne temporelle et fft temporelle



data_meantemp = np.zeros(data_m.shape)


data_meantemp = data_m #- np.mean(data)



#On fait la fft tempore lle pour chaque pixel

# Y = np.zeros(data.shape)

# for j in range (data.shape[0]):
#     Y[j,:] = fft.fft(data_meantemp[j,:])

Y = fft.fft(data_meantemp, axis = 1)
if display :
    num_fig += 1
    plt.figure(num_fig)
    plt.pcolormesh(abs(Y))
    plt.colorbar()
    plt.title('fft temporelle')

#%%demodualtion
t1 = np.linspace(0,50,51)
fpot = f_excitation + 5
c, ddemodule, t1 = demodulation_gld_sp_ba(t,data_m,fpot,t1)

plt.figure()
plt.plot(x, np.real(c))

#%% moyenne pour trouver pics

#pbl entre 1955 et 1930
num_fig += 1
plt.figure(num_fig)

# mean_fft1 = np.mean(Y[:1930], axis = 0)
# mean_fft2 = np.mean(Y[1955:], axis = 0)
# mean_fft = np.concatenate((mean_fft1, mean_fft2), axis = 0)


mean_fft = np.mean(Y[:,:], axis = 0)
plt.plot(abs(mean_fft), label = 'moyenne des fft de chaque pixel')
plt.legend()


#%% find peaks
plt.figure()
mean_fft = mean_fft #Y1
LL = len (mean_fft)


#on cherche les pics de taille entre prominence, de distance minimale "disatnce" entre chaque
#RDMF2 - th =5E-15 et w (1,2), 1 - th = 5E-2 et w (1,5), 3 - th = 3.5E-2 w (1,2), 4 - th = 3E-3 w (1,3), 5 - th = 8E-3 w(1,3), 6 - th = 1.2E-2 w(1,3), RSBP1 th = 8E-4 w(1,3)
peaks1, _ = find_peaks(abs(mean_fft[:int(LL/2)]), threshold = 0.0001, distance = 100, width=(1,5))
peaks2 , _ = find_peaks(abs(mean_fft[:int(LL/2)]), threshold = 0.000001, distance = 100, width=(3,10))


plt.plot(abs(mean_fft), label = 'fft' )
plt.plot(peaks1, abs(mean_fft[peaks1]), 'mx')
plt.plot(peaks2, abs(mean_fft[peaks2]), 'gx')

peaks = peaks1 #np.concatenate((peaks1,peaks2))
# peaks = np.append(peaks, 904)
# peaks = np.append(peaks, 688)
# peaks = np.append(peaks, 778)

# peaks = peaks[14:]
#enlever les sous harmoniques
f_excitation = 1000 / (2 * Tmot)

#%% sub pixelaire pour les pics

#Idée : on a un pic, on va chercher à faire passer la meilleure parabole
#qui décrit le sommet.
#Il s'agit d'un systeme d'équations simples à résoudre. Voir la thèse d'A.
#Marchand pour le détail du calcul.

Delta_k=(k(2)-k(1))*(abs(YY(ind+1))-abs(YY(ind-1)))/(2*(2*abs(YY(ind))-abs(YY(ind+1))-abs(YY(ind-1))))
k_detect=k(ind)+Delta_k


#%% Extraction de la fréquence sous-harmonique
# peaks = peaks

# ptscools = [16]

range_taille = 1
pas_taille = 2
min_corel = 0.6
index = []
# for i in range (len (ptscools)):
#     index.append(peaks[ptscools[i]])
index=peaks

size_plot = len(peaks)
# size_plot = len(ptscools)

longueur_donde = []
fig, axes = figurejolie(subplot = (size_plot,2))


for i in range (len(index)) :

    pos = index[i]
    
    sousharm = (Y[ :, pos ])
    
    lengthpix = len (sousharm)
    lengthfilm = len (Y[1,:])

    xpix = np.arange(lengthpix)
    
    tt_param_fit = find_best_lin(sousharm, range_taille = range_taille, pas_taille = pas_taille) 
    
    p = tt_param_fit[0]
    r = tt_param_fit[1]
    best_pos = tt_param_fit[2]
    best_taille = tt_param_fit[3]
    print (best_pos)
    print(best_taille)
    
    print (r)
   
    if r > min_corel :
        

        axes = joliplot('','', xpix, np.real(sousharm ), color = 1, title = r"Partie réelle " + str(i) + ", f = " + str (pos * facq / lengthfilm), legend = "Partie réelle", exp = False, subplot= (size_plot,2), fig = fig, axes = axes) 
        joliplot('', '', xpix, np.imag(sousharm), color = 2, title = False, legend = r"Partie imaginaire", exp = False)
        
        longueur_donde.append( [abs(2 * np.pi / p[0]), pos * facq / lengthfilm])
        
        xcos = np.arange(lengthpix)
        xcos = np.float_(xcos)
        cosinus = np.arange(len(xcos))
        cosinus = np.float_(cosinus)
        for m in range (len(xcos)):
            cosinus[m] = np.cos(2 * np.pi * xcos[m] / float(longueur_donde[-1][0] )) * float( max(np.real(sousharm)))
            
        joliplot('','', xcos, cosinus, color = 4, legend = r"Cosinus fitté", exp = False)
        
        x = np.arange(int (lengthpix/ pas_taille * best_pos ), int( (lengthpix/ (best_taille)) + (lengthpix / pas_taille * best_pos )))
        
        axes = joliplot('','', x, p[0] * (x) + p[1], color = 1, title = r"Phase unwrapped et fit" + str(i)+ ' r = '+ str(r), legend = r"fit lineaire", exp = False, subplot= (size_plot,2), fig = fig, axes = axes) 
        joliplot('','', xpix, np.unwrap(np.angle(sousharm )), color = 4, legend =  r'phase', exp = False)
        
        # axes.append( fig.add_subplot(len(peaks), 2, u) )
        # axes[-1].set_title(r"signal filtre median " + str(i)+ ' r = '+ str(r))
        # plt.axis()
        # u += 1
        # plt.plot(x,p[0] * (x ) + p[1], label = r"fit lineaire") 
        # plt.plot(np.unwrap(np.angle(sousharm )), label = r'phase')
        # plt.legend()
        # plt.grid()
        
 
plt.tight_layout()
        
    

if save_fig :   
    plt.savefig(path_images[:-15] + "resultats" + "/" + name_fig_FFT + "longueurs_dondes.png", dpi = 300)  
    
#fichier longueur d'onde avec lambda, f
for j in range (len (longueur_donde)):
    longueur_donde[j][0] = longueur_donde[j][0] * mmparpixely

#%% Sauvegarde des param et longueurs d'onde

plt.savefig(path_images[:-15] + "resultats" + "/" + name_fig_FFT + "longueurs_dondes_part2.png", dpi = 300)  
longueur_donde = np.asarray(longueur_donde)

np.savetxt(path_images[:-15] + "resultats" + "/longueur_donde_part2" + name_fig_FFT + ".txt", longueur_donde)
arr = []
for x in param_complets:
    arr.append(x)
param_complets = arr
param_complets.extend (["Paramètres de pre traitement :",  "regime_permanent = " + str(regime_permanent) ,"medsavgol = " + str(medsavgol) ,"med = " + str(med) ,"savgol = " + str(savgol) , "ordre_savgol = " + str(ordre_savgol) ,"taille_savgol = " + str(taille_savgol) ,"size_medfilt = " + str(),"size_medfilt = " + str(size_medfilt),"size_medfilt = " + str(size_medfilt) ])
param_complets.extend (["Paramètres de fft :",  "min_corel = " + str(min_corel),  "range_taille = " + str(range_taille),  "pas_taille = " + str(pas_taille), "nb de pics = " + str(len(peaks))] )
param_complets = np.asarray(param_complets)

np.savetxt(path_images[:-15] + "resultats" + "/Paramètres_FFT_part2" + name_fig_FFT + ".txt", param_complets, "%s")

