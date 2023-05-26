# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 11:17:53 2022

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
import os
from PIL import Image
from datetime import datetime
from scipy.optimize import minimize
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

openn_dico = True
if openn_dico :
    dico = open_dico()


#%%Paramètre de traitement


save = False
kappa_only = False
display = False
savefig = True

f_exc = round(Tmot)

grossissement =  dico[date][nom_exp]["grossissement"] #2.55726017002#2.80#5.7978508834100255 #import_angle(date, nom_exp, loc)

fichiers = os.listdir(loc)

lambda_tot = []

kappa_tot = []

nom_save_file = ""
for a in nom_exp :
    if not a.isnumeric() :
        nom_save_file += a

# paramètes éstimés, on va chercher des k qui correspondent plus ou moins pour D éstimé (mesuré auparavant)

D_estimé = 0.4E-5

#pour estimer un k à partir d'une fréquence
tension_surface = 0.05
g = 9.81
rho = 1000
dsurrho = D_estimé / rho

def RDD_comp (k, dsurrho):
   return np.sqrt(g * k + tension_surface/rho * k**3 + dsurrho * k**5)

x_est = np.arange(10,1000,5)

y_est = np.zeros(x_est.shape)

def diff(x,a):
    yt = RDD_comp(x,dsurrho)
    return (yt - a )**2

for idx,x_value in enumerate(x_est):
    res = minimize(diff, 1.0, args=(x_value), method='Nelder-Mead', tol=1e-6)
    y_est[idx] = res.x[0]


# fichiers.pop(-4)
# fichiers.pop(-2)


param_complets.extend([ "Date_du _jour = " + str(datetime.now())[:10],"nom_exp = " + str(nom_save_file),  "D_estimé = " + str(D_estimé),"grossissement = " + str(grossissement), "rho = " + str(rho), "tension_surface = " + str(tension_surface)])
#%% MAIN

for file in fichiers :
    
    nom_exp = file[8:13]
    
    path_images, liste_images, titre_exp = import_images(loc,nom_exp, "LAS")
    facq, texp, Tmot, Vmot, Hw, Larg_ice, Long_ice, tacq, type_exp = import_param (titre_exp, date)
    
    f_exc = dico[date][nom_exp]['fexc']#round(Tmot)
    
    folder_results = path_images[:-15] + "resultats"
    name_file = "positionLAS.npy"
    data_originale = np.load(folder_results + "\\" + name_file)
    
    data_originale = np.rot90(data_originale)
    
    debut_las = 100#100#250#400
    fin_las = np.shape(data_originale)[0] - 1200#50#200#300NBFT#100DAP
    
    
    t0 = 1
    tf = np.shape(data_originale)[1] - 1
    
    if display:
        figurejolie()
        [y,x] = np.histogram((data_originale[debut_las:fin_las,t0:tf]),10000)
        xc= (x[1:]+x[:-1]) / 2
        joliplot("x (pixel)", "Position du laser (pixel)", xc,y, exp = False)
        plt.yscale('log')
    
    # data_originale contient la detection de la ligne laser. C'est une matrice, avec
    # dim1=position (en pixel), dim2=temps (en frame) et valeur=position
    # verticale du laser en pixel.
    
    
    
    savgol = True
    im_ref = True
    
    ordre_savgol = 2
    taille_savgol = 21
    size_medfilt = 51
    
    [nx,nt] = data_originale[debut_las:fin_las,t0:tf].shape
    
    
    data = data_originale[debut_las:fin_las,t0:tf]
    
    
    #enlever moyenne pr chaque pixel
    
    if im_ref :
        mean_pixel = np.mean(data,axis = 1) #liste avec moyenne en temps de chaque pixel 
        for i in range (0,nt):
            data[:,i] = data[:,i] - mean_pixel #pour chaque temps, on enleve la moyenne temporelle de chaque pixel
    
    
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
        data = signalsv
    else :
        data = data_m
        
    if display:
        plt.figure()
        plt.pcolormesh(data)
        plt.xlabel("Temps (frame)")
        plt.ylabel("X (pixel)")
        cbar = plt.colorbar()
        cbar.set_label('Amplitude (m)')
    
        
    """ Analyse d'un signal temporel """
    
    
    if True:
        #On prend un point qcq en x, et on regarde le signal au cours du temps.
        
        i = 200
        
        # On va regarder la periode sur signal.
        
        Y1 = fft.fft(data[i,:]- np.mean(data[i,:]))
        
        P2 = abs(Y1/nt)
        P1 = P2[1:int(nt/2+1)]
        P1[2:-1] = 2*P1[2:-1]
        
        
        f = facq * np.arange(0,int(nt/2)) / nt 
        
        figurejolie()
        joliplot('f (Hz)', '|P1(f)|', f, P1, title = 'Single-Sided Amplitude Spectrum of X(t)', exp = False)
        
        
        if savefig :
            plt.savefig(path_images[:-15] + "resultats" + "/" + "FFT_temporelle_" + nom_exp + "_fexc_" + str(round(f_exc)) + "Hz_" + "pixel_" + str(i) + ".pdf", dpi = 1)
       
           
        
    f_0 = f_exc
    n_harm = int(input("Nombre d'harmoniques à chercher ? "))
    
    lambda_exp = []
    kappa_exp = []
    
    for u in range (1,n_harm+1):
        
        """ Demodulation et amplitude """
        
        # param_complets = param_complets.tolist()
        f_exc = f_0 * u
        if f_exc > facq / 2 :
            f_exc = facq - f_exc
            
        amp_demod = []
        cut_las = 0
        if f_0 * u > 80 :
            cut_las = 500
        if f_0 * u > 120 :
            cut_las = 700
            
        nx_new = nx - cut_las
        X = np.linspace(0, nx_new * mmparpixel/10, nx_new) #echelle des x en cm
        
        for i in range (nx_new):
            a = data[i,:]
            amp_demod.append(np.sum(a * np.exp(2*np.pi*1j*f_exc*t))*2/nx_new)
            
        if False : #display : 
            figurejolie()
            joliplot("temps(s)", "Amplitude (m)", t,a, exp = False)
        
        amp_demod = np.asarray(amp_demod)
        I = (np.abs(amp_demod))**2 #tableau intensite (avec amp en m)
         
        if False : #display :
            figurejolie()
            joliplot(r"x (cm)",r"amplitude (m)", X, np.abs(amp_demod), exp = False, log = False, legend = r"f = " + str(int(f_0 * u) ) + " Hz")
         
        def exppp(x, a, b):
            return a * np.exp(-b * x)
        
        attenuation,pcov = curve_fit (exppp, X, I, p0 = [1,0])
        attenuationA = curve_fit (exppp, X, np.abs(amp_demod), p0 = [1,0])
        
        if savefig :
            figurejolie()
            joliplot(r"x (cm)", r"I", X, I, color = 3, exp = False, log = True, legend = r"f = " + str(int(f_0 * u) ) + " Hz")
            joliplot(r"x (cm)", r"I", X, exppp(X, attenuation[0], attenuation[1]), color = 5, exp = False, log = True, legend = r"fit, on trouve $\kappa = $" + str(round(attenuation[1],4)))
            plt.xscale('linear')
            plt.yscale('log')
            plt.savefig(path_images[:-15] + "resultats" + "/" + "I(x)_fitkappa_" + str(round(f_0 * u)) + "Hz" + ".pdf", dpi = 1)
       
        if save :
            err_kappa = np.sqrt(np.diag(pcov))[1] #c'est la standard deviation (à priori exact (cf doc curve fit))
            kappa_exp.append([attenuation[1], err_kappa, f_0 * u])
            kappa_tot.append([attenuation[1], err_kappa, f_0 * u, nom_exp])
        
        if not kappa_only : #Si on veut pas que l'attenuation :
            
            """ Longueure d'onde """
            
            padding = 12
            
            ampfoispente = np.append( amp_demod * np.exp(attenuationA[0][1] * X), np.zeros(2**padding - nx_new))
            ampfoispente_0 = np.append( (amp_demod) , np.zeros(2**padding - nx_new))
            
            if savefig :
                figurejolie()
                joliplot("X (cm)", "Signal", X, np.real(ampfoispente[:nx_new]),color =2, legend = r'Signal démodulé * atténuation (m)', exp = False)
                joliplot("X (cm)", "Signal", X, np.real(ampfoispente_0[:nx_new]),color = 10, legend = r'Signal démodulé', exp = False)
                plt.savefig(path_images[:-15] + "resultats" + "/" + "Partie réelle_Signal_démodulé_" + str(round(f_0 * u)) + "Hz" + ".pdf", dpi = 1)
                
            
            nx_new
            FFT_demod_padding = fft.fft((ampfoispente)-np.mean((ampfoispente)))
            FFT_demod_padding_0 = fft.fft((ampfoispente_0)-np.mean((ampfoispente_0)))
            FFT_demod = fft.fft((ampfoispente_0[:nx_new])-np.mean((ampfoispente_0[:nx_new])))
            
            # P2 = abs(FFT_demod/nt)
            # P1 = P2[1:int(nt/2+1)]
            # P1[2:-1] = 2*P1[2:-1]
            k_padding = np.linspace(0, nx_new ,2**padding) 
            k = np.linspace(0,nx_new,nx_new)
            
            if display :
                figurejolie()
                joliplot(r'k ($mm^{-1}$)', r'|P1(k)|',k, np.abs(FFT_demod), exp = False, title = "FFT")
            
            if savefig and False :
                figurejolie()
                joliplot(r'k ($mm^{-1}$)', r'|P1(k)|',k_padding, np.abs(FFT_demod_padding),color = 5, exp = False, legend = "FFT zero padding 2**" + str(padding))
                joliplot(r'k ($mm^{-1}$)', r'|P1(k)|',k_padding, np.abs(FFT_demod_padding_0),color = 7, exp = False, legend = "FFT zero padding 2**" + str(padding) + " sans atténuation corrigée")
                plt.xlim((0,100))
                if f_exc != f_0 * u :
                    plt.xlim((2**padding-100, 2**padding))
                plt.savefig(path_images[:-15] + "resultats" + "/" + "FFT_spatiale_0padding_" + str(round(f_0 * u)) + "Hz" + ".pdf", dpi = 1)
                
            
            
            """ Find Peaks """
          
            
            k_theorique = np.interp(f_exc*2*np.pi,x_est,y_est)
            if f_exc != f_0 * u :
                k_theorique = np.interp((facq - f_exc)*2*np.pi,x_est,y_est)
            peaks_theorique =  k_theorique * 2**padding * mmparpixel / (2*np.pi*1000)
    
            if f_exc != f_0 * u :
                peaks_theorique = 2**padding - peaks_theorique
                
            mean_fft = FFT_demod_padding
            LL = len (FFT_demod_padding)
            
    
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
    
            peaks1, _ = find_peaks(abs(mean_fft), prominence = prominence/qtt_pics, distance = 2, width=(2,50))
    
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
                    
            for uuu in peaks1 : 
                if f_exc == f_0 * u :
                    if np.abs(uuu - peaks_theorique) < peaks_theorique * tolerance_k :
                        pics_lala.append(uuu)
                else :
                    if np.abs(uuu - peaks_theorique) < (2**padding - peaks_theorique) * tolerance_k :
                        pics_lala.append(uuu)
    
    
            seuil_arbitraire = 1.2 #ATTENTION, ici on fixe un seuil arbitraire qui peut être modifié. On considere 
                                    #qu'a partir du moment où il y a un pic 1.2 fois plus haut que tt les autres dans 
                                    #la gamme choisie alors il correspond à la longueure d'onde du signal
    
            moymoy = False
            for uuu in pics_lala :
                if max(np.abs(mean_fft[pics_lala])) != np.abs(mean_fft[uuu]):
                    if max(np.abs(mean_fft[pics_lala])) > np.abs(mean_fft[uuu]) * seuil_arbitraire :
                        peaks = np.where(max(np.abs(mean_fft[pics_lala])) == np.abs(mean_fft))[0]
                    else :
                        moymoy = True
                else :
                    peaks = pics_lala
            if moymoy :
                peaks = pics_lala
                if len(peaks) > 2 :
                    a = np.abs(mean_fft[peaks])
                    a = a.tolist()
                    a.sort()
                    if a[-2] > a[-3] * seuil_arbitraire :
                        peaks = np.concatenate( (np.where(a[-2] == np.abs(mean_fft))[0], np.where(a[-1] == np.abs(mean_fft))[0]), axis = 0)
    
    
            if savefig :
                figurejolie()
                plt.plot(abs(mean_fft), label = 'fft' )
                plt.vlines(x=peaks_theorique, ymin=0 , ymax = max(np.abs(mean_fft)), color = "C1", label = "k théorique" )
                plt.plot(peaks1, abs(mean_fft[peaks1]), 'mx', label = "Pics détéctés")
                plt.plot(peaks, abs(mean_fft[peaks]), 'ro', label = "Pics gardés")
                plt.legend()
                plt.xlim((0,200))
                if f_exc != f_0 * u :
                    plt.xlim((2**padding-200, 2**padding))
                plt.savefig(path_images[:-15] + "resultats" + "/" + "FFT_spatiale_pics_trouvés_" + str(round(f_0 * u)) + "Hz" + ".pdf", dpi = 1)
             
            
            """ Fit second degré """
    
            indice_tot = []
    
            for maxs in peaks :
                
                y0 = maxs 
                
                a = 3
                if y0 <=2 :
                    a = 1
                
                z = np.abs(FFT_demod_padding)[y0-a:y0+a]
                
                x = [u for u in range (y0-a,y0+a) - y0 + 1]
                
                p = np.polyfit(x,z,2)    #on peut utiliser fminsearch pour fitter par une fonction quelconque
                
                imax = -p[1]/(2*p[0])
                
                indice_tot.append(y0 + imax)
                
                
            indice = np.mean(indice_tot)
    
    
            longueur_donde = 2**padding * mmparpixel / (indice) / 1000
    
            if f_exc != f_0 * u :
                longueur_donde = 2**padding * mmparpixel / (2**padding - indice) / 1000
    
    
            if len(peaks) > 1 :
                error_lambda = (np.max(2**padding * mmparpixel / np.asarray(indice_tot) / 1000) - np.min(2**padding * mmparpixel / np.asarray(indice_tot) / 1000))/2
                if f_exc != f_0 * u :
                    error_lambda = (np.max(2**padding * mmparpixel / (2**padding -np.asarray(indice_tot)) / 1000) - np.min(2**padding * mmparpixel / (2**padding -np.asarray(indice_tot)) / 1000))/2      
                
            else : 
                error_lambda = 0
    
            if True :
                if u == 0 :
                    u = 1
                lambda_exp.append([longueur_donde, error_lambda, f_0 * u])
                lambda_tot.append([longueur_donde, error_lambda, f_0 * u])
            if u == 1 :
                dico = add_dico(dico,date,nom_exp,'lambda', longueur_donde)
                dico = add_dico(dico,date,nom_exp,'err_lambda', error_lambda)
    
    
    if False:
        lambda_exp = np.asarray(lambda_exp)
        np.savetxt(path_images[:-15] + "resultats" + "/lambda_err_fexc" + date + "_" + nom_exp + ".txt", lambda_exp, "%s")
        kappa_exp = np.asarray(kappa_exp)
        np.savetxt(path_images[:-15] + "resultats" + "/kappa_err_fexc" + date + "_" +  nom_exp + ".txt", kappa_exp, "%s")
        param_save = param_complets
        param_save.extend(["Paramètres de traitement :",  "debut_las = " + str(debut_las) ,"fin_las = " + str(fin_las),"t0 = " + str(t0) ,"tf = " + str(tf) ])
        param_save.extend(["savgol = " + str(savgol) ,"im_ref = " + str(im_ref), "seuil_arbitraire = " + str(seuil_arbitraire) ])
        param_save.extend(["padding = " + str(padding), "f_0 = " + str(f_0), "nom_exp " + str(nom_exp) ])
        param_save = np.asarray(param_save)
        np.savetxt(path_images[:-15] + "resultats" + "/Paramètres_FFT_lambda_kappa_" + date + "_" + nom_exp + ".txt", param_save, "%s")
        save_dico(dico)
        

#%% Sauvegarde des parametres et resultats
# param_complets = param_complets.tolist()


if True :
    param_complets.extend(["Paramètres de traitement :",  "debut_las = " + str(debut_las)])
    param_complets.extend(["savgol = " + str(savgol) ,"im_ref = " + str(im_ref)])
    param_complets.extend(["padding = " + str(padding), "seuil_arbitraire = " + str(seuil_arbitraire) ])
    param_complets = np.asarray(param_complets)
    np.savetxt(loc_resultats + "\\" + date + "_" + nom_save_file + "/Paramètres_FFT_lambda_kappa_date_traitement_" + str(datetime.now())[:10] + ".txt", param_complets, "%s")
    np.savetxt(loc_resultats + "\\" + date + "_" + nom_save_file + "/ATT" + "/kappa_f_exc_nom_exp_date_traitement_" + str(datetime.now())[:10] + ".txt", kappa_tot, "%s")
    np.savetxt(loc_resultats + "\\" + date + "_" + nom_save_file + "/RDD" + "/lambda_f_exc_nom_exp_date_traitement_" + str(datetime.now())[:10] + ".txt", lambda_tot, "%s")
    


