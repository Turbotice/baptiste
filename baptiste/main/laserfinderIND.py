# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 15:15:09 2022

@author: Banquise
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 13:08:41 2022

@author: Turbots
"""




import cv2 
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from skimage import feature
from scipy.signal import convolve2d
from scipy.signal import medfilt2d
import os
from PIL import Image
%run parametres_FSD.py
%run Functions_FSD.py
%run display_lib_gld.py



#Va chercher les images et stock leur chemin dasn liste images

path_images, liste_images, titre_exp = import_images(loc, nom_exp, "IND")

#Importe les paramètres de l'experience


facq, texp, Tmot, Vmot, Hw, Larg_ice, Long_ice, tacq, type_exp = import_param (titre_exp)    

mmparpixely, mmparpixelz = import_calibration(titre_exp,date)          
              
# Creates a file txt with all the parameters of the experience and of the analysis
    
param_complets = ["Paramètres d'adimensionnement :",  "lambda_vague = " + str(lambda_vague) , "Ampvague = " + str(Ampvague) ,  "Paramètres d'analyse : ", "debut = " + str(debut) ,"kernel_size = " + str(kernel_size), "kernel_iteration = " + str(kernel_iteration) ,"nbframe = " + str(nbframe) ,"maxsize = " + str(maxsize) , "minsize = " + str(minsize) , "threshold = " + str(threshold) , "sepwb = " + str(sepwb) , "size_crop = " + str(size_crop) , "bordersize = " + str(bordersize), "mmparpixel = " + str(mmparpixel), "Paramètres experimentaux : ", "facq = " + str(facq) , "texp = " + str(texp) , "Tmot = " + str(Tmot) , "Vmot = " + str(Vmot), "Hw = " + str(Hw), "Larg_ice = " + str(Larg_ice), "Long_ice = " + str(Long_ice), "tacq = " + str(tacq), "type_exp = " + str(type_exp)]

#%%

display = False


gradient = False
maximum = True
red_ratio = False
convconv = True

moyenner = False    #faire un moyennage local (convolve2d)
medfilter = True    #mettre un filtre médian et dilate horizontal

save = False
saveplot = False

size_subplot = 3


#%%Trouve et affiche le laser

if display and nbframe > 20 :
    continuer = input ("Attention plus de 20 figures à afficher, continuer ?")
    if not (continuer == "oui") :
        exit()

if convconv :
    maximum = True


im = cv2.imread(path_images + liste_images[0])
im2 = np.array(im)
im_2 = im2 [:,:,2]


y = np.zeros((len(liste_images),(np.shape(im_2)[1])))
xf = np.linspace(-25,25,50)
fconv = (1-xf**2/sigma)*np.exp(-xf**2/(2*sigma))
fconv = fconv - np.mean(fconv)


for i in range (debut, len (liste_images), 1): # int(len(liste_images)/ nbframe)) :
       
    if display :
        fig = plt.figure() 
        axes = []
        u = 1
    
    #Ouverture image et choix du canal Rouge
    
    im = cv2.imread(path_images + liste_images[i])
    
    # SELECTIONNE LA REGION D'INTERET
    
    im2 = np.array(im)
    im_red = im2 [:,:,2]
      
    if display :
        axes.append( fig.add_subplot(size_subplot, 2, u) )
        axes[-1].set_title("Image rouge")  
        plt.imshow(im_red)
        plt.axis()
        u += 1
    
    im_las = np.zeros(((np.shape(im_red)[0]),(np.shape(im_red)[1])))
    im_las_base = np.zeros(((np.shape(im_red)[0]),(np.shape(im_red)[1])))
    
    if moyenner :        
        #Convolution pour smoother l'image  
        siz = 5
        
        im_red = convolve2d (im_red, np.ones((siz,siz))/ siz ** 2 , mode = 'same')
        
    if medfilter :
        #filtre médian et dilate en horizontal
        im_red_traitee = medfilt2d(im_red, med_size)
        
        if display :
            axes.append( fig.add_subplot(size_subplot, 2, u) )
            axes[-1].set_title("Image filtre médian " + str (med_size) )  
            plt.imshow(im_red_traitee)
            plt.axis()
            u += 1
        
        kernel = np.ones((1,k_size), np.uint8)
        
        im_red_traitee = cv2.dilate(im_red_traitee, kernel, iterations=k_iteration)
        
        if display :
            axes.append( fig.add_subplot(size_subplot, 2, u) )
            axes[-1].set_title("Image dialted size " + str (k_size) + " iterations " + str(k_iteration))  
            plt.imshow(im_red_traitee)
            plt.axis()
            u += 1
            
            
     
    [nx,ny] = np.shape(im_red_traitee)
    
    
    for k in range (ny): # colonne
    
    # DETECTION MAX GRADIENT
        if convconv : 
            
            im_conv = np.convolve(im_red_traitee[:,k], fconv, mode = 'same' )
            
            if display and (k == 400 or k == 200) :
                axes.append( fig.add_subplot(size_subplot, 2, u) )
                axes[-1].set_title("Convolve avec profil " + str(k) + ' ieme colonne' )  
                plt.plot(im_red_traitee[:,k])
                plt.plot(im_conv)
                plt.axis()
                u += 1
            
              #DETECTION MAX INTENSITE 
                
                 
        if maximum :
            
            if convconv :
                colonne_k = im_conv
            else :
                colonne_k = im_red_traitee[:,k]
               
            #trouver le max
            border = 50

            indice =  np.argmax(colonne_k[border:-border])
            indice = indice + border
              
            if display:
                im_las_base[indice,k] = 255 
            
            #sub_pixellaire
            
            a=5 #zone pour le fit polynome
            
            imin = max(1,indice-a)
            imax = min(nx,indice+a)
              
            z = colonne_k[imin:imax]
            x = [ u for u in range (imin-indice,imax-indice)]
                
            p = np.polyfit(x,z,2)  #on peut utiliser fminsearch pour fitter par une fonction quelconque
                
            imax = -p[1]/(2*p[0])
            y[i,k] = indice + imax
            
            
            #creation d'une image pour la superposition
            if indice + imax > 0 and indice + imax < nx and display:
                im_las[int(indice + imax) - 1:int(indice + imax) + 1,k] = 255 
                
                
    #      MONTRE LA DETECTION EN DIRECTE SUPERPOSEE A L'IMAGE BRUTE  
    if display :  #length(x)<11
        im[:,:,2] = im[:,:,2]
        im[:,:,1] = im_las_base
        im[:,:,0] = im_las
        if display :
            axes.append( fig.add_subplot(size_subplot, 2, u) )
            axes[-1].set_title("Image " + str(i) + " / " + str(len(liste_images)))  
            plt.imshow(im)
            plt.axis()
        if save :
            plt.imsave(path_images[:-15] + "resultats" + "/image_superposée_wow.tiff" , im)   

    
    print("Image " + str(i + 1) + " sur " + str(len(liste_images)))
    
    if saveplot :
        plt.savefig(path_images[:-15] + "resultats" + "/" + name_fig_LAS + "plot_image_" + str(i) + ".png", dpi = 2000)
 
   
#%% ENREGISTRE LES DONNEES DANS UN FICHIER .txt


param_complets = np.array(param_complets)


save = True

if save :
    np.savetxt(path_images[:-15] + "resultats" + "/Paramètres_IND", param_complets, "%s")
    np.save(path_images[:-15] + "resultats" + "/positionLAS_IND.npy", y)
#%%charge les data dans y

folder_results = path_images[:-15] + "resultats"
name_file = "positionLAS_IND.npy"
y = np.load(folder_results + "\\" + name_file)

    
#%%fit en gaussienne
display = False
save = False
detrender = False
fit_gauss = False

import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import detrend
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy import misc
"""REMPLIR LE TROU"""

init = 1000 #IJSP2 226 IJSP3 155 IJSP4 1000
end = 1700 #IJSP2 2200 IJSP3 1680 IJSP4 1700
pas = 1


liste_e = []
liste_e_gauss = []
liste_sigma =[]
liste_A =[]
liste_ind = []

for j in range (init,end, pas): #(y.shape[0]):
    laser = y[j,:] * mmparpixelz
    
    if detrender :
        laser = detrend(laser)
    
    #pout TIPP1 : laser[1360:1528] = np.nan (position de la tige)
    
    debzone = 1270
    finzone = 1450
    
    laser[debzone:finzone] = np.nan 
    
    # laser[1370:1640] = np.nan #IJSP3
    # 1095 1370 #IJSP2
    # laser[1270:1450] = np.nan #IJSP4
    
    # f = interp1d(x,laser,kind = 'next',  fill_value = 'extrapolate')
    
    interplas = pd.DataFrame(laser)
    
    orderinterp = 1
    interplas = interplas.interpolate(method='polynomial', order=orderinterp)
    
    interplas = np.asarray(interplas)
    
    interplas = interplas[:,0]
    
    lensavgol = 401
    data_sav = savgol_filter(interplas,lensavgol,1)
    
    """FIT GAUSSIENNE"""
    
    def gauss(x, H, A, x0, sigma):
        return H + A *  - np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
    
    x = np.arange(0,len(interplas),1) * mmparpixely
    
    #pour TIPP1 : guess = [7 * mmparpixely,60 * mmparpixely,1460 * mmparpixely,195 * mmparpixely]
    
    
    guess = [7 * mmparpixely,60 * mmparpixely,1460 * mmparpixely,195 * mmparpixely]
    if fit_gauss :
        parameters, covariance = curve_fit(gauss, x, data_sav, p0 = guess)
    
    if display :
        figurejolie()
        
        joliplot(" ","",  x, interplas, color = 1, legend = 'trou bouché', exp = False)
        joliplot(" ","",  x, laser, color = 2, legend = 'laser détecté', exp = False)
        if fit_gauss :
            joliplot(" ","",  x, gauss(x,parameters[0],parameters[1],parameters[2],parameters[3]), color = 3, legend = 'gaussienne fittée', exp = False)
        joliplot(r"x (mm)",r"y (mm)",  x, data_sav, color = 4, legend = 'savgol', exp = False, title = "image num "+ str(int (j)) )
    
    """LONGUEUR DE LA DEFORMATION"""
    #1 lisser
    
    
    
    
    
    #2 droite pour avoir la longueur originale
    
    droite = np.linspace(interplas[0],interplas[-1], len(interplas))
    
    #3 calculer les longueurs
    len_droite = 0
    len_las = 0
    for i in range (len (interplas)-1):
        len_droite += np.sqrt( (droite[i]-droite[i+1])**2 + (x[i] - x[i+1])**2 )
        len_las += np.sqrt( (x[i] - x[i+1])**2 + (data_sav[i] - data_sav[i+1])**2 )
        
    e = len_las/len_droite
    
    
    liste_e.append(e)
    if fit_gauss :
        liste_sigma.append(parameters[3])
    
    if np.mod(int((j - init)/pas) + 1,100)==0:
        print('On processe l image numero: ' + str(int ((j - init)/pas)) + ' sur ' + str(int((end-init) / pas)))
        figurejolie()
        
        joliplot(r"x (mm)",r"y (mm)",  x, interplas, color = 2, legend = False, exp = False)
        joliplot(r"x (mm)",r"y (mm)", x, laser, color = 2, legend = 'laser détecté', exp = False)
        if fit_gauss :
            joliplot(r"x (mm)",r"y (mm)",  x, gauss(x,parameters[0],parameters[1],parameters[2],parameters[3]), color = 3, legend = 'gaussienne fittée', exp = False)
        joliplot(r"x (mm)",r"y (mm)",  x, data_sav, color = 4, legend = 'savgol', exp = False, title = "image num "+ str(int ((j - init)/pas)))


    #élongation locale
    # xgauss =  np.arange(0,len(interplas),1) * mmparpixely
    # def gaussfull(x):
    #     return parameters[1] *  - np.exp(-(x - parameters[2]) ** 2 / (2 * parameters[3] ** 2))

    # yp = misc.derivative(gaussfull, xgauss)
    
    # if display :
    #     figurejolie()
    #     plt.plot (xgauss, abs(yp))
    #     plt.plot(xgauss, gaussfull(xgauss),parameters[0],parameters[1],parameters[2],parameters[3])
        
    # liste_e_gauss.append(np.sqrt(1 + max(yp)**2))
    
    
    #Indentation (pos d'un point qui avance)
    if j == init:
        ind_init = data_sav[debzone - 20]
    liste_ind.append(data_sav[debzone - 20] - ind_init)
    
   
liste_j = np.arange(init,end,pas) 
figurejolie()
plt.plot(liste_j, liste_ind)

#%%

save = True


array_final_es = [liste_e] + [liste_sigma]
array_final_es = np.asarray(array_final_es)
array_imgind = [liste_j]+ [liste_ind]
array_imgind = np.asarray(array_imgind)
param_complets.extend(["Paramètres de traitement :",  "lensavgol = " + str(lensavgol) , "init = " + str(init), "end = " + str(end) ,"pas = " + str(pas), "zone bouchée = " + str( debzone) + " " + str(finzone) ,"mmparpixely = " + str(mmparpixely),"mmparpixelz = " + str(mmparpixelz), "type_exp = " + str(type_exp), "ordreinterp = " + str(orderinterp), "detrender = " + str(detrender)])
param_complets = np.asarray(param_complets)
array_final_egauss = [liste_j] + [liste_e_gauss]
if save :
    # np.savetxt(path_images[:-15] + "resultats" + "/elongation_sigma.txt", array_final_es, "%s")
    # np.savetxt(path_images[:-15] + "resultats" + "/Paramètres_IND.txt", param_complets, "%s")
    # np.savetxt(path_images[:-15] + "resultats" + "/image_elongationgauss.txt", array_final_egauss, "%s")
    np.savetxt(path_images[:-15] + "resultats" + "/image_indentation.txt", array_imgind, "%s")







