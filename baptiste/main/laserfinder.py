# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 13:08:41 2022

@author: Turbots
"""

#%%IMPORTATION DES DONNEES


import cv2 
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from skimage import feature
from scipy.signal import convolve2d
from scipy.signal import medfilt2d
from scipy import ndimage
import os
from PIL import Image

import baptiste.display.display_lib as disp
import baptiste.experiments.import_params as ip
import baptiste.files.file_management as fm
import baptiste.image_processing.image_processing
import baptiste.files.dictionaries as dic
import baptiste.tools.tools as tools


date = '240116'
nom_exp = 'CCM22'
exp = True
exp_type = 'LAS'


dico, params, loc = ip.initialisation(date, nom_exp, exp = True, display = False)



        
              
# Creates a file txt with all the parameters of the experience and of the analysis

    
# param_complets = ["Paramètres d'adimensionnement :",  "lambda_vague = " + str(lambda_vague) , "Ampvague = " + str(Ampvague) ,  "Paramètres d'analyse : ", "debut = " + str(debut) ,"kernel_size = " + str(kernel_size), "kernel_iteration = " + str(kernel_iteration) ,"nbframe = " + str(nbframe) ,"maxsize = " + str(maxsize) , "minsize = " + str(minsize) , "threshold = " + str(threshold) , "sepwb = " + str(sepwb) , "size_crop = " + str(size_crop) , "bordersize = " + str(bordersize), "mmparpixel = " + str(mmparpixel), "Paramètres experimentaux : ", "facq = " + str(facq) , "texp = " + str(texp) , "Tmot = " + str(Tmot) , "Vmot = " + str(Vmot), "Hw = " + str(Hw), "Larg_ice = " + str(Larg_ice), "Long_ice = " + str(Long_ice), "tacq = " + str(tacq), "type_exp = " + str(type_exp)]


#%%PARAMETRES


MODE = "go"
# MODE = "go"
params['nbframe'] = 20000
params['med_size'] = 20
params['debut'] = 0
params['bas_LAS'] = 300#500 CCCS2#570 # 380#180 RBT01
params['haut_LAS'] = 650#700 CCCS2#820 #RCSC1 780 CCCS1#260 RBT01
params['x_0'] = 30
params['x_f'] = 380 #crop

display = True
display_result_only = True


gradient = False
maximum = True
red_ratio = False
convconv = True

moyenner = False    #faire un moyennage local (convolve2d)
medfilter = False    #mettre un filtre médian et dilate horizontal

save = True
saveplot = False


if MODE == "test" :
    display = True
    save = False
    params['nbframe'] = 3
    
if MODE == "go" :
    display = False
    save = True
    params['nbframe'] = len (params['liste_images'])
    
    
if params['nbframe'] > len(params['liste_images']):
    params['nbframe'] = len (params['liste_images'])
    




#%%CONVOLUTION

if display and params['nbframe'] > 20 :
    continuer = input ("Attention plus de 20 figures à afficher, continuer ?")
    if not (continuer == "oui") :
        exit()

if convconv :
    maximum = True



size_conv = 30
precision_conv = 50
xf = np.linspace(-size_conv,size_conv,precision_conv)
sigma = 20
fconv = np.exp(-xf**2/(2*sigma))
# fconv = (1-xf**2/sigma)*np.exp(-xf**2/(2*sigma))
fconv = fconv - np.mean(fconv)
if False :
    disp.figurejolie()
    plt.plot(xf, fconv)




im = cv2.imread(params['path_images'] + params['liste_images'][0])
im_2 = im[params['bas_LAS']:params['haut_LAS'],:,2]
y = np.zeros((len(params['liste_images']),(np.shape(im_2)[1])))




#%%TRAITEMENT PRINCIPAL
for i in range (params['debut'], len (params['liste_images']), int(len(params['liste_images'])/ params['nbframe'] )) :
    
    #Ouverture image et choix du canal Rouge
    
    im = cv2.imread(params['path_images'] + params['liste_images'][i], cv2.IMREAD_UNCHANGED)
    
    # SELECTIONNE LA REGION D'INTERET
    
    im = np.array(im[params['bas_LAS']:params['haut_LAS'],:,:])
    
    im = np.array(im[:,params['x_0']:params['x_f'],:])
    
    I_min, I_max = np.min(im), np.max(im)
    im = ((im-I_min) / (I_max-I_min) * 255).astype(np.uint8)
    im_red = im[:,:,2].copy()
    # I_min, I_max = np.min(im_red), np.max(im_red)
    # im_red = ((im_red-I_min) / (I_max-I_min) * 255).astype(np.uint8)
    param1 = 30
    param2 = 30
    param3 = 16
    dst = cv2.fastNlMeansDenoising(im_red,None,param1,param2,param3)
    
    
    
    if display and not display_result_only :
        disp.figurejolie()
        disp.joliplot('', '', (), (), image = im_red, title = "Image rouge")
    
    im_las = np.zeros(((np.shape(im_red)[0]),(np.shape(im_red)[1])), dtype = np.uint8)
    im_las_base = np.zeros(((np.shape(im_red)[0]),(np.shape(im_red)[1])), dtype = np.uint8)
    
    if moyenner :        
        #Convolution pour smoother l'image  
        siz = 5
        
        im_red = convolve2d (im_red, np.ones((siz,siz))/ siz ** 2 , mode = 'same')
        
    if medfilter :
        #filtre médian et dilate en horizontal
        im_red_traitee = medfilt2d(im_red, params['med_size'])
        
        if display and not display_result_only:
            disp.figurejolie()
            disp.joliplot('', '', (), (), image = im_red_traitee, title = "Image filtre médian " + str (params['med_size']) )
        
        k_size = 8
        k_iteration = 4 #OPTI pour laser sur le téco
        
        kernel = np.ones((1,k_size), np.uint8)
        
        im_red_traitee = cv2.dilate(im_red_traitee, kernel, iterations=k_iteration)
        
        if display and not display_result_only :
            disp.figurejolie()
            disp.joliplot('', '', (), (), image = im_red_traitee, title = "Image dialted size " + str (k_size) + " iterations " + str(k_iteration))
            
    else :
        im_red_traitee = dst # im_red
        if display and not display_result_only:
            disp.figurejolie()
            disp.joliplot('', '', (), (), image = im_red_traitee, title = "Image filtre denoise " + str (param1) + " " + str (param2) + " "+ str (param3))
     
    [nx,ny] = np.shape(im_red_traitee)
    
    
    for k in range (ny): # colonne
    
    # DETECTION MAX GRADIENT
        if convconv : 
            xXxtentacion = np.arange(0,nx)
            im_conv = np.convolve(im_red_traitee[:,k], fconv, mode = 'same' )
            
            if display and  (k == 200 or k == 700 or k == 1500) and not display_result_only :
                disp.figurejolie()
                disp.joliplot('', '', xXxtentacion, im_red_traitee[:,k], title = "Convolve avec profil " + str(k) + ' ieme colonne', exp =False) 
                disp.joliplot('', '', xXxtentacion, im_conv, title = "Convolve avec profil " + str(k) + ' ieme colonne', exp = False)                
                 
        if maximum :
            
            if convconv :
                colonne_k = im_conv
            else :
                colonne_k = im_red_traitee[:,k]
               
            #trouver le max
            border = int(precision_conv/2)

            indice =  np.argmax(colonne_k[border:-border])
            indice = indice + border
              
            if display:
                im_las_base[indice,k] = 255 
            
            #sub_pixellaire
            
            a=4 #zone pour le fit polynome
            
            
            
            
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
    if display:  #length(x)<11
        im_display = np.zeros(im.shape, dtype = np.uint8)
        im_display[:,:,2] = im[:,:,2]   #Image de base
        im_display[:,:,1] = im_las_base #Laser detecté
        im_display[:,:,0] = im_las      #Laser detecté subpixellaire
        if display :
            disp.figurejolie()
            disp.joliplot('', '', (), (), image = im_display, title = "Image " + str(i) + " / " + str(len(params['liste_images']) ))
            plt.axis()
        
        if save :
            plt.imsave(params['path_images'][:-15] + "resultats" + "/image_superposée_wow.tiff" , im_display)   

    
    if np.mod(i,100)==0:
        print('On processe l image numero: ' + str(i) + ' sur ' + str(len(params['liste_images'])))
    
    if saveplot :
        plt.savefig(params['path_images'][:-15] + "resultats" + "/" + "plot_image_" + str(i) + ".pdf", dpi = 1)
    
    # writeVideo(vid,getframe(gcf))
    


# close(vid)


t = [u for u in range (0,np.shape(y)[0])]
x = np.array([u for u in range (0,np.shape(y)[1])]) * params['mmparpixel']
array_final = np.vstack([x, y])
t.insert(0,0)
t = np.array(t) / params['facq']

array_final = np.c_[t, array_final]

        # ENREGISTRE LES DONNEES DANS UN FICHIER .NPY
        


if save :
    np.save(params['path_images'][:-15] + "resultats" + "/positionLAS.npy", array_final)
    dic.save_dico(params, params['path_images'][:-15] + "resultats//params_las" + str(tools.datetimenow()) + ".pkl" )
    
    dic.add_dico(dico, date, nom_exp, 'Analyse_LAS', 'OUI')
    
    print('LASER SAVED')


