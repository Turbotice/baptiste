# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 14:03:38 2023

@author: Banquise
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))


def RemoveNodes(img):
    """
    Retire les pixels qui ont plus de 3 voisins d'une image skeletonisée 
    ie Retire les noeuds des branches des fissures
    Retourne la même image sans les noeuds
    """
    output = np.copy(img)
    img2 = img/255
    
    kernel1 = np.array([[1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]])
    
    convolve = cv2.filter2D(src=img2, ddepth=-1, kernel=kernel1)
    nodes = np.where(convolve >= 3)
    output[nodes] = 0
    
    return(output)


def cracks(img, crack_lenght_min):
    """
    Prend en entrée une image skeletonisée
    et la longueur minimale en pixel des fissures que l'on prend en compte
    
    Retourne:
    liste_arg: liste où est listé chaque fissure avec ses coordonnées cartésiennes (en pixel)
    use: liste des fissuures avec longueur > crack_lenght_min
    NX, NY: taille de l'image
    """

    img = 255-img

    img = RemoveNodes(img)

    cc = cv2.connectedComponents(img,connectivity=8)

    liste_arg = [[] for i in range(0,cc[0])]
    NY,NX = cc[1].shape

    for j in range(NX):
        for i in range(NY):
            liste_arg[cc[1][i,j]].append([i,j])

    del liste_arg[0]
 
    use = []
    for i,list_coord in enumerate(liste_arg):
        if len(list_coord) >= crack_lenght_min:
            use.append(i)
    
    return liste_arg, use, NX, NY
      
        
def analyze_cracks(liste_arg, use, plot = True, original = None):
    """
    Input:
    liste_arg: liste où est listé chaque fissure avec ses coordonnées cartésiennes (en pixel)
    use: liste des fissures que l'on va considérer
    plot: if True: trace les droites associées aux fit des fissures (optionnel)
    original: chemin de l'image original pour pouvoir supperposer les droites par dessus (optionnel)
    
    Output:
    angles: array contenant l'angle d'orientation en degré de chaque fissure
    weights: array contenant la longueur en pixel de chaque fissure
    
    """
    Ncomp = len(use)

    angles = np.zeros(Ncomp)
    weights = np.zeros(Ncomp)
    for i,k in enumerate(use):
        y = np.array(liste_arg[k])[:,0]
        x = np.array(liste_arg[k])[:,1]
        try:
            [a,b] = np.polyfit(x,y, 1)
            angles[i] = np.arctan(a)*180/np.pi
            weights[i] = len(x)
            if plot:
                plt.plot(x,a*x+b)
        except:  #les lignes parfaitement horizontales renvoient une erreur (pente infinie)
            angles[i] = 90.
            if plot:
                plt.plot([x[0],x[-1]],[y[0],y[-1]])
                
    if original != None:
        plt.imshow(cv2.imread(original,0),'gray')
    
    return angles, weights


def histo_angles(angles, weights, color=None):
    """
    Trace l'histogramme des angles pondéré par la taille des fissures
    """
    histo = plt.hist(angles,weights=weights,density=True,bins = 50, range=(-90,90),color=color)
    maxi = np.max(histo[0])
    plt.plot([-45,-45],[0,maxi],color='black',linestyle='dotted',alpha=0.7)
    plt.plot([45,45],[0,maxi],color='black',linestyle='dotted',alpha=0.7)
    plt.plot([0,0],[0,maxi],color='black',linestyle='dashed',alpha=0.7)
    plt.xlabel('angle (°)')
    plt.ylabel('densité')
    
    
    """ ### Fit les angles avec 2 gaussiennes
    def gaus(x,a1,a2,x1,x2,sigma1,sigma2):
        return a1*np.exp(-(x-x1)**2/(2*sigma1**2))+a2*np.exp(-(x-x2)**2/(2*sigma2**2))
    
    y = histo[0]
    x = (histo[1][:-1]+ histo[1][1:])/2
    
    neg = angles[np.where(angles<0)]
    negweights = weights[np.where(angles<0)]
    
    pos = angles[np.where(angles>0)]
    posweights = weights[np.where(angles>0)]
    
    s1 = weighted_avg_and_std(neg,negweights)[1]
    s2 = weighted_avg_and_std(pos,weights=posweights)[1]
       
    popt,pcov = curve_fit(gaus,x,y,p0=[0.5,0.5,np.average(neg,weights=negweights),np.average(pos,weights=posweights),s1,s2],bounds=((0, 0, -90, 0, 0, 0), (np.inf, np.inf, 0, 90, np.inf, np.inf)))
    
    x1 = popt[2]
    x2 = popt[3]
    s1 = popt[4]
    s2 = popt[5]
    
    x = np.linspace(-90,90,180)
    plt.plot(x,gaus(x,*popt),color='black')
    
    plt.title(r'$\theta_-=$'+str(int(np.round(x1,0)))+'$\pm$'+str(int(np.round(s1,0)))+'°      '+r'$\theta_+=$'+str(int(np.round(x2,0)))+'$\pm$'+str(int(np.round(s2,0)))+'°')
    """



#Fonctions éclatée qui marche pas
def gaus(x,a1,a2,x1,x2,sigma1,sigma2,histo,angles, weights):
  
  # Fit les angles avec 2 gaussiennes
   y = histo[0]
   x = (histo[1][:-1]+ histo[1][1:])/2
 
   neg = angles[np.where(angles<0)]
   negweights = weights[np.where(angles<0)]
 
   pos = angles[np.where(angles>0)]
   posweights = weights[np.where(angles>0)]
 
   s1 = weighted_avg_and_std(neg,negweights)[1]
   s2 = weighted_avg_and_std(pos,weights=posweights)[1]
    
   popt,pcov = curve_fit(gaus,x,y,p0=[0.5,0.5,np.average(neg,weights=negweights),np.average(pos,weights=posweights),s1,s2],bounds=((0, 0, -90, 0, 0, 0), (np.inf, np.inf, 0, 90, np.inf, np.inf)))
 
   x1 = popt[2]
   x2 = popt[3]
   s1 = popt[4]
   s2 = popt[5]
 
   x = np.linspace(-90,90,180)
   plt.plot(x,gaus(x,*popt),color='black')
 
   plt.title(r'$\theta_-=$'+str(int(np.round(x1,0)))+'$\pm$'+str(int(np.round(s1,0)))+'°      '+r'$\theta_+=$'+str(int(np.round(x2,0)))+'$\pm$'+str(int(np.round(s2,0)))+'°')

   return a1*np.exp(-(x-x1)**2/(2*sigma1**2))+a2*np.exp(-(x-x2)**2/(2*sigma2**2))
