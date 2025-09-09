# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 18:16:31 2023

@author: Banquise

outils utiles
"""
import numpy as np
from datetime import datetime


def normalize(x) :
    return (x - np.min(x)) / (np.max(x) - np.min(x)) 

def max_fit2(x, xmax, a = 5):
    
    nx = len(x)
    indice = xmax
    
    imin = max(1,indice-a)
    imax = min(nx,indice+a)
      
    z = x[imin:imax]
    zone_max = [ u for u in range (imin-indice,imax-indice)]
        
    p = np.polyfit(zone_max,z,2)  #on peut utiliser fminsearch pour fitter par une fonction quelconque
        
    imax = -p[1]/(2*p[0])
    pos_max = indice + imax
    
    val_max = p[1]*imax + p[2] + imax**2*p[0]
    
    return val_max, pos_max

def min_fit2(x, xmin, a = 5):
    
    nx = len(x)
    indice = xmin
    
    imin = max(1,indice-a)
    imax = min(nx,indice+a)
      
    z = x[imin:imax]
    zone_min = [ u for u in range (imin-indice,imax-indice)]
        
    p = np.polyfit(zone_min,z,2)  #on peut utiliser fminsearch pour fitter par une fonction quelconque
        
    imax = -p[1]/(2*p[0])
    pos_min = indice + imax
    
    val_min = p[1]*imax + p[2] + imax**2*p[0]
    
    return val_min, pos_min


def sort_listes(a,b, reverse=False) :
    """
    Trie deux listes dans l'ordre en fonction de la première'

    Parameters
    ----------
    a : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.
    reverse = False pour ordre croissant

    Returns
    -------
    a_sort : TYPE
        DESCRIPTION.
    b_sort : TYPE
        DESCRIPTION.

    """
    
    list_tot = [a,b]
    
    a_sort = np.asarray(sorted(np.rot90(np.asarray(list_tot)).tolist(), key=lambda student: student[0], reverse = reverse))[:,0]
    b_sort = np.asarray(sorted(np.rot90(np.asarray(list_tot)).tolist(), key=lambda student: student[0], reverse = reverse))[:,1]

    return a_sort, b_sort

def datetimenow (date = True, time = True, micro_sec = False) :
    date_time = ""
    if date :
        date_time += str(datetime.now())[:4] + str(datetime.now())[5:7] + str(datetime.now())[8:10]
    if time :
        if date_time != "" :
            date_time += "_"
        date_time += str(datetime.now())[11:13] + str(datetime.now())[14:16]+ str(datetime.now())[17:19]
    if micro_sec :
        if date_time != "" :
            date_time += "_"
        date_time += str(datetime.now())[20:]
    
    return date_time

def commun_space(f1, f2, phase1, phase2, pas_f): 
    f_new = np.arange( np.min( (np.min(f1), np.min(f2)) ), np.max( (np.max(f1), np.max(f2)) ) + pas_f , pas_f)
    f1_new = np.array([])
    f2_new = np.array([])
    phase1_new = np.array([])
    phase2_new = np.array([])
    
    for i in range(len(f1)) :
        f1[i] = round(f1[i],5)
    for j in range(len(f2)) :
        f2[j] = round(f2[j],5)
        

    for ii in range (len(f_new)) :
        if round(f_new[ii], 5) in f2 and round(f_new[ii],5) in f1 :
            f1_new = np.append(f1_new, f_new[ii])
            f2_new = np.append(f2_new, f_new[ii])
            phase1_new = np.append(phase1_new, phase1[np.where(round(f_new[ii],5) == f1)])
            phase2_new = np.append(phase2_new, phase2[np.where(round(f_new[ii],5) == f2)])
            
    return f1_new, f2_new, phase1_new, phase2_new
    