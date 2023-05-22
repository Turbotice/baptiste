# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 18:16:31 2023

@author: Banquise

outils utiles
"""
import numpy as np
from datetime import datetime

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