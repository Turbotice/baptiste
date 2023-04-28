# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 18:16:31 2023

@author: Banquise

outils utiles
"""
import numpy as np

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