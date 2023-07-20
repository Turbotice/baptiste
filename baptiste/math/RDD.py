# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 17:46:00 2023

@author: Banquise

Fonctions pour les RDD
"""
import numpy as np

def RDD_gravitaire(k):
    return np.sqrt(9.81 * k)

def RDD_pesante (k, drhoh) :
    return np.sqrt(9.81 * k * ((1 + drhoh * k)**(-1)))

def RDD_pesante_flexion (k,drhoh, Dsurrho) :
    return np.sqrt((9.81 * k + Dsurrho * k **5) * ((1 + drhoh * k)**(-1)))

def RDD_flexion (k, Dsurrho) :
    return np.sqrt(9.81 * k + Dsurrho * k **5)

def RDD_capillaire_flexion (k,gammasurrho, Dsurrho) :
    return np.sqrt(9.81 * k + gammasurrho * k**3 + Dsurrho * k **5)

def RDD_pesante_flexion_depth(k,drhoh, Dsurrho):
    return np.sqrt((9.81 * k + Dsurrho * k **5) * (( (np.tanh(2 * k))**(-1) + drhoh * k)**(-1)))

def RDD_hfit_depth(k, h):
    return np.sqrt( (9.81 * k + (6.07e9 * h**3 * k**5) / (12 * (1 - 0.37**2) * 1000) ) / (np.tanh( 2.6 * k)**-1 + (900 * h * k)/1000 ) )

def RDD_capilaire(k, gammasurrho):
    return np.sqrt(9.81 * k + k**3 * gammasurrho)

