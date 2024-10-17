# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 14:21:24 2023

@author: Banquise

traitement de signal à base de medfilter et tout ca
"""

from scipy.signal import savgol_filter

def savgol (data, length, order) :
    return savgol_filter(data, length, order)