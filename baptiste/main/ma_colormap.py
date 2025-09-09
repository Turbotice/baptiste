# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 11:47:49 2025

@author: Banquise
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from colorspacious import cspace_convert
import colorsys

def rgb_to_jch(rgb):
    cam = cspace_convert(rgb, "sRGB1", "CAM02-UCS")
    J = cam[:, 0]
    a = cam[:, 1]
    b = cam[:, 2]
    C = np.sqrt(a**2 + b**2)
    h = np.arctan2(b, a) * 180 / np.pi
    h %= 360
    return np.stack([J, C, h], axis=1)

def jch_to_rgb(jch):
    J, C, h = jch[:,0], jch[:,1], jch[:,2]
    h_rad = np.deg2rad(h)
    a = C * np.cos(h_rad)
    b = C * np.sin(h_rad)
    cam = np.stack([J, a, b], axis=1)
    rgb = cspace_convert(cam, "CAM02-UCS", "sRGB1")
    return np.clip(rgb, 0, 1)

def create_macolormap():
    # Couleur personnalisée HSV T187 L58 S35
    h = 187 / 360
    s = 0.35
    l = 0.58
    custom_rgb = colorsys.hls_to_rgb(h, l, s)
    
    # Points clés : départ, bleu foncé, couleur custom, bleu clair
    key_rgb = np.array([
        [0.0, 0.0, 0.1],           # Presque noir bleu
        [0.05, 0.1, 0.3],          # Bleu foncé
        custom_rgb,               # Couleur HSV donnée
        [0.7, 0.85, 1.0]           # Bleu très clair
    ])
    
    # Convertir en JCh
    key_jch = rgb_to_jch(key_rgb)
    
    # Paramètres
    n = 256
    J_start = key_jch[0,0] + 7 # noir plus lumineux
    J_end = key_jch[-1,0] + 7  # fin un peu plus lumineuse
    J = np.linspace(J_start, J_end, n)
    
    # Interpolation C et h sur J
    from scipy.interpolate import interp1d
    
    J_key = key_jch[:,0]
    C_key = key_jch[:,1]
    h_key = key_jch[:,2]
    
    interp_C = interp1d(J_key, C_key, kind='linear', fill_value="extrapolate")
    interp_h = interp1d(J_key, h_key, kind='linear', fill_value="extrapolate")
    
    C = interp_C(J)
    h = interp_h(J)
    
    # Construit la table JCh → RGB
    jch = np.stack([J, C, h], axis=1)
    rgb_interp = jch_to_rgb(jch)
    
    
    # Créer colormap
    ma_cm = ListedColormap(rgb_interp, name="ma_cm")
    import matplotlib.cm as cm
    
    # # # # Ajoute ta colormap à l'objet cm (fonctionne comme un dictionnaire)
    cm.register_cmap(name="ma_cm", cmap=ma_cm)
    setattr(cm, "ma_cm", ma_cm)
    

    cmap_inverse = ma_cm.reversed()

    cm.register_cmap(name="ma_cm_r", cmap=cmap_inverse)
    setattr(cm, "ma_cm_r", cmap_inverse)
    
    
create_macolormap()