# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 14:44:25 2023

@author: Banquise
"""

import numpy as np
import matplotlib.pyplot as plt

import baptiste.files.dictionaries as dic
import baptiste.display.display_lib as disp

path = 'W:\Banquise\Rimouski_2023\Traitements_donnees\\baptsarahantonin\\voie_Z_20230313_1714\\resultats\\20230704_180701_flexion+h+H26dm_f0-32Hz_lm7200\\params_20230704_180701_flexion+h+H26dm_228684.pkl'

uu = dic.open_dico(path)

plt.figure()

f = uu['228684']['data']['data']['data'][1][1] / 2 / np.pi

disp.joliplot(r'f (Hz)', r'$\theta$', f, uu['theta_f'], color = 2)