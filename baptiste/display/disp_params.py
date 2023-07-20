# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 11:53:37 2023

@author: Banquise

Display and import params from previous experiments
"""

import baptiste.files.dictionaries as dic
import baptiste.display.display_lib as disp
import baptiste.math.RDD as rdd
import os as os
import numpy as np
import matplotlib.pyplot as plt

path = 'W:\Banquise\Rimouski_2023\Traitements_donnees\\baptsarahantonin\\resultats\\voie_Z_20231303_1714\\k_de_omega_20230523_185336\\'
path = 'Y:\Banquise\Baptiste\\Présentations\\Images_presentation\\Figures\\fit_pesante_propre_20230616_154006\\'
path = 'W:\Banquise\Rimouski_2023\Traitements_donnees\\baptsarahantonin\\voie_Z_20230313_1714\\resultats\\phase_shift_histogrammé_fois4_6-12_20230620_110156\\'
path =' W:\Banquise\\Rimouski_2023\\Traitements_donnees\\baptsarahantonin\\voie_Z_20230313_1714\\resultats\\flexion_20230619_171308\\'
path = 'W:\\Banquise\\Rimouski_2023\\Traitements_donnees\\baptsarahantonin\\voie_Z_20230313_1714\\resultats\\theta_f_rawdata_20230620_121018\\'


files = os.listdir(path)

for i in files :
    if 'params' in i :
        params = dic.open_dico(path + i)
    if 'data_theta' in i :
        data = dic.open_dico(path + i)

# k = data['data']['data'][1][0]
# omega = data['data']['data'][1][1] * 2 * np.pi

# k_nice = np.linspace(0,np.max(k), 1000)

# disp.figurejolie()
# # disp.joliplot(data['data']['data'][0][0],data['data']['data'][0][1],k, omega)

# disp.joliplot(r'k ($m^{-1}$)', r'$\omega (Hz)$', k_nice, rdd.RDD_gravitaire(k_nice), exp = False, color = 5, legend = 'Gravitary dispersion relation')

# disp.joliplot(r'k ($m^{-1}$)', r'$\omega (Hz)$', k, omega, color = 2, legend = 'Experimental data', zeros=True)

# plt.grid('on')

