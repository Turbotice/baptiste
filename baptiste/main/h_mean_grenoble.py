# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 17:38:54 2024

@author: Banquise
"""

import os
import baptiste.files.file_management as fm
import numpy as np
import pandas

path ='K:\Gre24\\Data\\'




#%%

dates = ['20241126', '20241127','20241128','20241129','20241202','20241203','20241204']

h_mean = []
h_std = []
acq = []
date_acq = []

for date in dates :
    path_frac = path + date + '\\manip_relation_dispersion\\'
    file_acq = os.listdir(path_frac)
    for i in file_acq : 
        file_frac = os.listdir(path_frac + i)
        for j in file_frac :
            if 'h_tot' in j :
                h_tot = pandas.read_csv(path_frac + i + '\\' + j, sep = '\t', header = 0)
                h_tot = np.asarray(h_tot)
                hmean = h_mean.append(np.mean(h_tot))
                date_acq += [date]
                acq += [i]
                h_std += [np.std(h_tot)]
        
    