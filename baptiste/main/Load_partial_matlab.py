# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 15:09:42 2023

@author: Banquise
"""

import h5py

params = {}

loc_h = 'D:\\Banquise\\Baptiste\\Resultats_video\\d231013\\exp_analogue_geo\\test_2_13_10_23\\traitement\\sorted\\'

params['PIV_file'] = 'Height_in_micron_surf_100001to105001.mat'

f = h5py.File(loc_h + params['PIV_file'], 'r')
# u = f['H'][:,20,20]

# with h5py.File(loc_h + params['PIV_file']) as f:
#     data = [f[element[0]][0,0] for element in f['H']]