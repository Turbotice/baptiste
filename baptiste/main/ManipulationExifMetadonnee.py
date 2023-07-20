# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 15:10:37 2023

@author: Banquise

Meatdata des img
"""

from exif import Image
import pickle as pkl
 
path = "W:\Banquise\\Baptiste\\micmac\\test_stereo_tie16\\2images\\"

image = "Basler_a2A1920-160ucBAS__40232065__20230309_195209416_0000.tiff"



with open(path + image, 'rb') as img_file:
    img = Image(img_file)
    
print(img.has_exif)

# img.focal = "16mm"