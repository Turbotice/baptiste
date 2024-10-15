# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 12:25:04 2024

@author: Banquise
"""
import pandas
import pickle 
import cv2 
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from skimage import feature
from scipy.signal import convolve2d
from scipy.signal import savgol_filter, gaussian
from scipy.signal import medfilt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy import stats
import scipy.fft as fft
import os
from PIL import Image

import baptiste.display.display_lib as disp
import baptiste.experiments.import_params as ip
import baptiste.signal_processing.fft_tools as ft
import baptiste.image_processing.image_processing as imp
import baptiste.math.fits as fits
import baptiste.math.RDD as rdd
import baptiste.files.save as sv
import baptiste.files.dictionaries as dic
import baptiste.tools.tools as tools

#%%

# read file line by line
import numpy as np
path = 'K:\Share_hublot\\Data\\0221\\Drones\\bernache\\pos.txt'




file = open( path, "r")
lines = file.readlines()
lines = np.array(lines)

file.close()


lines[int(len(lines)/ 2)]
#%% Read str file (position drone)
import os 
pathh = 'K:\Share_hublot\\Data\\0223\\Drones\\mesange\\'

dossiers = os.listdir(pathh)

for i in dossiers :
    fichiers = os.listdir(pathh + i)
    for j in fichiers :
        if j[-3,:] == 'SRT' :
            file = open( pathh + i + "\\" + j ,"r")
            lines = file.readlines()
            lines = np.array(lines)
    
            file.close()
            
            print(lines[int(len(lines) / 6 / 2) + 1])
        
#%% read str
    
import os 
pathh = 'K:\Share_hublot\\Data\\0226\\Drones\\mesange\\'

dossiers = os.listdir(pathh)

i = dossiers[26]
fichiers = os.listdir(pathh + i)
for j in fichiers :
    if j[-3,:] == 'SRT' :
        file = open( pathh + i + "\\" + j, "r")
        lines = file.readlines()
        lines = np.array(lines)

        file.close()
        
        print(lines[int(len(lines) / 6 / 2) + 1])
        print(lines[int(len(lines) / 6 / 2) + 2])
        print(lines[int(len(lines) / 6 / 2) + 3])
        print(lines[int(len(lines) / 6 / 2) + 4])
        print(lines[int(len(lines) / 6 / 2) + 5])
        print(lines[int(len(lines) / 6 / 2) + 6])
    
#%% lat et long moyenne

path = 'K:\Share_hublot\\Data\\0226\\Drones\\mesange\\pos.txt'

pos = np.loadtxt(path)

mean_lat = np.mean(pos[:,0])
        
mean_long = np.mean(pos[:,1])


#%% Open flight record

for i in dossiers :
    files = os.listdir(path + i)
    for j in files :
        if 'DJIFlightRecord' in j :
            flight = pandas.read_csv(path + i + '\\' + j)



#%% path drone

path_drone = 'K:\Share_hublot\\Summary\\'

a = pandas.read_csv(path_drone + '0226_path_drone.txt', sep = ' ', header = None)

path_data = 'K:\Share_hublot\\Data\\0226\\Drones\\mesange'

files = os.listdir(path_data)

a = np.asarray(a)

datas = np.array([])

for i in a[:,1] :
    if i in files :
        datas = np.append(datas, path_data + '\\' + i)
        
np.savetxt(path_drone + 'dronesss.txt', datas)


#%% Timetable

date = '0226'

path_drones = 'K:\Share_hublot\\Summary\\'

datas = pandas.read_csv(path_drone + date + '_path_drone.txt', header= None)
datas = np.asarray(datas)[:,0]

time = dict()
tf = np.zeros(len(datas), dtype = object)
t0 = np.zeros(len(datas), dtype = object)

u = 0
blbl = np.array([], dtype = object)
for file in datas :
    print(file)
    
    a = os.listdir(file)
    blbl = np.array([], dtype = object)

    for i in a :
        if i[-3:] == 'SRT' :
            strr = open( file + '\\' + i, "r")
            lines = strr.readlines()
            lines = np.array(lines)

            strr.close()
            if file[37] == 'm' :                     #ATTTENTION : 'm' pour 0221, 0226, 'M' pour 0223
                t0[u] = int(i[-17:-11]) - 10000
                tf[u] = int(lines[-3][-13:-11]+ lines[-3][-10:-8]+lines[-3][-7:-5]) - 10000
                blbl = np.array([t0[u], tf[u]])
            if file[37] == 'b':
                t0[u] = int(i[-17:-11]) + 50000
                tf[u] = int(lines[-3][-13:-11]+ lines[-3][-10:-8]+lines[-3][-7:-5]) + 50000
                blbl = np.array([t0[u], tf[u]])

        if i[-3:] == 'JPG' :
            if file[37] == 'm':                  #ATTTENTION : 'm' pour 0221, 0226, 'M' pour 0223
                blbl = np.append(blbl, int(i[-17:-11]) - 10000)
            if file[37] == 'b':
                blbl = np.append(blbl, int(i[-17:-11]) + 50000)
            tf[u] = np.max(blbl)
            t0[u] = np.min(blbl)
        
    time[file] = blbl


    u += 1
    
timetable = np.vstack((datas, t0))
timetable = np.vstack((timetable, tf))

#%% plot timetable

t = range(np.min(t0), np.max(tf), 1)

disp.figurejolie(width = 15)


for j in range (timetable.shape[1]) :
    t00 = timetable[1,j]
    tff = timetable[2,j]
    tt = np.array([])
    x_t = np.array([])
    if timetable[0,j][37] =='m' :
        for i in t :
            if i <= tff and i >= t00 :
                tt = np.append(tt,i)
                x_t = np.append(x_t,1)
        disp.joliplot('t (UTC)', 'Instrument', tt,x_t, exp = False, color = 5, linewidth = 10)
        plt.annotate(timetable[0,j][46:48], [np.min(tt), 1])
    if timetable[0,j][37] =='b' :                                       #ATTENTION des fois 'b' des fois 'B', dépend de la date
        for i in t :
            if i <= tff and i >= t00 :
                tt = np.append(tt,i)
                x_t = np.append(x_t,0)
        disp.joliplot('t (UTC)', 'Instrument', tt,x_t, exp = False, color = 2, linewidth = 10)
        plt.annotate(timetable[0,j][47:49], [np.min(tt), 0])
    
plt.xlim(min(t), max(t))  #le temps est en 195030 (hhmmss) ce qui n'a aucun sens, il faudrait plot en UTC bien mais je sais pas faire
plt.yticks([0,1,2,3, 4], ['Bernache', 'Mésange', 'G', 'WB', 'T'])


#%% save timetable
path_save = 'K:\\Share_hublot\\Summary\\Timeline\\'

to_save = np.rot90(timetable)

df = pandas.DataFrame(to_save)

df.to_csv(path_save + date + "_timetable.txt", sep='\t', index=False)

dic.save_dico(time, path = path_save + date + 'times.pkl')


#%% Pos for timetable


def projection_real_space(x,y,x_0,y_0,h,alpha_0,f):

    # % Definition of x and y in real framework, camera sensor center is
    # % taken as a reference 
    # % Inputs : 
    # % - x: array of x-coordinates in pixels
    # % - y: array of y-coordinates in pixels
    # % - x_0 : x-coordinate of camera sensor center
    # % - y_0 : y-coordinate of camera sensor center
    # % - h : drone altitude in meter (above sea level)
    # % - alpha_0 : inclination angle of the camera, angle to the horizontal 
    # % - f : camera focal length
    
    yreal = (y - y_0) * h / np.sin(alpha_0) / (f*np.sin(alpha_0) + (y - y_0) * np.cos(alpha_0) )
    xreal = (x - x_0) * h / (f*np.sin(alpha_0) + (y - y_0) * np.cos(alpha_0) )

    yreal = -yreal
    return xreal,yreal




















