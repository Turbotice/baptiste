# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 17:12:28 2025

@author: Banquise
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt2d
from scipy.signal import find_peaks
from skimage.measure import profile_line
import scipy.fft as fft
import scipy.io as io
import h5py
import cv2
from scipy import interpolate
from scipy.signal import savgol_filter

import baptiste.display.display_lib as disp
import baptiste.experiments.import_params as ip
import baptiste.signal_processing.fft_tools as ft
import baptiste.image_processing.image_processing as imp
import baptiste.math.fits as fits
import baptiste.math.RDD as rdd
import baptiste.files.save as sv
import baptiste.files.dictionaries as dico


import icewave.tools.matlab2python as m2p
import icewave.baptiste.Fct_drone_1102 as fd

import icewave.multi_instruments.timeline as tl

path = "K:\Share_hublot\\Data\\0211\\Summary\\"
file = 'records_0211.pkl'

records = dico.open_dico(path + file)

date = '2024/02/11'

tl.display_records(records,date,date_format='2024/02/11',ax=None)


#%%

import pylab as plt
import numpy as np

import icewave.field.multi_instruments as multi
import icewave.geometry.display as display
import icewave.display.graphes as graphes

global colors
colors = disp.mcolors(range(1,9))

date_format='2024/02/11'



fig = disp.figurejolie(width = 8.6*1.8, height= 8.6/2*1.3)
ax = fig.add_subplot(111)

positions = {'Fulmar':0.8,'mesange':1,'Bernache':1.2,'geophones':1.5,'buoys':2.8,'phones':3.4}
    
import matplotlib.dates as mdates
h0 = 0

bs = {'Fulmar':0,'mesange':1,'Bernache':1}
cm = 2
record = records['drones']
for drone in records['drones'].keys():
    for name in records['drones'][drone].keys():#(drone,name) in selection:
        record = records['drones']#[drone]
        pos = positions[drone]
        try:
            title = name.split('-')[1]#.split('_')[0]
        except:
            title = name
        if name in record[drone].keys():
            times = record[drone][name][0]['time']# for key in record[name][0].keys()]
            x = [times[0],times[-1]]
            x = np.asarray([multi.convert_time(x[0]),multi.convert_time(x[0]) + 1200])

            r=np.random.random() / 10*bs[drone]-0.5
            y = pos#+r
            ax.plot(x/3600+h0, [y,y], color = colors[cm], marker = 'd')
            ax.text(x[0]/3600+h0,y+0.03,title)

key = 'geophones'
record = records[key]

cm = 4

pos = positions[key]
X={}
first={}
for i,num in enumerate(record.keys()):
    first[num]=True
    uuu = 0
    for k in record[num].keys():
#            date_format = str("2024/"+date[:2]+"/"+date[2:])#'2024/02/23'
        if record[num][k]['date']==date_format:# 2024/02/23date_format:
            uuu += 1
            
            t0 = record[num][k]['time'][0]
            t1 = record[num][k]['time'][-1]
            x = [t0,t1]
            if len(record[num][k]['time']) == 1:
                x = np.asarray([multi.convert_time(x[0]),multi.convert_time(x[0]) + 500])
                
            elif uuu == len(record[num].keys()) and i != 9:
                x = np.asarray([multi.convert_time(x[0]),multi.convert_time(x[1]) + 1000])
                
            else :
                x = np.asarray([multi.convert_time(x[0]),multi.convert_time(x[1])])


            y = pos+i*0.05#np.random.random()*b
            ax.plot(x/3600+h0, [y,y], color = colors[cm], marker = 'd')
            if first[num]:
                ax.text(66200/3600,y,r'$G$' + str(i+1))
                first[num]=False

x0 = 0.1

cm = 7

for key in ['buoys','phones']:
    record = records[key]
    pos = positions[key]
    X={}
    first={}
    if key == 'buoys' :
        cm = 6
    else :
        cm = 7
    
    for i,num in enumerate(record.keys()):
        first[num]=True
        for k in record[num].keys():
            elem = record[num][k]
            t0 = elem['time'][0]
            t1 = elem['time'][-1]
            x = [t0,t1]
            X[i] = np.asarray([multi.convert_time(x[0]),multi.convert_time(x[1])])
            #np.asarray([convert(t0)-3600*6,convert(t1)-3600*6])

            #print(i,X[i][0]/3600)
            if (X[i][0]/3600)<16:
                if i>0:
                    X[i]=X[i-1]
                else:
                    continue
            y = pos+i*0.1#np.random.random()*b
            ax.plot(X[i]/3600+h0, [y,y], color = colors[cm], marker = 'd')
            if first[num]:
                ax.text(X[i][0]/3600-x0+h0,y,num)
                first[num]=False
                
ax.set_ylim([0.5,4.2])

hours = np.asarray([11,12,13,14,15, 16])+6
ax.set_xticks(hours,[str(h)+':00' for h in hours])

# from matplotlib.lines import Line2D
# # access legend objects automatically created from data
# handles, labels = plt.gca().get_legend_handles_labels()
# names = {'Smartphones':('r','s'),'Buoys':('m','o'),'Geophones':('g','>'),'Bernache':('b','o'),'Mesange':('k','o')}
# legends = []
# for key in names.keys():
#     point = Line2D([0], [0], label=key, marker=names[key][1], markersize=20, 
#          markeredgecolor=names[key][0], markerfacecolor=names[key][0], linestyle='')
#     legends.append(point)

# # add manual symbols to auto legend
# handles.extend(legends)

# #ax.legend(handles=handles,fontsize=20)
# figs = graphes.legende('UTC time','Instruments',date,ax=ax)
#plt.show()



plt.xlim(18, 22)

#%% Situation map

import icewave.multi_instruments.situation_map as sm

sm.display_map(records,ax=None,BBox=None,project=False)

