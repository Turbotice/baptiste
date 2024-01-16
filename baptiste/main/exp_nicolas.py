# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 16:48:11 2023

@author: Banquise
"""

import cv2 
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from skimage import feature
from scipy.signal import convolve2d
from scipy.signal import savgol_filter, gaussian
from scipy.signal import medfilt
from scipy.signal import medfilt2d
from scipy.signal import find_peaks
from scipy.signal import detrend
from scipy.optimize import curve_fit
from scipy import ndimage
from scipy import stats
import scipy.fft as fft
import os
from PIL import Image
from datetime import datetime
from scipy.optimize import minimize
import matplotlib.cm as cm
import pandas as panda
import scipy.io as io
import scipy.signal as sig


import baptiste.display.display_lib as disp
import baptiste.experiments.import_params as ip
import baptiste.files.file_management as fm
import baptiste.image_processing.image_processing
import baptiste.files.dictionaries as dic
import baptiste.tools.tools as tools
import baptiste.math.RDD as rdd
import baptiste.math.fits as fits
import baptiste.signal_processing.fft_tools as ft

dico = dic.open_dico()

#%% Initialisation
params = {}
save = True

params['mmparpixel'] = 0.10111223458038422649
params['facq_las'] = 1000
params['facq_piv'] = 1
params['ratio_piv'] = 8



# params['path'] = 'E:\\Nicolas\\d230726\\230726_facq1.000Hz_10.86g_20.Hz_1.430V\\Data\\'
# params['path'] = 'E:\\Nicolas\\d230726\\230726_facq1000Hz_1086g_20Hz_1100V\\Data\\'
params['path'] = 'E:\\Nicolas\\d230726\\230726_facq1.000Hz_10.32g_80.Hz_4.340V\\Data\\'
# params['path'] = 'E:\\Nicolas\\d230726\\230726_facq1.000Hz_10.32g_80.Hz_4.960V\\Data\\'
# params['path'] = 'E:\\Nicolas\\d230726\\230726_facq1.000Hz_10.86g_20.Hz_1.540V\\Data\\'
# params['path'] = 'E:\\Nicolas\\d230725\\230725_facq0.100Hz_7.79g_80.Hz_5.360V\\Data\\'
# params['path'] = 'E:\\Nicolas\\d230725\\230725_facq0.100Hz_7.79g_40.Hz_1.720V\\Data\\'
# params['path'] = 'E:\\Nicolas\\d230725\\230725_facq0.100Hz_7.79g_40.Hz_3.260V\\Data\\'
# params['path'] = 'E:\\Nicolas\\d230727\\230727_facq1.000Hz_8.630g_100Hz_6.360V\\Data\\'
# params['path'] = 'E:\\Nicolas\\d230725\\230725_facq0.100Hz_7.79g_40.Hz_1.720V\\Data\\'
# params['path'] = 'E:\\Nicolas\\d230725\\230725_facq0.100Hz_7.79g_40.Hz_1.720V\\Data\\'
# params['path'] = 'E:\\Nicolas\\d230727\\230727_facq1.000Hz_9.690g_60.Hz_1.840V\\Data\\'
# params['path'] = 'E:\\Nicolas\\d230727\\230727_facq2.000Hz_10.35g_10.Hz_0.560V\\Data\\'
# params['path'] = 'E:\\Nicolas\\d230727\\230727_facq1.000Hz_11.84g_100Hz_5.560V\\Data\\'

#%%LAS NCOLAS




# params['file_las'] = 'facq1000Hz_2023-07-26 19.10.16.268_A0000.csv' #1.43V 20Hz
# params['file_las'] = 'facq1000Hz_20230726_18.50.17.785_A0000.csv' #1.1V 20Hz
# params['file_las'] = 'facq2000Hz_2023-07-26 15.20.00.000_A0002.csv' #4.34V 80Hz
# params['file_las'] = 'facq2000Hz_2023-07-26 15.40.25.108_A0000.csv' #4.34V 80Hz
# params['file_las'] = 'facq2000Hz_2023-07-26 15.41.53.687_A0002.csv' #4.96V 80Hz
# params['file_las'] = 'facq2000Hz_2023-07-26 16.03.04.884_A0002.csv' #4.96V 80Hz
# params['file_las'] = 'facq2000Hz_2023-07-25_18.49.18.204_A0000.csv' #5.36V 80Hz
# params['file_las'] = 'facq2000Hz_2023-07-25_19.01.39.172_A0001.csv' #5.36V 80Hz
# params['file_las'] = 'facq2000Hz_2023-07-25 19.13.05.082_A0002.csv' #1.72V 40 Hz
# params['file_las'] = 'facq2000Hz_2023-07-25 19.25.04.747_A0002.csv' #3.26V 40Hz
# params['file_las'] = 'facq1000Hz_2023-07-27 15.00.09.276_A0000.csv' #6.36V 100Hz



params['facq_las'] = float(params['file_las'][4:8])

las = panda.read_csv(params['path']  + params['file_las'] , sep = ',', header= 2)

t_las = np.asarray(las['Protocol TimeStamp'])
x_las = np.asarray(las['Distance [mm]'])

x_las = x_las - np.mean(x_las)
t_las = t_las - t_las[0]

disp.figurejolie()

disp.joliplot('t', 'x', t_las, x_las, exp = False)
if save :
    plt.savefig(params['path'] + 'x_de_t_' + params['file_las'][:-4] + '.pkl')

disp.figurejolie()

FFT, f = ft.fft_bapt(x_las, params['facq_las'])

# f = np.linspace(0,facq_las, len(t_las))

disp.joliplot('f', '|A|', f, np.abs(FFT/len(FFT)))
if save :
    plt.savefig(params['path'] + 'fft_' + params['file_las'][:-4] + '.pkl')



#%% Technique RMS : Marche

save = False
nt = len(t_las)

# FFT, f = ft.fft_bapt(x_las, facq_las)

# t_nperiodes = (int(facq/fexc) + 1) * n_periodes
# t_n_frac = int(fracs[j,0] - t_nperiodes - int(t_nperiodes/n_periodes * n_periodesavantfracture))
# t_0_frac = int(fracs[j,0] - int(t_nperiodes/n_periodes * n_periodesavantfracture) )


params['fexc_1'] = float ( params['path'][params['path'].index("g_") + 2:params['path'].index("g_") + 5]) / 2
params['d_f_1'] = 2

[b,a] = sig.butter(3, [params['fexc_1'] - params['d_f_1'], params['fexc_1'] + params['d_f_1']], btype='bandpass', analog=False, output='ba',fs=params['facq_las'])
params['yfilt_10'] = sig.filtfilt(b, a, x_las)

params['fexc_2']  = float ( params['path'][params['path'].index("g_") + 2:params['path'].index("g_") + 5])
params['d_f_2'] = 2

[b,a] = sig.butter(3, [params['fexc_2'] - params['d_f_2'], params['fexc_2'] + params['d_f_2']], btype='bandpass', analog=False, output='ba',fs=params['facq_las'])
params['yfilt_20'] = sig.filtfilt(b, a, x_las)

Y1 = fft.fft(x_las)
Y10 = fft.fft(params['yfilt_10'])
Y20 = fft.fft(params['yfilt_20'])

P2 = np.abs(Y1)
P10 = np.abs(Y10)
P20 = np.abs(Y20)

params['amp_reelle'] = np.sqrt( np.sum ( np.abs(x_las) **2) / nt *2)

params['amp_FFT'] = np.sqrt(np.sum(np.abs(P2 /np.sqrt(nt) )**2 ) / nt*2)

params['amp_FFT_10Hz'] =  np.sqrt( np.sum ( np.abs(params['yfilt_10']) **2) / nt*2)

params['amp_reelle_10Hz'] =  np.sqrt(  np.sum(np.abs(P10 /np.sqrt(nt) )**2 ) / nt*2)

params['amp_FFT_20Hz'] = np.sqrt(  np.sum ( np.abs(params['yfilt_20']) **2) / (nt))

params['amp_reelle_20Hz'] =  np.sqrt(  np.sum(np.abs(P20/np.sqrt(nt) )**2 ) / nt*2)

print('amp_reelle',params['amp_reelle'])
# print('amp_FFT',params['amp_FFT'])
print('amp_FFT' + str(params['fexc_1']),params['amp_FFT_10Hz'])
# print('amp_reelle_10Hw',params['amp_reelle_10Hz'])
print('amp_FFT' + str(params['fexc_2']),params['amp_FFT_20Hz'])
# print('amp_reelle_20Hw',params['amp_reelle_20Hz'])
print('ratio '+ str(params['fexc_1']) + ' / reel',params['amp_reelle_10Hz'] / params['amp_reelle'])
print('ratio '+ str(params['fexc_1']) + ' / ' + str(params['fexc_2']), params['amp_reelle_10Hz'] / params['amp_reelle_20Hz'])
print('ratio '+ str(params['fexc_2']) + ' / reel',params['amp_reelle_20Hz'] / params['amp_reelle'])
print('ratio bruit', 1 - np.sqrt(params['amp_reelle_10Hz']**2 + params['amp_reelle_20Hz']**2)/params['amp_reelle'] )




if save :
    dic.save_dico(params, params['path'] + 'params_pointeur_' + params['file_las'][:-4] + '.pkl')
    
    
    
#%% Demod LAS : MARCHE PAS

fexc = 10
d_f = 1
demo_10 = np.zeros(len(FFT), dtype = complex)
demo_10[int(len(FFT) * fexc/facq_las)-int(len(FFT) * d_f/facq_las):int(len(FFT) * fexc/facq_las)+int(len(FFT) * d_f/facq_las)] = FFT[int(len(FFT) * fexc/facq_las)-int(len(FFT) * d_f/facq_las):int(len(FFT) * fexc/facq_las)+int(len(FFT) * d_f/facq_las)]
if d_f == 0 :
    demo_10[int(len(FFT) * fexc/facq_las)] = FFT[int(len(FFT) * fexc/facq_las)]
i_demo_10 = fft.ifft(demo_10)
disp.figurejolie()
disp.joliplot('t', 'x', t_las, np.real(i_demo_10), exp = False, title = 'signal demodulé ' + str(fexc) + 'Hz df = ' + str(d_f))

#%%Tecnique butter : MARCHE PAS

fexc = 10
d_f = 0.5



[b,a] = sig.butter(2, [fexc - d_f, fexc + d_f], btype='bandpass', analog=False, output='ba',fs=1000)
yfilt = sig.filtfilt(b, a, x_las)

disp.figurejolie()
disp.joliplot('t', 'x', t_las, yfilt, exp = False, title = 'signal demodulé ' + str(fexc) + 'Hz df = ' + str(d_f))


#%% Amp 

pix = np.real(i_demo_10)
fexc = 10
distance = int(facq_las/fexc)
a = 5

peaks_max_0 = find_peaks(pix, distance = distance)

peaks_max = np.zeros(len(peaks_max_0[0]))
vals_max = np.zeros(len(peaks_max_0[0]))

for v in range (len(peaks_max_0[0])) : #fit polynomial du max
    val_max, pos_max = tools.max_fit2(pix, peaks_max_0[0][v], a = a)
    peaks_max[v] = pos_max
    vals_max[v] = val_max

peaks_max = peaks_max


disp.joliplot("t (frames)", "z (m)", t_las[peaks_max_0[0]], vals_max, exp = True)

peaks_min_0 = find_peaks(-pix, distance = distance)

peaks_min = np.zeros(len(peaks_min_0[0]))
vals_min = np.zeros(len(peaks_min_0[0]))

for w in range (len(peaks_min_0[0])) : #fit polynomial du min
    val_min, pos_min = tools.max_fit2(pix, peaks_min_0[0][w], a = a)
    peaks_min[w] = pos_min
    vals_min[w] = val_min

peaks_min = peaks_min


disp.joliplot("t (frames)", "z (m)", t_las[peaks_min_0[0]], vals_min, exp = True)

nb_peaks = np.min( (len(peaks_max), len(peaks_min)) )
Amp_t_x = np.zeros(nb_peaks)

for j in range (nb_peaks):
    Amp_t_x[j] = vals_max[j] - vals_min[j]

# Amp_t["Amp"][str(i)] = Amp_t_x
# Amp_t["peaks_min"][str(i)] = peaks_min
# Amp_t["peaks_max"][str(i)] = peaks_max

 
disp.figurejolie()
disp.joliplot("t", "Amp", peaks_max[:len(Amp_t_x)], Amp_t_x)

print('amp_max', np.max(Amp_t_x))
print('amp_moy', np.mean(Amp_t_x))
# Amp_max[i - params['x0_A']] = np.max(Amp_t_x)
# T_max[i - params['x0_A']] = int(np.where(np.max(Amp_t_x) == Amp_t_x)[0][0])
# Amp_moy[i - params['x0_A']] = np.mean(Amp_t_x[T_max[i - params['x0_A']]:])

#%% PIV Nicolas

# file_piv = 'PIVlab_16_0_300.mat'
params['file_piv'] = 'PIVlab_0_300_8_80Hz_4.34V.mat'
# params['file_piv'] = 'PIVlab_8_0_300_facq0.1Hz_fexc40Hz_1.72V.mat'
# params['file_piv'] = 'PIVlab_900_1300_8.mat'
# params['file_piv'] = 'PIVlab_8_0_500.mat'
# params['file_piv'] = 'PIVlab_0_500_8.mat'

data_brut = io.loadmat(params['path'] + params['file_piv'])
display = True
params['mparpixel_piv'] = params['mmparpixel'] * params['ratio_piv'] /1000

#%% Sort data

champ_u = data_brut["u_original"]
champ_v = data_brut["v_original"]

data_u = []
data_v = []

x_0 = 50
x_f = -50
y_0 = 50
y_f = -50


for i in range (np.shape(champ_u)[0] -10):
    data_u.append(champ_u[i][0]- np.nanmean(champ_u[i][0]))
data_u = np.asarray(data_u)

for i in range (np.shape(champ_v)[0] - 10):
    data_v.append(champ_v[i][0]- np.nanmean(champ_v[i][0]))
data_v = np.asarray(data_v)


# data = np.sqrt(np.power(data_v,2) + np.power(data_u,2))
data = data_u



data = np.transpose(data, (2,1,0)) #Pour passer de s(t,y,x) à s(x,y,t)
data = np.flip(data, 1)

data = data[x_0:x_f,y_0:y_f, :] * params['facq_piv'] / params['mparpixel_piv'] # On crop et met data en m.s-1


[nx,ny,nt] = data.shape
x = np.linspace(0, (nx -1) * params['mparpixel_piv'] , nx)  #Taille X en pixel
y = np.linspace(0, (ny - 1) * params['mparpixel_piv'] , ny)  #Taille Y en pixel
t = np.linspace(0, (nt - 1) /params['facq_piv'], nt)  

if display :
    disp.figurejolie()
    disp.joliplot("X (m)","Y (m)",x,y,table = data[:,:,20], tcbar = "Champ u")
    plt.clim(0,np.nanquantile(data[:,:,0],0.9))
    # plt.clim(0,10)
    
# display_video = True
# if display_video :
#     disp.figurejolie()
#     plt.pcolormesh(y, x, data[:,:,0], shading='auto')
#     plt.xlabel("Y (m)")
#     plt.ylabel("X (m)")
#     cbar = plt.colorbar()
#     cbar.set_label("Champ u")
#     plt.axis("equal")
#     plt.clim(0, 150)
#     # plt.clim(0,np.nanquantile(data[:,:,20],0.8))
#     for mmm in range (1,20):
#         # disp.figurejolie()
#         plt.pcolormesh(y, x, data[:,:,mmm], shading='auto')
#         # plt.clim(0, np.nanquantile(data[:,:,mmm],0.8))
#         plt.clim(0,10)
#         plt.pause(0.1)
        
        # cbar = plt.colorbar()
        # plt.clim(-2,2)
    
#%% Demodulé

fexc_piv =40

#[0.02931119 0.24914509 0.48363459] 3 fréquences fondamentales pour videos complete

demod_piv = ft.demodulation(t,data,fexc_piv)

if display :
    disp.figurejolie() 
    disp.joliplot("X (m)","Y (m)",x,y,table = (np.real(demod_piv)), tcbar = 'Champ u démodulé à ' + str(fexc_piv) + "Hz")  
    
    plt.clim(0, np.nanquantile(data[:,:,0],0.95))

#%% Filtre autour de 10Hz butter (march pas car fexc < facq)

fexc = 9.9
d_f = 0.5



[b,a] = sig.butter(2, [fexc - d_f, fexc + d_f], btype='bandpass', analog=False, output='ba',fs=1)
yfilt = sig.filtfilt(b, a, data)

disp.figurejolie()
disp.joliplot('x', 'y', x, y, table = yfilt[:,:,0], title = 'signal demodulé ' + str(fexc) + 'Hz df = ' + str(d_f))
plt.clim(0,np.nanquantile(data[:,:,0],0.9))

#%% fft spatiale 2d (marche pas si ya des nan)

#enleve les nan
data[np.where(np.isnan(data) == True)] = 0

dk_x = 2 * np.pi / params['mparpixel_piv']
dk_y = 2 * np.pi / params['mparpixel_piv']

# deom_padding = ft.add_padding(demod_piv, [10,10])
deom_padding = demod_piv
Y, kx, ky = ft.fft_bapt(deom_padding, dk_x, dk_y)

# Y, kx, ky = ft.fft_bapt(data[:,:,20], dk_x, dk_y)

Y = fft.fftshift(Y)
disp.figurejolie()
disp.joliplot('kx', 'ky', kx, ky, table = np.abs(Y))


# Y = fft.fftshift(Y)
np.where(np.max(np.abs(Y)) == np.abs(Y))


#%% LINE PROFILE

y_las = 0.054 #en m 
x_las = 0.1596
long_onde = 0.031 #lambda mesuré avant
pourcent_lambda = 2

disp.figurejolie()
disp.joliplot("X (m)","Y (m)",x,y,table = np.mean(data, axis = 2), tcbar = "Champ u")
plt.clim(0,800)

bl = np.zeros(nx) + y_las

plt.plot(x, bl)
plt.plot(x_las, y_las, 'kx')

# line = demod_piv[int(0/ 200 * nx),int(200/ 200 * ny) :int(54/ 200 * ny)]
line = data[int(y_las/ np.max(x) * nx),int(0/ 200 * ny) :int(200/ 200 * ny), 20]

disp.figurejolie()
disp.joliplot('x', 'u', x[:len(line)], np.real(line), exp = False)

plt.plot(x_las, 0., 'kx')


u_mean = np.zeros(ny)
u_std = np.zeros(ny)

for j in range (nx) :
    x_las = j
    u_max = np.zeros(nt)
    u_min = np.zeros(nt)
    u_las = np.zeros(nt)
    if x_las > int( (pourcent_lambda * long_onde)/ np.max(x) * nx) and x_las < nx -int( (pourcent_lambda * long_onde)/ np.max(x) * nx) :
        for i in range (nt) : 
            line = data[int(0/ 200 * nx) :int(200/ 200 * nx),int(y_las/ np.max(y) * ny), i]
            u_max[i] = np.nanmax(line[x_las - int( (pourcent_lambda * long_onde)/ np.max(x) * nx):x_las + int( (pourcent_lambda * long_onde) / np.max(x) * nx)] )
            u_min[i] = np.nanmin(line[x_las - int( (pourcent_lambda * long_onde)/ np.max(x) * nx):x_las + int( (pourcent_lambda * long_onde) / np.max(x) * nx)] )
            u_las[i] = line[x_las]
            
        u_mean[j] = np.nanmean((u_las - u_min)/(u_max - u_min))
        u_std[j] = np.nanstd((u_las - u_min)/(u_max - u_min))


# u_max = np.zeros(nt)
# u_min = np.zeros(nt)
# u_las = np.zeros(nt)
# for i in range (nt) : 
#     line = data[int(pos_line/ np.max(x) * nx),int(0/ 200 * ny) :int(200/ 200 * ny), i]
#     u_max[i] = np.max(line[int( (y_las - pourcent_lambda * long_onde)/ np.max(x) * nx):int( (y_las + pourcent_lambda * long_onde) / np.max(x) * nx)] )
#     u_min[i] = np.min(line[int( (y_las - pourcent_lambda * long_onde)/ np.max(x) * nx):int( (y_las + pourcent_lambda * long_onde) / np.max(x) * nx)] )
#     u_las[i] = line[int(y_las/ np.max(x) * nx)]    

# disp.figurejolie()
# disp.joliplot('t', 'amp/amp_max', t, (u_las - u_min)/(u_max - u_min), exp = False)

print('mean pourcent amp', np.nanmean((u_las - u_min)/(u_max - u_min)))
print('std pourcent amp', np.nanstd((u_las - u_min)/(u_max - u_min)))

disp.figurejolie()
disp.joliplot('x', 'mean pourcent amp', x, u_mean, exp = False)

disp.figurejolie()
disp.joliplot('x', 'std pourcent amp', x, u_std, exp = False)

# Y_x, kx = ft.fft_bapt(line, dk_x)
# disp.figurejolie()
# disp.joliplot('kx', '|A|', kx, np.abs(Y_x) , exp = False)

#%% Profile line au niveau de y_las avec moy temp

y_las = 0.054

line = np.mean(data, axis = 2)[int(0/ 200 * nx) :int(200/ 200 * nx), int(y_las/ np.max(y) * ny)]

disp.figurejolie()
disp.joliplot('x', 'u', x,np.mean(data, axis = 2)[int(0/ 200 * nx) :int(200/ 200 * nx), int(y_las/ np.max(y) * ny)], exp = False)

x_las = 0.1596

u_max = np.nanmax(line[int( (x_las - pourcent_lambda * long_onde)/ np.max(x) * nx):int( (x_las + pourcent_lambda * long_onde) / np.max(x) * nx)] )
u_min = np.nanmin(line[int( (x_las - pourcent_lambda * long_onde)/ np.max(x) * nx):int( (x_las + pourcent_lambda * long_onde) / np.max(x) * nx)] )
u_las = line[int(x_las/np.max(x) * nx)]


print('mean pourcent amp', np.nanmean((u_las - u_min)/(u_max - u_min)))
print('std pourcent amp', np.nanstd((u_las - u_min)/(u_max - u_min)))


















