# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 16:37:39 2025

@author: Banquise
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 18:16:23 2025

@author: Banquise
"""

#%% MODULES

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


import icewave.tools.matlab2python as m2p
import icewave.baptiste.Fct_drone_1102 as fd


#%% Import transformation

path = "W:\SagWin2024\\Data\\0211\\Drones\\transformations\\"

#Mesange
file = "mesange_structure.mat"
params_mesange = fd.open_mat(path, file)

#Bernache
file = "bernache_structure.mat"
params_bernache = fd.open_mat(path, file)


#%% IMPORT DATA

# path = "K:\Share_hublot\\Data\\0211\\Drones\\bernache\\matData\\18-stereo_001\\"
path = "Y:\Banquise\\Baptiste\\Resultats\\Analyse_1102\\Data\\"


filename = "PIV_processed_i011500_N15500_Dt4_b1_W32_xROI1_width3839_yROI1_height2159_scaled_bernache.mat"
filename = 'PIV_processed_i011496_N15496_Dt4_b1_W32_xROI1_width3839_yROI1_height2159_scaled_mesange.mat'

save_path = 'Y:\Banquise\\Baptiste\\Resultats\\Analyse_1102\\Mesange'

f = h5py.File(path + filename,'r') 

matdata = open(path + filename, 'r')

data = m2p.mat_to_dict(f, f )

Vz = data['m']['Vz'] #t,y,x en s, m, m
Vz_og = data['m']['Vz']

# Vz = np.transpose(Vz, (2,1,0)) #Pour passer de s(t,y,x) à s(x,y,t)
# Vz = np.flip(Vz, 1)

X = data['m']['X'] #m ATTONTION y,x



Y = data['m']['Y'] #m ATTONTION y,x



T = data['m']["t"] #s 

facq = 29.97

#%% Conversion en coordonnées globales

path = "W:\SagWin2024\\Data\\0211\\Drones\\transformations\\"


filename = "mesange_structure.mat"

f = h5py.File(path + filename,'r') 
f = m2p.mat_to_dict(f, f )
param = f['param_struct']

xxxxxxx = np.array([10,11,12])

fd.drone_pix2real(xxxxxxx,xxxxxxx,xxxxxxx,param)

# fd.drone_real2pix(0,0,0,param)

    
#%% Vz en zeta  filtre
from scipy import signal
facq = 29.97
dT = 1 / facq


Vz = data['m']['Vz'] #t,y,x en s, m, m

Vz = np.transpose(Vz, (2,1,0)) #Pour passer de s(t,y,x) à s(x,y,t)  
Vz = np.flip(Vz, 1)


f_cut = [0.1,2]

[b,a] = signal.butter(3, f_cut, fs = facq, btype = "bandpass")
Vz_filt = signal.filtfilt(b, a, Vz, axis = 2)


# Vz_filt = Vz.butter(2, 5, btype = "low")
eta = np.cumsum(Vz_filt, axis = 2) * dT
# zeta = np.cumsum(Vz, axis = 2) * dT

fexc = 0.2
V_demod = ft.demodulation(T, Vz, fexc)

#%% V demod
save = False
from scipy.interpolate import RegularGridInterpolator

plt.figure()
f_demod = np.linspace(0.1,0.5,20)
X_new = np.transpose(X, (1,0))
Y_new = np.transpose(Y, (1,0))

for i in range (len(f_demod)):
    fexc = f_demod[i]
    V_demod_i = ft.demodulation(T, Vz, fexc)
    V_demod_i = np.flip(V_demod_i, 1)
    
    Fz = RegularGridInterpolator((data['m']['PIXEL']['x_pix'],data['m']['PIXEL']['y_pix']),V_demod_i)
    
    V_demod_full = np.zeros((1000,1000))
    
    

    
    plt.pcolormesh(X_new,Y_new,np.real(V_demod_i))
    plt.title('V demod ' + str(fexc) + ' Hz')
    # plt.pause(0.2)
    if save :    
        sv.save_graph(save_path, 'Champ_demod_' + str(fexc) + 'Hz', pdf = False)



#%% Plot line in real space

save = False

# Vz = data['m']['Vz'] #t,y,x en s, m, m

# Vz = np.transpose(Vz, (2,1,0)) #Pour passer de s(t,y,x) à s(x,y,t)
# Vz = np.flip(Vz, 1)
# V = Vz_filt
# V = eta
V = Vz_filt


display = True
# xpix_0 = 165
# ypix_0 = 70
# d = 45 #m
# theta = 41.22 * np.pi / 180 #radians

# t0 = 1648
# dt = 10


xpix_0 = 202
ypix_0 = 10
d = 24 #m
theta = 90* np.pi / 180 #41.22 * np.pi / 180 #radians

t0 = 1600
dt = 1000



t_plot = t0 + dt

lines = np.zeros( (1000,2*dt) )
lines_eta = np.zeros( (1000,2*dt) )

x_pix, y_pix, dist_p0 = fd.px_to_real_line(data, xpix_0, ypix_0, d, theta)

if display :
    t_plot = t0 + dt
    xp = np.linspace(0,V.shape[0]-1, V.shape[0])
    yp = np.linspace(0,V.shape[1]-1, V.shape[1])
    plt.figure()
    disp.joliplot('x (pixel)','y (pixel)',xp,yp, table = V[ :,:, t_plot])
    plt.clim(np.quantile(V[ :,:, t_plot], 0.1), np.quantile(V[ :,:, t_plot], 0.9))
    plt.plot(xpix_0, ypix_0, 'kx')
    plt.plot(np.array(x_pix/16,dtype = int),  np.array(y_pix/16,dtype = int), 'r-')
    
    plt.axis('equal')
    
    if save : 
        sv.save_graph(save_path, 'Champ_Vz_filtre_t_' + str(t0 + dt) + 'frames_line', pdf = False)

display = False
if display :
    disp.figurejolie()
    
for t in  range (t0- dt,t0 + dt) :
    colors = disp.vcolors( int(( t- (t0 - dt) )  / dt / 2 * 9)) 
    
    Vz_line = fd.extract_line_Vz(data, V, x_pix, y_pix, t)
    eta_line = fd.extract_line_Vz(data, eta, x_pix, y_pix, t)
    lines[:,t- (t0 - dt) ] = Vz_line
    lines_eta[:,t- (t0 - dt) ] = eta_line
    if display :
        if t == t0 :
            disp.joliplot('x (m)', r'Vz (m.s$^{-1}$)' , dist_p0, Vz_line ,color = 2, exp = False, linewidth= 5)
            #r'$\zeta$ (m)'
        else :   
            plt.plot(dist_p0, Vz_line, color=colors)
            
#%% Spectro en xi yi : trouver f_max et t (f_max)

save = False
display = False
from scipy import signal

facteur_f = 0

sss = Vz[220,2,:]
f, t, Sxx = signal.spectrogram(sss, facq, nperseg = 300)

f_cut = 0.5 #Hz
i_f_cut = int(f_cut / (np.max(f) / f.shape) + 1)


for j in range (Sxx.shape[1]) :
    Sxx[:,j] = Sxx[:,j] * f**facteur_f

plt.figure()
plt.pcolormesh(t, f, Sxx, shading='gouraud')
plt.colorbar()
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.clim(np.min(Sxx[:i_f_cut,:]), np.max(Sxx[:i_f_cut,:]))
plt.ylim(0,0.5)

if save : 
    sv.save_graph(save_path, 'spectro_f_t_x_' + str(100) + '_y_' + str(100) + '_nperseg_' + str(300), pdf = False)
    
    

# t_f_m = np.zeros(i_f_cut, dtype = int)
# for j in range(i_f_cut) :
#     t_f_m[j] = int(np.where(Sxx[:i_f_cut,:] == np.sort(Sxx[:i_f_cut,:])[j,-1])[1][0])
    
# f1 = np.where(Sxx[np.arange(0,i_f_cut),t_f_m] == np.sort(Sxx[np.arange(0,i_f_cut),t_f_m])[-1])[0][0]
# t1 = t_f_m[np.where(Sxx[np.arange(0,i_f_cut),t_f_m] == np.sort(Sxx[np.arange(0,i_f_cut),t_f_m])[-1])[0][0]]
# f2 = np.where(Sxx[np.arange(0,i_f_cut),t_f_m] == np.sort(Sxx[np.arange(0,i_f_cut),t_f_m])[-2])[0][0]
# t2 = t_f_m[np.where(Sxx[np.arange(0,i_f_cut),t_f_m] == np.sort(Sxx[np.arange(0,i_f_cut),t_f_m])[-2])[0][0]]



# plt.figure()
# plt.pcolormesh(Sxx[:i_f_cut,:], shading='gouraud')
# plt.plot(t_f_m, np.arange(0,i_f_cut), 'rx')

n_tenta = 100
n_seg = np.linspace(100,1000,n_tenta, dtype = int)
# t_max = np.zeros(n_tenta)

# f_max = np.zeros(n_tenta)

#temps de la plus basse fréquence
t_max_1 = np.zeros(n_tenta)
#plus basse frequence parmi les deux fréquences avec le plus de signal
f_max_1 = np.zeros(n_tenta)
#temps de la plus haute fréquence
t_max_2 = np.zeros(n_tenta)
#plus haute frequence parmi les deux fréquences avec le plus de signal
f_max_2 = np.zeros(n_tenta)
#Amplitude de la plus basse fréquence
A_max_1 = np.zeros(n_tenta)
#Amplitude de la plus haute fréquence
A_max_2 = np.zeros(n_tenta)


err_f = np.zeros(n_tenta)
err_t = np.zeros(n_tenta)

if display :
    plt.figure()
for i in range (len(n_seg)) :
    f, t, Sxx = signal.spectrogram(sss, facq, nperseg = n_seg[i])
    
    # for j in range (Sxx.shape[1]) :
    #     Sxx[:,j] = Sxx[:,j] * f**facteur_f
        
    t_f_m = np.zeros(i_f_cut, dtype = int)
    for j in range(i_f_cut) :
        t_f_m[j] = int(np.where(Sxx[:i_f_cut,:] == np.sort(Sxx[:i_f_cut,:])[j,-1])[1][0])
        
    f1_bl = np.where(Sxx[np.arange(0,i_f_cut),t_f_m] == np.sort(Sxx[np.arange(0,i_f_cut),t_f_m])[-1])[0][0]
    f2_bl = np.where(Sxx[np.arange(0,i_f_cut),t_f_m] == np.sort(Sxx[np.arange(0,i_f_cut),t_f_m])[-2])[0][0]

    f1 = np.min((f1_bl, f2_bl))
    A1 = Sxx[f1,t_f_m[f1]]
    t1 = t[t_f_m[f1]]
    f1 = f[f1]
    
    f2 = np.max((f1_bl, f2_bl))
    A2 = Sxx[f2,t_f_m[f2]]
    t2 = t[t_f_m[f2]]
    f2 = f[f2]

    f_max_1[i] = f1
    f_max_2[i] = f2
    err_f[i] = f[1]
    t_max_1[i] = t1
    t_max_2[i] = t2
    err_t[i] = t[1]
    A_max_1[i] = A1
    A_max_2[i] = A2
    
    # f_max[i] = f[np.where(Sxx[:i_f_cut,:] == np.max(Sxx[:i_f_cut,:]))[0][0]]
    # err_f[i] = f[1]
    # t_max[i] = t[np.where(Sxx[:i_f_cut,:] == np.max(Sxx[:i_f_cut,:]))[1][0]]
    # err_t[i] = t[1]
    if display : 
        plt.pcolormesh(t, f, Sxx, shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.clim(np.min(Sxx[:i_f_cut,:]), np.max(Sxx[:i_f_cut,:]))
        plt.ylim(0,0.5)
        # plt.plot(t_max[i], f_max[i], 'rx')
        
        plt.plot(t_max_1[i], f_max_1[i], 'rx')
        plt.plot(t_max_2[i], f_max_2[i], 'bx')
        
        plt.pause(0.2)
    
plt.figure()
plt.plot(n_seg, f_max_1, 'rx')
plt.errorbar(n_seg, f_max_1, yerr = err_f)
f_xy = np.polyfit(n_seg, f_max_1, 0, w = np.sqrt(1/err_f) )[0]
disp.joliplot('taille segment (frames)', 'f moyenne (Hz)', n_seg, [f_xy for u in range(n_tenta)], color = 5, exp = False, legend = 'plus basse des deux fréquences')

plt.plot(n_seg, f_max_2, 'kx')
plt.errorbar(n_seg, f_max_2, yerr = err_f)
f_xy = np.polyfit(n_seg, f_max_2, 0, w = np.sqrt(1/err_f) )[0]
disp.joliplot('taille segment (frames)', 'f moyenne (Hz)',n_seg, [f_xy for u in range(n_tenta)], color = 8, exp = False, legend = 'plus haute des deux fréquences')


if save : 
    sv.save_graph(save_path, 'f_max_x100_y100', pdf = False)


plt.figure()
plt.plot(n_seg, t_max_1, 'rx')
plt.errorbar(n_seg, t_max_1, yerr = err_t)
t_xy = np.polyfit(n_seg, t_max_1, 0, w = np.sqrt(1/err_t) )[0]
disp.joliplot('taille segment (frames)', 't moyen (s)',n_seg, [t_xy for u in range(n_tenta)],color = 5, exp = False, legend = 'plus basse des deux fréquences')

plt.plot(n_seg, t_max_2, 'kx')
plt.errorbar(n_seg, t_max_2, yerr = err_t)
t_xy = np.polyfit(n_seg, t_max_2, 0, w = np.sqrt(1/err_t) )[0]
disp.joliplot('taille segment (frames)', 't moyen (s)',n_seg, [t_xy for u in range(n_tenta)], color = 8, exp = False, legend = 'plus haute des deux fréquences')

if save : 
    sv.save_graph(save_path, 't_max_x100_y100', pdf = False)
    
    
plt.figure()
plt.plot(n_seg, A_max_1* f_max_1)
plt.plot(n_seg, A_max_2* f_max_2)




# plt.figure()
# plt.plot(n_seg, f_max, 'kx')
# plt.errorbar(n_seg, f_max, yerr = err_f)
# f_xy = np.polyfit(n_seg, f_max, 0, w = np.sqrt(1/err_f) )[0]
# plt.plot(n_seg, [f_xy for u in range(n_tenta)], color = 'r')


# if save : 
#     sv.save_graph(save_path, 'f_max_x100_y100', pdf = False)


# plt.figure()
# plt.plot(n_seg, t_max, 'kx')
# plt.errorbar(n_seg, t_max, yerr = err_t)
# t_xy = np.polyfit(n_seg, t_max, 0, w = np.sqrt(1/err_t) )[0]
# plt.plot(n_seg, [t_xy for u in range(n_tenta)], color = 'r')

# if save : 
#     sv.save_graph(save_path, 't_max_x100_y100', pdf = False)

#%% Spectro : f_max et t_max pour tout x et y 
save = False

t_xy_tot = np.zeros(np.shape(Vz)[:-1])
f_xy_tot = np.zeros(np.shape(Vz)[:-1])

n_tenta = 100
n_seg = np.linspace(100,1000,n_tenta, dtype = int)

facteur_f = 0
f_cut = 0.5 #Hz

for i in range (np.shape(Vz)[0]) :
    #en x
    if i % 5 == 0 :
        print(str(int(i/np.shape(Vz)[0] * 100)) + ' %')
    for j in range (np.shape(Vz)[1]) :
        #en y
        t_max = np.zeros(n_tenta)
        f_max = np.zeros(n_tenta)
        err_f = np.zeros(n_tenta)
        err_t = np.zeros(n_tenta)
        
        for k in range (len(n_seg)) :
            sss = Vz[i,j,:]
                           
            f, t, Sxx = signal.spectrogram(sss, facq, nperseg = n_seg[k])
            
            i_f_cut = int(f_cut / (np.max(f) / f.shape) + 1)
            
            for l in range (Sxx.shape[1]) :
                Sxx[:,l] = Sxx[:,l] * f**facteur_f
            
            f_max[k] = f[np.where(Sxx[:i_f_cut,:] == np.max(Sxx[:i_f_cut,:]))[0][0]]
            err_f[k] = f[1]
            t_max[k] = t[np.where(Sxx[:i_f_cut,:] == np.max(Sxx[:i_f_cut,:]))[1][0]]
            err_t[k] = t[1]
        
        #on fait plusieurs decoupages pour le spectro, on prend la moyenne ponderée par le pas de fréquence/temps (err) (sinon discretisatiion car on a pas une plage de temps assez grande)
        f_xy_tot[i,j] = np.polyfit(n_seg, f_max, 0, w = np.sqrt(1/err_f) )[0]
        t_xy_tot[i,j] = np.polyfit(n_seg, t_max, 0, w = np.sqrt(1/err_t) )[0]
        
        
        
    
        
plt.figure()
plt.pcolormesh(X, Y, np.rot90(f_xy_tot))
plt.colorbar(label = r'f (Hz)')
plt.clim(0.18, 0.24)
plt.axis('equal')
plt.xlabel(r'X (m)')
plt.ylabel(r'Y (m)')

if save : 
    sv.save_graph(save_path, 'Champ_f_max_0.18_0.24_Hz_1000tentas', pdf = False)
    
plt.figure()
plt.pcolormesh(X, Y, np.rot90(t_xy_tot))
plt.colorbar(label = r't (s)')
plt.contour(X, Y, np.rot90(t_xy_tot), 13, colors='k')

plt.axis('equal')
plt.xlabel(r'X (m)')
plt.ylabel(r'Y (m)')


if save : 
    sv.save_graph(save_path, 'Champ_t_max_100tentas_isocontours', pdf = False)

#%% Spectro : f_max et t_max pour tout x et y, AVEC 2 FREQUENCES
save = False

t_xy_tot_1 = np.zeros(np.shape(Vz)[:-1])
f_xy_tot_1 = np.zeros(np.shape(Vz)[:-1])
t_xy_tot_2 = np.zeros(np.shape(Vz)[:-1])
f_xy_tot_2 = np.zeros(np.shape(Vz)[:-1])
A_xy_tot_1 = np.zeros(np.shape(Vz)[:-1])
A_xy_tot_2 = np.zeros(np.shape(Vz)[:-1])



n_tenta = 100
n_seg = np.linspace(100,1000,n_tenta, dtype = int)

facteur_f = 0
f_cut = 0.5 #Hz

err_f = np.zeros(n_tenta)
err_t = np.zeros(n_tenta)

for i in range (np.shape(Vz)[0]) :
    #en x
    if i % 1 == 0 :
        print(str(int(i/np.shape(Vz)[0] * 100)) + ' %')
    for j in range (np.shape(Vz)[1]) :
        #en y
        #temps de la plus basse fréquence
        t_max_1 = np.zeros(n_tenta)
        #plus basse frequence parmi les deux fréquences avec le plus de signal
        f_max_1 = np.zeros(n_tenta)
        #temps de la plus haute fréquence
        t_max_2 = np.zeros(n_tenta)
        #plus haute frequence parmi les deux fréquences avec le plus de signal
        f_max_2 = np.zeros(n_tenta)
        #Amplitude de la plus basse fréquence
        A_max_1 = np.zeros(n_tenta)
        #Amplitude de la plus haute fréquence
        A_max_2 = np.zeros(n_tenta)
        
        for k in range (len(n_seg)) :
            sss = Vz[i,j,:]
                           
            f, t, Sxx = signal.spectrogram(sss, facq, nperseg = n_seg[k])
            
            i_f_cut = int(f_cut / (np.max(f) / f.shape) + 1)
            
            t_f_m = np.zeros(i_f_cut, dtype = int)
            for l in range(i_f_cut) :
                t_f_m[l] = int(np.where(Sxx[:i_f_cut,:] == np.sort(Sxx[:i_f_cut,:])[l,-1])[1][0])
                
            f1_bl = np.where(Sxx[np.arange(0,i_f_cut),t_f_m] == np.sort(Sxx[np.arange(0,i_f_cut),t_f_m])[-1])[0][0]
            f2_bl = np.where(Sxx[np.arange(0,i_f_cut),t_f_m] == np.sort(Sxx[np.arange(0,i_f_cut),t_f_m])[-2])[0][0]

            f1 = np.min((f1_bl, f2_bl))
            A1 = Sxx[f1,t_f_m[f1]]
            t1 = t[t_f_m[f1]]
            f1 = f[f1]
            
            f2 = np.max((f1_bl, f2_bl))
            A2 = Sxx[f2,t_f_m[f2]]
            t2 = t[t_f_m[f2]]
            f2 = f[f2]

            f_max_1[k] = f1
            f_max_2[k] = f2
            err_f[k] = f[1]
            t_max_1[k] = t1
            t_max_2[k] = t2
            err_t[k] = t[1]
            A_max_1[k] = A1
            A_max_2[k] = A2
            

        
        #on fait plusieurs decoupages pour le spectro, on prend la moyenne ponderée par le pas de fréquence/temps (err) (sinon discretisatiion car on a pas une plage de temps assez grande)
        f_xy_tot_1[i,j] = np.polyfit(n_seg, f_max_1, 0, w = np.sqrt(1/err_f) )[0]
        t_xy_tot_1[i,j] = np.polyfit(n_seg, t_max_1, 0, w = np.sqrt(1/err_t) )[0]
        f_xy_tot_2[i,j] = np.polyfit(n_seg, f_max_2, 0, w = np.sqrt(1/err_f) )[0]
        t_xy_tot_2[i,j] = np.polyfit(n_seg, t_max_2, 0, w = np.sqrt(1/err_t) )[0]
        A_xy_tot_1[i,j] = np.mean(A_max_1)
        A_xy_tot_2[i,j] = np.mean(A_max_2)
        
        
        
    
save = True    
plt.figure()
plt.pcolormesh(X, Y, np.rot90(f_xy_tot_1))
plt.colorbar(label = r'f1 (Hz)')
plt.clim(0.12, 0.25)
plt.axis('equal')
plt.xlabel(r'X (m)')
plt.ylabel(r'Y (m)')

if save : 
    sv.save_graph(save_path, 'n100_bichro_Champ_f_BF_max_0.12_0.25_Hz', pdf = False)


plt.figure()
plt.pcolormesh(X, Y, np.rot90(f_xy_tot_2))
plt.colorbar(label = r'f2 (Hz)')
plt.clim(0.15, 0.30)
plt.axis('equal')
plt.xlabel(r'X (m)')
plt.ylabel(r'Y (m)')

if save : 
    sv.save_graph(save_path, 'n100_bichro_Champ_f_HF_max_0.15_0.30_Hz', pdf = False)
    
plt.figure()
plt.pcolormesh(X, Y, np.rot90(t_xy_tot_1))
plt.colorbar(label = r't1 (s)')
plt.contour(X, Y, np.rot90(t_xy_tot_1), 13, colors='k')

plt.axis('equal')
plt.xlabel(r'X (m)')
plt.ylabel(r'Y (m)')

if save : 
    sv.save_graph(save_path, 'n100_bichro_Champ_t_BF_isocontours', pdf = False)


plt.figure()
plt.pcolormesh(X, Y, np.rot90(t_xy_tot_2))
plt.colorbar(label = r't2 (s)')
plt.contour(X, Y, np.rot90(t_xy_tot_2), 13, colors='k')

plt.axis('equal')
plt.xlabel(r'X (m)')
plt.ylabel(r'Y (m)')


if save : 
    sv.save_graph(save_path, 'n100_bichro_Champ_t_HF_isocontours', pdf = False)
    
plt.figure()
plt.pcolormesh(X, Y, np.rot90(A_xy_tot_1/A_xy_tot_2))
plt.colorbar(label = r'A1 / A2')
plt.clim(0.5,2)
# plt.contour(X, Y, np.rot90(A_xy_tot_1/A_xy_tot_2), 10, colors='k')

plt.axis('equal')
plt.xlabel(r'X (m)')
plt.ylabel(r'Y (m)')

if save : 
    sv.save_graph(save_path, 'n100_bichro_Champ_ABF_sur_AHF', pdf = False)


    


#%% Detection Front final : en x,y,t
save = False
d = 24 #m
theta = 90* np.pi / 180 
seuil = 0.08 #Bernache 0.1

def deriv1(f,x,h,smooth = 1):
    return (f(x+(smooth * h))-f(x))/ (smooth * h)

#Demi longueur autour de x pour trouver t front (en pixels ?)
l_findsigma = 50

#t à partir duquel on cherche la fracture
t0_frac = 1400 #Bernache 1500
t = 2600 #tps pour chercher seuil en x

#Les y qu'on regarde (en pixels)
table_y = np.linspace(0,119,120)
table_y = np.asarray(table_y, dtype = int)

#Taille sur laquelle on dérive en espace (en pixels)
smooth = 20


x_new = dist_p0[smooth+1:-smooth-1]
dx = np.abs(dist_p0[1] - dist_p0[0])

# disp.figurejolie(width = 16)


t_front = np.zeros(len(table_y), dtype = int)

pos_front = np.zeros(len(table_y), dtype = int)

for ypix_0 in table_y: #,120) :
    if ypix_0 > 80 : #BERNACHE : 40
        xpix_0 = 190 #bernache
        xpix_0 = 60 #mesange
    else :
        xpix_0 = 80 #205 bernache
    x_pix, y_pix, dist_p0 = fd.px_to_real_line(data, xpix_0, ypix_0, d, theta)
    
    Vz_line = fd.extract_line_Vz(data, V, x_pix, y_pix, t)
    eta_line = fd.extract_line_Vz(data, eta, x_pix, y_pix, t)
    
    F_eta = interpolate.interp1d( dist_p0,eta_line)
    
    d_eta = deriv1(F_eta, x_new, dx, smooth)
    
    # disp.joliplot('x (m)', r'Vz (m.s$^{-1}$)' , dist_p0, Vz_line ,color = 2, exp = False, linewidth= 5)
    
    
    # disp.figurejolie(width = 16)
    # disp.joliplot('x (m)', r'$\eta$ (m)' , dist_p0, eta_line ,color = 2, exp = False, linewidth= 5)
    # disp.joliplot('x (m)', r'$\frac{\delta \eta}{\delta x}$' , x_new, d_eta ,color = 2, exp = False, linewidth= 3)
    
    pos_front[ypix_0] = int( round(x_pix[np.argmax(np.abs(d_eta)) +smooth + 1]/16))
    
    # pour trouver le bon t
    x_frac = np.argmax(np.abs(d_eta)) +smooth + 1
    x_new_frac = dist_p0[x_frac - l_findsigma + smooth: x_frac + l_findsigma - smooth]
    sigma_zx = np.zeros( (l_findsigma * 2 - 2 * smooth) )
    #i est le temps auquel on regarde, on commence à t0_frac (en frames)
    i = t0_frac
    #Si la zone de recherche de t sors du domaine on passe
    if x_frac + l_findsigma > 1000 or  x_frac - l_findsigma < 0:
        pass
    else :
        #Tant que le seuil n'est pas dépassé
        while np.abs(sigma_zx[int(l_findsigma - smooth)]) < seuil :
            Vz_line = fd.extract_line_Vz(data, V, x_pix, y_pix, i)
            Fz = interpolate.interp1d( dist_p0,Vz_line)
            d_Vz = deriv1(Fz, x_new_frac, dx, smooth)
            #On remesure et stock le cisaillement vertical
            sigma_zx = d_Vz
            i += 1
            #Si le temps est trop grand, on passe
            if i > 2400 :
                break
    t_front[ypix_0] = i
    
    if ypix_0 % 10 == 0:
        print(str(int(ypix_0/119 * 100)) + '%' )
        
        
        # if np.abs(kappa[xi, ti]) > seuil and threshnotfound :
        #     t_crack[xi] = ti + t0 - dt #la frame ou ca depasse le seuil à partir de t0
        #     threshnotfound = False
        # elif ti == t_crack[xi + 1] - t0 + dt -1 and threshnotfound:
        #     t_crack[xi] = t_crack[xi+1]


filtre_poly2 = False
filtre_savgol = True

taille_savgol = 10


# plt.figure()
# plt.plot(pos_front, table_y, 'r-')

# plt.plot(pos_front[best_y_front], best_y_front)

# if save : 
#     sv.save_graph(save_path, 'Champs_VZ_avec_x_frac_y_frac_filtre', pdf = False)
    
# plt.figure()
# plt.plot(table_y, t_front)


# if save : 
#     sv.save_graph(save_path, 'y_frac_t_frac', pdf = False)
    
#selectionner les pts qui se tiennent : correlation de x_frac et t_frac

r = np.corrcoef(t_front, pos_front)
 
plt.figure()
plt.plot(table_y, pos_front/t_front)

# if save : 
#     sv.save_graph(save_path, 'y_frac_v_frac', pdf = False)

# best_t_front = t_front[np.where( np.abs(pos_front/t_front - 0.105) < 0.005 )]
# best_x_front = pos_front[np.where( np.abs(pos_front/t_front - 0.105) < 0.005 )]
best_y_front= table_y[np.where( np.abs(pos_front/t_front - 0.105) < 0.005 )]

best_y_front = table_y

# plt.figure()
# plt.plot(best_y_front, t_front[best_y_front])

# if save : 
#     sv.save_graph(save_path, 'y_frac_t_frac_filtre', pdf = False)

# plt.figure()
# plt.plot(best_y_front, pos_front[best_y_front])

# if save : 
#     sv.save_graph(save_path, 'y_frac_x_frac_filtre', pdf = False)


y_m_front = Y[best_y_front, pos_front[best_y_front]]
#On fit x pour qu'il soir régulier en mètres
if filtre_poly2 :
    p_fit_x = np.polyfit(y_m_front,X[best_y_front, pos_front[best_y_front]],2)
    x_m_front = y_m_front**2 * p_fit_x[0] + y_m_front * p_fit_x[1] + p_fit_x[2]
    
if filtre_savgol :
    x_m_front = savgol_filter(X[best_y_front, pos_front[best_y_front]], taille_savgol, 2)
    
    
#On remet en pixel le résultat précedent
x_pix_front, y_pix_front = fd.real_to_px(data, x_m_front,y_m_front)


#A partir de ce nouveau x_front, on fit y pour qu'il soit régulier en m, EN FAIT NON
# p_fit_y = np.polyfit(x_m_front,Y[y_pix_front, x_pix_front],2)
# y_m_front = x_m_front**2 * p_fit_y[0] + x_m_front * p_fit_y[1] + p_fit_y[2]
# #et on remet en pixel
# x_pix_front, y_pix_front = fd.real_to_px(data, x_m_front,y_m_front)

disp_smooth_front = True
if disp_smooth_front : 
    disp.figurejolie(width = 12)
    disp.joliplot('','',xp,yp, table = V[:,:,t_plot])
    plt.colorbar()
    plt.clim(np.quantile(Vz[ :,:, t_plot], 0.01), np.quantile(Vz[ :,:, t_plot], 0.99))
    disp.joliplot('x (pix)', 'y (pix)',x_pix_front, y_pix_front, exp = False, color = 2, legend = 'Position front smooth')
    disp.joliplot('x (pix)', 'y (pix)',pos_front[best_y_front], best_y_front, exp = False, color = 3, legend = 'Position front moche')
    
    if save : 
        sv.save_graph(save_path, 'SAVGOL10_y_front_og_et_smooth', pdf = False)
    
    
if filtre_poly2 :
    p_fit_t = np.polyfit(y_m_front, T[t_front[best_y_front]],2)
    t_front_s = y_m_front**2 * p_fit_t[0] + y_m_front * p_fit_t[1] + p_fit_t[2]
    
if filtre_savgol :
    t_front_s = savgol_filter(T[t_front[best_y_front]], taille_savgol, 2)

t_frame_front = np.array(t_front_s*facq, dtype = int)

plt.figure()

plt.plot(y_m_front, t_frame_front)
plt.plot(y_m_front, t_front[best_y_front])
plt.title('t front og et smooth')


if save : 
    sv.save_graph(save_path, 'SAVGOL10_t_front_og_et_smooth', pdf = False)



size_mean_x = 20
f1_mean_10 = np.zeros(len(x_pix_front))
for i in range (len(x_pix_front)) :
    f1_mean_10[i] = np.mean(f_xy_tot_1[x_pix_front[i] -size_mean_x:x_pix_front[i] +size_mean_x ,y_pix_front[i]])
    
f2_mean_10 = np.zeros(len(x_pix_front))
for i in range (len(x_pix_front)) :
    f2_mean_10[i] = np.mean(f_xy_tot_2[x_pix_front[i] -size_mean_x:x_pix_front[i] +size_mean_x ,y_pix_front[i]])
    
disp.figurejolie(width = 12)

disp.joliplot(r'y (m)', r'f (Hz)', y_m_front,f1_mean_10, color = 2, legend = 'f1 moyennée en x' )

disp.joliplot(r'y (m)', r'f (Hz)', y_m_front,f2_mean_10, color = 5, legend = 'f2 moyennée en x'  )


if save : 
    sv.save_graph(save_path, 'SAVGOL10_n100_f_front_bichromatique_moyenne_x_20_pixels', pdf = False)

#%% Comparaison t_front et t_xy_max

save = False

disp.figurejolie(width = 12)

# disp.joliplot('y (m)', 't front (s)',y_pix_front, T[t_front[best_y_front]], exp = False, color = 2, legend = 'Methode Cisaillement' )
# disp.joliplot('y (m)', 't front (s)',y_pix_front, t_front_s, exp = False, color = 4, legend = 'Methode Cisaillement smooth' )

disp.joliplot('y (m)', 't front (s)',y_m_front, t_xy_tot_1[x_pix_front, y_pix_front], exp = False, color = 2, legend = 'Methode Spectro BF' )
disp.joliplot('y (m)', 't front (s)',y_m_front, t_xy_tot_2[x_pix_front, y_pix_front], exp = False, color = 5, legend = 'Methode Spectro HF' )
disp.joliplot('y (m)', 't front (s)',y_m_front, t_front_s, exp = False, color = 8, legend = 'Methode Cisaillement smooth' )
# disp.joliplot('y (m)', 't front (s)',y_m_front, T[t_front[best_y_front]], exp = False, color = 4, legend = 'Methode Cisaillement' )
if save : 
    sv.save_graph(save_path, 'SAVGOL10_n100_t_front_double_methode_bichromatique', pdf = False)
    
    
disp.figurejolie(width = 12)

disp.joliplot('y (m)', 't front (s)',y_m_front, t_xy_tot_1[x_pix_front, y_pix_front] -t_xy_tot_2[x_pix_front, y_pix_front], exp = False, color = 2, legend = 'Methode Spectro BF - HF' )
disp.joliplot('y (m)', 't front (s)',y_m_front, y_m_front - y_m_front, exp = False, color = 8, legend = '0' )

if save : 
    sv.save_graph(save_path, 'SAVGOL10_n100_t_front_BF_moins_HF_double_methode_bichromatique', pdf = False)

disp.figurejolie(width = 12)

disp.joliplot('y (m)', 'f front (Hz)', y_m_front, f_xy_tot_1[x_pix_front, y_pix_front], exp = False, color = 2)
disp.joliplot('y (m)', 'f front (Hz)', y_m_front, f_xy_tot_2[x_pix_front, y_pix_front], exp = False, color = 5)

if save : 
    sv.save_graph(save_path, 'SAVGOL10_f_front_bichromatique', pdf = False)


disp.figurejolie(width = 12)

disp.joliplot('y (m)', 'Amplitude pic (UA)', y_m_front, A_xy_tot_1[x_pix_front, y_pix_front], exp = False, color = 2, legend = 'Amplitude BF')
disp.joliplot('y (m)', 'Amplitude pic (UA)', y_m_front, A_xy_tot_2[x_pix_front, y_pix_front], exp = False, color = 5, legend = 'Amplitude HF')

if save : 
    sv.save_graph(save_path, 'SAVGOL10_A_fréquence_front_bichromatique', pdf = False)


disp.figurejolie(width = 12)

disp.joliplot('y (m)', 'Amplitude pic (UA)', y_m_front, A_xy_tot_1[x_pix_front, y_pix_front] * f_xy_tot_1[x_pix_front, y_pix_front]**2, exp = False, color = 2, legend = 'Amplitude*f2 BF')
disp.joliplot('y (m)', 'Amplitude pic (UA)', y_m_front, A_xy_tot_2[x_pix_front, y_pix_front] * f_xy_tot_2[x_pix_front, y_pix_front]**2, exp = False, color = 5, legend = 'Amplitude*f2 HF')

if save : 
    sv.save_graph(save_path, 'SAVGOL10_A_f_carre_fréquence_front_bichromatique', pdf = False)

disp.figurejolie(width = 12)

disp.joliplot('y (m)', 'Amplitude pic 1 / 2 ', y_m_front, A_xy_tot_1[x_pix_front, y_pix_front]/A_xy_tot_2[x_pix_front, y_pix_front], exp = False, color = 2, legend = 'Amplitude BF / Amplitude HF')
disp.joliplot('y (m)', 'Amplitude pic*f**2 1 / 2', y_m_front, (A_xy_tot_2[x_pix_front, y_pix_front] * f_xy_tot_1[x_pix_front, y_pix_front]**4)/(A_xy_tot_2[x_pix_front, y_pix_front] * f_xy_tot_2[x_pix_front, y_pix_front]**4) , exp = False, color = 5, legend = 'Amplitude*f2 BF / A f2 HF')
disp.joliplot('y (m)', 'Amplitude pic (UA)', y_m_front, y_m_front**0, exp = False, color = 8, legend = '1')

if save : 
    sv.save_graph(save_path, 'SAVGOL10_Rapport_de_A_fréquence_front_bichromatique', pdf = False)

#%% Ac autour du front final

save = True

temps = True
espace = True

#Mesure de Ac et kappa_c
Ac_avant = np.zeros(len(y_pix_front))
Ac_apres = np.zeros(len(y_pix_front))
Ac_avant_x = np.zeros(len(y_pix_front))
Ac_apres_x = np.zeros(len(y_pix_front))

t_avant_frac = 150
t_apres_frac = 150

d = 5 #m
theta = 90 * np.pi / 180
aa = 4 #m

for i in range(len(y_pix_front)) :
    t_demi_T = int(1/ f_xy_tot_2[x_pix_front[i], y_pix_front[i]] / 2 * facq)
    lambda_sur2 = 1.5614 * (2*t_demi_T/facq)**2 / 2 #en m
    pas_y = (np.max(X[y_pix_front[i],:]) - np.min(X[y_pix_front[i],:])) / X.shape[0] #m/px
    lambda_sur2 = int(lambda_sur2 / pas_y)

    colors = disp.vcolors( int(( i / len(y_pix_front) * 9)) )
    
    if temps :
        tmax_avant = np.argmax(eta[x_pix_front[i], y_pix_front[i], t_frame_front[i] - 3 * t_demi_T:t_frame_front[i] - t_demi_T]) + t_frame_front[i] - 3*t_demi_T
        tmin_avant = np.argmin(eta[x_pix_front[i], y_pix_front[i], t_frame_front[i] - 3 * t_demi_T:t_frame_front[i] - t_demi_T]) + t_frame_front[i] - 3*t_demi_T
        tmax_apres = np.argmax(eta[x_pix_front[i], y_pix_front[i], t_frame_front[i] - t_demi_T:t_frame_front[i] + t_demi_T]) + t_frame_front[i] - t_demi_T
        tmin_apres = np.argmin(eta[x_pix_front[i], y_pix_front[i], t_frame_front[i] - t_demi_T:t_frame_front[i] + t_demi_T]) + t_frame_front[i] - t_demi_T
        
        Ac_avant[i] = (eta[x_pix_front[i], y_pix_front[i],tmax_avant] - eta[x_pix_front[i], y_pix_front[i],tmin_avant])/2
        Ac_apres[i] = (eta[x_pix_front[i], y_pix_front[i],tmax_apres] - eta[x_pix_front[i], y_pix_front[i],tmin_apres])/2
        plt.figure(55)
        plt.plot(eta[x_pix_front[i], y_pix_front[i], t_frame_front[i] - 3*t_demi_T:t_frame_front[i] - t_demi_T], color = colors)
        plt.figure(56)
        plt.plot(eta[x_pix_front[i], y_pix_front[i], t_frame_front[i] - t_demi_T:t_frame_front[i]+ t_demi_T], color = colors)
    
    if espace :
        #On regarde min et max à +- lambda/2 autour de la fracure
        #Puis min max à fracture et fracture + lambda, à tfrac + T/2
    
        xmax_avant = np.argmax(eta[x_pix_front[i] - lambda_sur2:x_pix_front[i] + lambda_sur2, y_pix_front[i], t_frame_front[i]]) + x_pix_front[i] - lambda_sur2
        xmin_avant = np.argmin(eta[x_pix_front[i] - lambda_sur2:x_pix_front[i] + lambda_sur2, y_pix_front[i], t_frame_front[i]]) + x_pix_front[i] - lambda_sur2
        xmax_apres = np.argmax(eta[x_pix_front[i]:x_pix_front[i] + int(2*lambda_sur2), y_pix_front[i], t_frame_front[i] + int(t_demi_T)]) + x_pix_front[i]
        xmin_apres = np.argmin(eta[x_pix_front[i]:x_pix_front[i] + int(2*lambda_sur2), y_pix_front[i], t_frame_front[i] + int(t_demi_T)]) + x_pix_front[i]
        
        
        
        Ac_avant_x[i] = (eta[xmax_avant, y_pix_front[i],t_frame_front[i]] - eta[xmin_avant, y_pix_front[i],t_frame_front[i]])/2
        Ac_apres_x[i] = (eta[xmax_apres, y_pix_front[i],t_frame_front[i]+ int(t_demi_T)] - eta[xmin_apres, y_pix_front[i],t_frame_front[i]+ int(t_demi_T)])/2
        plt.figure(57)
        plt.plot(eta[x_pix_front[i] - lambda_sur2:x_pix_front[i] + lambda_sur2, y_pix_front[i], t_frame_front[i]], color = colors)
        plt.figure(58)
        plt.plot(eta[x_pix_front[i]:x_pix_front[i] + int(2*lambda_sur2), y_pix_front[i], t_frame_front[i] + int(t_demi_T)], color = colors)
        
    
    
    

    


if temps :
    
    if save : 
        plt.figure(55)
        plt.title('eta_t_avant_frac')
        sv.save_graph(save_path, 'SAVGOL10_eta_t_avant_frac_fct_y_', pdf = False)
        plt.figure(56)
        plt.title('eta_t_apres_frac')
        sv.save_graph(save_path, 'SAVGOL10_eta_t_apres_frac_fct_y_', pdf = False)
    disp.figurejolie(width = 12)
    disp.joliplot('y (m)', r'$A_c$(m)', y_m_front, Ac_avant, exp = False, color = 2, legend = "T avant fracture") #rouge, entre T/2 et 3 T/2 avant fracture, la période avant fracture
    disp.joliplot('y (m)', r'$A_c$(m)', y_m_front, Ac_apres, exp = False, color = 12, legend = "T à la fracture") #violet, autour de la fracture +- T/2
    
    if save : 
        sv.save_graph(save_path, 'SAVGOL10_Ac_y_pr_tfrac_xfrac', pdf = False)
    
    
    A_c = (Ac_avant + Ac_apres) / 2
    
    err_Ac = np.abs(Ac_avant - Ac_apres) / 2
    
    
    meanAC = np.zeros(len(y_pix_front))
    for i in range(len(y_pix_front)) :
        meanAC[i] = np.mean(A_c[i:])
    
    disp.figurejolie(width = 12)
    
    disp.joliplot(r'y cut', r'mean $A_c$', y_m_front, meanAC, exp = False, color = 2)    
        
    disp.figurejolie(width = 12)
    disp.joliplot(r'y (m)', r'$A_c$ (m)', y_m_front, A_c, exp = True, color = 4)
    plt.errorbar(y_m_front, A_c, err_Ac, None, fmt = 'none')

    if save : 
        sv.save_graph(save_path, 'SAVGOL10_Ac_x_y_pr_tfrac_xfrac_avec_err', pdf = False)
        
        
if espace : 

    if save : 
        plt.figure(57)
        plt.title('x_à_frac')
        sv.save_graph(save_path, 'SAVGOL10_eta_x_avant_frac_fct_y_t_' + str(t_avant_frac) + 'frames', pdf = False)
        plt.figure(58)
        plt.title('x_apres_frac')
        sv.save_graph(save_path, 'SAVGOL10_eta_x_apres_frac_fct_y_t_' + str(t_apres_frac) + 'frames', pdf = False)
        
    disp.figurejolie(width = 12)
    disp.joliplot('y (m)', r'$A_c$(m)', y_m_front, Ac_avant_x, exp = False, color = 2, legend = "x à la fracture") #rouge, entre T/2 et 3 T/2 avant fracture, la période avant fracture
    disp.joliplot('y (m)', r'$A_c$(m)', y_m_front, Ac_apres_x, exp = False, color = 12, legend = "x apres la fracture") #violet, autour de la fracture +- T/2
    
    if save : 
        sv.save_graph(save_path, 'SAVGOL10_Ac_y_pr_tfrac_xfrac', pdf = False)  
        
    A_c_x = (Ac_avant_x + Ac_apres_x) / 2
    
    err_Ac_x = np.abs(Ac_avant_x - Ac_apres_x) / 2
    
    
    meanAC_x = np.zeros(len(y_pix_front))
    for i in range(len(y_pix_front)) :
        meanAC_x[i] = np.mean(A_c_x[i:])
    
    disp.figurejolie(width = 12)
    
    disp.joliplot(r'y cut', r'mean $A_cx$', y_m_front, meanAC_x, exp = False, color = 2)
        
    disp.figurejolie(width = 12)
    disp.joliplot(r'y (m)', r'$A_cx$ (m)', y_m_front, A_c_x, exp = True, color = 4)
    plt.errorbar(y_m_front, A_c_x, err_Ac_x, None, fmt = 'none')
    
    if save : 
        sv.save_graph(save_path, 'SAVGOL10_Ac_x_y_pr_tfrac_xfrac_avec_err', pdf = False)
    

#%% kappa_c autour du front final
import icewave.drone.drone_projection as dp

save_video = False
save = True

x_0 = pos_front[best_y_front[20]]
y_0 = best_y_front[20]
t_0 = t_front[best_y_front[20]]
d_x = 5 #m
d_y = 5 #m

size = 100

if save_video :
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as manimation
    
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=20, metadata=metadata)
    
    fig = plt.figure()
    
    with writer.saving(fig, "animation_cosinus.mp4", 100):
        for i in range(t_0 - 270, t_0+60, 2) :
            dist_p0_x, dist_p0_y, rectangle = fd.extract_rectangle(data, x_0, y_0, i, eta, d_x, d_y, size = 1000)
    
            plt.pcolormesh(dist_p0_x, dist_p0_y,rectangle)
            plt.axis('equal')
            # plt.pause(2)
            writer.grab_frame()
            
            if i % 10 == 0:
                print(str(int(i/165 * 100)) + '%' )
    

def divergence(rectangle, x, y): 
    return np.asarray(np.gradient(rectangle))[0,:,:] + np.asarray(np.gradient(rectangle))[1,:,:]

def courbure(rectangle, pas_x, pas_y) :
    return 0.5 * (np.asarray(np.gradient(np.asarray(np.gradient(rectangle, pas_x))[0,:,:], pas_x))[0,:,:] + np.asarray(np.gradient(np.asarray(np.gradient(rectangle, pas_y))[1,:,:], pas_y))[1,:,:])


disp.figurejolie()    
# div_rectangle = divergence(rectangle, dist_p0_x, dist_p0_y)

# plt.figure()
# plt.pcolormesh(dist_p0_x, dist_p0_y, div_rectangle)
# plt.colorbar()

# mean_curv = - 0.5 * divergence( div_rectangle / np.abs(div_rectangle), dist_p0_x, dist_p0_y)
# curvature = - 0.5 * divergence( div_rectangle, dist_p0_x, dist_p0_y)

# plt.figure()
# plt.pcolormesh(dist_p0_x, dist_p0_y, mean_curv)
# plt.colorbar()

# max_kappa = np.zeros(len(best_y_front))
t_max_kappa = np.zeros(len(best_y_front))


kappa_c_avant = np.zeros(len(best_y_front))
kappa_c_apres = np.zeros(len(best_y_front))



for j in range (len(y_pix_front)) :
    x_0 = x_pix_front[j]
    y_0 = y_pix_front[j]
    t_0 = t_frame_front[j]
    
    t_demi_T = int(1/ f_xy_tot_2[x_pix_front[j], y_pix_front[j]] / 2 * facq)
    
    kappa_2D = np.zeros(30)
    
    for i in range(30) :
        dist_p0_x, dist_p0_y, rectangle = fd.extract_rectangle(data,x_0, y_0, int(t_0 + (t_demi_T * 2)/ 30 * (i + 1)) , eta, d_x, d_y, size = size)
        pas_x = dist_p0_x[-1] - dist_p0_x[-2]
        pas_y = dist_p0_y[-1] - dist_p0_y[-2]
        rectangle_kappa = cv2.blur(rectangle[:len(dist_p0_x), :len(dist_p0_y)], (int(size/4), int(size/4)) )
        curvature = courbure(rectangle_kappa, pas_x, pas_y)
        kappa_2D[i] = np.nanmax(curvature)
        
    colors = disp.vcolors( int(j/len(y_pix_front) * 9) ) 
    plt.figure(61)
    plt.plot(kappa_2D, color = colors)
    
    if j % 5 == 0:
        print(str(int(j/len(y_pix_front) * 100)) + '%' )
    kappa_c_avant[j] = np.nanmax(kappa_2D)
    t_max_kappa[j] = np.argmax(kappa_2D)#* 10 + t0 - 150
    
    
    kappa_2D = np.zeros(30)
    
    for i in range(30) :
        dist_p0_x, dist_p0_y, rectangle = fd.extract_rectangle(data,x_0, y_0, int(t_0 + (t_demi_T * 2)/ 30 * (i + 1) + (t_demi_T * 2)) , eta, d_x, d_y, size = size)
        pas_x = dist_p0_x[-1] - dist_p0_x[-2]
        pas_y = dist_p0_y[-1] - dist_p0_y[-2]
        rectangle_kappa = cv2.blur(rectangle[:len(dist_p0_x), :len(dist_p0_y)], (int(size/4), int(size/4)) )
        curvature = courbure(rectangle_kappa, pas_x, pas_y)
        kappa_2D[i] = np.nanmax(curvature)
    
    kappa_c_apres[j] = np.nanmax(kappa_2D)
    colors = disp.vcolors( int(j/len(y_pix_front) * 9) ) 
    plt.figure(62)
    plt.plot(kappa_2D, color = colors)
    
# kappa = (xp*ypp - yp * xpp) / (xp**2 + yp**2)**(1.5) #c'est quoi xp ???

# #mean curvature : -0.5 div(div(f) / abs(div(f)))

kappa_c = (kappa_c_avant + kappa_c_apres) / 2

err_kappa_c = np.abs(kappa_c_apres - kappa_c_avant) / 2


disp.figurejolie(width = 12)
disp.joliplot('y (m)', r'$\kappa_c$ (m$^{-1}$)', y_m_front, kappa_c, exp = True, color = 2, legend = r'$\kappa_c$')
plt.errorbar(y_m_front, kappa_c, err_kappa_c, fmt = 'none')

if save : 
    sv.save_graph(save_path, 'SAVGOL10_kappa_c_y_pr_tfrac_xfrac_avec_err_d_5m', pdf = False)


disp.figurejolie(width = 12)
disp.joliplot('y (m)', r'$\kappa_c$ (m$^{-1}$)', y_m_front, kappa_c_avant, exp = False, color = 2, legend = r'$\kappa_c$ avant')
disp.joliplot('y (m)', r'$\kappa_c$ (m$^{-1}$)', y_m_front, kappa_c_apres, exp = False, color = 5, legend = r'$\kappa_c$ après')


if save : 
    sv.save_graph(save_path, 'SAVGOL10_kappa_c_y_avant_apres_d_5_m', pdf = False)
    
disp.figurejolie(width = 12)
disp.joliplot('x (m)', r'y (m)', np.arange(0,size) / size * d_x * 2, np.arange(0,size) / size * d_y * 2, table = rectangle, exp = False, color = 5, legend = r'$\kappa_c$ après')

disp.figurejolie(width = 12)
disp.joliplot('x (m)', r'y (m)', np.arange(0,size) / size * d_x * 2, np.arange(0,size) / size * d_y * 2, table =  cv2.blur(rectangle[:len(dist_p0_x), :len(dist_p0_y)], (int(size/4), int(size/4)) ), exp = False, color = 5, legend = r'$\kappa_c$ après')

if save : 
    sv.save_graph(save_path, 'SAVGOL10_rectangle_pour_kappa_d_5_m', pdf = False)
disp.figurejolie(width = 12)
disp.joliplot('x (m)', r'y (m)', np.arange(0,size) / size * d_x * 2, np.arange(0,size) / size * d_y * 2, table = courbure(cv2.blur(rectangle[:len(dist_p0_x), :len(dist_p0_y)], (int(size/4), int(size/4)) ), pas_x, pas_y), exp = False, color = 5, legend = r'$\kappa_c$ après')



#%% RESULTATS (kappac(Ac))
save = True

disp.figurejolie()
disp.joliplot(r'y (m)', r'$A_c$ (m)', y_m_front, A_c, exp = True, color = 4)
plt.errorbar(y_m_front, A_c, err_Ac, None, fmt = 'none')

if save : 
    sv.save_graph(save_path, 'Ac_y_pr_tfrac_xfrac_avec_err', pdf = False)


disp.figurejolie()
disp.joliplot(r'y (m)', r'$\kappa_c$ (m$^{-1}$)', y_m_front, kappa_c, exp = True, color = 4)
plt.errorbar(y_m_front, kappa_c, err_kappa_c, None, fmt = 'none')

if save : 
    sv.save_graph(save_path, 'kappa_c_y_pr_tfrac_xfrac_avec_err', pdf = False)


len_err = 5
err_fit = np.zeros(len_err)
err_max_range = np.linspace(0.1, 0.6, len_err)
alpha = np.zeros(len_err)

for i in range(len_err):
    err_max = err_max_range[i]
    A_c_filt = A_c[(err_Ac < A_c * err_max) * (err_kappa_c < kappa_c * err_max)]
    kappa_c_filt = kappa_c[(err_Ac < A_c * err_max) * (err_kappa_c < kappa_c * err_max)]
    err_Ac_filt = err_Ac[(err_Ac < A_c * err_max) * (err_kappa_c < kappa_c * err_max)]
    err_kappa_c_filt = err_kappa_c[(err_Ac < A_c * err_max) * (err_kappa_c < kappa_c * err_max)]
    
    disp.figurejolie()
    disp.joliplot(r'$A_c$ (m)', r'$\kappa_c$ (m$^{-1}$)', A_c_filt, kappa_c_filt, exp = True, color = 4, title = 'err = ' + str(err_max))
    plt.errorbar(A_c_filt, kappa_c_filt, err_kappa_c_filt, err_Ac_filt, fmt = 'none')
    
    popt, pcov = fits.fit_powerlaw(A_c_filt, kappa_c_filt, display = True, legend = 'err = ' + str(err_max))
    
    if save : 
        sv.save_graph(save_path, 'kappa_c_A_c_filtre_err_' + str(err_max), pdf = False)
    
    err_fit[i] = np.sqrt(np.diag(pcov)[0]) / popt[0]
    alpha[i] = popt[0]
    
disp.figurejolie()
disp.joliplot(r'erreur (\%)', 'erreur fit', err_max_range, err_fit, color = 2)

if save : 
    sv.save_graph(save_path, 'err_fit_err_max_toleree', pdf = False)

disp.figurejolie()
disp.joliplot(r'erreur (\%)', 'slope', err_max_range, alpha, color = 8)

if save : 
    sv.save_graph(save_path, 'pente_err_max_toleree', pdf = False)



#%% Courbure (t) sur eta
save = False
facq = 29.97
a = 50
err = 1000
imax = np.zeros(dt*2, dtype = int) 
hmax = np.zeros(dt*2)
popt_max = np.zeros(dt*2, dtype = object)
kmax = np.zeros(dt*2)
err_max = np.zeros(dt*2)

     
for i in range(lines.shape[1]) :
    forme = lines[:,i]
    n = forme.shape[0]
    t = i / facq

    imax[i] = int(np.argmax(forme[a:lines.shape[0] - a]) + a)  
    hmax[i] = forme[imax[i]]
    
    yfit = forme[imax[i]-a:imax[i]+a]
    xfit = dist_p0[imax[i]-a:imax[i]+a]
    
    popt_max[i] = np.polyfit(xfit,yfit, 2, full = True)
    
    yth = np.polyval(popt_max[i][0], xfit)
    err_max[i] = np.sqrt(popt_max[i][1][0]) / np.abs(np.max(hmax[i]))
    if err_max[i] > err :
        kmax[i] = None
        err_max[i] = None
        hmax[i] = None
    else :
        kmax[i] = np.abs(popt_max[i][0][0]*2)

if display :
    disp.figurejolie()    
    plt.plot(dist_p0[imax], hmax, 'r^')    
    for ww in range (lines.shape[1]) :
        colors = disp.vcolors( int(ww / lines.shape[1] * 9))
        disp.joliplot('x (m)', r'$\zeta$ (m)', [], [] ,color = 2, exp = False, linewidth= 5)
        plt.plot(dist_p0, lines[:,ww],color=colors)
    for i in range(lines.shape[1]) :
        if not np.isnan(hmax[i]) :
            xfit = dist_p0[imax[i]-a:imax[i]+a]
            yth = np.polyval(popt_max[i][0], xfit)
            plt.plot(xfit, yth, 'r-') 
            
    if save : 
        sv.save_graph(save_path, 'Courbure_mesure_a' + str(a), pdf = False)

# disp.figurejolie()
# disp.joliplot(r'x (m)', r'$\zeta$ (m)', hmax, kmax, color = 8)

tt = np.linspace(0, (lines.shape[1] -1) * dT, lines.shape[1])
disp.figurejolie()
disp.joliplot(r't (s)', r'$\kappa$ (m$^{-1}$)', tt, kmax, color = 8)

if save : 
    sv.save_graph(save_path, 'kappa_t_a' + str(a), pdf = False)

#%% shear vertical
tt = np.linspace(0, (2 * dt-1) / facq, 2 * dt)

F = np.zeros(dt*2, dtype = object)
dF = np.zeros(dt*2, dtype = object)

dx = np.abs(dist_p0[1] - dist_p0[0])
smooth = 20

kappa = np.zeros( (lines.shape[0] - 2* smooth - 2, dt*2) )
x_new = dist_p0[smooth+1:-smooth-1]

def deriv1(f,x,h,smooth = 1):
    return (f(x+(smooth * h))-f(x))/ (smooth * h)

#Une fonction interpolée par temps
for t in range (t0 - dt, t0 + dt) :
    Fz = interpolate.interp1d( dist_p0,lines[:,t - t0 - dt])
    F[t - t0 - dt] = Fz
    dF[t - t0 - dt] = deriv1(Fz, x_new, dx, smooth)
    kappa[:,t - t0 - dt] = dF[t - t0 - dt]#**2 / ( 1 + Fz(x_new)**2 )**(1.5)
    
    

#%%Plot shear
save = False
plt.figure()
rrange0 = 0
rrange1 = dt * 2

seuil = 0.03

for i in range (rrange0, rrange1) :
    # plt.plot(dist_p0, F[i](dist_p0))
    colors = disp.vcolors( int(( i / (rrange1 -rrange0) * 9)) )
    plt.plot(x_new, kappa[:,i], color = colors)
    disp.joliplot('x (m)', r'$\sigma_{xz}$ (s$^{-1}$)' , [], [] ,color = 2, exp = False, linewidth= 5)
plt.plot(x_new, np.linspace(seuil,seuil,len(x_new)))

if save : 
    sv.save_graph(save_path, 'cisaillement_x', pdf = False)

#spatio temporel de la courbure
disp.figurejolie()
disp.joliplot('x (m)','t (s)', x_new,T[t0 - dt + rrange0:t0 - dt + rrange1], table = kappa[:,rrange0:rrange1])
plt.clim(np.quantile(kappa[:,rrange0:rrange1], 0.5), np.quantile(kappa[:,rrange0:rrange1], 0.95))

plt.colorbar()
# plt.clim(seuil, seuil)

if save : 
    sv.save_graph(save_path, 'spatio_temporel_cisaillement', pdf = False)

#%% Find threshold
save = False
# x_crack = np.zeros()
t_crack = np.zeros(len(x_new), dtype = int) + t0 + dt
# t_crack[-1] = t0 + dt

for xi in range(len(x_new)-2, -1, -1) :
    thresh_notfound = True
    for ti in range (0, t_crack[xi + 1] - t0 + dt) : #ne peut pas casser à un temps superieur pour un x plus petit
        if np.abs(kappa[xi, ti]) > seuil and thresh_notfound :
            t_crack[xi] = ti + t0 - dt #la frame ou ca depasse le seuil à partir de t0
            thresh_notfound = False
        elif ti == t_crack[xi + 1] - t0 + dt -1 and thresh_notfound:
            t_crack[xi] = t_crack[xi+1]
            
            
plt.plot(x_new, T[t_crack], 'r-')

if save : 
    sv.save_graph(save_path, 'spatio_temporel_cisaillement_seuil_' + str(seuil), pdf = False)

#%% Spatio autour de t_frac
save = False
tps_etudi = 180
tps_pr_max = 90
tps_apres_max = 90

aa = 2 #a en metres
a = int( len(x_new) / (np.max(x_new) - np.min(x_new)) * aa ) 


plt.figure()
# t_top = np.asarray(t_crack - t0 + dt -60, dtype = int)
# t_top2 = np.asarray(t_crack - t0 + dt, dtype = int)

spatio_frac = np.zeros((len(x_new), tps_etudi+tps_apres_max))
t_maxs = np.zeros(len(x_new), dtype = int)
kappa_c = np.zeros(len(x_new))
A_c = np.zeros(len(x_new))

for i in range(len(x_new)-1) :
    
    imax = np.argmax(lines_eta[i, t_crack[i] - t0 + dt -tps_pr_max :t_crack[i] - t0 + dt+tps_apres_max])
    t_maxs[i] = t_crack[i] - tps_pr_max + imax
    
    spatio_frac[i, :] = lines_eta[i, t_crack[i] - t0 + dt + imax - tps_etudi - tps_pr_max:  t_crack[i]- t0 + dt + imax - tps_pr_max+tps_apres_max]
    
    #A
    A_c[i] = lines_eta[i, t_crack[i] - t0 + dt + imax - tps_pr_max]


#kappa




for j in range(len(t_maxs)) :
    forme = lines_eta[:,t_maxs[j]]
    if j > a  and len(x_new) - j > a :
        imax = j
        kappa_c[j] = fd.mesure_kappa(forme, x_new, a, imax)   
    
    
    
    

# nice_events = np.asarray(kappa[:,t_top:t_top2], dtype = int)

ttt = np.linspace(0,tps_etudi-1+tps_pr_max, tps_etudi+tps_apres_max)

disp.joliplot('x', 't', x_new, ttt, table = spatio_frac)

if False : 
    sv.save_graph(save_path, 'spatio_temporel_tcrack', pdf = False)

disp.figurejolie()
disp.joliplot('x', 'Ac', x_new, A_c, exp = False, color = 4, linewidth= 3)

if save : 
    sv.save_graph(save_path, 'Ac_x_pr_tcrack', pdf = False)

disp.figurejolie()
disp.joliplot('t', r'$\kappa_c$ (m$^{-1}$)', x_new, kappa_c, color = 8)

if save : 
    sv.save_graph(save_path, 'kappac_t_a' + str(aa) + 'm', pdf = False)

#%% profil avant/apres fracture : Lk
save = True
i_crack = 300


aa = 10 #a en metres
a = int( len(x_new) / (np.max(x_new) - np.min(x_new)) * aa )

disp.figurejolie()

# disp.joliplot('x', 'A', x_new[i_crack-a:i_crack + a], lines_eta[i_crack-a:i_crack + a, t_maxs[i_crack] - t0 + dt], exp = False, color = 6)
# plt.plot(x_new[i_crack],lines_eta[i_crack, t_maxs[i_crack] - t0 + dt], 'rx' )

for i in range(-30,180,5) : #on regarde un peu apres et beaucoup avant tmax (tmax c'est le temps ou il y a le max d'amplitude le plus proche de t_crack)
    colors = disp.vcolors(int((i+30) / 210 * 9))
    
    plt.plot(x_new[i_crack-a:i_crack + a], lines_eta[i_crack-a:i_crack + a, t_maxs[i_crack] - t0 + dt + i], color = colors)
    #Le temps va de violet à jaune
    
    
    # if i == 0 :
    #     disp.joliplot('x', 'A', x_new[i_crack-a:i_crack + a], lines_eta[i_crack-a:i_crack + a, t_maxs[i_crack] - t0 + dt + i], exp = False, color = 5)
    # else :
    #     disp.joliplot('x', 'A', x_new[i_crack-a:i_crack + a], lines_eta[i_crack-a:i_crack + a, t_maxs[i_crack] - t0 + dt + i], exp = False, color = 6)
    
    # plt.plot(x_new[i_crack],lines_eta[i_crack, t_maxs[i_crack] - t0 + dt], 'rx' )


if save : 
    sv.save_graph(save_path, 'Profils_avant_apres_tcrack_x_' + str(x_new[i_crack]) + 'm', pdf = False)


