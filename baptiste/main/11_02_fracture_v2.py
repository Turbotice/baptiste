# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 18:14:30 2025

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
import baptiste.tools.tools as to


import icewave.tools.matlab2python as m2p
import icewave.baptiste.Fct_drone_1102 as fd






#%% IMPORT DATA

# path = "K:\Share_hublot\\Data\\0211\\Drones\\bernache\\matData\\18-stereo_001\\"
path = "Y:\Banquise\\Baptiste\\Resultats\\Analyse_1102\\Data\\"
path = "K:\Share_hublot\\Data\\0211\\Drones\\"

filename = "PIV_processed_i011500_N15500_Dt4_b1_W32_xROI1_width3839_yROI1_height2159_scaled_bernache.mat"
filename = 'PIV_processed_i011496_N15496_Dt4_b1_W32_xROI1_width3839_yROI1_height2159_scaled_mesange.mat'
filename = 'real_field_stereo_0211_2024_rectangular_grid.h5'

save_path = 'Y:\Banquise\\Baptiste\\Resultats\\Analyse_1102\\Mesange'

f = h5py.File(path + filename,'r') 

def h5_to_dict_fast(h5obj):
    """Convertit récursivement un fichier ou groupe h5py en dict avec numpy arrays."""
    out = {}
    for key, item in h5obj.items():
        if isinstance(item, h5py.Dataset):
            # Lecture directe en numpy array
            out[key] = item[()]  
        elif isinstance(item, h5py.Group):
            # Récursion rapide pour les groupes
            out[key] = h5_to_dict_fast(item)
    return out
# matdata = open(path + filename, 'r')

data = h5_to_dict_fast(f)


Vz = np.asarray(data['u'][2]) #t,y,x en s, m, m
Vz_og = np.asarray(data['u'][2])

# Vz = np.transpose(Vz, (2,1,0)) #Pour passer de s(t,y,x) à s(x,y,t)
# Vz = np.flip(Vz, 1)


Vx = np.asarray(data['u'][0])

Vy = np.asarray(data['u'][1])


X = data['X'] #m 

Y = data['Y'] #m 

T = data["t"] #s 

facq = data['SCALE']['facq_t']
mparpixel = 1/data['SCALE']['facq_x']


nx = X.shape[0]
ny = Y.shape[1]
x = X[0,:]

y = Y[:,0]
x_pix = np.linspace(0, nx-1, nx, dtype=int)
y_pix = np.linspace(0, ny-1, ny, dtype=int)

x= x_pix * mparpixel
y= y_pix * mparpixel

#%% Vz en zeta  filtre
from scipy import signal
facq = 29.97
dT = 1 / facq


f_cut = [2]

[b,a] = signal.butter(3, f_cut, fs = facq, btype = "lowpass")
Vz_filt = signal.filtfilt(b, a, Vz, axis = 2)


# Vz_filt = Vz.butter(2, 5, btype = "low")
# eta = np.cumsum(Vz_filt, axis = 2) * dT
eta = np.cumsum(Vz, axis = 2) * dT


#%% V demod plot
save = False


# disp.figurejolie(width = 8.6 * 4 / 5, height = 8.6 * 4 / 6)
f_demod = np.linspace(0.20,0.30,3)

for i in range (len(f_demod)):
    fexc = f_demod[i]
    V_demod_i = ft.demodulation(T, Vz, fexc)
    disp.figurejolie(width = 8.6 * 104.8 / 120, height = 8.6 * 73.6 / 120)
    
    plt.pcolormesh(y,x,np.flip(np.flip(np.real(V_demod_i),1),0), cmap = 'RdBu')
    plt.title('V demod ' + str(fexc) + ' Hz')
    cbar = plt.colorbar()
    cbar.set_label(r'$U_z$ démodulé à ' + str(f_demod[i]) + ' Hz')

    plt.xlabel(r'$X$ (m)')
    plt.ylabel(r'$Y$ (m)')
    # plt.pause(0.3)
    if save :    
        sv.save_graph(save_path, 'Champ_demod_' + str(fexc) + 'Hz', pdf = False)

disp.figurejolie(width = 8.6 * 104.8 / 120, height = 8.6 * 73.6 / 120)

plt.pcolormesh(y,x, np.flip(np.flip(Vz[:,:,905],1),0), cmap = 'RdBu')
# plt.title('V demod ' + str(fexc) + ' Hz')
cbar = plt.colorbar()
cbar.set_label(r'$U_z$ (m.s$^{-1}$)' )

plt.xlabel(r'$X$ (m)')
plt.ylabel(r'$Y$ (m)')



#%% theta et k (f)

# f_demod = np.linspace(0.184,0.370,50)
f_demod = np.linspace(0.1,0.4,100)
f_demod = np.linspace(1 / (T[-1] - T[0]), facq / 2, int(len(T)/2))

padding = 9   #puissance de 2 pour le 0 padding
k_xx = []
k_yy = []
kk = []
theta = []
fff = []
vmax_FFT = []

plotplot = False

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

for i in range(len(f_demod)) :
    fexc = f_demod[i]
    # print('fréquence : ', fexc)
    demod = ft.demodulation(T - T[0],Vz,fexc)
    demod_padding = np.zeros((2**padding,2**padding), dtype = 'complex128')
    # for l in range (nx):
    #     for m in range(ny):
    #         demod_padding[l,m] = demod_padding[l,m] + demod[l,m]
    demod_padding[:nx,:ny] = demod
    Y_FFT = fft.fft2(demod_padding)
    Y_FFT = fft.fftshift(Y_FFT)
    nx_FFT = np.shape(Y_FFT)[0]
    ny_FFT = np.shape(Y_FFT)[1]
    
    #zone kx>0 et ky <0 :
    # [int(nx_FFT/2):,:int(nx_FFT/2)]
    # max_fft = np.asarray([(np.where(np.max(np.abs(Y_FFT[int(nx_FFT/2):,:int(ny_FFT/2)])) == np.abs(Y_FFT[int(nx_FFT/2):,:int(ny_FFT/2)]))[0][0])/ nx_FFT,
    #                       (np.where(np.max(np.abs(Y_FFT[int(nx_FFT/2):,:int(ny_FFT/2)])) == np.abs(Y_FFT[int(nx_FFT/2):,:int(ny_FFT/2)]))[1][0] - ny_FFT/2) / ny_FFT]) /mparpixel * 2 * np.pi
    
    
    max_fft = ((np.asarray([(np.where(np.max(np.abs(Y_FFT)) == np.abs(Y_FFT))[0][0]), (np.where(np.max(np.abs(Y_FFT)) == np.abs(Y_FFT))[1][0])] )) - nx_FFT/2) / nx_FFT /mparpixel* 2 * np.pi
    
    k_x = np.linspace(-2*np.pi/(2* mparpixel), 2*np.pi/(mparpixel* 2), nx_FFT)
    k_y = np.linspace(-2*np.pi/(2* mparpixel), 2*np.pi/(mparpixel* 2), ny_FFT)

    
    k_xx.append(max_fft[0])
    k_yy.append(max_fft[1])
    kk.append(cart2pol(max_fft[0],max_fft[1])[0])
    theta.append(cart2pol(max_fft[0],max_fft[1])[1])
    fff.append(fexc)
    vmax_FFT.append(np.max(np.abs(Y_FFT)))

    
    
    if plotplot :
        disp.figurejolie(width = 8.6)
        plt.pcolormesh(k_y,k_x, np.abs(Y_FFT),shading='auto', cmap = 'ma_cm')
        # plt.pcolormesh(np.abs(Y_FFT),shading='auto', cmap = 'ma_cm')
        plt.plot(k_yy[-1],k_xx[-1], 'ro', label = 'MAX FFT')
        plt.xlabel(r'$k_x$ (m$^{-1}$)' )
        plt.ylabel(r'$k_y$ (m$^{-1}$)' )
        # plt.legend()
 
k_xx = np.asarray(k_xx)
k_yy = np.asarray(k_yy)
kk = np.asarray(kk)
theta = np.asarray(theta)
fff = np.asarray(fff)
vmax_FFT = np.asarray(vmax_FFT)
omega = fff * 2 * np.pi


#%% AFFICHAGE RDD et ANGLE

disp.figurejolie()
# disp.joliplot('f(Hz)', 'angle (radian)',fff,theta, exp = False)
disp.joliplot('f(Hz)', r'$\theta$', fff, np.cos(theta), exp = True, cm = 6, zeros = False)

disp.figurejolie(width = 8.6 * 4 / 5)
# disp.joliplot('f(Hz)', 'angle (radian)',fff,theta, exp = False)
# disp.joliplot('f(Hz)', r'$\theta$', np.sqrt(9.81 / kk), np.cos(theta), exp = False, cm = 6, zeros = True)

def linlin(x,a,b) :
    return a * x + b

def fit_theta0_U(x, U, theta0) :
    return (x / U - np.sin(theta) * np.sin(theta0)) / np.cos(theta0)
    
popt, pcov = fits.fit(fit_theta0_U, np.sqrt(9.81 / kk), np.cos(theta), display = True, err = False, nb_param = 2, p0 = [10,0], bounds = False, 
        zero = True, th_params = False, xlabel = r'k (m$^{-1}$)', ylabel = r'cos($\theta$)', legend_data = r'Experimental Data', 
        legend_fit = 'h = ', log = False, cm = 3)






disp.figurejolie(width = 8.6 * 4 / 5)
# disp.joliplot('f(Hz)', r'K (m$^{-1}$)', fff*2, kk, color = 4)
disp.joliplot(r'k (m$^{-1}$)', r'$\omega$ (s$^{-1}$)', fff, kk, cm = 6, zeros = True)

disp.figurejolie()
popt, pcov = fits.fit(rdd.RDD_pesante, kk, omega, display = True, err = False, nb_param = 1, p0 = [0], bounds = False, 
        zero = False, th_params = False, xlabel = r'k (m$^{-1}$)', ylabel = r'$\omega$', legend_data = r'Experimental Data', 
        legend_fit = 'h = ', log = False, cm = 4)
klin = np.linspace(np.min(kk), np.max(kk), 1000)
disp.joliplot(r'k (m$^{-1}$)', r'$\omega$ (s$^{-1}$)', klin, rdd.RDD_gravitaire(klin), color = 2, exp = False, log = True)



def RDD_pesante_v2 (k,h) :
    return np.sqrt( (9.81 * k) / (1 + 1.1 * h * k) )

disp.figurejolie(width = 8.6 * 4 / 5)
popt, pcov = fits.fit(RDD_pesante_v2, kk, omega, display = True, err = False, nb_param = 1, p0 = [0.1], bounds = False, 
        zero = False, th_params = False, xlabel = r'k (m$^{-1}$)', ylabel = r'$\omega$', legend_data = r'Experimental Data', 
        legend_fit = 'h = ', log = False, cm = 5)
klin = np.linspace(np.min(kk), np.max(kk), 1000)
disp.joliplot(r'k (m$^{-1}$)', r'$\omega$ (s$^{-1}$)', klin, rdd.RDD_gravitaire(klin), color = 2, exp = False, log = True)
plt.ylim(1,3)

# E = 2.2e9
# h = 0.12
# D = E * h**3 / 10

# plt.plot(klin,RDD_pesante_v2(klin, 0.12) )


Y_t = np.fft.fft(Vz, axis=2)

f_fft = 1/(T - T[0])/2

disp.figurejolie()
disp.joliplot(r'k (m$^{-1}$)', r'$\omega$ (s$^{-1}$)', f_fft, np.abs(np.mean(np.mean(Y_t, axis = 0), axis = 0) - np.mean(Y_t)), cm = 6, exp = False, log = True)


disp.figurejolie()
disp.joliplot(r'k (m$^{-1}$)', r'max FFT', kk, vmax_FFT, cm = 6, exp = False, log = True)


#%% Densité spectrale en energie

# f_demod = np.linspace(0.184,0.370,50)
f_demod = np.linspace(0.1,0.4,100)

f_demod = np.linspace(1 / (T[-1] - T[0]), facq / 2, int(len(T)/2))

padding = 9   #puissance de 2 pour le 0 padding

fff = []
vmax_FFT = []
composante_f_FFT = []

plotplot = False

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

for i in range(len(f_demod)) :
    fexc = f_demod[i]
    # print('fréquence : ', fexc)
    demod = ft.demodulation(T - T[0],eta,fexc)
    demod_padding = np.zeros((2**padding,2**padding), dtype = 'complex128')
    # for l in range (nx):
    #     for m in range(ny):
    #         demod_padding[l,m] = demod_padding[l,m] + demod[l,m]
    demod_padding[:nx,:ny] = demod
    Y_FFT = fft.fft2(demod_padding)
    Y_FFT = fft.fftshift(Y_FFT)
    nx_FFT = np.shape(Y_FFT)[0]
    ny_FFT = np.shape(Y_FFT)[1]
    
    #zone kx>0 et ky <0 :
    # [int(nx_FFT/2):,:int(nx_FFT/2)]
    # max_fft = np.asarray([(np.where(np.max(np.abs(Y_FFT[int(nx_FFT/2):,:int(ny_FFT/2)])) == np.abs(Y_FFT[int(nx_FFT/2):,:int(ny_FFT/2)]))[0][0])/ nx_FFT,
    #                       (np.where(np.max(np.abs(Y_FFT[int(nx_FFT/2):,:int(ny_FFT/2)])) == np.abs(Y_FFT[int(nx_FFT/2):,:int(ny_FFT/2)]))[1][0] - ny_FFT/2) / ny_FFT]) /mparpixel * 2 * np.pi
    
    
    fff.append(fexc)
    vmax_FFT.append(np.max(np.abs(Y_FFT)))
    composante_f_FFT.append(np.sum(np.abs(Y_FFT)**2) * mparpixel**2 / nx / ny)
    
    
    if plotplot :
        disp.figurejolie(width = 8.6)
        plt.pcolormesh(k_y,k_x, np.abs(Y_FFT),shading='auto', cmap = 'ma_cm')
        # plt.pcolormesh(np.abs(Y_FFT),shading='auto', cmap = 'ma_cm')
        plt.plot(k_yy[-1],k_xx[-1], 'ro', label = 'MAX FFT')
        plt.xlabel(r'$k_x$ (m$^{-1}$)' )
        plt.ylabel(r'$k_y$ (m$^{-1}$)' )
        # plt.legend()
 

fff = np.asarray(fff)
vmax_FFT = np.asarray(vmax_FFT)

composante_f_FFT = np.asarray(composante_f_FFT)
omega = fff * 2 * np.pi


disp.figurejolie(width = 8.6 * 4 / 5)
disp.joliplot(r'$\omega$ (s$^{-1}$)', r'$|Y| \omega^2$', omega, composante_f_FFT * omega**2, cm = 6, exp = False, log = True)

disp.figurejolie(width = 8.6 * 4 / 5)
disp.joliplot(r'k (m$^{-1}$)', r'$|Y| \omega^2$', omega**2 / 9.81, composante_f_FFT * omega**2, cm = 6, exp = False, log = True)

#%% Plot line in real space

save = False

# Vz = data['Vz'] #t,y,x en s, m, m

# Vz = np.transpose(Vz, (2,1,0)) #Pour passer de s(t,y,x) à s(x,y,t)
# Vz = np.flip(Vz, 1)
# V = Vz_filt
# V = eta
V = Vz


display = True
# xpix_0 = 165
# ypix_0 = 70
# d = 45 #m
# theta = 41.22 * np.pi / 180 #radians

# t0 = 1648
# dt = 10


xpix_0 = 50
ypix_0 = 50
d = 10 #m
theta = 90* np.pi / 180 #41.22 * np.pi / 180 #radians

t0 = 1600
dt = 1000



t_plot = t0 + dt

lines = np.zeros( (1000,2*dt) )
lines_eta = np.zeros( (1000,2*dt) )

# x_pix, y_pix, dist_p0 = fd.px_to_real_line(data, xpix_0, ypix_0, d, theta)

if display :
    t_plot = t0 + dt
    xp = np.linspace(0,V.shape[0]-1, V.shape[0])
    yp = np.linspace(0,V.shape[1]-1, V.shape[1])
    plt.figure()
    disp.joliplot('x (pixel)','y (pixel)',xp,yp, table = V[ :,:, t_plot])
    plt.clim(np.quantile(V[ :,:, t_plot], 0.1), np.quantile(V[ :,:, t_plot], 0.9))
    plt.plot(xpix_0, ypix_0, 'kx')
    # plt.plot(np.array(x_pix/16,dtype = int),  np.array(y_pix/16,dtype = int), 'r-')
    
    plt.axis('equal')
    
    if save : 
        sv.save_graph(save_path, 'Champ_Vz_filtre_t_' + str(t0 + dt) + 'frames_line', pdf = False)

display = False
if display :
    disp.figurejolie()
    
for t in  range (t0- dt,t0 + dt) :
    colors = disp.vcolors( int(( t- (t0 - dt) )  / dt / 2 * 9)) 
    
    Vz_line, dist_p0 = fd.px_to_real_line_TOP(xpix_0, ypix_0, d ,theta, Vz, x, y, t, size = 1000)
    # eta_line = fd.extract_line_Vz(data, eta, x_pix, y_pix, t)
    # lines[:,t- (t0 - dt) ] = Vz_line
    # lines_eta[:,t- (t0 - dt) ] = eta_line
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

sss = Vz[50,2,:]
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
        
        
#%% Plot spectro
    
save = False    
plt.figure()
plt.pcolormesh(X, Y, f_xy_tot_1, cmap = 'ma_cm')
plt.colorbar(label = r'f1 (Hz)')
# plt.clim(0.12, 0.25)
plt.axis('equal')
plt.xlabel(r'X (m)')
plt.ylabel(r'Y (m)')

if save : 
    sv.save_graph(save_path, 'n100_bichro_Champ_f_BF_max_0.12_0.25_Hz', pdf = False)


plt.figure()
plt.pcolormesh(X, Y, f_xy_tot_2, cmap = 'ma_cm')
plt.colorbar(label = r'f2 (Hz)')
# plt.clim(0.15, 0.30)
plt.axis('equal')
plt.xlabel(r'X (m)')
plt.ylabel(r'Y (m)')

if save : 
    sv.save_graph(save_path, 'n100_bichro_Champ_f_HF_max_0.15_0.30_Hz', pdf = False)
    
plt.figure()
plt.pcolormesh(X, Y, t_xy_tot_1, cmap = 'ma_cm')
plt.colorbar(label = r't1 (s)')
plt.contour(X, Y, t_xy_tot_1, 13, colors='k')

plt.axis('equal')
plt.xlabel(r'X (m)')
plt.ylabel(r'Y (m)')

if save : 
    sv.save_graph(save_path, 'n100_bichro_Champ_t_BF_isocontours', pdf = False)


plt.figure()
plt.pcolormesh(X, Y, t_xy_tot_2, cmap = 'ma_cm')
plt.colorbar(label = r't2 (s)')
plt.contour(X, Y, t_xy_tot_2, 13, colors='k')

plt.axis('equal')
plt.xlabel(r'X (m)')
plt.ylabel(r'Y (m)')


if save : 
    sv.save_graph(save_path, 'n100_bichro_Champ_t_HF_isocontours', pdf = False)
    
plt.figure()
plt.pcolormesh(X, Y, A_xy_tot_1/A_xy_tot_2, cmap = 'ma_cm')
plt.colorbar(label = r'A1 / A2')
# plt.clim(0.5,2)
# plt.contour(X, Y, np.rot90(A_xy_tot_1/A_xy_tot_2), 10, colors='k')

plt.axis('equal')
plt.xlabel(r'X (m)')
plt.ylabel(r'Y (m)')

if save : 
    sv.save_graph(save_path, 'n100_bichro_Champ_ABF_sur_AHF', pdf = False)


    


#%% Detection Front final : en x,y,t
save = False
d = 40#m
theta = 90* np.pi / 180 
seuil = 0.1 #Bernache 0.1

def deriv1(f,x,dx,smooth = 1):
    return (f(x+(smooth * dx))-f(x))/ (smooth * dx)

#Demi longueur autour de x pour trouver t front (en pixels ?)
l_findsigma = 10


#t à partir duquel on cherche la fracture
t0_frac = 1000 #Bernache 1500
t = 2900 #tps pour chercher seuil en x

#Les y qu'on regarde (en pixels)
table_y = np.linspace(0,ny-1,ny)
table_y = np.asarray(table_y, dtype = int)

#Taille sur laquelle on dérive en espace (en pixels)
smooth = 5
xpix_0 = 40
x_interp = x[xpix_0:xpix_0 +int(d / mparpixel)]
x_new = x_interp[smooth+1:-smooth-1]
dx = np.abs(x[1] - x[0])

# disp.figurejolie(width = 16)


t_front = np.zeros(len(table_y), dtype = int)

pos_front = np.zeros(len(table_y), dtype = int)

for ypix_0 in table_y: #,120) :
    # if ypix_0 > 80 : #BERNACHE : 40
    #     xpix_0 = 20 #bernache
    #     xpix_0 = 20 #mesange
    # else :
    #     xpix_0 = 20 #205 bernache

        

    
    Vz_line = Vz[xpix_0:xpix_0 +int(d / mparpixel),ypix_0,t]
    eta_line = eta[xpix_0:xpix_0 +int(d / mparpixel),ypix_0,t]
    F_eta = interpolate.interp1d(x_interp,eta_line)
    d_eta = deriv1(F_eta, x_new, dx, smooth)
    
    # disp.joliplot('x (m)', r'Vz (m.s$^{-1}$)' , dist_p0, Vz_line ,color = 2, exp = False, linewidth= 5)
    
    
    # disp.figurejolie(width = 16)
    # disp.joliplot('x (m)', r'$\eta$ (m)' , dist_p0, eta_line ,color = 2, exp = False, linewidth= 5)
    # disp.joliplot('x (m)', r'$\frac{\delta \eta}{\delta x}$' , x_new, d_eta ,color = 2, exp = False, linewidth= 3)
    
    pos_front[ypix_0] = int( round(x_pix[np.argmax(np.abs(d_eta)) +smooth + 1 + xpix_0]))
    
    # pour trouver le bon t
    x_frac = np.argmax(np.abs(d_eta)) +smooth + 1 + xpix_0
    
    x_new_frac = x[x_frac - l_findsigma-1: x_frac + l_findsigma+1]
    x_new_frac_smooth = np.asarray(x[x_frac - l_findsigma + smooth: x_frac + l_findsigma - smooth], dtype = float)
    
    sigma_zx = np.zeros( (l_findsigma * 2 - 2 * smooth) )
    #i est le temps auquel on regarde, on commence à t0_frac (en frames)
    #Si la zone de recherche de t sors du domaine on passe
    # if x_frac + l_findsigma > 1000 or  x_frac - l_findsigma < 0:
    #     pass
    # else :
        #Tant que le seuil n'est pas dépassé
        
    i = 1000    
    
    while np.abs(sigma_zx[int(l_findsigma - smooth)]) < seuil :
        Vz_line = Vz_line = Vz[x_frac - l_findsigma-1: x_frac + l_findsigma+1,ypix_0,i]
        F_Vz = interpolate.interp1d( x_new_frac,Vz_line)
        d_Vz = deriv1(F_Vz, x_new_frac_smooth, dx, smooth)
        #On remesure et stock le cisaillement vertical
        sigma_zx = d_Vz
        i += 1
        #Si le temps est trop grand, on passe
        if i > 2700 :
            break
    t_front[ypix_0] = i
    
    if ypix_0 % 10 == 0:
        print(str(int(ypix_0/len(table_y) * 100)) + '%' )
        
        
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
plt.plot(table_y, t_front)


plt.figure()
plt.plot(table_y, pos_front)

# if save : 
#     sv.save_graph(save_path, 'y_frac_v_frac', pdf = False)

# best_t_front = t_front[np.where( np.abs(pos_front/t_front - 0.105) < 0.005 )]
# best_x_front = pos_front[np.where( np.abs(pos_front/t_front - 0.105) < 0.005 )]
# best_y_front= table_y[np.where( np.abs(pos_front/t_front - 0.105) < 0.005 )]

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
    
#%% Detection front v2
plt.figure()
x_interp = x[xpix_0:]
x_new = x_interp[smooth+1:-smooth-1]

ypix_0 = 50

for t in range(1000, 3400, 30) :

    Vz_line = Vz[xpix_0:,ypix_0,t]
    eta_line = eta[xpix_0:,ypix_0,t]
    F_eta = interpolate.interp1d(x_interp,eta_line)
    d_eta = deriv1(F_eta, x_new, dx, smooth)
    
    # plt.plot(x_interp, Vz_line, 'r')
    plt.plot(x_new, d_eta, 'g')

def divergence(rectangle, x, y): 
    return np.asarray(np.gradient(rectangle))[1,:,:] + np.asarray(np.gradient(rectangle))[1,:,:]

def courbure(rectangle, pas_x, pas_y) :
    return 0.5 * (np.asarray(np.gradient(np.asarray(np.gradient(rectangle, pas_x))[0,:,:], pas_x))[0,:,:] + np.asarray(np.gradient(np.asarray(np.gradient(rectangle, pas_y))[1,:,:], pas_y))[1,:,:])

def rotationnel(rectangle, pas_x, pas_y) :
    dQdx = np.asarray(np.gradient(rectangle))[1,:,:]
    dPdy = np.asarray(np.gradient(rectangle))[0,:,:]
    return dQdx - dPdy



disp.figurejolie(width = np.max(y) / np.max(x) * 8.6, height = np.max(x) / np.max(x) * 8.6)
plt.pcolormesh(y, x, np.flip(np.mean(divergence(Vz[:,:,:], mparpixel, mparpixel), axis = 2),1), cmap = 'ma_cm')
# plt.colorbar()
plt.xlabel(r'$X$ (m)')
plt.ylabel(r'$Y$ (m)' )



mean_div = np.mean(divergence(Vz[:,:,:], mparpixel, mparpixel), axis = 2)

bin_div = np.abs(mean_div) < (np.mean(np.abs(mean_div)) / 4)
from scipy import ndimage

size = 4

for i in range (2) :
    bin_div = ndimage.binary_erosion(bin_div, structure = np.ones((size,size)))
    bin_div = ndimage.binary_dilation(bin_div, structure = np.ones((size,size)))

bin_div = ndimage.binary_dilation(bin_div, structure = np.ones((size,size)))

# disp.figurejolie()
# plt.pcolormesh(y, x, bin_div)

bin_div = np.asarray(bin_div[:,1:], dtype = float) - np.asarray(bin_div[:,:-1], dtype = float)

disp.figurejolie()
plt.pcolormesh(bin_div)


pos_front = np.where(bin_div == -1)

x_front = pos_front[0]
y_front = pos_front[1]

spatio_vz_front = np.zeros((len(y_front), len(T)))

disp.figurejolie()
for i in range(len(y_front)) :
    disp.joliplot('t', 'U', T, eta[x_front[i],y_front[i],:], exp = False, cm = int(i / len(y_front)* 9))
    spatio_vz_front[i,:] = eta[x_front[i],y_front[i],:]



disp.figurejolie()
plt.pcolormesh(spatio_vz_front)

# plt.plot(np.argmax(rotationnel(spatio_vz_front, mparpixel, mparpixel), axis = 1), x_front, 'r')

plt.plot(np.argmax(spatio_vz_front, axis = 1), x_front, 'b')

# t_front = np.argmax(rotationnel(spatio_vz_front, mparpixel, mparpixel), axis = 1)

t_front = np.argmax(spatio_vz_front, axis = 1)

uuuu = x_front
x_front = y_front
y_front = uuuu

import baptiste.main.Show_image_georectifiée as si


t = 1800
si.plot_img(X, Y, t)

plt.plot(x_front * mparpixel + np.min(X), y_front* mparpixel + np.min(Y), 'r', linewidth=4)
plt.plot(x_front * mparpixel + np.min(X) - 4, y_front* mparpixel + np.min(Y), color = disp.mcolors(8), linewidth=4)

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

save = False

temps = False
espace = True

#Mesure de Ac et kappa_c
Ac_avant = np.zeros(len(y_front))
Ac_apres = np.zeros(len(y_front))
Ac_avant_x = np.zeros(len(y_front))
Ac_apres_x = np.zeros(len(y_front))

t_avant_frac = 150
t_apres_frac = 150

f_vague = 0.25 #Hz

d = 5 #m
theta = 90 * np.pi / 180
aa = 4 #m

# f_cut = [0.1,3]

# [b,a] = signal.butter(3, f_cut, fs = facq, btype = "bandpass")
# Vz_filt = signal.filtfilt(b, a, Vz, axis = 2)

for i in range(len(y_front)) :
    t_demi_T = int(1/ f_vague / 2 * facq)
    lambda_sur2 = 1.5614 * (2*t_demi_T/facq)**2 / 2 #en m
    lambda_sur2 = int(lambda_sur2 / mparpixel)

    colors = disp.mcolors( int(( i / len(y_front) * 9)) )
    
    if temps :
        tmax_avant = np.argmax(eta[x_front[i], y_front[i], t_front[i] - t_demi_T:t_front[i] + t_demi_T]) + t_front[i] - t_demi_T
        tmin_avant = np.argmin(eta[x_front[i], y_front[i], t_front[i] - t_demi_T:t_front[i] + t_demi_T]) + t_front[i] - t_demi_T
        tmax_apres = np.argmax(eta[x_front[i], y_front[i], t_front[i] + t_demi_T:t_front[i] + 3 * t_demi_T]) + t_front[i] + t_demi_T
        tmin_apres = np.argmin(eta[x_front[i], y_front[i], t_front[i] + t_demi_T:t_front[i] + 3 * t_demi_T]) + t_front[i] + t_demi_T
        
        Ac_avant[i] = (eta[x_front[i], y_front[i],tmax_avant] - eta[x_front[i], y_front[i],tmin_avant])/2
        Ac_apres[i] = (eta[x_front[i], y_front[i],tmax_apres] - eta[x_front[i], y_front[i],tmin_apres])/2
        plt.figure(55)
        plt.plot(eta[x_front[i], y_front[i], t_front[i] - t_demi_T:t_front[i] + t_demi_T] - np.mean(eta[x_front[i], y_front[i], t_front[i] - t_demi_T:t_front[i] + t_demi_T]), color = colors)
        plt.plot(np.argmax(eta[x_front[i], y_front[i], t_front[i] - t_demi_T:t_front[i] + t_demi_T]), np.max(eta[x_front[i], y_front[i], t_front[i] - t_demi_T:t_front[i] + t_demi_T])- np.mean(eta[x_front[i], y_front[i], t_front[i] - t_demi_T:t_front[i] + t_demi_T]), 'kx')
        plt.figure(56)
        plt.plot(eta[x_front[i], y_front[i], t_front[i] + t_demi_T:t_front[i] + 3 * t_demi_T] - np.mean(eta[x_front[i], y_front[i], t_front[i] + t_demi_T:t_front[i] + 3 * t_demi_T]), color = colors)
        plt.plot(np.argmax(eta[x_front[i], y_front[i], t_front[i] + t_demi_T:t_front[i] + 3 * t_demi_T]), np.max(eta[x_front[i], y_front[i], t_front[i] + t_demi_T:t_front[i] + 3 * t_demi_T]) - np.mean(eta[x_front[i], y_front[i], t_front[i] + t_demi_T:t_front[i] + 3 * t_demi_T]), 'kx')
    
    if espace :
        #On regarde min et max à +- lambda/2 autour de la fracure
        #Puis min max à fracture et fracture + lambda, à tfrac + T/2
        
        eta_avant = eta[x_front[i]:x_front[i] + 2 * lambda_sur2, y_front[i], t_front[i]]
        eta_apres = eta[x_front[i] - 2 * lambda_sur2:x_front[i], y_front[i], t_front[i]]
        
        xmax_avant = np.argmax(eta_avant) + x_front[i]
        xmin_avant = np.argmin(eta_avant) + x_front[i]
        xmax_apres = np.argmax(eta_apres) + x_front[i] - 2*lambda_sur2
        xmin_apres = np.argmin(eta_apres) + x_front[i] - 2*lambda_sur2
        
        
        
        Ac_avant_x[i] = (eta[xmax_avant, y_front[i],t_front[i]] - eta[xmin_avant, y_front[i],t_front[i]])/2
        Ac_apres_x[i] = (eta[xmax_apres, y_front[i],t_front[i]] - eta[xmin_apres, y_front[i],t_front[i]])/2
        plt.figure(57)
        plt.plot(eta_avant, color = colors)
        plt.figure(58)
        plt.plot(eta_apres, color = colors)
        
    
    
    

    


if temps :
    y_front_m = y_front * mparpixel + np.min(Y)
    if save : 
        plt.figure(55)
        plt.title('eta_t_avant_frac')
        sv.save_graph(save_path, 'SAVGOL10_eta_t_avant_frac_fct_y_', pdf = False)
        plt.figure(56)
        plt.title('eta_t_apres_frac')
        sv.save_graph(save_path, 'SAVGOL10_eta_t_apres_frac_fct_y_', pdf = False)
    disp.figurejolie(width = 12)
    disp.joliplot('y (m)', r'$A_c$(m)', y_front_m, Ac_avant, exp = False, color = 2, legend = "T avant fracture") #rouge, entre T/2 et 3 T/2 avant fracture, la période avant fracture
    disp.joliplot('y (m)', r'$A_c$(m)', y_front_m, Ac_apres, exp = False, color = 12, legend = "T à la fracture") #violet, autour de la fracture +- T/2
    
    if save : 
        sv.save_graph(save_path, 'SAVGOL10_Ac_y_pr_tfrac_xfrac', pdf = False)
    
    
    A_c = (Ac_avant + Ac_apres) / 2
    
    err_Ac = np.abs(Ac_avant - Ac_apres) / 2
    
    
    meanAC = np.zeros(len(y_front))
    for i in range(len(y_front)) :
        meanAC[i] = np.mean(A_c[i:])
    
    disp.figurejolie(width = 12)
    
    disp.joliplot(r'y cut', r'mean $A_c$', y_front_m, meanAC, exp = False, color = 2)    
        
    disp.figurejolie(width = 12)
    disp.joliplot(r'y (m)', r'$A_c$ (m)', y_front_m, A_c, exp = True, color = 4)
    plt.errorbar(y_front_m, A_c, err_Ac, None, fmt = 'none')

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
    disp.joliplot('y (m)', r'$A_c$(m)', y_front * mparpixel, Ac_avant_x, exp = False, color = 2, legend = "x à la fracture") #rouge, entre T/2 et 3 T/2 avant fracture, la période avant fracture
    disp.joliplot('y (m)', r'$A_c$(m)', y_front * mparpixel, Ac_apres_x, exp = False, color = 12, legend = "x apres la fracture") #violet, autour de la fracture +- T/2
    
    if save : 
        sv.save_graph(save_path, 'SAVGOL10_Ac_y_pr_tfrac_xfrac', pdf = False)  
        
    A_c_x = (Ac_avant_x + Ac_apres_x) / 2
    
    err_Ac_x = np.abs(Ac_avant_x - Ac_apres_x) / 2
    
    
    meanAC_x = np.zeros(len(y_front))
    for i in range(len(y_front)) :
        meanAC_x[i] = np.mean(A_c_x[i:])
    
    disp.figurejolie(width = 12) 
    
    disp.joliplot(r'y cut', r'mean $A_cx$', y_front * mparpixel, meanAC_x, exp = False, color = 2)
        
    disp.figurejolie(width = 12)
    disp.joliplot(r'y (m)', r'$A_cx$ (m)', y_front * mparpixel, A_c_x, exp = True, color = 4)
    plt.errorbar(y_m_front, A_c_x, err_Ac_x, None, fmt = 'none')
    
    if save : 
        sv.save_graph(save_path, 'SAVGOL10_Ac_x_y_pr_tfrac_xfrac_avec_err', pdf = False)

#%% Mesure AC v2

from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter

disp.figurejolie()
f_vague = 0.3
range_t = 1000


Ac = []
kappa = []

for i in range(len(x_front)):
    if i > 0 and i < 42 :
        jj = 15
    else :
        jj = 0 # x_front[i] - 5
    

    data_eta = eta[x_front[i] - 5, y_front[i], t_front[i] - range_t:t_front[i] + range_t] - np.mean(eta[x_front[i] - 5, y_front[i], t_front[i] - range_t:t_front[i] + range_t])
    plt.plot(data_eta)
    peaks_p, _ = find_peaks(data_eta, distance = int( 1 / f_vague * facq))
    peaks_m, _ = find_peaks( -data_eta, distance = int( 1 / f_vague * facq))
    plt.scatter(peaks_p, data_eta[peaks_p], marker='^')
    plt.scatter(peaks_m, data_eta[peaks_m], marker='v')
    
    max_len = max(len(data_eta[peaks_p]), len(data_eta[peaks_m]))

# Compléter avec des NaN
    eta_max_pad = np.pad(data_eta[peaks_p], (0, max_len - len(data_eta[peaks_p])), constant_values=np.nan)
    eta_min_pad = np.pad(data_eta[peaks_m], (0, max_len - len(data_eta[peaks_m])), constant_values=np.nan)
    
    Ac.append(np.nanmax(eta_max_pad - eta_min_pad )/2)
    
    


    # Appliquer un filtre de moyenne 3x3 sur les 2 premières dimensions
    # size = (3, 3, 1) => 3x3 sur H,W et rien sur C
    A_filtre = uniform_filter(eta[np.max((x_front[i] - 5 - jj, 0)):x_front[i] - jj, np.max((y_front[i]-2,0)):y_front[i]+2, t_front[i] - range_t:t_front[i] + range_t], size=(5, 5, 1))

    # [int(A_filtre.shape[0]/2),int(A_filtre.shape[1]/2),:]

    if i > 41 and i < 92 :
        kappa.append(np.quantile(courbure(A_filtre, mparpixel, mparpixel), 0.95))
        # kappa.append(np.max(courbure(A_filtre, mparpixel, mparpixel)[int(A_filtre.shape[0]/2):int(A_filtre.shape[0]/2)+2,int(A_filtre.shape[1]/2): int(A_filtre.shape[1]/2)+2,:]))
    else :
        kappa.append(np.max(courbure(A_filtre, mparpixel, mparpixel)))
    
    # kappa.append(np.max(courbure(A_filtre, mparpixel, mparpixel)))
    
    # kappa.append(np.max(courbure(A_filtre, mparpixel, mparpixel)[int(A_filtre.shape[0]/2):int(A_filtre.shape[0]/2)+2,int(A_filtre.shape[1]/2): int(A_filtre.shape[1]/2)+2,:]))
    
yf_min = 32
yf_max = 45
    
disp.figurejolie(width = 8.6 * 4 / 5)
disp.joliplot(r'$Y$ (m)', r'$A_{max}$ (m)', y_front * mparpixel, Ac, exp = True, cm = 3)

disp.figurejolie(width = 8.6 * 4 / 5)
disp.joliplot(r'$Y$ (m)', r'$\kappa$ (m$^{-1}$)', y_front * mparpixel, kappa, exp = True, cm = 3, zeros = True)

disp.figurejolie(width = 8.6 * 4 / 5)
disp.joliplot(r'$Y$ (m)', r'$\kappa$ (m$^{-1}$)', Ac, kappa, exp = True, cm = 3, zeros = True)
uu = np.linspace(0,np.max(Ac),100)
disp.joliplot(r'$A_{max}$ (m)', r'$\kappa$ (m$^{-1}$)', uu, uu * 0.28**2, exp = False, cm = 8, zeros = True)

#%% Mesure kappa (x > xf, y > yf)



eta_filt_space = uniform_filter(eta, size=(5, 5, 1))

# array_cut = np.zeros_like(eta)
from scipy.ndimage import uniform_filter
kappa_tout = courbure(eta_filt_space, mparpixel, mparpixel)

# for i in range (len(x_front)):

#     array_cut[(nx - x_front[i]):, y_front[i], :] = kappa_tout[(nx - x_front[i]):, y_front[i], :]
    
mask = np.zeros((nx, ny), dtype=bool)

for j in range(len(x_front)):
    if j < 48 :
        jjj = 5
    elif j > 71:
        jjj = 0
        if j > 80:
            jjj = 2
    else :
        jjj = 3
    mask[j,:x_front[j] - jjj] = True  # pour chaque colonne y, garder x > xf[y]

# Étendre le masque sur la dimension temporelle (t)
mask = mask[:, :, np.newaxis]  # devient (nx, ny, 1)
mask = np.broadcast_to(mask, kappa_tout.shape)  # shape (nx, ny, nt)

# Appliquer le masque
arr_filtered = np.where(mask, kappa_tout, 0)
arr_filtered_eta = np.where(mask, eta, 0)


arr_filtered[arr_filtered == 0] = np.nan

# kappa_th_max_lin = 0.5 * (2 * np.pi / 25)**2
# kappa_tout[kappa_tout < kappa_th_max_lin] = np.nan

uuu = np.nanmax(np.abs(arr_filtered_eta[:,:,:1700]), axis=(1, 2))
uu = np.nanmin(np.abs(arr_filtered_eta[:,:,:1700]), axis=(1, 2))

#%% Plot kappa
max_par_ligne = np.nanmax(np.abs(arr_filtered[:,:,:1700]), axis=(1, 2))

tmax_par_ligne = np.argmax(np.abs(arr_filtered[:,:,:1700]), axis=2)
tmax_par_ligne = np.max(tmax_par_ligne, axis=1)

map_t = to.normalize(tmax_par_ligne)


mask = np.isclose(arr_filtered, max_par_ligne[:, np.newaxis, np.newaxis])

# 3️⃣ Réduire le masque sur la dimension temps (True s’il y a au moins un t qui correspond)
mask2d = np.any(mask, axis=2)  # shape (nx, ny)

# 4️⃣ Créer le tableau final (NaN partout, 1 là où max)
result = np.full((nx, ny), np.nan)
result[mask2d] = 1.0

ttt = 1100

# for i in range (10):
#     ttt = 1400 + 30* i
#     # disp.figurejolie()
#     si.plot_img(X, Y, ttt)
    
#     # plt.plot(x_front * mparpixel + np.min(X), y_front* mparpixel + np.min(Y), 'r', linewidth=4)
#     plt.pcolormesh(X,Y, np.abs(arr_filtered[:,:,ttt]), cmap = 'ma_cm', alpha = 0.8)
#     plt.colorbar()
# # plt.plot(x_front * mparpixel, y_front* mparpixel, 'r', linewidth=4)
#     plt.clim(0, np.nanquantile(np.abs(arr_filtered[:,:,ttt]), 1))
    
#     plt.pcolormesh(X,Y, result, cmap = 'RdBu', alpha = 1)




disp.figurejolie(width = 8.6 * 4 / 5)
disp.joliplot(r'$Y$ (m)', r'$\kappa$ (m$^{-1}$)', y_front * mparpixel, max_par_ligne, exp = True, cm = 3, zeros = False)

disp.figurejolie()
for i in range(len(max_par_ligne)) :
    plt.plot(x[i], max_par_ligne[i], 'd', color = disp.mcolors(int(map_t[i] * 9)))
    
disp.figurejolie(width = 8.6 * 4 / 5)
lalal = np.concatenate( ([tmax_par_ligne], [tmax_par_ligne]))
plt.pcolormesh( (lalal - np.min(lalal)) / facq, cmap = 'ma_cm')
plt.colorbar()

disp.figurejolie(width = 8.6 * 4 / 5)
disp.joliplot(r'$Y$ (m)', r'$t$(s)', y_front * mparpixel, (tmax_par_ligne - np.min(tmax_par_ligne)) / facq, exp = False, cm = 3, zeros = False)

# disp.figurejolie()
# plt.plot(y, np.nanmax(np.abs(arr_filtered[:,:,:1700]), axis=(0, 2)))

disp.figurejolie(width = 8.6 * 4 / 5)
disp.joliplot(r'$A$ (m)', r'$\kappa$ (m$^{-1}$)', Ac, max_par_ligne, exp = True, cm = 3, zeros = False)


Amax_par_ligne = np.nanmax(np.abs(arr_filtered_eta[:,:,:1200]), axis=(1, 2))
Amin_par_ligne = np.nanmin(np.abs(arr_filtered_eta[:,:,:1200]), axis=(1, 2))

disp.figurejolie(width = 8.6 * 4 / 5)
disp.joliplot(r'$Y$ (m)', r'$A_{max}$ (m)', y_front * mparpixel, (Amax_par_ligne - Amin_par_ligne) / 2 , exp = False, cm = 3, zeros = False)


# disp.figurejolie(width = 8.6 * 4 / 5)
# disp.joliplot( r'$A_{max}$ (m)',r'$\kappa$ (m$^{-1}$)', (Amax_par_ligne - Amin_par_ligne) / 2 ,max_par_ligne,  exp = True, cm = 3, zeros = False)
# uuu = np.linspace(0, np.max((Amax_par_ligne - Amin_par_ligne) / 2),100 )
# disp.joliplot(r'$A_{max}$ (m)',r'$\kappa$ (m$^{-1}$)',  uuu, uuu * 0.21**2, exp = False, cm = 6, zeros = True)


#%% Seuil SP : A max (x,y), dt kappa, kappa, seuil dtkappa

kappa_tout = courbure(eta, mparpixel, mparpixel)

A_max_tot = np.zeros((nx, ny))

f_vague = 0.23
# disp.figurejolie()
ututu = 0
for i in range (nx):
    for j in range (ny):
        
        eta_t = eta[i,j,:]

        peaks_p, _ = find_peaks(eta_t, distance = int( 1 / f_vague * facq*0.9))
        peaks_m, _ = find_peaks( -eta_t, distance = int( 1 / f_vague * facq*0.9))
        # plt.scatter(peaks_p, eta_t[peaks_p], marker='^')
        # plt.scatter(peaks_m, eta_t[peaks_m], marker='v')
        
        max_len = max(len(eta_t[peaks_p]), len(eta_t[peaks_m]))
        
        if len(eta_t[peaks_p]) != len(eta_t[peaks_m]) :
            ututu +=1     
    # Compléter avec des NaN
        eta_max_pad = np.pad(eta_t[peaks_p], (0, max_len - len(eta_t[peaks_p])), constant_values=np.nan)
        eta_min_pad = np.pad(eta_t[peaks_m], (0, max_len - len(eta_t[peaks_m])), constant_values=np.nan)
        # A_max_tot[i,j] = np.nanmean(np.sort(eta_max_pad - eta_min_pad)[-10:])/2
        A_max_tot[i,j] = np.nanmax(eta_max_pad - eta_min_pad )/2
        
disp.figurejolie()
plt.pcolormesh(A_max_tot)
plt.colorbar()

seuil_kappa = A_max_tot * (f_vague * 2 * np.pi)**3 / 9.81

dkappa_dt = np.gradient(kappa_tout, 1/ facq, axis=2)

#%% Un evenement

f_vague = 0.23
seuil_kappa = A_max_tot * (f_vague * 2 * np.pi)**3 / 9.81

def rotationnel(Vx, Vy, dx, dy = 0):
    if dy == 0 :
        dy = dx 
    dUy_dx = np.gradient(Vy, dy, axis=1)
    dUx_dy = np.gradient(Vx, dx, axis=0)
    
    
    return dUy_dx - dUx_dy


kappa_tout = courbure(eta, mparpixel, mparpixel)
dkappa_dt = np.gradient(kappa_tout, 1/ facq, axis=2)
rot_tot = rotationnel(Vx, Vy, mparpixel)


max_temp = np.max(rot_tot, axis = 2)
tmax_temp = np.argmax(rot_tot, axis = 2)

xmax = np.where(max_temp == np.max(max_temp))[0][0]
ymax = np.where(max_temp == np.max(max_temp))[1][0]

tmaxmax = tmax_temp[xmax, ymax]

plt.figure()

plt.plot(dkappa_dt[xmax, ymax, tmaxmax-150:tmaxmax+150])
plt.figure()
plt.plot(rot_tot[xmax, ymax, tmaxmax-150:tmaxmax+150])

#%% Amax apres front

arr_filtered_Amax = np.where(mask2d, A_max_tot, 0)
arr_filtered_Amax[arr_filtered_Amax ==0] = np.nan

Amax_par_ligne = np.nanmax(arr_filtered_Amax, axis= 1)

si.plot_img(X, Y, 1500)
plt.pcolormesh(X,Y,arr_filtered_Amax, cmap = 'ma_cm', alpha = 0.8)
plt.colorbar()


disp.figurejolie(width = 8.6 * 4 / 5)
disp.joliplot('$Y$ (m)', '$A_{max}$ (m)', x,  Amax_par_ligne, cm = 5, exp = False)

#%% omega (Y)


nx, ny, nt = arr_filtered_eta.shape


# --- 1) FFT selon le temps (axe t = 2) ---
A_fft = np.fft.fft(arr_filtered_eta - np.mean(arr_filtered_eta), axis=2)

# --- 2) Axe des fréquences ---
freqs = np.fft.fftfreq(nt, d=1/facq)

# --- 3) On garde seulement les fréquences positives ---
pos = freqs >= 0
freqs_pos = freqs[pos]

# --- 4) Amplitude du spectre sur fréquences positives ---
A_amp = np.abs(A_fft[:, :, pos])  # shape = (nx, ny, nt/2)

# --- 5) Moyenne spatiale selon x pour avoir un spectre par y ---
A_mean_y = A_amp.mean(axis=1)     # shape = (ny, nt/2)

# --- 6) Pour chaque y, on trouve la fréquence max ---
indices_max = np.argmax(A_mean_y, axis=0)
fmax_y = freqs_pos[indices_max]   # shape = (ny,)

plt.figure()
plt.plot(fmax_y)
plt.xlabel("y (index)")
plt.ylabel("Fréquence dominante (Hz)")
plt.grid(True)
plt.show()

#%% Detection fracture avec depassement dt kappa

# disp.figurejolie()
# plt.pcolormesh(dkappa_dt[:,:,1500])
# plt.colorbar()

mask_dtkappa = dkappa_dt > seuil_kappa[:, :, np.newaxis]
first_cross_t = np.argmax(mask_dtkappa, axis=2) #Donne bien le premier, pas la liste des maxs

first_cross_t = np.array(first_cross_t, dtype = float)

first_cross_t[first_cross_t == 0] = np.nan


si.plot_img(X, Y, 1500)
plt.pcolormesh(X,Y, first_cross_t, cmap = 'ma_cm', alpha = 0.8)
plt.colorbar()




#%% Test HF vs BF

disp.figurejolie()
f_cut = [1]

yyyy = 50

[b,a] = signal.butter(3, f_cut, fs = facq, btype = "lowpass")
eta_filt = signal.filtfilt(b, a, eta, axis = 2)

disp.joliplot('x', 'eta', x, eta[:,yyyy,t_front[yyyy]], exp = False, cm = 2)
disp.joliplot('x', 'eta', x, eta_filt[:,yyyy,t_front[yyyy]], exp = False, cm = 6)


disp.figurejolie()

for i in range (40) :
    disp.joliplot('x', 'eta', x[x_front[yyyy] - 10:x_front[yyyy] + 10] , 
                  eta[x_front[yyyy] - 10:x_front[yyyy] + 10,yyyy,t_front[yyyy] - 20 + i] -eta_filt[x_front[yyyy] - 10:x_front[yyyy] + 10,yyyy,t_front[yyyy] - 20 + i] - np.mean(eta[x_front[yyyy] - 10:x_front[yyyy] + 10,yyyy,t_front[yyyy] - 20 + i] -eta_filt[x_front[yyyy] - 10:x_front[yyyy] + 10,yyyy,t_front[yyyy] - 20 + i]) , 
                  exp = False, cm = int(i / 40 * 9))

plt.plot(x[x_front[yyyy]], 0, 'kx' )



max_eta = np.where(eta[x_front[yyyy] - 10:x_front[yyyy] + 10,yyyy,t_front[yyyy] - 20: t_front[yyyy] + 20] -eta_filt[x_front[yyyy] - 10:x_front[yyyy] + 10,yyyy,t_front[yyyy] - 20: t_front[yyyy] + 20] - np.mean(eta[x_front[yyyy] - 10:x_front[yyyy] + 10,yyyy,t_front[yyyy] - 20: t_front[yyyy] + 20] -eta_filt[x_front[yyyy] - 10:x_front[yyyy] + 10,yyyy,t_front[yyyy] - 20: t_front[yyyy] + 20]) 
          == np.max(eta[x_front[yyyy] - 10:x_front[yyyy] + 10,yyyy,t_front[yyyy] - 20: t_front[yyyy] + 20] -eta_filt[x_front[yyyy] - 10:x_front[yyyy] + 10,yyyy,t_front[yyyy] - 20: t_front[yyyy] + 20] 
                    - np.mean(eta[x_front[yyyy] - 10:x_front[yyyy] + 10,yyyy,t_front[yyyy] - 20: t_front[yyyy] + 20] -eta_filt[x_front[yyyy] - 10:x_front[yyyy] + 10,yyyy,t_front[yyyy] - 20: t_front[yyyy] + 20])) )


plt.plot(x[x_front[yyyy] - 10 + max_eta[0]], 0, 'ro')

plt.plot(x[x_front[yyyy] - 10:x_front[yyyy] + 10], eta[x_front[yyyy] - 10:x_front[yyyy] + 10,yyyy,t_front[yyyy] + 7])


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
    
    

#%% Plot shear
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


