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

import baptiste.display.display_lib as disp
import baptiste.experiments.import_params as ip
import baptiste.signal_processing.fft_tools as ft
import baptiste.image_processing.image_processing as imp
import baptiste.math.fits as fits
import baptiste.math.RDD as rdd
import baptiste.files.save as sv


import icewave.tools.matlab2python as m2p
import icewave.baptiste.Fct_drone_1102 as fd






#%% IMPORT DATA

path = "K:\Share_hublot\\Data\\0211\\Drones\\bernache\\matData\\18-stereo_001\\"
filename = "PIV_processed_i011500_N15500_Dt4_b1_W32_xROI1_width3839_yROI1_height2159_scaled.mat"

f = h5py.File(path + filename,'r') 

matdata = open(path + filename, 'r')

data = m2p.mat_to_dict(f, f )

Vz = data['m']['Vz'] #t,y,x en s, m, m

Vz = np.transpose(Vz, (2,1,0)) #Pour passer de s(t,y,x) à s(x,y,t)
Vz = np.flip(Vz, 1)

X = data['m']['X'] #m

Y = data['m']['Y'] #m ATTONTION y,x

T = data['m']["t"] #s ATTONTION y,x

facq = 29.97

#%% AFFICHAGE

plt.figure()
for i in range (1468,1800,1) :
    plt.pcolormesh(Vz[:,:,i])
    plt.pause(0.02)


    
#%% Profile line sur l'image en pixels

xm = 29
ym = 39
d = 7 #pixel
theta = 3*np.pi/4 #radians

t0 = 839
dt = 100

x_pix = np.linspace(0,Vz.shape[0]-1, Vz.shape[0])
y_pix = np.linspace(0,Vz.shape[1]-1, Vz.shape[1])

x0 = int(xm - d * np.sin(theta))
xf = int(xm + d * np.sin(theta))
y0 = int(ym - d * np.cos(theta))
yf = int(ym + d * np.cos(theta))

plt.figure()
# plt.pcolormesh(np.flip(Vz[t0,:,:], 0))
disp.joliplot('','',x_pix,y_pix, table = Vz[ :,:, t0])
plt.plot(xm, ym, 'kx')
plt.plot([x0,xf], [y0,yf], 'r-')
plt.axis('equal')


plt.figure()
for t in  range (t0- dt,t0 + dt) :
    line = profile_line(Vz[:,:,t],[x0,y0], [xf,yf])
    colors = disp.vcolors( int(( t- (t0 - dt) )  / dt / 2 * 9)) 
    if t == t0 :
        plt.plot(line - np.mean(line),'r-', lw = 8)
    else :   
        plt.plot(line - np.mean(line),color=colors)
#%% Vz en zeta
facq = 29.97
dT = 1 / facq


zeta = np.cumsum(Vz, axis = 2) * dT

       
#%% Plot line in real space


Vz = data['m']['Vz'] #t,y,x en s, m, m

Vz = np.transpose(Vz, (2,1,0)) #Pour passer de s(t,y,x) à s(x,y,t)
Vz = np.flip(Vz, 1)
# Vz = zeta

display = True
# xpix_0 = 165
# ypix_0 = 70
# d = 10 #m
# theta = 41.22 * np.pi / 180 #radians

# t0 = 1648
# dt = 30


xpix_0 = 140
ypix_0 = 60
d = 40 #m
theta = np.pi/ 2#41.22 * np.pi / 180 #radians

t0 = 1400
dt = 300



t_plot = t0 + dt

lines = np.zeros( (1000,2*dt) )

x_pix, y_pix, dist_p0 = fd.px_to_real_line(data, xpix_0, ypix_0, d, theta)

if display :
    t_plot = t0 + dt
    xp = np.linspace(0,Vz.shape[0]-1, Vz.shape[0])
    yp = np.linspace(0,Vz.shape[1]-1, Vz.shape[1])
    plt.figure()
    disp.joliplot('x (pixel)','y (pixel)',xp,yp, table = Vz[ :,:, t_plot])
    plt.clim(np.quantile(Vz[ :,:, t_plot], 0.1), np.quantile(Vz[ :,:, t_plot], 0.9))
    plt.plot(xpix_0, ypix_0, 'kx')
    plt.plot(np.array(x_pix/16,dtype = int),  np.array(y_pix/16,dtype = int), 'r-')
    
    plt.axis('equal')


if display :
    disp.figurejolie()
    
for t in  range (t0- dt,t0 + dt) :
    colors = disp.vcolors( int(( t- (t0 - dt) )  / dt / 2 * 9)) 
    
    Vz_line = fd.extract_line_Vz(data, Vz, x_pix, y_pix, t)
    lines[:,t- (t0 - dt) ] = Vz_line
    if display :
        if t == t0 :
            disp.joliplot('x (m)', r'Vz (m.s$^{-1}$)' , dist_p0, Vz_line ,color = 2, exp = False, linewidth= 5)
            #r'$\zeta$ (m)'
        else :   
            plt.plot(dist_p0, Vz_line, color=colors)
            
            
#%% Courbure (t) sur zeta

facq = 29.97
a = 20
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

# disp.figurejolie()
# disp.joliplot(r'x (m)', r'$\zeta$ (m)', hmax, kmax, color = 8)

tt = np.linspace(0, (lines.shape[1] -1) * dT, lines.shape[1])
disp.figurejolie()
disp.joliplot(r't (s)', r'$\kappa$ (m$^{-1}$)', tt, kmax, color = 8)

#%% Courbure propre sur Vz
tt = np.linspace(0, (2 * dt-1) / facq, 2 * dt)
from scipy import interpolate
from scipy.misc import derivative
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
    kappa[:,t - t0 - dt] = dF[t - t0 - dt]**2 / ( 1 + Fz(x_new)**2 )**(1.5)
    
    

#%%Plot courbure

plt.figure()

for i in range (2 * dt) :
    # plt.plot(dist_p0, F[i](dist_p0))
    colors = disp.vcolors( int(( i / dt / 2 * 9)) )
    plt.plot(x_new, kappa[:,i], color = colors)
    disp.joliplot('x (m)', r'$\kappa$ (m$^{-1}$)' , [], [] ,color = 2, exp = False, linewidth= 5)

#spatio temporel de la courbure
disp.figurejolie()
disp.joliplot('x (m)','t (s)', x_new,T[t0 - dt:t0 + dt], table = kappa)
plt.clim(np.quantile(kappa, 0.5), np.quantile(kappa, 0.97))
