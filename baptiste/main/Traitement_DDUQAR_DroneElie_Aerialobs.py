# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 16:02:21 2022

@author: Banquise
"""

#%% INITIALISATION (ATTENTION EFFACE LES PARAMETRES)

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt2d
from scipy.signal import find_peaks
from skimage.measure import profile_line
import scipy.fft as fft
import scipy.io as io

import baptiste.display.display_lib as disp
import baptiste.experiments.import_params as ip
import baptiste.signal_processing.fft_tools as ft
import baptiste.image_processing.image_processing as imp
import baptiste.math.fits as fits
import baptiste.math.RDD as rdd
import baptiste.files.save as sv


date = '230516'
nom_exp = 'UQARG'

dico, params, loc = ip.initialisation(date)

loc_h = 'D:\Banquise\Baptiste\Resultats_video\d221104\d221104_PIVA6_PIV_44sur026_facq151Hz_texp5000us_Tmot010_Vmot410_Hw12cm_tacq020s/'
loc_h = 'W:\Banquise\\Data_DD_UQAR\\'
# loc_h = 'W:\\Banquise\\Baptiste\\baie_HAHA_drone\\'
# loc_h = 'D:\\Banquise\\Baptiste\\Drone\\Traitements\\'

params['special_folder'] = 'Traitement_baieHaha_drone'

path = loc_h 



"""
                                                    HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
                                                                    PARTIE 1
                                                         Traitement de signal 3D, s(t,y,x)
                                                    HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
"""


#%% DONNEES PIVLAB

params['PIV_file'] = 'PIVlab_6m30_7m10_3couches_64_32_16_traitement230412'            #GSL BQ
# params['PIV_file'] = 'PIVlab_64_32_16_pasfragments_HQ_230515'                           #GSL HQ banquise continue
# params['PIV_file'] = 'PIVlab_64_32_16_GSL_full_HQ_230516'                             #GSL HQ full
# params['PIV_file'] = 'PIVlab_BB_HQ_1700_3200_8'                                       #BB HQ
# data_brut = io.loadmat(loc_h + 'PIVlab_3000.mat')

# params['PIV_file_u'] = 'matlab_u_og_baiehahadrone'
# params['PIV_file_v'] = 'matlab_v_og_baiehahadrone'

# params['PIV_file'] = 'PIVlab_7_10_23_32_16_100frames_file11'

data_brut = io.loadmat(loc_h + params['PIV_file']) 

# data_brut_u = io.loadmat(loc_h + params['PIV_file_u']) 
# data_brut_v = io.loadmat(loc_h + params['PIV_file_v']) 


#%%PARAMETRES

display = True
save = False
  
params['facq'] =29.97 #29.97 #151
params['ratio_PIV'] = 16
params['mparpixel'] = 0.03 * 3 * params['ratio_PIV'] # *3 si BQ # +- 0.0005 pour GSL (dapres Elie), #0.07988 1ere mesure BQ #0.0300 2 eme mesure HQ #0.031 Elie
# params['mparpixel'] = 0.03538 * params['ratio_PIV'] * 3 #3.538 +/- 0.0507 cm/px pour BB
kacqx = 2 * np.pi / params['mparpixel']
kacqy = 2 * np.pi / params['mparpixel']



#%%MISE EN FORME

champ_u = data_brut["u_original"]
champ_v = data_brut["v_original"]

# champ_u = data_brut_u["u_original"]
# champ_v = data_brut_v["v_original"]

data_u = []
data_v = []


for i in range (np.shape(champ_u)[0] -10):
    data_u.append(champ_u[i][0])#- np.nanmean(champ_u[i][0]))
data_u = np.asarray(data_u)

for i in range (np.shape(champ_v)[0]-10):
    data_v.append(champ_v[i][0])#- np.nanmean(champ_v[i][0]))
data_v = np.asarray(data_v)


# data = np.sqrt(np.power(data_v,2) + np.power(data_u,2))
data = data_u



data = np.transpose(data, (2,1,0)) #Pour passer de s(t,y,x) à s(x,y,t)
data = np.flip(data, 1)

[nx,ny,nt] = data.shape


params['x0'] = 0        #en pixels
params['xf'] = nx
params['y0'] = 0
params['yf'] = ny
params['t0'] = 0 * params['facq']        #en secondes
params['tf'] = nt

data = data[int(params['x0']):int(params['xf']),int(params['y0']):int(params['yf']),int(params['t0']):int(params['tf'])]

[nx,ny,nt] = data.shape

x = np.linspace(0, nx * params['mparpixel'] , nx)  #Taille X en pixel
y = np.linspace(0, ny * params['mparpixel'] , ny)  #Taille Y en pixel
t = np.linspace(0, nt /params['facq'], nt)       


#On enleve la moyenne temporelle pour chaque pixel
# data = imp.substract_mean(data, space = True, temp = False)

#%%
# champ_u = data_brut_u["u_original"]
# champ_v = data_brut_v["v_original"]

data_u = np.zeros( (np.shape(champ_u)[0],champ_u[0][0].shape[0],champ_u[0][0].shape[1])  )
data_v = 0


for i in range (np.shape(champ_u)[0] - 5):
    data_u[i] = champ_u[i][0]- np.nanmean(champ_u[i][0])
data_u = np.asarray(data_u)

data = data_u



data = np.transpose(data, (2,1,0)) #Pour passer de s(t,y,x) à s(x,y,t)
data = np.flip(data, 1)

[nx,ny,nt] = data.shape


params['x0'] = 0        #en pixels
params['xf'] = nx - 20
params['y0'] = 0
params['yf'] = ny
params['t0'] = 0        #en secondes
params['tf'] = nt - 10

data = data[int(params['x0']):int(params['xf']),int(params['y0']):int(params['yf']),int(params['t0']):int(params['tf'])]

[nx,ny,nt] = data.shape

x = np.linspace(0, nx * params['mparpixel'] , nx)  #Taille X en pixel
y = np.linspace(0, ny * params['mparpixel'] , ny)  #Taille Y en pixel
t = np.linspace(0, nt /params['facq'], nt)       


#On enleve la moyenne temporelle pour chaque pixel
data = imp.substract_mean(data, space = True, temp = False)

#%% Affichage
save = False       
              
data[np.isnan(data)] = 0 #met les nan à 0


# Petit film du champ de vitesse

display_video = False
if display_video :
    disp.figurejolie()
    plt.pcolormesh(y, x, data[:,:,0], shading='auto')
    plt.xlabel("Y (m)")
    plt.ylabel("X (m)")
    cbar = plt.colorbar()
    cbar.set_label("Champ u")
    plt.axis("equal")
    for mmm in range (1,20):
        # disp.figurejolie()
        plt.pcolormesh(y, x, data[:,:,mmm], shading='auto')
        plt.pause(0.1)
        # cbar = plt.colorbar()
        # plt.clim(-2,2)

# Affichage du champ de vitesse de la première image

if display :
    params = disp.figurejolie(params, nom_fig = 'champ_vitesse_img_1') 
    params[str(params['num_fig'][-1])]['data'] = disp.joliplot("X (m)","Y (m)",x,y,table = data[:,:,0], tcbar = "Champ u")
    plt.clim(0,np.quantile(data[:,:,0],0.90))
    
    if save :
        sv.save_graph (path, 'test_data0', params = params, num_fig = False, nc = False, pkl = True)


#%% MOUVEMENT D'UN PIXEL
save = False

params['x_pixel'] = 20
params['y_pixel'] = 33


disp.figurejolie(num_fig = params['num_fig'][-1])
params[str(params['num_fig'][-1])]['data'] = disp.joliplot(r'$x_0$',r'$y_0$',[params['x_pixel']* params['mparpixel']] ,[params['y_pixel']* params['mparpixel']], color = 6)

if save :
    sv.save_graph (path, 'test_pixel', params = params, num_fig = False, nc = False, pkl = True)


params = disp.figurejolie(params, nom_fig = 'pos_pixel(t)')

data_pixel = data[params['x_pixel'],params['y_pixel'],:]
params[str(params['num_fig'][-1])]['data'] = disp.joliplot('t','x',t, data_pixel, exp = False)
plt.title('pos pixel en fct du tps')

if save :
    sv.save_graph (path, 'mvt_pixel', params = params, num_fig = False, nc = False, pkl = True)


params = disp.figurejolie(params)
Y1, f = ft.fft_bapt(data[params['x_pixel'],params['y_pixel'],:], params['facq'])
ft.plot_fft(Y1, f, log = False)

plt.title('fft de ce mvt')


if save :
    params[str(params['num_fig'][-1])]['data'] = sv.data_to_dict(['f','a'], [f,[]], data = Y1)
    sv.save_graph (path, 'fft_mvt_pixel', params = params, num_fig = False, nc = False, pkl = True)



#%%DEMODULATION


params["f_exc"] =0.4
#[0.02931119 0.24914509 0.48363459] 3 fréquences fondamentales pour videos complete

demod = ft.demodulation(t,data,params["f_exc"])

if display :
    params = disp.figurejolie(params, nom_fig = 'champ_demodule_' + str(params["f_exc"]) + "Hz") 
    params[str(params['num_fig'][-1])] = disp.joliplot("X (m)","Y (m)",x,y,table = (np.real(demod)), tcbar = 'Champ u démodulé à ' + str(params["f_exc"]) + "Hz", div = True)  
    plt.clim(0, np.quantile(data[:,:,0],0.50))
    if save :
        sv.save_graph (path, 'FFT_2D', params = params, num_fig = False, nc = False, pkl = True)

#%%FFT 2D 

padpad = False

if padpad :
    padding = [9,9,7]
    data_pad = ft.add_padding(data, padding)
    YY, kkx, kky, ff = ft.fft_bapt(data_pad, kacqx, kacqy, params['facq'], og_shape = [nx,ny,nt])

else :
    YY, kkx, kky, ff = ft.fft_bapt(data, kacqx,  kacqy, params['facq'])
    

    
params = disp.figurejolie(params) 
ft.plot_fft(YY, f1 = kkx, f2 = kky, f3 = ff, log = False)


params[str(params['num_fig'][-1])] = sv.data_to_dict(['kx','ky', 'f'], [kkx,kky,ff], data = YY)
# sv.save_graph (path, 'FFT_2D', params = params, num_fig = False, nc = False, pkl = True)

#%% omega(k) avec fft MARCHE PAS


Yk = np.zeros((nx*ny ,nt), dtype = 'complex')
YYis = fft.ifftshift(YY)
for i in range (nx) :
    for j in range (ny) :
        Yk[i*j, :] = YYis[i,j,:]
        
k = np.linspace(0, np.sqrt(max(kkx * 2)**2 + max(kky * 2)**2) , nx*ny)

ft.plot_fft(np.rot90(Yk), f1 = ff, f2 = k, log = False, xlabel = r'f(Hz)', ylabel = r'k$(m^{-1})$')


#%%FFT temporelle moyenne

YY_temp = np.mean(np.abs(YY), axis = 0)
YY_temp = np.mean(np.abs(YY_temp), axis = 0)
# YY_temp = fft.fftshift(YY_temp)
disp.figurejolie()
disp.joliplot('f (Hz)', 'Amp (arbitraire)', ff, YY_temp, exp = False, color = 9)
#0.8 pour 1190 pts, 0.05 pour 2**11
peaks, rrr = find_peaks(YY_temp,threshold = 1 )
disp.joliplot('f (Hz)', 'Amp (arbitraire)', peaks / (len(ff)-1) * params['facq']  - params['facq'] / 2, YY_temp[peaks], color = 4, exp = True, legend = 'Pics')

f_fonda  = peaks / (len(ff)-1) * params['facq']
print(f_fonda[:3])

#%%RDD
# 2) Demoduler pour angle
save_demod = False

fmin = 0.1 #2/nt * params['facq'] #fréquence minimale résolue pour avoir au moins 2 périodes par fréquence
fmax = 1
nb_f = 20
padding = [9,9]    #puissance de 2 pour le 0 padding
k_xx = []
k_yy = []
kk = []
theta = []
fff = []

# mparpixel = 0.07988064791133845 * ratio_PIV #mesuré avec la taille du bateau...


plotplot = True
nb_plot = 5

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

huhu = 0
for i in np.linspace(fmin, fmax, nb_f) :
    huhu += 1
    if np.mod(huhu-1,20)==0:
        print('iteration ' + str(huhu) + ' sur ' + str(nb_f))
    demod = ft.demodulation(t,data,i)
    demod_padding = ft.add_padding(demod, padding)
    Y_FFT, k_x, k_y = ft.fft_bapt(demod_padding, kacqx, kacqy)
    nx_FFT = np.shape(Y_FFT)[0]
    ny_FFT = np.shape(Y_FFT)[1]
    max_fft, maxx = ft.max_fft(Y_FFT, f1 = k_x,f2 = k_y, display = False)
    k_xx.append(max_fft[0])
    k_yy.append(max_fft[1])
    kk.append(cart2pol(max_fft[0],max_fft[1])[0])
    theta.append(max_fft[1]/max_fft[0])
    fff.append(i)
    


    if save_demod :
        sv.save_mat(demod, loc_h, title = "champ_demod_f_" + str(i))
    
    if plotplot :
        if int(huhu/nb_f * nb_plot ) == (huhu/nb_f * nb_plot):
            disp.figurejolie()
            ft.plot_fft(Y_FFT, k_x, k_y, tcbar = r'Démodulation à f = ' + str(round(i, 3)) + " Hz")
            plt.plot(max_fft[0], max_fft[1], 'ro')
 
k_xx = np.asarray(k_xx)
k_yy = np.asarray(k_yy)
kk = np.asarray(kk)
theta = np.asarray(theta)
fff = np.asarray(fff)


#%% Affichage RDD
save = False

params = disp.figurejolie(params, nom_fig = 'theta_de_f') 
params[str(params['num_fig'][-1])]['data'] = disp.joliplot(r"f (Hz)",r"$\theta$", fff, theta, color = 1)
 
if save :
    sv.save_graph (path, params[str(params['num_fig'][-1])]['nom_fig'], params = params)


params = disp.figurejolie(params, nom_fig = 'k_de_f') 
params[str(params['num_fig'][-1])]['data'] = disp.joliplot(r"f (Hz)",r"K (m$^{-1}$)", fff, kk, color = 2)
 
if save :
    sv.save_graph (path, params[str(params['num_fig'][-1])]['nom_fig'], params = params)


#%%Region d'interet

#en Hz
# zone1 = np.array([fmin,1.4])
zone1 = np.array([0.51,0.75])
zone2 = np.array([0.2,0.442])
zone3 = np.array([0.442,0.6775])

# zone2 = np.array([fmin,0.4])
# zone3 = np.array([0.4,0.6])
zone2 = np.array([fmin,0.6])

""" Banquise totale HQ"""
# zone1 = np.array([fmin,0.18]) 
# zone2 = np.array([0.21,0.43])
# zone3 = np.array([0.44,0.68])
zone4 = np.array([0.758,0.767])

#conversion en numérique
zone1 = np.array((zone1 - fmin)/(fmax - fmin) * nb_f, dtype = 'int64')
zone2 = np.array((zone2 - fmin)/(fmax - fmin) * nb_f, dtype = 'int64')
zone3 = np.array((zone3 - fmin)/(fmax - fmin) * nb_f, dtype = 'int64')
zone4 = np.array((zone4 - fmin)/(fmax - fmin) * nb_f, dtype = 'int64')

#ZONE 1-2-3
ff_new = np.concatenate((fff[zone1[0]:zone1[1]] * 2,fff[zone2[0]:zone2[1]],fff[zone3[0]:zone3[1]]*2/3))
kk_new = np.concatenate((kk[zone1[0]:zone1[1]],kk[zone2[0]:zone2[1]],kk[zone3[0]:zone3[1]]))


#ZONE 2-3-4
ff_new = np.concatenate((fff[zone2[0]:zone2[1]], fff[zone3[0]:zone3[1]] * 2/3, fff[zone4[0]:zone4[1]]*2/3))
kk_new = np.concatenate((kk[zone2[0]:zone2[1]], kk[zone3[0]:zone3[1]], kk[zone4[0]:zone4[1]]))

#ZONE 3-4
ff_new = np.concatenate((fff[zone3[0]:zone3[1]] * 2/3, fff[zone4[0]:zone4[1]]*2/3))
kk_new = np.concatenate((kk[zone3[0]:zone3[1]], kk[zone4[0]:zone4[1]]))

#ZONE 2-3
ff_new = np.concatenate((fff[zone2[0]:zone2[1]], fff[zone3[0]:zone3[1]]*2/3))
kk_new = np.concatenate((kk[zone2[0]:zone2[1]], kk[zone3[0]:zone3[1]]))

disp.figurejolie()
plt.plot(ff_new,kk_new,'ko')
plt.plot(fff[zone1[0]:zone1[1]] * 2,kk[zone1[0]:zone1[1]],'rx')
plt.plot(fff[zone2[0]:zone2[1]],kk[zone2[0]:zone2[1]],'bx')
plt.plot(fff[zone3[0]:zone3[1]]*2/3,kk[zone3[0]:zone3[1]],'gx')
plt.plot(fff[zone4[0]:zone4[1]]*2/3,kk[zone4[0]:zone4[1]],'yx')

#%%Fit RDD :

logomega = np.log(2 * np.pi * fff)
logk = np.log(kk)

popt = fits.fit_powerlaw( kk, 2 * np.pi * fff, display = True,  xlabel = r'log(k)', ylabel = 'log($\omega$)', legend = 'Experimental data')

popt1 = fits.fit_powerlaw( kk[zone1[0]:zone1[1]], 2 * np.pi * fff[zone1[0]:zone1[1]], display = True, xlabel = r'log(k)', ylabel = 'log($\omega$)', legend = 'zone 1')
popt2 = fits.fit_powerlaw( kk[zone2[0]:zone2[1]], 2 * np.pi * fff[zone2[0]:zone2[1]], display = True, xlabel = r'log(k)', ylabel = 'log($\omega$)', legend = 'zone 2')
popt3 = fits.fit_powerlaw( kk[zone3[0]:zone3[1]], 2 * np.pi * fff[zone3[0]:zone3[1]], display = True, xlabel = r'log(k)', ylabel = 'log($\omega$)', legend = 'zone 3')
popt4 = fits.fit_powerlaw( kk[zone4[0]:zone4[1]], 2 * np.pi * fff[zone4[0]:zone4[1]], display = True, xlabel = r'log(k)', ylabel = 'log($\omega$)', legend = 'zone 4')


logff_new = np.log(2 * np.pi * ff_new)
logkk_new = np.log(kk_new)

disp.figurejolie()
popt_new = fits.fit_powerlaw( kk_new, ff_new * 2 * np.pi, display = True, xlabel = r'log(k)', ylabel = 'log($\omega$)', legend = '2-3-4')


# disp.figurejolie()
# import seaborn as sns
# sns.regplot(logk[zone1[0]:zone1[1]],logomega[zone1[0]:zone1[1]], color ='blue')
# sns.regplot(logk[zone2[0]:zone2[1]],logomega[zone2[0]:zone2[1]], color ='red')
# sns.regplot(logk[zone3[0]:zone3[1]],logomega[zone3[0]:zone3[1]], color ='green')

#%% RANSAC pour le fit :
    
from skimage import io

no_use_data = np.zeros(len(logk), dtype = 'bool')


no_use_data[zone1[0]:zone1[1]] = True
no_use_data[zone2[0]:zone2[1]] = True
no_use_data[zone3[0]:zone3[1]] = True
no_use_data[zone4[0]:zone4[1]] = True

   
# model_robust1, inliers1, outliers1 = fits.fit_ransac(logk[zone1[0]:zone1[1]], logomega[zone1[0]:zone1[1]], thresh = 0.4, display = True)

model_robust2, inliers2, outliers2 = fits.fit_ransac(logk[zone2[0]:zone2[1]], logomega[zone2[0]:zone2[1]], thresh = 0.001, display = False, newfig = True)

# model_robust3, inliers3, outliers3 = fits.fit_ransac(logk[zone3[0]:zone3[1]], logomega[zone3[0]:zone3[1]], thresh = 0.1, display = False, newfig = True)

model_robust_new, inliers_new, outliers_new = fits.fit_ransac(logkk_new, logff_new, thresh = 0.1, display = False, newfig = True)


no_use_data = (no_use_data == False)
disp.figurejolie()
disp.joliplot(r'log(k)', r'log($\omega$)', logk[no_use_data], logomega[no_use_data], color= 11, exp = True, legend = 'Data without fit')

plt.legend(loc='lower right')
plt.show()  

disp.figurejolie()
disp.joliplot(r'log(k)', r'log($\omega$)', logk[zone2[0]:zone2[1]], logomega[zone2[0]:zone2[1]], color= 11, exp = True, legend = 'Data without fit')



# g1 = np.mean(np.exp(2 * logomega[zone1[0]:zone1[1]][inliers1] - logk[zone1[0]:zone1[1]][inliers1]))
g2 = np.mean(np.exp(2 * logomega[zone2[0]:zone2[1]][inliers2] - logk[zone2[0]:zone2[1]][inliers2]))
# g3 = np.mean(np.exp(2 * logomega[zone3[0]:zone3[1]][inliers3] - logk[zone3[0]:zone3[1]][inliers3]))
g_new = np.mean(np.exp(2 * logff_new[inliers_new] - logkk_new[inliers_new]))
# print('g1 = ' + str(round(g1,3)))
print('g2 = ' + str(round(g2,3)))
# print('g3 = ' + str(round(g3,3)))
print('g_new = ', g_new)


  

#%%Fit avec RDD pesante
zone = 1
zone = 2
# zone = 3
# zone = 23
zone = '2-3-4'
save = False

# if zone == 2 :
#     data_pes2 = np.stack((kk[zone2[0]:zone2[1]],fff[zone2[0]:zone2[1]] * 2 * np.pi), axis = -1)
# if zone == 3 :
#     data_pes3 = np.stack((kk[zone3[0]:zone3[1]],fff[zone3[0]:zone3[1]] * 2 * np.pi * 2 / 3), axis = -1)

    

params = disp.figurejolie(params, nom_fig = 'fit_pesante_full') 

#     fits.fit(rdd.RDD_pesante, np.append(kk[zone2[0]:zone2[1]][inliers2],kk[zone3[0]:zone3[1]][inliers3]) , 
#              np.append(fff[zone2[0]:zone2[1]][inliers2] * 2 * np.pi,fff[zone3[0]:zone3[1]][inliers3] * 2 * np.pi * 2 / 3) , 
#              err = True, p0 = [0.2], bounds=([0], [1]), zero = True, th_params = 0.01)

zone = 2
if zone == 2 :
    
    fits.fit(rdd.RDD_pesante, kk[zone2[0]:zone2[1]], fff[zone2[0]:zone2[1]] * 2 * np.pi, err = True, p0 = [0.2], bounds=([0], [1]),
              zero = True, th_params = 0.01)
    if save :
        params[str(params['num_fig'][-1])]['data'] = sv.data_to_dict(['k','omega'], [kk[zone2[0]:zone2[1]],fff[zone2[0]:zone2[1]]], data = [kk[zone2[0]:zone2[1]],fff[zone2[0]:zone2[1]]])
        if save :
            sv.save_graph(path, params[str(params['num_fig'][-1])]['nom_fig'] , params = params)
    

# if zone == 3 :
    
#     fits.fit(rdd.RDD_pesante, kk[zone3[0]:zone3[1]][inliers3], fff[zone3[0]:zone3[1]][inliers3] * 2 * np.pi * 2 / 3, err = True, p0 = [0.2], bounds=([0], [1]),
#              zero = True, th_params = 0.01)
    

# joliplot(r'k (m$^{-1}$)', r'$\omega$', kk_range2, RDD_pesante(kk_range2, 0.9 * 0.2), exp = False, legend = r'RDD pesante théorique ($\delta\rho$ = 0.9 et h = 0.2m)', color = 9,zeros = True)

params = disp.figurejolie(params, nom_fig = 'fit_pesante_2_384') 
zone = '2-3-4'
if zone == '2-3-4' :
    fits.fit(rdd.RDD_pesante, kk_new, ff_new * 2 * np.pi, err = True, p0 = [0.2], bounds=([0], [1]),
             zero = True, th_params = 0.01)
    if save :
        params[str(params['num_fig'][-1])]['data'] = sv.data_to_dict(['k','omega'], [kk_new,ff_new], data = [kk_new,ff_new])
        if save :
            sv.save_graph(path, params[str(params['num_fig'][-1])]['nom_fig'] , params = params)
    
    



#%%Fit avec RDD pesante et flexion


# data_pes1 = np.stack((kk[15:59],(fff[15:59] * 2 * np.pi)**2), axis = -1)
data_pes2 = np.stack((kk[zone2[0]:zone2[1]],fff[zone2[0]:zone2[1]] * 2 * np.pi), axis = -1)
# data_pes3 = np.stack((kk[161:275],(fff[161:275] * 2 * np.pi)**2), axis = -1)

disp.figurejolie()
popt2, pcov2 = fits.fit(rdd.RDD_pesante_flexion, kk[zone2[0]:zone2[1]], fff[zone2[0]:zone2[1]] * 2 * np.pi, err = False, p0 = [0.3,1e4],
                      bounds=([0.2,1e2], [1, 1e8]), zero = True, nb_param = 2, th_params = [0.4, 8e2])
plt.plot(np.linspace(0,np.max(kk)))


# popt3, pcov3 = fits.fit(rdd.RDD_pesante_flexion, kk[zone3[0]:zone3[1]][inliers3], fff[zone3[0]:zone3[1]][inliers3] * 2 * np.pi * 2 / 3, err = False, p0 = [0.3,1e4], bounds=([0,1e4], [0.5, 1e5]),
#           zero = True, th_fct = rdd.RDD_gravitaire, nb_param = 2)



#%% SAVE
save = False
if save :
    parametres = []
    disp.figurejolie()
    # joliplot('f(Hz)', 'angle (radian)',fff,theta, exp = False)
    disp.joliplot('f(Hz)', 'angle (degrés)',fff,theta * 180 / np.pi, exp = False, color = 5)
    plt.savefig()
    disp.figurejolie()
    disp.joliplot('f(Hz)', r'K (m$^{-1}$)', fff, kk, color = 4)
    plt.savefig()
    parametres.extend(["facq = " + str(facq), "ratio_PIV = " + str(ratio_PIV), "mparpixel = " + str(mparpixel)])
    parametres.extend(["facq = " + str(facq), "ratio_PIV = " + str(ratio_PIV), "mparpixel = " + str(mparpixel)])
    parametres.extend(["facq = " + str(facq), "ratio_PIV = " + str(ratio_PIV), "mparpixel = " + str(mparpixel)])



#%% FFT

padpad = True   #Est ce qu'on applique du 0 padding
padding = 10     #puissance de 2 pour le 0 padding

zone_dinteret_x = np.asarray([0, nx])
zone_dinteret_y = np.asarray([0,ny])

X_interet = np.linspace(zone_dinteret_x[0],zone_dinteret_x[1],zone_dinteret_x[1]-zone_dinteret_x[0] ) 
Y_interet = np.linspace(zone_dinteret_y[0],zone_dinteret_y[1],zone_dinteret_y[1]-zone_dinteret_y[0] ) 

if display :
    figurejolie()
    plt.pcolormesh(Y_interet, X_interet, (np.real(demod[zone_dinteret_x[0]:zone_dinteret_x[1],zone_dinteret_y[0]:zone_dinteret_y[1]])), shading = 'auto')
    plt.axis('equal')
    cbar = plt.colorbar()
    plt.xlabel("Y (cm)")
    cbar.set_label("Champ u démodulé à " + str(f_exc) + "Hz COUPE")
    plt.ylabel("X (cm)")

if padpad :
    demod_padding = np.zeros((2**padding,2**padding), dtype = 'complex128')
    for i in range (nx):
        for j in range(ny):
            demod_padding[i,j] = demod_padding[i,j] + demod[i,j]
    Y_FFT = fft.fft2(demod_padding)

else :
    Y_FFT = fft.fft2(demod[zone_dinteret_x[0]:zone_dinteret_x[1],zone_dinteret_y[0]:zone_dinteret_y[1]])
    

nx_FFT = np.shape(Y_FFT)[0]
ny_FFT = np.shape(Y_FFT)[1]
Y_FFT = fft.fftshift(Y_FFT)
if padpad :    
    title_FFT = "FFT 2D spatiale 0 padding (2**" + str(padding) + "), démodulé à " + str(f_exc) + "Hz"
else :
    title_FFT = "FFT 2D spatiale, démodulé à " + str(f_exc) + "Hz"
    
k_x = np.linspace(-.5, .5, nx_FFT)
k_y = np.linspace(-.5, .5, ny_FFT)
    
if display :
    figurejolie()
    plt.pcolormesh(k_y , k_x, np.abs(Y_FFT), shading = 'auto')
    plt.xlabel(r"$k_y (m^{-1})$")
    plt.ylabel(r"$k_x (m^{-1})$")
    cbar = plt.colorbar()
    cbar.set_label(title_FFT)

#max de la FFT en cooedonnées kx ky
k_max_fft_x = (np.where(np.max(np.abs(Y_FFT)) == np.abs(Y_FFT))[0][0] - nx_FFT/2) / nx_FFT
k_max_fft_y = (np.where(np.max(np.abs(Y_FFT)) == np.abs(Y_FFT))[1][0] - ny_FFT/2) / ny_FFT
k_max_fft = np.sqrt(k_max_fft_x**2 + k_max_fft_y**2)

if True : #f_exc > facq/2 :
    lambda_demod = nx_FFT/(np.transpose(np.where(np.max(np.abs(Y_FFT)) == np.abs(Y_FFT)))[0][0])
else :
    lambda_demod = nx_FFT / (nx_FFT-np.transpose(np.where(np.max(np.abs(Y_FFT)) == np.abs(Y_FFT)))[0][0])

err_lambda = nx / nx_FFT
           
        
if display :
    plt.plot(k_max_fft_y,k_max_fft_x, 'ro', label = 'MAX FFT')
    # plt.plot(0,0, 'ro', label = 'MAX FFT')
    plt.legend()

print ("lambda = ", lambda_demod)
print ("k = ", k_max_fft)

k = 160
lambda_demod = 2 * np.pi / k


#%%SAUVEGARDE LAMBDA
save = True
if save :
    list_lambda.append([lambda_demod, err_lambda, f_exc])
    

#%%Save
parametres.extend(["PARAMETRES DE TRAITEMENT :","padpad = " + str(padpad), "padding = " + str(padding)])
np.savetxt(path_mat[:-13] + "resultats" + "/Paramètres_" + nom_exp + "_" + date + ".txt", parametres, "%s")
np.savetxt(path_mat[:-13] + "resultats" + "/lambda_err_fexc_PIV_" + nom_exp + "_" + date + ".txt", list_lambda, "%s")



#%%Profil moyen pour attenuation

#max de la FFT en indices
x_max_FFT = np.transpose(np.where(np.max(np.abs(Y_FFT)) == np.abs(Y_FFT)))[0][0]
y_max_FFT = np.transpose(np.where(np.max(np.abs(Y_FFT)) == np.abs(Y_FFT)))[0][1]


y_new_x = np.zeros(np.shape(Y_FFT), dtype = 'complex128') #FFT dans le sens de propagation
y_new_moinsx = np.zeros(np.shape(Y_FFT), dtype = 'complex128') #FFT dans le sens reflechis

#On choisi le quart de FFT à sélectionner en fonction de son max. ax/bx/ay/by vont déterminer les coordonnées du carré

if x_max_FFT < nx_FFT/2 :
    bx = int(nx_FFT/2)
    ax = 0
else :
    bx = nx_FFT
    ax = int(nx_FFT/2) + 1 
    
if y_max_FFT < ny_FFT/2 :
    by = int(ny_FFT/2)
    ay = 0 
else :
    by = ny_FFT
    ay = int(ny_FFT/2) + 1
    
ay = 0
by = ny_FFT

y_new_x[ax:bx,ay:by] = Y_FFT[ax:bx,ay:by]

if x_max_FFT < nx_FFT/2 :
    bx = nx_FFT
    ax = int(nx_FFT/2) + 1 
else :
    bx = int(nx_FFT/2)
    ax = 0
    
y_new_moinsx[ax:bx,ay:by] = Y_FFT[ax:bx,ay:by]

if display :
    figurejolie()
    plt.pcolormesh(np.abs(y_new_x))
    figurejolie()
    plt.pcolormesh(np.abs(y_new_moinsx))



demod_stat_x = fft.ifft2(y_new_x)
demod_stat_moinsx = fft.ifft2(y_new_moinsx)

if display :
    figurejolie()
    plt.pcolormesh(y , x ,np.real(demod_stat_x))
    plt.title("FFT inverse sens de propagation")
    
    figurejolie()
    plt.pcolormesh(y , x, np.abs(demod_stat_moinsx))
    plt.title("FFT inverse onde réflechie")

profil_amp_x = []
profil_amp_moinsx = []
x_ATT= np.linspace(25 * mmparpixel_PIV / 10, (nx-25) * mmparpixel_PIV / 10, nx-50)


for j in x_ATT:
    uuu = profile_line((np.abs(demod_stat_x)), (j, 0), (j, 66), mode = 'reflect')
    uuu.sort()
    uuu = uuu[10:-10]
    profil_amp_x.append(np.mean(uuu))

figurejolie()
plt.semilogy(profil_amp_x)
profil_amp_x = np.asarray(profil_amp_x)
profil_amp_x = profil_amp_x**2

for j in x_ATT:
    uuu = profile_line((np.abs(demod_stat_x)), (j, 0), (j, 66), mode = 'reflect')
    uuu.sort()
    uuu = uuu[10:-10]
    profil_amp_moinsx.append(np.mean(uuu))

figurejolie()
plt.semilogy(profil_amp_moinsx)
profil_amp_moinsx = np.asarray(profil_amp_moinsx)
profil_amp_moinsx = profil_amp_x**2



def exppp(x, a, b):
    return a * np.exp(-b * x)

attenuation_x = np.polyfit(x_ATT, np.log(profil_amp_x), 1)
attenuation_moinsx = np.polyfit(x_ATT, np.log(profil_amp_moinsx), 1)

figurejolie()
joliplot(r"x (cm)", r"I", x_ATT, profil_amp_x, color = 3, exp = False, log = True, legend = r"f = " + str(int(f_exc) ) + " Hz")
joliplot(r"x (cm)", r"I", x_ATT, x_ATT * attenuation_x[0] + attenuation_x[1], color = 5, exp = False, log = True, legend = r"fit, on trouve $\kappa = $" + str(round(attenuation_x[1],4)))
plt.xscale('linear')
plt.yscale('log')


figurejolie()
joliplot(r"x (cm)", r"I", x_ATT, profil_amp_moinsx, color = 3, exp = False, log = True, legend = r"f = " + str(int(f_exc) ) + " Hz")
joliplot(r"x (cm)", r"I", x_ATT, x_ATT * attenuation_moinsx[0] + attenuation_moinsx[1], color = 5, exp = False, log = True, legend = r"fit, on trouve $\kappa = $" + str(round(attenuation_x[1],4)))
plt.xscale('linear')
plt.yscale('log')





# plt.figure()
# for i in range (0,66) :
#     p = profile_line(np.log(np.abs(demod_stat)), (55, i), (0, i))
#     plt.plot(p, 'x')
#%% k_extraction

import numpy as np
from radialavg2 import radialavg2

def kextraction(data, fitlength, step_ana):
    """
    Extracts the wavenum oneach point of the wavefield.
    It calls for the function radialavg2 to reconstruct the Bessel function of first order on each ooint of the 2D matrix data.
    :param: 
   
        * data : complex demodulated field;
        * fitelength : resolution of the reconstructed bessel function;
        * step_ana : step of analysis.
   
    :return:
   
        Return k the wavenum field
    
    Example
    -------
    >>> step_ana = 1
    >>> fitlength = 30
    >>> kfield  = kextraction(c, fitlength, step_ana)
    """

    [nx,ny] = data.shape
    cx = 0 
    k2 = np.zeros((int((ny-fitlength)/step_ana), int((nx-fitlength)/step_ana)-1))
    phase_locale = np.ones((2*fitlength,2*fitlength))
    signal_local = np.zeros(phase_locale.shape)
    for x0 in range(fitlength, nx-fitlength+1, step_ana):
        if np.mod(x0,60)==0:
            print(str(np.round(x0*100/(nx-fitlength),0))+ ' % ')
        cy = 0
        for y0 in range(fitlength, ny-fitlength+1, step_ana):
            phase_locale = np.ones((2*fitlength,2*fitlength))*np.exp(1j*np.angle(data[x0,y0]))
            signal_local = np.real(data[x0-fitlength:x0+fitlength, y0-fitlength:y0+fitlength]*phase_locale)
            [r2,zr2] = radialavg2(signal_local, 1, fitlength+1, fitlength+1)
            xx = r2[0:fitlength]
            xx2 = np.concatenate((np.flipud(-xx),xx))
            test = np.abs(zr2[0:fitlength])
            test2 = np.concatenate((np.flipud(test),test))
            pp = np.polyfit(xx2,test2,deg=2)
            pp[0]=np.abs(pp[0])
            pp[2]=np.abs(pp[2])
            k2[cy,cx]=np.sqrt(4*pp[0]/pp[2])
            cy+=1
        cx += 1
    return k2




fit_length = 5
champ_k = kextraction(demod_stat_x, fit_length, 1)
#%%
k_tot = np.transpose(champ_k[:-fit_length,:-fit_length])
#AFFICHAGE CHAMP K
k_tot = medfilt2d(k_tot, kernel_size=5)
figurejolie()
plt.pcolormesh(k_tot**(5/3))
plt.title('champ k, démodulé à ' + str(f_exc) + " Hz")
cbar = plt.colorbar()
cbar.set_label('k (m?)')
# plt.clim(vmin = 0, vmax = 20)
plt.axis("equal")

