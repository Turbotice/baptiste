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



dico = dic.open_dico()

#%% Importation params utiles
save = False
save_path = 'E:\\Baptiste\\Resultats_exp\\All_RDD\\Resultats\\'

tableau_1 = pandas.read_csv('E:\\Baptiste\\Resultats_exp\\Tableau_params\\Tableau1_Params_231117_240116\\tableau_1.txt', sep = '\t', header = 0)
tableau_1 = np.asarray(tableau_1)

g = 9.81
A = np.array(tableau_1[:,9], dtype = float)
long_onde = np.array(tableau_1[:,6], dtype = float)
k = 2 * np.pi / long_onde
H = np.array(tableau_1[:,10], dtype = float)
omega = np.sqrt(g * k + np.tanh(k * H))
x = np.zeros(len(omega), dtype = object)
t = np.zeros(len(omega), dtype = object)
for i in range (len(omega)) :
    x[i] = np.linspace(0, long_onde[i], 500)
    t[i] = np.pi / omega[i] * 2 * 3 / 4





#verification du domaine de validité :
    
    
# disp.figurejolie()
# disp.joliplot('long_onde', 'Ordre2 / Ordre 1', long_onde, 3/8*A * k / k**3 / H **3)


#%% Graphes pour ordre 2 Stokes

eta1 = np.zeros(len(omega), dtype = object)
eta2 = np.zeros(len(omega), dtype = object)
eta3 = np.zeros(len(omega), dtype = object)
terme1 = np.zeros(len(omega), dtype = float)
terme2 = np.zeros(len(omega), dtype = float)
std_eta2 = np.zeros(len(omega), dtype = object)
t_plot = np.linspace(0,1,500)

for i in range( len(omega)) :
    terme1[i] = A[i]**2 * k[i] / 4
    terme2[i] = 3 * (1 / np.tanh(k[i] * H[i]))**3 - 1/ np.tanh(k[i] * H[i])
    eta1[i] = A[i] * np.sin(k[i] * x[i] - omega[i] * t[i])
    eta2[i] = terme1[i] * terme2[i] * np.cos(2 * (k[i] * x[i] - omega[i] * t[i])) #- A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i]) #+ A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i])
    eta3[i] = (1 - 3* A[i]**2 * k[i]**2 / 8) * A[i] * np.cos(k[i] * x[i] - omega[i] * t[i]) + A[i] **2 * k[i] * np.cos(2 * (k[i] * x[i] - omega[i] * t[i])) + 3 * A[i]**3 * k[i] ** 2 * np.cos(3 * (k[i] * x[i] - omega[i] * t[i])) / 8
    std_eta2[i] = np.std(eta2[i])
    

disp.figurejolie()
for i in range (len(omega)) :
    disp.joliplot('t (0 à T)', 'x (m)', t_plot, eta1[i] + eta2[i] - np.mean(eta1[i] + eta2[i]), exp = False, color = 2)#, legend = '$\eta_1 + \eta_2$')
    # disp.joliplot('t (0 à T)', 'x (m)', t_plot, eta1[i], exp = False, color = 4)#, legend = '$\eta_1$')

disp.figurejolie()
for i in range (len(omega)) :
    disp.joliplot('t (0 à T)', 'x (m)', t_plot, eta1[i] + eta2[i] + eta3[i] - np.mean(eta1[i] + eta2[i] + eta3[i]), exp = False, color = 2)#, legend = '$\eta_1 + \eta_2 + \eta_3$ ')
    disp.joliplot('t (0 à T)', 'x (m)', t_plot, eta1[i] + eta2[i] - np.mean(eta1[i] + eta2[i]), exp = False, color = 4)#, legend = '$\eta_1 + \eta_2$')
    disp.joliplot('t (0 à T)', 'x (m)', t_plot, eta1[i], exp = False, color = 12)#, legend = '$\eta_1$')


disp.figurejolie()    
for i in range (len(omega)) :
    disp.joliplot('t (0 à T)', '$\eta_1$', t_plot, eta1[i], exp = False, color = 4)

    
disp.figurejolie()
for i in range (len(omega)) :
    disp.joliplot('t (0 à T)', '$\eta_2$', t_plot, eta2[i] - np.mean(eta2[i]), exp = False, color = 5)
    
disp.figurejolie()
for i in range (len(omega)) :
    disp.joliplot('$\lambda$', 'terme2 / A', long_onde, terme2 / A, exp = True, color = 5)
    
disp.figurejolie()
disp.joliplot(r'$\lambda$', r'std($\frac{\eta_2}{A}$)', long_onde, std_eta2 / A)

#%% Graphes pour ordre 2 Stokes stationnaire

eta1 = np.zeros(len(omega), dtype = object)
eta_1 = np.zeros(len(omega), dtype = object)
eta2 = np.zeros(len(omega), dtype = object)
eta_2 = np.zeros(len(omega), dtype = object)
eta3 = np.zeros(len(omega), dtype = object)
eta_3 = np.zeros(len(omega), dtype = object)
terme1 = np.zeros(len(omega), dtype = float)
terme2 = np.zeros(len(omega), dtype = float)
std_eta2 = np.zeros(len(omega), dtype = object)
t_plot = np.linspace(0,1,500)
for i in range( len(omega)) :
    terme1[i] = A[i]**2 * k[i] / 4
    terme2[i] = 3 * (1 / np.tanh(k[i] * H[i]))**3 - 1/ np.tanh(k[i] * H[i])
    eta1[i] = A[i] * np.sin(k[i] * x[i] - omega[i] * t[i])
    eta_1[i] = A[i] * np.sin( -k[i] * x[i] - omega[i] * t[i])
    eta2[i] = terme1[i] * terme2[i] * np.cos(2 * (k[i] * x[i] - omega[i] * t[i])) #- A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i]) #+ A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i])
    eta_2[i] = terme1[i] * terme2[i] * np.cos(2 * (-k[i] * x[i] - omega[i] * t[i]))
    eta3[i] = (1 - 3* A[i]**2 * k[i]**2 / 8) * A[i] * np.cos(k[i] * x[i] - omega[i] * t[i]) + A[i] **2 * k[i] * np.cos(2 * (k[i] * x[i] - omega[i] * t[i])) + 3 * A[i]**3 * k[i] ** 2 * np.cos(3 * (k[i] * x[i] - omega[i] * t[i])) / 8
    eta_3[i] = (1 - 3* A[i]**2 * -k[i]**2 / 8) * A[i] * np.cos(-k[i] * x[i] - omega[i] * t[i]) + A[i] **2 * -k[i] * np.cos(2 * (-k[i] * x[i] - omega[i] * t[i])) + 3 * A[i]**3 * -k[i] ** 2 * np.cos(3 * (-k[i] * x[i] - omega[i] * t[i])) / 8
    
    

disp.figurejolie()
for i in range (len(omega)) :
    disp.joliplot('t (0 à T)', 'x (m)', t_plot, eta1[i] + eta2[i] + eta_1[i] + eta_2[i] - np.mean(eta1[i] + eta2[i]), exp = False, color = 2)#, legend = '$\eta_1 + \eta_2$')
    # disp.joliplot('t (0 à T)', 'x (m)', t_plot, eta1[i], exp = False, color = 4)#, legend = '$\eta_1$')

disp.figurejolie()    
for i in range (len(omega)) :
    disp.joliplot('t (0 à T)', '$\eta_1$', t_plot, eta1[i]+ eta_1[i], exp = False, color = 4)
    
    
disp.figurejolie()
for i in range (len(omega)) :
    disp.joliplot('t (0 à T)', 'x (m)', t_plot, eta1[i] + eta2[i] + eta_1[i] + eta_2[i]+ eta3[i] + eta_3[i], exp = False, color = 2)


    

    



#%% Graphes pour solitons
c = 0.5
A = 0.02
x = np.linspace(0, 1, 100)
H = 0.1


eta = 2 * A / (1 + np.cosh(np.sqrt(3 * A) * (x - c) / H**3))

disp.figurejolie()
disp.joliplot('x (m)', r'$\eta$ (m)', x, eta, exp = False, color = 5)

#%% Comparaison soliton exp


date = '240109'
nom_exp = 'QSC10'
exp = True
exp_type = 'LAS'

save = False
display = True

dico, params, loc = ip.initialisation(date, nom_exp, exp = True, display = False)

folder_results = params['path_images'][:-15] + "resultats"
name_file = "positionLAS.npy"
data_originale = np.load(folder_results + "\\" + name_file)

if True : #cam_dessus :
    data_originale = np.rot90(data_originale)
    data_originale = np.flip(data_originale, 0)


    
# data_originale = data_originale - np.nanmean(data_originale, axis = 0)    


params['debut_las'] = 150
params['fin_las'] = np.shape(data_originale)[0] - 400
#100 / 400 Pour DAP
#650 / 300 Pour CCCS1
#1 / 200 pr CCCS2

params['t0'] = 1
params['tf'] = np.shape(data_originale)[1] - 1


[nx,nt] = data_originale[params['debut_las']:params['fin_las'],params['t0']:params['tf']].shape


data = data_originale[params['debut_las']:params['fin_las'],params['t0']:params['tf']]

#mise à l'échelle en m
data_m = data *  params['mmparpixely'] / 1000

data_m = data_m / params['grossissement']



params['savgol'] = True
params['ordre_savgol'] = 2
params['taille_savgol'] = 150
signalsv = np.zeros(data.shape)
for i in range(0,nt):  
    signalsv[:,i] = savgol_filter(data_m[:,i], params['taille_savgol'],params['ordre_savgol'], mode = 'nearest')
    if np.mod(i,1000)==0:
        print('On processe l image numero: ' + str(i) + ' sur ' + str(nt))
print('Done !')


if params['savgol'] :
    data = signalsv.copy()
else :
    data = data_m.copy()



 

t = np.arange(0,nt)/params['facq']
x = np.arange(0,nx)*params['mmparpixelz'] / 1000

# data = data_m.copy()
# data = data - np.nanmean(data, axis = 0)  #enleve la moyenne spatiale pr chaque tps
data = data - np.nanmean(data)


if display:
    disp.figurejolie()
    # plt.pcolormesh(t, x, data,shading='auto')
    plt.pcolormesh(data,shading='auto')
    plt.xlabel("Temps (frames)")
    plt.ylabel("X (pixel)")
    cbar = plt.colorbar()
    cbar.set_label('Amplitude (m)')
    plt.clim(-0.01,0.01)
    

t0 = 150
tf = 155
liste_t = [150, 162, 175, 186]
liste_popt = np.zeros(len(liste_t), dtype = object)

for i in range (len(liste_t)) :
    x0 = 100
    xf = 1300
    tt = int(liste_t[i])
    x_exp = np.linspace(0, (xf-x0) *params['mmparpixelz'] / 1000, xf-x0 )
    
    forme = data[x0:xf, tt]
    
    
    c = (xf-x0) / 2 * params['mmparpixelz'] / 1000
    A = np.nanmax(forme)
    H = dico[date][nom_exp]['Hw'] / 100
    eta = 2 * A / (1 + np.cosh(np.sqrt(3 * A) * (x_exp - c) / H**3))
    
    def fct(x, A, c, H, me) :
        return 2 * A / (1 + np.cosh(np.sqrt(3 * A) * (x - c) / H**3)) - me
    
    liste_popt[i], pcov = curve_fit(fct, x_exp, forme)
    
    disp.figurejolie()
    disp.joliplot('x (m)', r'$\eta$ (m)', x_exp, forme, exp = False, color = 5, legend = 'experience')
    # disp.joliplot('x (m)', r'$\eta$ (m)', x_exp, eta, exp = False, color = 2, legend = 'test soliton')
    disp.joliplot('x (m)', r'$\eta$ (m)', x_exp, fct(x_exp, liste_popt[i][0], liste_popt[i][1], liste_popt[i][2], liste_popt[i][3]), exp = False, color = 2, legend = 'fit soliton')
#%% Fit exp stokes 3 (marche pas)
  
date = '240109'
nom_exp = 'QSC10'
exp = True
exp_type = 'LAS'

save = False
display = True

dico, params, loc = ip.initialisation(date, nom_exp, exp = True, display = False)

folder_results = params['path_images'][:-15] + "resultats"
name_file = "positionLAS.npy"
data_originale = np.load(folder_results + "\\" + name_file)

if True : #cam_dessus :
    data_originale = np.rot90(data_originale)
    data_originale = np.flip(data_originale, 0)


    
# data_originale = data_originale - np.nanmean(data_originale, axis = 0)    


params['debut_las'] = 150
params['fin_las'] = np.shape(data_originale)[0] - 400
#100 / 400 Pour DAP
#650 / 300 Pour CCCS1
#1 / 200 pr CCCS2

params['t0'] = 1
params['tf'] = np.shape(data_originale)[1] - 1


[nx,nt] = data_originale[params['debut_las']:params['fin_las'],params['t0']:params['tf']].shape


data = data_originale[params['debut_las']:params['fin_las'],params['t0']:params['tf']]

#mise à l'échelle en m
data_m = data *  params['mmparpixely'] / 1000

data_m = data_m / params['grossissement']



params['savgol'] = True
params['ordre_savgol'] = 2
params['taille_savgol'] = 150
signalsv = np.zeros(data.shape)
for i in range(0,nt):  
    signalsv[:,i] = savgol_filter(data_m[:,i], params['taille_savgol'],params['ordre_savgol'], mode = 'nearest')
    if np.mod(i,1000)==0:
        print('On processe l image numero: ' + str(i) + ' sur ' + str(nt))
print('Done !')


if params['savgol'] :
    data = signalsv.copy()
else :
    data = data_m.copy()



 

t = np.arange(0,nt)/params['facq']
x = np.arange(0,nx)*params['mmparpixelz'] / 1000

# data = data_m.copy()
# data = data - np.nanmean(data, axis = 0)  #enleve la moyenne spatiale pr chaque tps
data = data - np.nanmean(data)


if display:
    disp.figurejolie()
    # plt.pcolormesh(t, x, data,shading='auto')
    plt.pcolormesh(data,shading='auto')
    plt.xlabel("Temps (frames)")
    plt.ylabel("X (pixel)")
    cbar = plt.colorbar()
    cbar.set_label('Amplitude (m)')
    plt.clim(-0.01,0.01)
    

t0 = 150
tf = 155
liste_t = [150, 162, 175, 186]
liste_popt = np.zeros(len(liste_t), dtype = object)

lambda_exp = float(dico[date][nom_exp]['lambda'])
k_exp = 2 * np.pi / lambda_exp
omega_exp = 2 * np.pi * float(dico[date][nom_exp]['fexc'])
H_exp = float(dico[date][nom_exp]['Hw'])
facq = float(dico[date][nom_exp]['facq'])

boost_nonlineaire = 1
nb_periodes = 1
x = np.linspace(1 *lambda_exp / 16, 1*lambda_exp / 16 + nb_periodes * lambda_exp * (xf-x0) / (lambda_exp / params['mmparpixel'] * 1000), (xf-x0))

for i in range (len(liste_t)) :
    x0 = 100
    xf = 1300
    tt = int(liste_t[i])
    t  = tt / facq
    x_exp = np.linspace(0, (xf-x0) *params['mmparpixelz'] / 1000, xf-x0 )
    
    forme = data[x0:xf, tt]
    
    Amp_exp_eta2 = np.max(data[x0:xf, liste_t]) * boost_nonlineaire
    
    k_exp_prime = k_exp * (1 - Amp_exp_eta2 ** 2 * k_exp ** 2)
    terme1 = Amp_exp_eta2**2 * k_exp / 4
    terme2 = 3 * (1 / np.tanh(k_exp * H_exp))**3 - 1/ np.tanh(k_exp * H_exp)
    terme_1 = Amp_exp_eta2**2 * -k_exp / 4
    terme_2 = 3 * (1 / np.tanh(-k_exp * H_exp))**3 - 1/ np.tanh(-k_exp * H_exp)
    #termes en x et -x (_)
    eta1 = Amp_exp_eta2 * np.sin(k_exp * x - omega_exp * t)
    eta_1 = Amp_exp_eta2 * np.sin(-k_exp * x - omega_exp * t)
    eta2 = terme1 * terme2 * np.cos(2 * (k_exp * x - omega_exp * t)) #- A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i]) #+ A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i])
    eta_2 = terme_1 * terme_2 * np.cos(2 * (-k_exp * x - omega_exp * t))
    eta3 = (1 - 3* Amp_exp_eta2**2 * k_exp**2 / 8) * Amp_exp_eta2 * np.cos(k_exp_prime * x - omega_exp * t) + Amp_exp_eta2 **2 * k_exp * np.cos(2 * (k_exp_prime * x - omega_exp * t)) + 3 * Amp_exp_eta2**3 * k_exp ** 2 * np.cos(3 * (k_exp_prime * x - omega_exp * t)) / 8
    eta_3 = (1 - 3* Amp_exp_eta2**2 * -k_exp**2 / 8) * Amp_exp_eta2 * np.cos(-k_exp_prime * x - omega_exp * t) + Amp_exp_eta2 **2 * -k_exp * np.cos(2 * (-k_exp_prime * x - omega_exp * t)) + 3 * Amp_exp_eta2**3 * -k_exp ** 2 * np.cos(3 * (-k_exp_prime * x - omega_exp * t)) / 8
    
    t = 6*np.pi / omega_exp / 4 - 0.03
    
    
    def f_eta_2(x, Amp_exp_eta2, H_exp, t) :
        omega_exp = 10.053096491487338
        k_exp = 11.976831538313936
        terme1 = Amp_exp_eta2**2 * k_exp / 4
        terme2 = 3 * (1 / np.tanh(k_exp * H_exp))**3 - 1/ np.tanh(k_exp * H_exp)
        terme_1 = Amp_exp_eta2**2 * -k_exp / 4
        terme_2 = 3 * (1 / np.tanh(-k_exp * H_exp))**3 - 1/ np.tanh(-k_exp * H_exp)
        eta1 = Amp_exp_eta2 * np.sin(k_exp * x - omega_exp * t)
        eta_1 = Amp_exp_eta2 * np.sin(-k_exp * x - omega_exp * t)
        eta2 = terme1 * terme2 * np.cos(2 * (k_exp * x - omega_exp * t)) #- A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i]) #+ A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i])
        eta_2 = terme_1 * terme_2 * np.cos(2 * (-k_exp * x - omega_exp * t))
        
        return eta1 + eta_1 + eta2 + eta_2

    def f_eta_3(x, Amp_exp_eta2, H_exp, t) :
        omega_exp = 10.053096491487338
        k_exp = 11.976831538313936
        terme1 = Amp_exp_eta2**2 * k_exp / 4
        terme2 = 3 * (1 / np.tanh(k_exp * H_exp))**3 - 1/ np.tanh(k_exp * H_exp)
        terme_1 = Amp_exp_eta2**2 * -k_exp / 4
        terme_2 = 3 * (1 / np.tanh(-k_exp * H_exp))**3 - 1/ np.tanh(-k_exp * H_exp)
        eta1 = Amp_exp_eta2 * np.sin(k_exp * x - omega_exp * t)
        eta_1 = Amp_exp_eta2 * np.sin(-k_exp * x - omega_exp * t)
        eta2 = terme1 * terme2 * np.cos(2 * (k_exp * x - omega_exp * t))
        eta_2 = terme_1 * terme_2 * np.cos(2 * (-k_exp * x - omega_exp * t))
        eta3 = (1 - 3* Amp_exp_eta2**2 * k_exp**2 / 8) * Amp_exp_eta2 * np.cos(k_exp_prime * x - omega_exp * t) + Amp_exp_eta2 **2 * k_exp * np.cos(2 * (k_exp_prime * x - omega_exp * t)) + 3 * Amp_exp_eta2**3 * k_exp ** 2 * np.cos(3 * (k_exp_prime * x - omega_exp * t)) / 8
        eta_3 = (1 - 3* Amp_exp_eta2**2 * -k_exp**2 / 8) * Amp_exp_eta2 * np.cos(-k_exp_prime * x - omega_exp * t) + Amp_exp_eta2 **2 * -k_exp * np.cos(2 * (-k_exp_prime * x - omega_exp * t)) + 3 * Amp_exp_eta2**3 * -k_exp ** 2 * np.cos(3 * (-k_exp_prime * x - omega_exp * t)) / 8
       
        return eta1 + eta_1 + eta2 + eta_2 + eta3 + eta_3
    
    liste_popt[i], pcov = curve_fit(f_eta_2, x_exp, forme)
    
    disp.figurejolie()
    disp.joliplot('x (m)', r'$\eta$ (m)', x_exp, forme, exp = False, color = 5, legend = 'experience')
    # disp.joliplot('x (m)', r'$\eta$ (m)', x_exp, eta, exp = False, color = 2, legend = 'test soliton')
    disp.joliplot('x (m)', r'$\eta$ (m)', x_exp, f_eta_2(x_exp, liste_popt[i][0], liste_popt[i][1], liste_popt[i][2]), exp = False, color = 2, legend = 'fit soliton')


#%% test 1 point

H = 0.11
long_onde = 0.21
k = 2 * np.pi / long_onde
A = 0.008
x = long_onde/2
omega = np.sqrt(g * k + np.tanh(k * H))
t = np.linspace(0, 2 * np.pi / omega, 500)
terme1 = A**2 * k / 4
terme2 = 3 * (1 / np.tanh(k * H))**3 - 1/ np.tanh(k * H)
eta1 = A * np.sin(k * x + omega * t)
eta2 = terme1 * terme2 * np.cos(2 * (k * x - omega * t)) - A * k**2 / 2 / np.sinh(2 * k * H) #+ A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i])

disp.figurejolie()

disp.joliplot('t', 'eta1 + eta2', t, eta1 + eta2 - np.mean(eta1 + eta2), exp = False, color = 2)
disp.joliplot('t', 'eta1 + eta2', t, eta2 - np.mean(eta2), exp = False, color = 5)
disp.joliplot('t', 'eta1 + eta2', t, eta1, exp = False, color = 7)
#%% DDP retracé avec la nouvelle amp (mauvaise idée)


# disp.joliplot(r'$\lambda$', 'amp', long_onde, A, zeros = True, color = 2, legend = 'DDP OG')
popt_1, pcov_1 = fits.fit_powerlaw(long_onde, A, display = True, legend = 'DDP OG', xlabel =r"$\lambda$ (m)", ylabel = "Amplitude (m)", new_fig= True, color = 2)

#DDP avec min(eta1 + eta2) à la place de A
A_new = np.zeros(len(omega))
for i in range (len(omega)) :
    A_new[i] = np.abs(np.min(eta1[i] + eta2[i] - np.mean(eta1[i] + eta2[i])))
    
# disp.joliplot(r'$\lambda$', 'amp', long_onde, A_new, zeros = True, color = 5, legend = 'Amp refaite')
popt_2, pcov_2 = fits.fit_powerlaw(long_onde, A_new, display = True, legend = 'Amp refaite', xlabel =r"$\lambda$ (m)", ylabel = "Amplitude (m)", new_fig= False, color = 8)


disp.figurejolie()
disp.joliplot(r'$\lambda$ (m)', 'amp (m)', long_onde, A, zeros = True, color = 2, legend = 'DDP OG')
disp.joliplot(r'$\lambda$ (m)', 'amp (m)', long_onde, A_new, zeros = True, color = 5, legend = 'amp refaite')
plt.xlim((0, np.max(long_onde) + 0.01))
plt.ylim((0,np.max(A_new) + 0.0005))
#%% Courbure

kappa = A* k**2 #* (1 + A * k * terme2)
kappa_corr = A* k**2 * (1 + A * k * terme2)
popt_1, pcov_1 = fits.fit_powerlaw(long_onde, kappa, display = True, legend = r'$\kappa = Ak^{2}$ (m$^{-1}$)', xlabel =r"$\lambda$ (m)", ylabel = "$\kappa$ (m$^{-1}$)", new_fig= True, color = 8)
popt_1, pcov_1 = fits.fit_powerlaw(long_onde, kappa_corr, display = True, legend = r'$\kappa_{corrected}$ (m$^{-1}$)', xlabel =r"$\lambda$ (m)", ylabel = "$\kappa$ (m$^{-1}$)", new_fig= False, color = 2)


#%% Charge donées exp

date = '240109'
nom_exp = 'QSC10'
exp = True
exp_type = 'LAS'

save = False
display = True

dico, params, loc = ip.initialisation(date, nom_exp, exp = True, display = False)

folder_results = params['path_images'][:-15] + "resultats"
name_file = "positionLAS.npy"
data_originale = np.load(folder_results + "\\" + name_file)

if True : #cam_dessus :
    data_originale = np.rot90(data_originale)
    data_originale = np.flip(data_originale, 0)


    
# data_originale = data_originale - np.nanmean(data_originale, axis = 0)    


params['debut_las'] = 150
params['fin_las'] = np.shape(data_originale)[0] - 1
#100 / 400 Pour DAP
#650 / 300 Pour CCCS1
#1 / 200 pr CCCS2

params['t0'] = 111
params['tf'] = np.shape(data_originale)[1] - 1


[nx,nt] = data_originale[params['debut_las']:params['fin_las'],params['t0']:params['tf']].shape


data = data_originale[params['debut_las']:params['fin_las'],params['t0']:params['tf']]

#mise à l'échelle en m
data_m = data *  params['mmparpixely'] / 1000

data_m = data_m / params['grossissement']



params['savgol'] = False
params['ordre_savgol'] = 2
params['taille_savgol'] = 20
signalsv = np.zeros(data.shape)
for i in range(0,nt):  
    signalsv[:,i] = savgol_filter(data_m[:,i], params['taille_savgol'],params['ordre_savgol'], mode = 'nearest')
    if np.mod(i,1000)==0:
        print('On processe l image numero: ' + str(i) + ' sur ' + str(nt))
print('Done !')


if params['savgol'] :
    data = signalsv.copy()
else :
    data = data_m.copy()



 

t = np.arange(0,nt)/params['facq']
x = np.arange(0,nx)*params['mmparpixelz'] / 1000

# data = data_m.copy()
# data = data - np.nanmean(data, axis = 0)  #enleve la moyenne spatiale pr chaque tps
data = data - np.nanmean(data)


if display:
    disp.figurejolie()
    # plt.pcolormesh(t, x, data,shading='auto')
    plt.pcolormesh(data,shading='auto')
    plt.xlabel("Temps (frames)")
    plt.ylabel("X (pixel)")
    cbar = plt.colorbar()
    cbar.set_label('Amplitude (m)')
    plt.clim(-0.01,0.01)
#%% kappa (x)

#DMLO5
t00 = 255
tff = 265
x0 = 200
xf = 960

#EDTH
# t00 = 1
# tff = 200
# x0 = 0
# xf = 550


lambda_exp = float(dico[date][nom_exp]['lambda'])
k_exp = 2 * np.pi / lambda_exp
omega_exp = 2 * np.pi * float(dico[date][nom_exp]['fexc'])
H_exp = float(dico[date][nom_exp]['Hw'])
facq = float(dico[date][nom_exp]['facq'])

boost_nonlineaire = 0.5
nb_periodes = 1
liste_t = np.linspace(t00,tff,tff-t00, dtype = int)
x = np.linspace(1 *lambda_exp / 16, 1*lambda_exp / 16 + nb_periodes * lambda_exp * (xf-x0) / (lambda_exp / params['mmparpixel'] * 1000), (xf-x0))
x_plotexp = np.linspace(0, (xf-x0) * params['mmparpixel'] / 1000, (xf-x0))

err = 1000000

liste_t = [257]


a = 20
sig = 0.002
stddd = np.std(data[x0:xf,:], axis = 0)
meann = np.mean(data[x0:xf, stddd < sig])

hmin = np.zeros(np.shape(liste_t))
hmax = np.zeros(np.shape(liste_t))
popt_min = np.zeros(np.shape(liste_t), dtype = object)
popt_max = np.zeros(np.shape(liste_t), dtype = object)
err_min = np.zeros(np.shape(liste_t))
err_max = np.zeros(np.shape(liste_t))
kmin = np.zeros(np.shape(liste_t))
kmax = np.zeros(np.shape(liste_t))
imin = np.zeros(np.shape(liste_t), dtype = int)
imax = np.zeros(np.shape(liste_t), dtype = int)




for i in range(len(liste_t)) :
    t0 = liste_t[i]
    forme = data[x0:xf, t0] - meann
    n = forme.shape[0]
    t = 1/omega_exp #t0 / facq
    
    '''        THEORIE
    
    Amp_exp_eta2 = np.max(data[x0:xf, liste_t]) * boost_nonlineaire
    
    k_exp_prime = k_exp * (1 - Amp_exp_eta2 ** 2 * k_exp ** 2)
    terme1 = Amp_exp_eta2**2 * k_exp / 4
    terme2 = 3 * (1 / np.tanh(k_exp * H_exp))**3 - 1/ np.tanh(k_exp * H_exp)
    terme_1 = Amp_exp_eta2**2 * -k_exp / 4
    terme_2 = 3 * (1 / np.tanh(-k_exp * H_exp))**3 - 1/ np.tanh(-k_exp * H_exp)
    #termes en x et -x (_)
    eta1 = Amp_exp_eta2 * np.sin(k_exp * x - omega_exp * t)
    eta_1 = Amp_exp_eta2 * np.sin(-k_exp * x - omega_exp * t)
    eta2 = terme1 * terme2 * np.cos(2 * (k_exp * x - omega_exp * t)) #- A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i]) #+ A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i])
    eta_2 = terme_1 * terme_2 * np.cos(2 * (-k_exp * x - omega_exp * t))
    eta3 = (1 - 3* Amp_exp_eta2**2 * k_exp**2 / 8) * Amp_exp_eta2 * np.cos(k_exp_prime * x - omega_exp * t) + Amp_exp_eta2 **2 * k_exp * np.cos(2 * (k_exp_prime * x - omega_exp * t)) + 3 * Amp_exp_eta2**3 * k_exp ** 2 * np.cos(3 * (k_exp_prime * x - omega_exp * t)) / 8
    eta_3 = (1 - 3* Amp_exp_eta2**2 * -k_exp**2 / 8) * Amp_exp_eta2 * np.cos(-k_exp_prime * x - omega_exp * t) + Amp_exp_eta2 **2 * -k_exp * np.cos(2 * (-k_exp_prime * x - omega_exp * t)) + 3 * Amp_exp_eta2**3 * -k_exp ** 2 * np.cos(3 * (-k_exp_prime * x - omega_exp * t)) / 8
    
    
    eta1_2_3 = Amp_exp_eta2 * np.sin(k_exp * x - omega_exp * t)
    terme1_2 = Amp_exp_eta2**2 * k_exp / 4
    terme2_2 = 3 * (1 / np.tanh(k_exp * H_exp))**3 - 1/ np.tanh(k_exp * H_exp)
    eta2 = terme1_2 * terme2_2 * np.cos(2 * (k_exp * x - omega_exp * t)) #- A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i]) #+ A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i])

    t = 6*np.pi / omega_exp / 4 - 0.03
    eta1_2 = Amp_exp_eta2 * np.sin(k_exp * x - omega_exp * t)
    terme1_2 = Amp_exp_eta2**2 * k_exp / 4
    terme2_2 = 3 * (1 / np.tanh(k_exp * H_exp))**3 - 1/ np.tanh(k_exp * H_exp)
    eta2_1 = terme1_2 * terme2_2 * np.cos(2 * (k_exp * x - omega_exp * t)) #- A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i]) #+ A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i])
    '''
    

    #QCS
    imin[i] = np.argmin(forme[int(1*n/8):int(7*n/8)]) + int(1*n/8)
    imax[i] = np.argmax(forme[int(1*n/8):int(7*n/8)]) + int(1*n/8)
    #CCM
    # imin[i] = np.argmin(forme[a:int(1*n/2)]) + a
    # imax[i] = np.argmax(forme[a:int(1*n/2)]) + a
    #NPDP
    # imin[i] = np.argmin(a + forme[int(0 * n / 5):int(9 * n / 9) - a]) + a + int(0 * n / 5)
    # imax[i] = np.argmax(a + forme[int(0 * n / 5):int(9 * n / 9) - a]) + a + int(0 * n / 5)
    
    
    
    
    hmin[i] = forme[imin[i]]
    hmax[i] = forme[imax[i]]
    


    yfit = forme[imin[i]-a:imin[i]+a]
    xfit = x_plotexp[imin[i]-a:imin[i]+a]
    popt_min[i] = np.polyfit(xfit,yfit, 2, full = True)
    yth = np.polyval(popt_min[i][0], xfit)
    err_min[i] = np.sqrt(popt_min[i][1][0])/ np.abs( popt_min[i][0][0] )* 100
    if err_min[i] > err :
        kmin[i] = None
        err_min[i] = None
        hmin[i] = None
    else :
        kmin[i] = popt_min[i][0][0]*2
    
    yfit = forme[imax[i]-a:imax[i]+a]
    xfit = x_plotexp[imax[i]-a:imax[i]+a]
    popt_max[i] = np.polyfit(xfit,yfit, 2, full = True)
    yth = np.polyval(popt_max[i][0], xfit)
    err_max[i] = np.sqrt(popt_max[i][1][0])/ np.abs( popt_max[i][0][0] )* 100
    if err_max[i] > err :
        kmax[i] = None
        err_max[i] = None
        hmax[i] = None
    else :
        kmax[i] = popt_max[i][0][0]*2
save = False
disp.figurejolie()

plt.plot(x_plotexp[imin], hmin, 'rv')
plt.plot(x_plotexp[imax], hmax, 'r^')

for i in range(len(liste_t)) :
    
    if not np.isnan(hmin[i]) :
        xfit = x_plotexp[imin[i]-a:imin[i]+a]
        yth = np.polyval(popt_min[i][0], xfit)
        plt.plot(xfit, yth, 'r-') 
    if not np.isnan(hmax[i]) :
        xfit = x_plotexp[imax[i]-a:imax[i]+a]
        yth = np.polyval(popt_max[i][0], xfit)
        plt.plot(xfit, yth, 'r-') 

  
disp.joliplot(r'x (m)', '$\eta$ (m)', x_plotexp, data[x0:xf, liste_t] - meann, exp = False, color = 14)
if save :
    plt.savefig('E:\\Baptiste\\Resultats_exp\\Courbure\\' + params['date']  + '_' + params['nom_exp'] + '_' + 'exp+fitcourbure_' + tools.datetimenow() + '.pdf')


disp.figurejolie()


disp.joliplot(r'$\frac{\eta - \eta_{min}}{\eta_{max} - \eta_{min}}$' ,r'$\kappa$ (m$^{-1}$)', (-hmin - np.nanmin(-hmin)) / (np.nanmax(-hmin)- np.nanmin(-hmin)), kmin, exp = True, color = 8, legend = 'Curvature at minimum')
plt.errorbar( (-hmin - np.nanmin(-hmin)) / (np.nanmax(-hmin)- np.nanmin(-hmin)), kmin, yerr = err_min, fmt = 'none',capsize = 5, ecolor = 'black',elinewidth = 3)


disp.joliplot(r'$\frac{\eta - \eta_{min}}{\eta_{max} - \eta_{min}}$' ,r'$\kappa$ (m$^{-1}$)', (hmax - np.nanmin(hmax)) / (np.nanmax(hmax)- np.nanmin(hmax)), -kmax, exp = True, color = 2, legend = 'Curvature at maximum')
plt.errorbar((hmax - np.nanmin(hmax)) / (np.nanmax(hmax)- np.nanmin(hmax)), -kmax, yerr = err_max, fmt = 'none',capsize = 5, ecolor = 'red',elinewidth = 3)

h_plot = np.linspace(0,1,100)
kappa_th = np.zeros(100) + np.nanmax(np.abs(hmin)) * k_exp**2

disp.joliplot(r'$\frac{\eta - \eta_{min}}{\eta_{max} - \eta_{min}}$' ,r'$\kappa$ (m$^{-1}$)', h_plot, kappa_th, color = 14, legend = r'$\kappa_{th}$', exp = False)

if save :
    plt.savefig('E:\\Baptiste\\Resultats_exp\\Courbure\\' + params['date']  + '_' + params['nom_exp'] + '_' + 'exp+fitcourbure_' + tools.datetimenow() + '.pdf')

forme = data[x0:xf,liste_t[0]]
x_kappa = np.linspace(a* params['mmparpixel'] / 1000, (xf-x0 - a) * params['mmparpixel'] / 1000, (xf-x0 - 2 * a) )
popt_x = np.zeros(len(x_kappa))
x_plotexp = np.linspace(0, (xf-x0) * params['mmparpixel'] / 1000, (xf-x0) )
disp.figurejolie()


for j in range (len(forme) - 2 * a) :
    yfit = forme[a + j: a + j + 2*a]
    xfit = x_plotexp[a +j:a +j + 2*a] 
    popt = np.polyfit(xfit,yfit, 2)
    popt_x[j] = -popt[0] * 2
    # yth = np.polyval(popt, xfit)
    # disp.joliplot('x (m)', r'$\kappa$ (m$^{-1}$)', xfit, yth, exp = False, color = 5)

disp.joliplot(r'x (m)', '$\eta$ (m)', x_plotexp , data[x0:xf, liste_t[0]], exp = False, color = 14)

   
disp.figurejolie()
disp.joliplot('x (m)', r'$\kappa$ (m$^{-1}$)', x_kappa + a* params['mmparpixel'] / 1000, popt_x, exp = False, color = 2)

disp.figurejolie()
disp.joliplot('x (m)', r'$\kappa$ (m$^{-1}$)', x, eta1 + eta2+ eta3+ eta_1 +eta_2 +eta_3 , exp = False, color = 2)
disp.joliplot(r'x (m)', '$\eta$ (m)', x_plotexp , data[x0:xf, liste_t[0]], exp = False, color = 14)

#%% Comparaison exp : ESPACE


#trace forme du laser superposé à forme de la théorie. On prend les params de l'exp. EN ESPACE

#FORME EXP (x)

lambda_exp = float(dico[date][nom_exp]['lambda'])
k_exp = 2 * np.pi / lambda_exp
omega_exp = 2 * np.pi * float(dico[date][nom_exp]['fexc'])
H_exp = float(dico[date][nom_exp]['Hw'])
facq = float(dico[date][nom_exp]['facq'])

# x_l2 = 737
# x0 = x_l2 - lambda_exp/2 * params['mmparpixel'] / 1000
# xf = x_l2 + lambda_exp/2 * params['mmparpixel'] / 1000
# t0 = 86
# tf = t0 + 13


x_l2 = 737
x0 = 1    #x_l2 - int(lambda_exp/4 / params['mmparpixel'] * 1000)
xf = 230 # x_l2 + int(lambda_exp/4 / params['mmparpixel'] * 1000)
t0 = 71

#ATTENTION, ON MET MOINS (-) CAR LE HAUT EST EN BAS POUR LE LASER

forme = -data[x0:xf, t0]

x_plotexp = np.linspace(0, (xf-x0) * params['mmparpixel'] / 1000, (xf-x0))

#FORME THEORIQUE (x)

Amp_exp_eta3 = np.max(forme) * 0.5
Amp_exp_eta2 = np.max(forme) * 1.1
Amp_exp_eta1 = np.max(forme)


t = (71 - t0) / facq + 2*np.pi / omega_exp / 4

x = np.linspace(0, 0 + lambda_exp * (xf-x0) / (lambda_exp / params['mmparpixel'] * 1000), (xf-x0))



eta1 = Amp_exp_eta1 * np.sin(k_exp * x - omega_exp * t)

eta1_2 = Amp_exp_eta2 * np.sin(k_exp * x - omega_exp * t)
terme1_2 = Amp_exp_eta2**2 * k_exp / 4
terme2_2 = 3 * (1 / np.tanh(k_exp * H_exp))**3 - 1/ np.tanh(k_exp * H_exp)
eta2 = terme1_2 * terme2_2 * np.cos(2 * (k_exp * x - omega_exp * t)) #- A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i]) #+ A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i])


eta1_3 = Amp_exp_eta3 * np.sin(k_exp * x - omega_exp * t)
terme1_3 = Amp_exp_eta3**2 * k_exp / 4
terme2_3 = 3 * (1 / np.tanh(k_exp * H_exp))**3 - 1/ np.tanh(k_exp * H_exp)
eta2_3 = terme1_3 * terme2_3 * np.cos(2 * (k_exp * x - omega_exp * t)) #- A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i]) #+ A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i])
k_exp_prime = k_exp * (1 - Amp_exp_eta3 ** 2 * k_exp ** 2)
eta3 = (1 - 3* Amp_exp_eta3**2 * k_exp**2 / 8) * Amp_exp_eta3 * np.cos(k_exp_prime * x - omega_exp * t) + Amp_exp_eta3 **2 * k_exp * np.cos(2 * (k_exp_prime * x - omega_exp * t)) + 3 * Amp_exp_eta3**3 * k_exp ** 2 * np.cos(3 * (k_exp_prime * x - omega_exp * t)) / 8


disp.figurejolie()
disp.joliplot(r'x (0 à $\frac{\lambda}{2}$)', 'x (m)', x_plotexp, eta1, exp = False, color = 4, legend = '$\eta_1$')
disp.joliplot(r'x (0 à $\frac{\lambda}{2}$)', 'x (m)', x_plotexp, eta1_2 + eta2, exp = False, color = 2, legend = '$\eta_1 + \eta_2$')
# disp.joliplot('t (0 à T)', 'x (m)', t_exp, eta1_3 + eta2_3 + eta3 - np.mean(eta1_3 + eta2_3 + eta3), exp = False, color = 12, legend = '$\eta_1 + \eta_2 + \eta_3$')
disp.joliplot(r'x (0 à $\frac{\lambda}{2}$)', 'x (m)', x_plotexp, forme, exp = False, color = 14, legend = 'Expérience')

#%% POUR UNE VIDEO
lambda_exp = float(dico[date][nom_exp]['lambda'])
k_exp = 2 * np.pi / lambda_exp
omega_exp = 2 * np.pi * float(dico[date][nom_exp]['fexc'])
H_exp = float(dico[date][nom_exp]['Hw'])
facq = float(dico[date][nom_exp]['facq'])

boost_nonlineaire = 2
t00 = 100
tff = 146
x0 = 20    #x_l2 - int(lambda_exp/4 / params['mmparpixel'] * 1000)
xf = 1350 # x_l2 + int(lambda_exp/4 / params['mmparpixel'] * 1000)

# t00 = 408
# tff = 438
# x0 = 250
# xf = 1250


nb_periodes = 1
liste_t = np.linspace(t00,tff,tff-t00, dtype = int)
x = np.linspace(1 *lambda_exp / 16, 1*lambda_exp / 16 + nb_periodes * lambda_exp * (xf-x0) / (lambda_exp / params['mmparpixel'] * 1000), (xf-x0))
x_plotexp = np.linspace(0, (xf-x0) * params['mmparpixel'] / 1000, (xf-x0))

err = 0.3

# liste_t = [400, 412,424]


a = 50
sig = 0.002
stddd = np.std(data[x0:xf,:], axis = 0)
meann = np.mean(data[x0:xf, stddd < sig])

hmin = np.zeros(np.shape(liste_t))
hmax = np.zeros(np.shape(liste_t))
popt_min = np.zeros(np.shape(liste_t), dtype = object)
popt_max = np.zeros(np.shape(liste_t), dtype = object)
err_min = np.zeros(np.shape(liste_t))
err_max = np.zeros(np.shape(liste_t))
kmin = np.zeros(np.shape(liste_t))
kmax = np.zeros(np.shape(liste_t))
imin = np.zeros(np.shape(liste_t), dtype = int)
imax = np.zeros(np.shape(liste_t), dtype = int)




for i in range(len(liste_t)) :
    t0 = liste_t[i]
    forme = data[x0:xf, t0] - meann
    n = forme.shape[0]
    t = t0 / facq
    
    '''         THEORIE
    
    Amp_exp_eta2 = np.max(data[x0:xf, liste_t]) * boost_nonlineaire
    
    k_exp_prime = k_exp * (1 - Amp_exp_eta2 ** 2 * k_exp ** 2)
    terme1 = Amp_exp_eta2**2 * k_exp / 4
    terme2 = 3 * (1 / np.tanh(k_exp * H_exp))**3 - 1/ np.tanh(k_exp * H_exp)
    terme_1 = Amp_exp_eta2**2 * -k_exp / 4
    terme_2 = 3 * (1 / np.tanh(-k_exp * H_exp))**3 - 1/ np.tanh(-k_exp * H_exp)
    eta1 = Amp_exp_eta2 * np.sin(k_exp * x - omega_exp * t)
    eta_1 = Amp_exp_eta2 * np.sin(-k_exp * x - omega_exp * t)
    eta2 = terme1 * terme2 * np.cos(2 * (k_exp * x - omega_exp * t)) #- A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i]) #+ A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i])
    eta_2 = terme_1 * terme_2 * np.cos(2 * (-k_exp * x - omega_exp * t))
    eta3 = (1 - 3* Amp_exp_eta2**2 * k_exp**2 / 8) * Amp_exp_eta2 * np.cos(k_exp_prime * x - omega_exp * t) + Amp_exp_eta2 **2 * k_exp * np.cos(2 * (k_exp_prime * x - omega_exp * t)) + 3 * Amp_exp_eta2**3 * k_exp ** 2 * np.cos(3 * (k_exp_prime * x - omega_exp * t)) / 8
    eta_3 = (1 - 3* Amp_exp_eta2**2 * -k_exp**2 / 8) * Amp_exp_eta2 * np.cos(-k_exp_prime * x - omega_exp * t) + Amp_exp_eta2 **2 * -k_exp * np.cos(2 * (-k_exp_prime * x - omega_exp * t)) + 3 * Amp_exp_eta2**3 * -k_exp ** 2 * np.cos(3 * (-k_exp_prime * x - omega_exp * t)) / 8

    eta1_2_3 = Amp_exp_eta2 * np.sin(k_exp * x - omega_exp * t)
    terme1_2 = Amp_exp_eta2**2 * k_exp / 4
    terme2_2 = 3 * (1 / np.tanh(k_exp * H_exp))**3 - 1/ np.tanh(k_exp * H_exp)
    eta2 = terme1_2 * terme2_2 * np.cos(2 * (k_exp * x - omega_exp * t)) #- A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i]) #+ A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i])

    t = 6*np.pi / omega_exp / 4 - 0.03
    eta1_2 = Amp_exp_eta2 * np.sin(k_exp * x - omega_exp * t)
    terme1_2 = Amp_exp_eta2**2 * k_exp / 4
    terme2_2 = 3 * (1 / np.tanh(k_exp * H_exp))**3 - 1/ np.tanh(k_exp * H_exp)
    eta2_1 = terme1_2 * terme2_2 * np.cos(2 * (k_exp * x - omega_exp * t)) #- A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i]) #+ A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i])
    '''
    

    #QCS
    imin[i] = np.argmin(forme[int(1.5*n/6):int(7*n/8)]) + int(1.5*n/6)
    imax[i] = np.argmax(forme[int(1.5*n/6):int(7*n/8)]) + int(1.5*n/6)
    #CCM
    # imin[i] = np.argmin(forme[a:int(1*n/2)]) + a
    # imax[i] = np.argmax(forme[a:int(1*n/2)]) + a
    #NPDP
    # imin[i] = np.argmin(a + forme[int(0 * n / 5):int(9 * n / 9) - a]) + a + int(0 * n / 5)
    # imax[i] = np.argmax(a + forme[int(0 * n / 5):int(9 * n / 9) - a]) + a + int(0 * n / 5)
    
    
    
    
    hmin[i] = forme[imin[i]]
    hmax[i] = forme[imax[i]]
    


    yfit = forme[imin[i]-a:imin[i]+a]
    xfit = x_plotexp[imin[i]-a:imin[i]+a]
    popt_min[i] = np.polyfit(xfit,yfit, 2, full = True)
    yth = np.polyval(popt_min[i][0], xfit)
    err_min[i] = np.sqrt(popt_min[i][1][0])/ np.abs( popt_min[i][0][0] )* 100
    if err_min[i] > err :
        kmin[i] = None
        err_min[i] = None
        hmin[i] = None
    else :
        kmin[i] = popt_min[i][0][0]*2
    
    yfit = forme[imax[i]-a:imax[i]+a]
    xfit = x_plotexp[imax[i]-a:imax[i]+a]
    popt_max[i] = np.polyfit(xfit,yfit, 2, full = True)
    yth = np.polyval(popt_max[i][0], xfit)
    err_max[i] = np.sqrt(popt_max[i][1][0])/ np.abs( popt_max[i][0][0] )* 100
    if err_max[i] > err :
        kmax[i] = None
        err_max[i] = None
        hmax[i] = None
    else :
        kmax[i] = popt_max[i][0][0]*2
    

    
    # disp.joliplot(r'x (0 à $\frac{\lambda}{2}$)', 'x (m)', x_plotexp, eta1, exp = False, color = 4)#, legend = '$\eta_1$')
    # disp.joliplot(r'x (0 à $\frac{\lambda}{2}$)', 'x (m)', x_plotexp, (eta1_2_3 + eta2)/boost_nonlineaire, exp = False, color = 2)#, legend = '$\eta_1 + \eta_2$')
    # disp.joliplot(r'x (0 à $\frac{\lambda}{2}$)', 'x (m)', x_plotexp, (eta1_2 + eta2_1)/boost_nonlineaire, exp = False, color = 4)
    # disp.joliplot(r'x (0 à $\frac{\lambda}{2}$)', 'x (m)', x_plotexp, (eta1 + eta_1 + eta2 + eta_2) / boost_nonlineaire, exp = False, color = 4)
    # disp.joliplot(r'x (0 à $\frac{\lambda}{2}$)', 'x (m)', x_plotexp, (eta1 + eta2) / boost_nonlineaire, exp = False, color = 2)
    # disp.joliplot(r'x (0 à $\frac{\lambda}{2}$)', 'x (m)', x_plotexp, (eta_1 + eta_2) / boost_nonlineaire, exp = False, color = 8)
    # disp.joliplot(r'x (0 à $\frac{\lambda}{2}$)', 'x (m)', x_plotexp, -(eta1 + eta_1 + eta2 + eta_2 + eta3 + eta_3) / boost_nonlineaire/ 4, exp = False, color = 8)
    # disp.joliplot(r'x (0 à $\frac{\lambda}{2}$)', 'x (m)', x_plotexp, forme, exp = False, color = 14)#, legend = 'Expérience')
    # plt.grid('off')
    # plt.title(str(t0))
    # plt.pause(0.3)
#%% film profil


path = 'E:\Baptiste\\Resultats_exp\\Courbure\\Resultats\\'

data_1 = -data[x0:xf, liste_t[0]:liste_t[-1]]

[nx,nt] = data_1.shape

nb_img_film = nt
h_film = 1000 #hauteur de l'image en pixel
pourcent_img = 1

multiplicateur_1 =  h_film / (np.nanmax(data_1) - np.nanmin(data_1) ) / pourcent_img # pour que le laser bouge sur 1/pourcent_img de l'image
im_brut = np.zeros( (h_film,nx,nb_img_film,3) ) + 255

data_plot_1 = data_1 * multiplicateur_1

for i in range (nb_img_film) :
    for j in range (nx) :
        im_brut[int(data_plot_1[j,i]) + int(h_film/2) - 2: int(data_plot_1[j,i]) + int(h_film/2) + 2,j,i,0] = 0 
        im_brut[int(data_plot_1[j,i]) + int(h_film/2) - 2: int(data_plot_1[j,i]) + int(h_film/2) + 2,j,i,1] = 0 
        
        

frame = im_brut

fps = 7

image_temp_name = 'image.png'

# video_name = 'video_demod_' + str(f_exc) + 'Hz_' + str(fps) + 'fps_'+ str(nb_img_film) + 'images_moinsx.avi'
# video_name = 'video_demod_' + str(f_exc) + 'Hz_' + str(fps) + 'fps_'+ str(nb_img_film) + 'images.avi'
video_name = "profil_laser_QSC10_" + str(fps) + 'fps_'+ str(nb_img_film) + ".avi"

folder_video = path + video_name

height, width = frame.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(folder_video, 0, fps, (width,height), isColor = True)# isColor = False)

for i in range (nb_img_film):
    cv2.imwrite(path + image_temp_name, frame[:,:,i])
    im = cv2.imread(path + image_temp_name)#, cv2.IMREAD_GRAYSCALE)
    video.write(im)

cv2.destroyAllWindows()
video.release()








#%% Test pour a

date = '240109'
nom_exp = 'QSC10'
exp = True
exp_type = 'LAS'

save = False
display = True

dico, params, loc = ip.initialisation(date, nom_exp, exp = True, display = False)

folder_results = params['path_images'][:-15] + "resultats"
name_file = "positionLAS.npy"
data_originale = np.load(folder_results + "\\" + name_file)

if True : #cam_dessus :
    data_originale = np.rot90(data_originale)
    data_originale = np.flip(data_originale, 0)
   

    
data_originale = data_originale - np.nanmean(data_originale, axis = 0)    


params['debut_las'] = 150
params['fin_las'] = np.shape(data_originale)[0] - 250
#100 / 400 Pour DAP
#650 / 300 Pour CCCS1
#1 / 200 pr CCCS2

params['t0'] = 111
params['tf'] = np.shape(data_originale)[1] - 1800


[nx,nt] = data_originale[params['debut_las']:params['fin_las'],params['t0']:params['tf']].shape


data = data_originale[params['debut_las']:params['fin_las'],params['t0']:params['tf']]

params['ordre_savgol'] = 2
params['taille_savgol'] = 11
params['size_medfilt'] = 51


#mise à l'échelle en m
data_m = data *  params['mmparpixely'] / 1000

data_m = data_m / params['grossissement']


t = np.arange(0,nt)/params['facq']
x = np.arange(0,nx)*params['mmparpixelz'] / 1000

data = data_m.copy()


lambda_exp = float(dico[date][nom_exp]['lambda'])
k_exp = 2 * np.pi / lambda_exp
omega_exp = 2 * np.pi * float(dico[date][nom_exp]['fexc'])
H_exp = float(dico[date][nom_exp]['Hw'])
facq = float(dico[date][nom_exp]['facq'])

t00 = 1
tff = 150
x0 = 20    #x_l2 - int(lambda_exp/4 / params['mmparpixel'] * 1000)
xf = 280 # x_l2 + int(lambda_exp/4 / params['mmparpixel'] * 1000)

t00 = 65
tff = 90
x0 = 20
xf = 1350



nb_periodes = 1
liste_t = np.linspace(t00,tff,tff-t00, dtype = int)
x = np.linspace(1 *lambda_exp / 16, 1*lambda_exp / 16 + nb_periodes * lambda_exp * (xf-x0) / (lambda_exp / params['mmparpixel'] * 1000), (xf-x0))
x_plotexp = np.linspace(0, (xf-x0) * params['mmparpixel'] / 1000, (xf-x0))

err = 0.2


sig = 0.001
stddd = np.std(data[x0:xf,:], axis = 0)
meann = np.mean(data[x0:xf, stddd < sig])




liste_a = np.linspace(5,95,10, dtype = int)
somme_err = np.zeros(len(liste_a))
kappa_max = np.zeros(len(liste_a))
kappa_min = np.zeros(len(liste_a))

for j in range(len(liste_a)) :
    a = liste_a[j]
    hmin = np.zeros(np.shape(liste_t))
    hmax = np.zeros(np.shape(liste_t))
    popt_min = np.zeros(np.shape(liste_t), dtype = object)
    popt_max = np.zeros(np.shape(liste_t), dtype = object)
    err_min = np.zeros(np.shape(liste_t))
    err_max = np.zeros(np.shape(liste_t))
    kmin = np.zeros(np.shape(liste_t))
    kmax = np.zeros(np.shape(liste_t))
    imin = np.zeros(np.shape(liste_t), dtype = int)
    imax = np.zeros(np.shape(liste_t), dtype = int)
    for i in range(len( liste_t)) :
        t0 = liste_t[i]
        forme = -data[x0:xf, t0] + meann
        n = forme.shape[0]
        t = t0 / facq
        
        #QCS
        imin[i] = np.argmin(forme[int(1.5*n/6):int(7*n/8)]) + int(1.5*n/6)
        imax[i] = np.argmax(forme[int(1.5*n/6):int(7*n/8)]) + int(1.5*n/6)
        #CCM
        # imin[i] = np.argmin(forme[a:int(1*n/2)]) + a
        # imax[i] = np.argmax(forme[a:int(1*n/2)]) + a
        # #NPDP
        # imin[i] = np.argmin(forme[a + int(0 * n / 5):int(8 * n / 9)]) + a + int(0 * n / 5)
        # imax[i] = np.argmax(forme[a + int(0 * n / 5):int(8 * n / 9)]) + a + int(0 * n / 5)
        
        hmin[i] = forme[imin[i]]
        hmax[i] = forme[imax[i]]
        
        yfit = forme[imin[i]-a:imin[i]+a]
        xfit = x_plotexp[imin[i]-a:imin[i]+a]
        popt_min[i] = np.polyfit(xfit,yfit, 2, full = True)
        yth = np.polyval(popt_min[i][0], xfit)
        err_min[i] = np.sqrt(popt_min[i][1][0])/ np.abs(np.max(hmin[i]))
        if err_min[i] > err :
            kmin[i] = None
            err_min[i] = None
            hmin[i] = None
        else :
            kmin[i] = popt_min[i][0][0]*2
        
        yfit = forme[imax[i]-a:imax[i]+a]
        xfit = x_plotexp[imax[i]-a:imax[i]+a]
        popt_max[i] = np.polyfit(xfit,yfit, 2, full = True)
        yth = np.polyval(popt_max[i][0], xfit)
        err_max[i] = np.sqrt(popt_max[i][1][0]) / np.abs(np.max(hmax[i]))
        if err_max[i] > err :
            kmax[i] = None
            err_max[i] = None
            hmax[i] = None
        else :
            kmax[i] = popt_max[i][0][0]*2
        
    disp.figurejolie()
    plt.plot(x_plotexp[imin], hmin, 'rv')
    plt.plot(x_plotexp[imax], hmax, 'r^')
    for i in range(len(liste_t)) :
        if not np.isnan(hmin[i]) :
            xfit = x_plotexp[imin[i]-a:imin[i]+a]
            yth = np.polyval(popt_min[i][0], xfit)
            plt.plot(xfit, yth, 'r-') 
        if not np.isnan(hmax[i]) :
            xfit = x_plotexp[imax[i]-a:imax[i]+a]
            yth = np.polyval(popt_max[i][0], xfit)
            plt.plot(xfit, yth, 'r-')
    disp.joliplot(r'x (0 à $\frac{\lambda}{2}$)', 'x (m)', x_plotexp, -data[x0:xf, liste_t] + meann, exp = False, color = 14)
    if save :
        plt.savefig('E:\\Baptiste\\Resultats_exp\\Courbure\\' + params['date']  + '_' + params['nom_exp'] + '_' + 'exp+fitcourbure_' + tools.datetimenow() + '.pdf')


    disp.figurejolie()
    disp.joliplot(r'$\frac{\eta - \eta_{min}}{\eta_{max}}$' ,r'$\kappa$ (m$^{-1}$)', (-hmin - np.nanmin(np.abs(-hmin))) / np.nanmax(-hmin), kmin, exp = True, color = 8, legend = 'Curvature at minimum')
    plt.errorbar((-hmin - np.nanmin(np.abs(-hmin))) / np.nanmax(-hmin), kmin, yerr = err_min, fmt = 'none',capsize = 5, ecolor = 'black',elinewidth = 3)
    disp.joliplot(r'$\frac{\eta - \eta_{min}}{\eta_{max}}$' ,r'$\kappa$ (m$^{-1}$)', (hmax - np.nanmin(np.abs(hmax))) / np.nanmax(hmax), -kmax, exp = True, color = 2, legend = 'Curvature at maximum')
    plt.errorbar((hmax - np.nanmin(np.abs(hmax))) / np.nanmax(hmax), -kmax, yerr = err_max, fmt = 'none',capsize = 5, ecolor = 'red',elinewidth = 3)
    disp.joliplot(r'$\frac{\eta - \eta_{min}}{\eta_{max}}$' ,r'$\kappa$ (m$^{-1}$)', 0.02, np.nanmax(np.abs(hmin)) * k_exp**2, color = 10)
    plt.title(str(a))
    if save :
        plt.savefig('E:\\Baptiste\\Resultats_exp\\Courbure\\' + params['date']  + '_' + params['nom_exp'] + '_' + 'exp+fitcourbure_' + tools.datetimenow() + '.pdf')
        
    somme_err[j] = (np.nansum(err_max) + np.nansum(err_min))/a
    kappa_max[j] = np.nanstd(kmax) #np.nanmax(np.abs(kmax))
    kappa_min[j] = np.nanstd(kmin) #np.nanmax(np.abs(kmin)) 
    # kappa_moins[j] = np.nansum(k)
    
disp.figurejolie()
disp.joliplot('a', r'$\sigma_{err}$', liste_a, somme_err, exp = False, color = 2)
disp.figurejolie()
disp.joliplot('a', r'$std(\kappa_{max})$', liste_a, kappa_max, exp = True, color = 2)
disp.figurejolie()
disp.joliplot('a', r'$std(\kappa_{min})$', liste_a, kappa_min, exp = True, color = 2)




#%% PLOT courbure
save = False
disp.figurejolie()

plt.plot(x_plotexp[imin], hmin, 'rv')
plt.plot(x_plotexp[imax], hmax, 'r^')

for i in range(len(liste_t)) :
    
    if not np.isnan(hmin[i]) :
        xfit = x_plotexp[imin[i]-a:imin[i]+a]
        yth = np.polyval(popt_min[i][0], xfit)
        plt.plot(xfit, yth, 'r-') 
    if not np.isnan(hmax[i]) :
        xfit = x_plotexp[imax[i]-a:imax[i]+a]
        yth = np.polyval(popt_max[i][0], xfit)
        plt.plot(xfit, yth, 'r-') 

  
disp.joliplot(r'x (m)', '$\eta$ (m)', x_plotexp, data[x0:xf, liste_t] - meann, exp = False, color = 14)
if save :
    plt.savefig('E:\\Baptiste\\Resultats_exp\\Courbure\\' + params['date']  + '_' + params['nom_exp'] + '_' + 'exp+fitcourbure_' + tools.datetimenow() + '.pdf')


disp.figurejolie()


disp.joliplot(r'$\frac{\eta - \eta_{min}}{\eta_{max} - \eta_{min}}$' ,r'$\kappa$ (m$^{-1}$)', (-hmin - np.nanmin(-hmin)) / (np.nanmax(-hmin)- np.nanmin(-hmin)), kmin, exp = True, color = 8, legend = 'Curvature at minimum')
plt.errorbar( (-hmin - np.nanmin(-hmin)) / (np.nanmax(-hmin)- np.nanmin(-hmin)), kmin, yerr = err_min, fmt = 'none',capsize = 5, ecolor = 'black',elinewidth = 3)


disp.joliplot(r'$\frac{\eta - \eta_{min}}{\eta_{max} - \eta_{min}}$' ,r'$\kappa$ (m$^{-1}$)', (hmax - np.nanmin(hmax)) / (np.nanmax(hmax)- np.nanmin(hmax)), -kmax, exp = True, color = 2, legend = 'Curvature at maximum')
plt.errorbar((hmax - np.nanmin(hmax)) / (np.nanmax(hmax)- np.nanmin(hmax)), -kmax, yerr = err_max, fmt = 'none',capsize = 5, ecolor = 'red',elinewidth = 3)

h_plot = np.linspace(0,1,100)
kappa_th = np.zeros(100) + np.nanmax(np.abs(hmin)) * k_exp**2

disp.joliplot(r'$\frac{\eta - \eta_{min}}{\eta_{max} - \eta_{min}}$' ,r'$\kappa$ (m$^{-1}$)', h_plot, kappa_th, color = 14, legend = r'$\kappa_{th}$', exp = False)

if save :
    plt.savefig('E:\\Baptiste\\Resultats_exp\\Courbure\\' + params['date']  + '_' + params['nom_exp'] + '_' + 'exp+fitcourbure_' + tools.datetimenow() + '.pdf')

#%% Save courbure.

if save :
    params['courbure'] = {}
    params['courbure']['hmin'] = hmin
    params['courbure']['hmax'] = hmax
    params['courbure']['imin'] = imin
    params['courbure']['imax'] = imax
    params['courbure']['popt_min'] = popt_min
    params['courbure']['popt_max'] = popt_max
    params['courbure']['kmin'] = kmin
    params['courbure']['kmax'] = kmax
    params['courbure']['err_min'] = err_min
    params['courbure']['err_max'] = err_max
    params['courbure']['x_plotexp'] = x_plotexp
    params['courbure']['x0'] = x0
    params['courbure']['xf'] = xf
    params['courbure']['t00'] = t00
    params['courbure']['tff'] = tff
    params['courbure']['a'] = a
    params['courbure']['sig'] = sig
    params['courbure']['err'] = err

    dic.save_dico(params, 'E:\\Baptiste\\Resultats_exp\\Courbure\\params_courbure_' + tools.datetimenow() + '.pkl')


#%% Courbure sur tout une exp 

save = False
display = True
date = '231122'
exp = True
exp_type = 'LAS'

nb_exps = 8

err = 0.15
# sig = 0.002 #QSC
sig = 0.00075 #NPDP
# sig = 0.016 #RLPY

# # NPDP
x0 = 120    
xf = 420
t00 = 10
tff = 300

#RLPY
# t00 = 600
# tff = 650
# x0 = 70
# xf = 400

#QSC
# t00 = 1
# tff = 150
# x0 = 20
# xf = 1350


#ECTD
# t00 = 200
# tff = 400
# x0 = 1
# xf = 559

# #MLO
# t00 = 150
# tff = 300
# x0 = 1
# xf = 769

# #EJCJ
# t00 = 300
# tff = 500
# x0 = 1
# xf = 350


# #TNB
# t00 = 180
# tff = 400
# x0 = 1
# xf = 1100

# #EDTH
# t00 = 50
# tff = 150
# x0 = 1
# xf = 300

# #DMLO
# t00 = 140
# tff = 290
# x0 = 140
# xf = 750


a = 30

k_maxmax = np.zeros(nb_exps)
k_minmin = np.zeros(nb_exps)
h_maxmax = np.zeros(nb_exps)
h_mimmin = np.zeros(nb_exps)
amplitude_exp = np.zeros(nb_exps)
k_exp_th = np.zeros(nb_exps)

for j in range (1,nb_exps) :
    
    '''CHARGE LES DATA'''
    
    if j < 10 :
        nom_exp = 'NPDP' + str(j)
    else : 
        nom_exp = 'NPD' + str(j)
    
    print(nom_exp)        
    
    # if j == 3:
    #     t00 = 100
    #     tff = 200
    #     # x0 = 1
    #     # err = 0.06
    # elif j == 1 :
    #     t00 = 60
    #     tff = 150
    #     # x0 = 1
    #     # err = 0.08
    # elif j == 2 :
    #     t00 = 60
    #     tff = 150
    #     # err = 0.07
    # else :
    #     t00 = 20
    #     tff = 120
    #     # x0 = 1
    #     # xf = 300
    #     # err = 0.07
        
    if j == 10 :
        pass
    # elif j == 2:
    #     pass
    else :
        path_save = 'E:\\Baptiste\\Resultats_exp\\Courbure\\' + date + '_' + nom_exp + '\\'
        
        dico, params, loc = ip.initialisation(date, nom_exp, exp = True, display = False)
        if save :
            os.mkdir(path_save)
        folder_results = params['path_images'][:-15] + "resultats"
        name_file = "positionLAS.npy"
        data_originale = np.load(folder_results + "\\" + name_file)
        data_originale = np.rot90(data_originale)
        data_originale = np.flip(data_originale, 0)
        # enleve moyenne moyenne en espace pour chaque temps (pas ouf)
        # data_originale = data_originale - np.nanmean(data_originale, axis = 0)    
    
        params['debut_las'] = 150
        params['fin_las'] = np.shape(data_originale)[0] - 1
    
        params['t0'] = 111
        params['tf'] = np.shape(data_originale)[1] - 1
        
        lambda_exp = float(dico[date][nom_exp]['lambda'])
        k_exp = 2 * np.pi / lambda_exp
        omega_exp = 2 * np.pi * float(dico[date][nom_exp]['fexc'])
        H_exp = float(dico[date][nom_exp]['Hw'])
        facq = float(dico[date][nom_exp]['facq'])
        amp_exp = float(dico[date][nom_exp]['Amp_max'])
        
        
        
        [nx,nt] = data_originale[params['debut_las']:params['fin_las'],params['t0']:params['tf']].shape
    
        data = data_originale[params['debut_las']:params['fin_las'],params['t0']:params['tf']]
        
        params['im_ref'] = False
    
        #enlever moyenne pr chaque pixel
    
        if params['im_ref'] :
            mean_pixel = np.nanmean(data,axis = 1) #liste avec moyenne en temps de chaque pixel 
            for i in range (0,nt):
                data[:,i] = data[:,i] - mean_pixel #pour chaque pixel, on enleve la moyenne temporelle de chaque pixel
    
        #mise à l'échelle en m
        data_m = data *  params['mmparpixely'] / 1000
        data_m = data_m / params['grossissement']
        
        #Filtre savgol
        params['savgol'] = False
        params['ordre_savgol'] = 2
        params['taille_savgol'] = int(a) * 2 + 1
        signalsv = np.zeros(data.shape)
        # for i in range(0,nt):  
        #     signalsv[:,i] = savgol_filter(data_m[:,i], params['taille_savgol'],params['ordre_savgol'], mode = 'nearest')
        #     if np.mod(i,1000)==0:
        #         print('On processe l image numero: ' + str(i) + ' sur ' + str(nt))
        # print('Done !')
    
    
        if params['savgol'] :
            data = signalsv.copy()
        else :
            data = data_m.copy()
            
            
        data = data - np.nanmean(data)
        
        t = np.arange(0,nt)/params['facq']
        x = np.arange(0,nx)*params['mmparpixelz'] / 1000
        
        
        
        '''INITIE PARAMS'''
        
        
    
        liste_t = np.linspace(t00,tff,tff-t00, dtype = int)
        
        x = np.linspace(1 *lambda_exp / 16, 1*lambda_exp / 16 + lambda_exp * (xf-x0) / (lambda_exp / params['mmparpixel'] * 1000), (xf-x0))
        
        x_plotexp = np.linspace(0, (xf-x0) * params['mmparpixel'] / 1000, (xf-x0))
    
        stddd = np.std(data[x0:xf,:], axis = 0)
        meann = np.mean(data[x0:xf, stddd < sig])
    
        hmin = np.zeros(np.shape(liste_t))
        hmax = np.zeros(np.shape(liste_t))
        popt_min = np.zeros(np.shape(liste_t), dtype = object)
        popt_max = np.zeros(np.shape(liste_t), dtype = object)
        err_min = np.zeros(np.shape(liste_t))
        err_max = np.zeros(np.shape(liste_t))
        kmin = np.zeros(np.shape(liste_t))
        kmax = np.zeros(np.shape(liste_t))
        imin = np.zeros(np.shape(liste_t), dtype = int)
        imax = np.zeros(np.shape(liste_t), dtype = int)
    
    
    
    
        for i in range(len( liste_t)) :
            t0 = liste_t[i]
            forme = data[x0:xf, t0] - meann
            n = forme.shape[0]
            t = t0 / facq
            
            #QSC
            imin[i] = np.argmin(forme[int(1.5*n/6):int(7*n/8)]) + int(1.5*n/6)
            imax[i] = np.argmax(forme[int(1.5*n/6):int(7*n/8)]) + int(1.5*n/6)        
            #NPDP
            imin[i] = np.argmin(forme[a + int(2 * n / 9):int(7 * n / 9)]) + a + int(2 * n / 9)
            imax[i] = np.argmax(forme[a + int(2 * n / 9):int(7 * n / 9)]) + a + int(2 * n / 9)
            
            hmin[i] = forme[imin[i]]
            hmax[i] = forme[imax[i]]
            
            yfit = forme[imin[i]-a:imin[i]+a]
            xfit = x_plotexp[imin[i]-a:imin[i]+a]
            popt_min[i] = np.polyfit(xfit,yfit, 2, full = True)
            yth = np.polyval(popt_min[i][0], xfit)
            err_min[i] = np.sqrt(popt_min[i][1][0])/ np.abs(np.max(hmin[i]))
            if err_min[i] > err :
                kmin[i] = None
                err_min[i] = None
                hmin[i] = None
            else :
                kmin[i] = popt_min[i][0][0]*2
            
            yfit = forme[imax[i]-a:imax[i]+a]
            xfit = x_plotexp[imax[i]-a:imax[i]+a]
            
            popt_max[i] = np.polyfit(xfit,yfit, 2, full = True)
            popt_max[i]
            
            yth = np.polyval(popt_max[i][0], xfit)
            err_max[i] = np.sqrt(popt_max[i][1][0]) / np.abs(np.max(hmax[i]))
            if err_max[i] > err :
                kmax[i] = None
                err_max[i] = None
                hmax[i] = None
            else :
                kmax[i] = popt_max[i][0][0]*2
        
        if display :
            disp.figurejolie()    
            plt.plot(x_plotexp[imin], hmin, 'rv')
            plt.plot(x_plotexp[imax], hmax, 'r^')    
            for i in range(len(liste_t)) :
                if not np.isnan(hmin[i]) :
                    xfit = x_plotexp[imin[i]-a:imin[i]+a]
                    yth = np.polyval(popt_min[i][0], xfit)
                    plt.plot(xfit, yth, 'r-') 
                if not np.isnan(hmax[i]) :
                    xfit = x_plotexp[imax[i]-a:imax[i]+a]
                    yth = np.polyval(popt_max[i][0], xfit)
                    plt.plot(xfit, yth, 'r-') 
                    
                    
            # Pour visualiser le temps en couleur de courbe
            for ww in range (len(liste_t)) :
                colors = disp.vcolors( int(ww / len(liste_t) * 9) )
                plt.scatter(x_plotexp,data[x0:xf, liste_t[ww] ] - meann,color=colors)
            
            # disp.joliplot(r'x (0 à $\frac{\lambda}{2}$)', 'x (m)', x_plotexp, data[x0:xf, liste_t] - meann, exp = False, color = 14)
            
            
            if save :
                plt.savefig(path_save + 'exp+fitcourbure_' + tools.datetimenow() + '.pdf')
        
            disp.figurejolie()    
            disp.joliplot(r'$\frac{\eta - \eta_{min}}{\eta_{max}}$' ,r'$\kappa$ (m$^{-1}$)', (-hmin - np.nanmin(-hmin)) / (np.nanmax(-hmin)- np.nanmin(-hmin)), kmin, exp = True, color = 8, legend = 'Curvature at minimum')
            plt.errorbar((-hmin - np.nanmin(-hmin)) / (np.nanmax(-hmin)- np.nanmin(-hmin)), kmin, yerr = err_min, fmt = 'none',capsize = 5, ecolor = 'black',elinewidth = 3)
            disp.joliplot(r'$\frac{\eta - \eta_{min}}{\eta_{max}}$' ,r'$\kappa$ (m$^{-1}$)', (hmax - np.nanmin(hmax)) / (np.nanmax(hmax)- np.nanmin(hmax)), -kmax, exp = True, color = 2, legend = 'Curvature at maximum')
            plt.errorbar((hmax - np.nanmin(hmax)) / (np.nanmax(hmax)- np.nanmin(hmax)), -kmax, yerr = err_max, fmt = 'none',capsize = 5, ecolor = 'red',elinewidth = 3)
            
            h_plot = np.linspace(0,1,100)
            kappa_th = np.zeros(100) + amp_exp * k_exp**2
            disp.joliplot(r'$\frac{\eta - \eta_{min}}{\eta_{max} - \eta_{min}}$' ,r'$\kappa$ (m$^{-1}$)', h_plot, kappa_th, color = 14, legend = r'$\kappa_{th}$', exp = False)
            
            if save :
                plt.savefig(path_save + 'Courburemax_courburemin_courburetherique_0a1' + tools.datetimenow() + '.pdf')
    
    
        k_maxmax[j] = np.nanmax(np.abs(kmax))
        k_minmin[j] = np.nanmax(np.abs(kmin))
        h_maxmax[j] = np.nanmax(np.abs(hmax))
        h_mimmin[j] = np.nanmax(np.abs(hmin))
        amplitude_exp[j] = amp_exp
        k_exp_th[j] = k_exp

courbure_th = amplitude_exp * k_exp_th**2
disp.figurejolie()
disp.joliplot('A (m)', r'$\kappa$ (m$^{-1}$)', amplitude_exp, k_maxmax, legend = 'Courbure au max', color = 2)
disp.joliplot('A (m)', r'$\kappa$ (m$^{-1}$)', amplitude_exp, k_minmin, legend = 'Courbure au min', color = 8)
disp.joliplot('A (m)', r'$\kappa$ (m$^{-1}$)', amplitude_exp, amplitude_exp * k_exp_th**2, legend = 'Courbure théorique', color = 14)
if save :
    plt.savefig(path_save[:-13] + date + '_' + nom_exp[:3] + '_couburemax_courburemin_courburetheorique_A_' + tools.datetimenow() + '.pdf')

#%% Save params

if save :
    params['courbure'] = {}
    params['courbure']['k_maxmax'] = k_maxmax
    params['courbure']['k_minmin'] = k_minmin
    params['courbure']['h_maxmax'] = h_maxmax
    params['courbure']['h_mimmin'] = h_mimmin
    params['courbure']['k_exp_th'] = k_exp_th
    params['courbure']['courbure_th'] = courbure_th
    params['courbure']['amplitude_exp'] = amplitude_exp
    params['courbure']['x_plotexp'] = x_plotexp
    params['courbure']['x0'] = x0
    params['courbure']['xf'] = xf
    params['courbure']['t00'] = t00
    params['courbure']['tff'] = tff
    params['courbure']['a'] = a
    params['courbure']['sig'] = sig
    params['courbure']['err'] = err

    dic.save_dico(params, 'E:\\Baptiste\\Resultats_exp\\Courbure\\231124_RLPY\\' + tools.datetimenow() + '_params_courbure' + '.pkl')


    
#%% l (lambda)

# aa = [4,5,6,7]
# disp.figurejolie()
# for ll in range(len(aa)) :
#     a = int(aa[ll])

path_save = 'E:\Baptiste\\Resultats_exp\\Courbure\\Resultats\\20240527_l_lmabda_a6_complet\\'

save = False
display = True

dates = ['231120', '231121', '231122', '231124', '231129', '231129', '231130', '240109', '240115', '240116' ]
nom_exps = ['ECTD9', 'EDTH8', 'NPDP2', 'RLPY3', 'EJCJ6', 'MLO23', 'DMLO1', 'QSC07', 'TNB03', 'CCM03']
t0s = [800, 500, 300, 1000, 600, 350, 400, 280, 500, 600]
x0s = [400, 390, 150, 270, 260, 500, 630, 120, 670, 50]
xfs = [600, 570, 680, 500, 480, 880, 950, 1520, 920, 200]

# x0s = [400, 390, 360, 270, 260, 500, 630, 170, 670, 50]
# xfs = [600, 570, 460, 500, 480, 880, 950, 1420, 920, 200]

a = 6 # mm
liste_exp = [0,1,2,3,4,5,6,7,8,9]

liste_exp = [2]

def filtre_mean(data, n) :
    result = np.zeros(len(data))
    for i in range (len(data)) :
        result[i] = np.mean(data[i - n: i + n])
    return result

def LMH(x,y) :
    half_max = np.nanmax(y) * 1/ 2
    d = np.sign(half_max - y)
    k = np.array(np.where(d == -1))
    
    return x[k[0,-1]] - x[k[0,0]]

nb_t = 2
nb_exp = 10
      
l = np.zeros(nb_exp, dtype = float)
l_exp = np.zeros(nb_exp, dtype = object)
long_onde = np.zeros(nb_exp)
kappa = np.zeros(nb_exp, dtype = object)


for i in liste_exp : #range (len (dates)) :
    
    #import les data laser
    date = dates[i]
    nom_exp = nom_exps[i]
    
    if nom_exp == 'CCM03' :
        a = 2

    dico, params, loc = ip.initialisation(date, nom_exp, exp = True, display = False)
    
    lambda_exp = float(dico[date][nom_exp]['lambda'])
    k_exp = 2 * np.pi / lambda_exp
    omega_exp = 2 * np.pi * float(dico[date][nom_exp]['fexc'])
    H_exp = float(dico[date][nom_exp]['Hw'])
    facq = float(dico[date][nom_exp]['facq'])
    
    folder_results = params['path_images'][:-15] + "resultats"
    name_file = "positionLAS.npy"
    data_originale = np.load(folder_results + "\\" + name_file)

    data_originale = np.rot90(data_originale)
    data_originale = np.flip(data_originale, 0)
    
    data_originale = data_originale - np.nanmean(data_originale) 
    params['debut_las'] = 1
    params['fin_las'] = np.shape(data_originale)[0] - 1
    if nom_exp == 'CCM03' :
        params['fin_las'] = np.shape(data_originale)[0] - 1600
        # a = 3
    
    params['t0'] = 1
    params['tf'] = np.shape(data_originale)[1] - 1
    
    [nx,nt] = data_originale[params['debut_las']:params['fin_las'],params['t0']:params['tf']].shape
    
    data = data_originale[params['debut_las']:params['fin_las'],params['t0']:params['tf']]
    
    #mise à l'échelle en m
    data_m = data *  params['mmparpixely'] / 1000
    data_m = data_m / params['grossissement']
    
    #filtre de la taille 2 * a pour diminuer le bruit
    params['savgol'] = True
    params['ordre_savgol'] = 2
    params['taille_savgol'] = int( a / params['mmparpixel']*2)
    signalsv = np.zeros(data.shape)
    for w in range(0,nt):  
        signalsv[:,w] = savgol_filter(data_m[:,w], params['taille_savgol'],params['ordre_savgol'], mode = 'nearest')
        if np.mod(w,1000)==0:
            print('On processe l image numero: ' + str(w) + ' sur ' + str(nt))
    print('Done !')

    
    if params['savgol'] :
        data = signalsv.copy()
    else :
        data = data_m.copy() - np.nanmean(data_m)

    if display :
        disp.figurejolie()
        # plt.pcolormesh(t, x, data,shading='auto')
        plt.pcolormesh(data,shading='auto')
        plt.xlabel("Temps (frames)")
        plt.ylabel("X (pixel)")
        cbar = plt.colorbar()
        cbar.set_label('Amplitude (m)')
        # plt.clim(-0.01,0.01)
    
    x0 = x0s[i]
    xf = xfs[i]
    t0 = t0s[i]
    
    #On selectionne les 10 temps avec la plus forte amplitude
    peaks_x = []
    peaks_t = []
    
    if nom_exp == 'RLPY3' :
        for bl in range(520,t0) :
            peaks_t.append(bl)
            peaks_x.append(np.nanmax(data[x0:xf, bl]))
    else :
        for bl in range(t0) :
            peaks_t.append(bl)
            peaks_x.append(np.nanmax(data[x0:xf, bl]))
        
    
    xs,ts = tools.sort_listes(peaks_x, peaks_t)
    
    
    ts = np.array(ts, dtype = int)
    liste_t = ts[-nb_t:]
    kappa_max = np.zeros(nb_t)
    l_t = np.zeros(nb_t)
    #dans ces 10 temps, on regarde kappa (x)
    for p in range (len(liste_t)) :
    
        forme = data[x0:xf,liste_t[p]]
        
        x_kappa = np.linspace(a / 1000 , (xf-x0 - a/ params['mmparpixel']) * params['mmparpixel'] / 1000, xf-x0 - int(2 * a/ params['mmparpixel']) )
        popt_x = np.zeros(len(x_kappa))
        b_x = np.zeros(len(x_kappa))
        x_plotexp = np.linspace(0, (xf-x0) * params['mmparpixel'] / 1000, (xf-x0) )
        
        
    
        # fit d'un polynome sur tt les pts et on en extrait la courbure en fct de x
        for j in range (len(forme) - int(2 * a/ params['mmparpixel'])) :
            yfit = forme[a + j: a + j + 2*a]
            xfit = x_plotexp[a +j:a +j + 2*a] 
            popt = np.polyfit(xfit,yfit, 2)
            popt_x[j] = -popt[0] * 2
            b_x[j] = popt[1]
            # yth = np.polyval(popt, xfit)
            # disp.joliplot('x (m)', r'$\kappa$ (m$^{-1}$)', xfit, yth, exp = False, color = 5)
        if nom_exp == 'MLO23' or nom_exp == 'ECTD9' or nom_exp == 'DMLO1' or nom_exp == 'TNB03' or nom_exp == 'NPDP2' or nom_exp == 'EJCJ6' or nom_exp == 'EDTH8' :
            popt_x[np.where(popt_x < 0)] = 0
        
        # popt_x_mean = filtre_mean(popt_x ** 2, int(a / 2 / params['mmparpixel']))
        
        popt_x_mean = filtre_mean(popt_x, int(a / 2 / params['mmparpixel']))

        
        if display :
            disp.figurejolie()
            disp.joliplot(r'x (m)', '$\eta$ (m)', x_plotexp , forme, exp = False, color = 14)
            if save :
                plt.savefig(path_save + tools.datetimenow() + 'forme_t_'+ str(p) + '_' + nom_exp + '.pdf')
                plt.savefig(path_save + tools.datetimenow() + 'forme_t_'+ str(p) + '_' + nom_exp + '.png')
        # if display :
        #     disp.figurejolie()
        #     disp.joliplot(r'x (m)', '$b$ (m-1)', x_kappa , b_x, exp = False, color = 14)
        
        #on trace kappa**2(x) brut et lissé (lissage moyen de taille a)
        if display :
            disp.figurejolie()
            disp.joliplot('x (m)', r'$\kappa^{2}$ (m$^{-1}$)', x_kappa, popt_x_mean, exp = False, color = 8)
            # disp.joliplot('x (m)', r'$\kappa^{2}$ (m$^{-1}$)', x_kappa, popt_x ** 2, exp = False, color = 2)
            disp.joliplot('x (m)', r'$\kappa^{2}$ (m$^{-1}$)', x_kappa, popt_x, exp = False, color = 2)
            plt.grid('on')
            if save :
                plt.savefig(path_save + tools.datetimenow() + 'kappa_carre_x_t_'+ str(p) + '_' + nom_exp + '.pdf')
                plt.savefig(path_save + tools.datetimenow() + 'kappa_carre_x_t_'+ str(p) + '_' + nom_exp + '.png')
        
        #on cherche la largeur à mi hauteur
        l_t[p] = LMH(x_kappa, popt_x_mean )
        kappa_max[p] = np.nanmax(popt_x_mean)
    
    #Tri des résultats : si la taille trouvée fait  lambda /2 ou plus on enleve, c'est qu'il y a dautres maximums sur les cotés et ca fausse la mesure de la lmh  
    l_exp[i] = l_t[np.where(l_t < x_kappa[-1] / 2)]
    l[i] = np.mean(l_exp[i])
    kappa[i] = np.mean(kappa_max[np.where(l_t < x_kappa[-1] / 2)])
    long_onde[i] = lambda_exp
        
    if nom_exp == 'MLO23' :
        if a == 10 :
            l_exp[i] = l_t[np.where(l_t < 0.04)]
            l[i] = np.mean(l_exp[i])
            kappa[i] = np.mean(kappa_max[np.where(l_t < 0.04)])
        if a == 6 :
            l_exp[i] = [l_t[6],l_t[7], l_t[9]]
            l[i] = np.mean(l_exp[i])
            
    if nom_exp == 'EDTH5' :
        l_exp[i] = l_t[np.where(l_t < 0.028)]
        l[i] = np.mean(l_exp[i])
        kappa[i] = np.mean(kappa_max[np.where(l_t < 0.028)])
    if nom_exp == 'EDTH8' :
        if a == 6 :
            '''MESURE A LA MAIN 2024/05/24'''
            l_t[0] = 0.01438
            l_t[1] = 0.01471
            l_t[2] = 0.01785
            l_t[3] = 0.01794
            l_t[6] = 0.01689
            l_t[9] = 0.01543
            l_exp[i] = [l_t[0],l_t[1],l_t[2],l_t[3],l_t[6],l_t[9]]
            l[i] = np.mean(l_exp[i])
        
    if nom_exp == 'QSC07' :
        
        if a == 6 :
            l_exp[i] = [l_t[0],l_t[3],l_t[5]]
            l[i] = np.mean(l_t[np.where(l_t < 0.05)])
            l[i] = np.mean(l_exp[i])
        kappa[i] = np.mean(kappa_max[np.where(l_t < 0.05)])

    if nom_exp == 'CCM03' :
        if a ==2 :
            "MESURE A LA MAIN 2024/05/27"
            l_t[1] = 0.00512
            l_t[3] = 0.00529
            l_t[7] = 0.00504
            l_t[8] = 0.00509
            l_exp[i] = [l_t[1],l_t[3],l_t[7],l_t[8]]
            l[i] = np.mean(l_exp[i])
        if a == 6 :
            l_exp[i] = l_t[np.where(l_t < 0.012)]
            l[i] = np.mean(l_exp[i])
            kappa[i] = np.mean(kappa_max[np.where(l_t < 0.012)])
    if nom_exp == 'ECTD9' :
        if a == 6 :
            '''MESURE A LA MAIN 2024/05/24'''
            l_t[0] = 0.0212
            l_t[1] = 0.0199
            l_t[4] = 0.02127
            l_t[6] = 0.02464
            l_t[8] = 0.02243
            l_t[9] = 0.0221
            l_exp[i] = [l_t[0],l_t[1],l_t[4],l_t[6],l_t[8],l_t[9]]
            l[i] = np.mean(l_exp[i])
    if nom_exp == 'TNB03' :
        if a == 10 :
            l_exp[i] = [l_t[3],l_t[6],l_t[7],l_t[8],l_t[9]]
            l[i] = np.mean(l_exp[i])
        if a == 6 :
            '''MESURE A LA MAIN 2024/05/16'''
            l_t[3] = 0.0265
            l_t[6] = 0.0289
            l_t[7] = 0.0277
            l_t[9] = 0.0294
            l_exp[i] = [l_t[2],l_t[3],l_t[6],l_t[7],l_t[9]]
            l[i] = np.mean(l_exp[i])     
    if nom_exp == 'DMLO1' :
        if a == 6 :
            l_exp[i] = [l_t[2]]
            l[i] = np.mean(l_exp[i])
    if nom_exp == 'EJCJ6' :
        if a == 6 :
            '''MESURE A LA MAIN 2024/05/27'''
            l_t[0] = 0.0199
            l_t[1] = 0.0260
            l_t[2] = 0.0224
            l_t[6] = 0.0162
            l_t[7] = 0.0160
            l_t[9] = 0.0231
            l_exp[i] = [l_t[0],l_t[1],l_t[2],l_t[6],l_t[7],l_t[9]]
            l[i] = np.mean(l_exp[i])     
    if nom_exp == 'RLPY3' :
        if a == 6 :
            '''MESURE A LA MAIN 2024/05/27'''
            l_t[7] = 0.0252
            l_exp[i] = [l_t[0],l_t[7],l_t[8],l_t[9]]
            l[i] = np.mean(l_exp[i])     
    # if nom_exp == 'NPDP2' :
    #     if a == 6 :
    #         l_exp[i] = [l_t[2],l_t[4],l_t[6],l_t[7],l_t[8]]
    #         l[i] = np.mean(l_exp[i])     
            
            
    if display : 
        disp.figurejolie()
        disp.joliplot( r'$\kappa$ (m$^{-1}$)', r'l (m)', kappa_max, l_t, color = 2)
        disp.joliplot( r'$\kappa$ (m$^{-1}$)', r'l (m)', kappa_max[:len(l_exp[i])], l_exp[i], color = 1)
        if save :
            plt.savefig(path_save + tools.datetimenow() + 'l_kappa_' + nom_exp + '.pdf')
            plt.savefig(path_save + tools.datetimenow() + 'l_kappa_png_' + nom_exp + '.png')
    
#%%

# popt_x_mean
# forme
# x_kappa

disp.figurejolie()
disp.joliplot(r'$x$ (mm)', '$\eta$ (mm)', x_plotexp*1000, forme*1000, color = 18, exp = False)




#%% Plot results and save params

disp.figurejolie()


# colors = disp.vcolors( int(ll / len(aa) * 9) )
# plt.scatter(long_onde,l,color=colors)
# plt.xlabel(r'$\lambda$ (m)')
# plt.ylabel('l (m)')
# plt.colorbar()
# plt.clim(np.min(aa), np.max(aa))
disp.joliplot('$\lambda$ (m)', 'l (m)', long_onde, l, color = 17, zeros = True)
for i in range (len(dates)) :
    plt.annotate(nom_exps[i], (long_onde[i], l[i]))

if save :
    plt.savefig(path_save + tools.datetimenow() + 'l_lambda_a6_annotated' + '.pdf')
    plt.savefig(path_save + tools.datetimenow() + 'l_lambda_a6_annotated' + '.png', dpi = 200)
    

disp.figurejolie()


disp.joliplot('$\lambda$ (m)', 'l (m)', long_onde, l, color = 17, zeros = True)

if save :
    plt.savefig(path_save + tools.datetimenow() + 'l_lambda_a6' + '.pdf')
    plt.savefig(path_save + tools.datetimenow() + 'l_lambda_a6' + '.png', dpi = 200)

def fit_x(x,a) :
    return a * x

popt, pcov = curve_fit(fit_x, long_onde, l)

plot_l, plot_lambda = tools.sort_listes(l, long_onde)

plot_lambda = np.append([0], plot_lambda)

plot_lambda = np.linspace(0,0.564, 100)

disp.joliplot('$\lambda$ (m)', 'l (m)', plot_lambda, plot_lambda * popt[0], color = 8, zeros = True, exp = False)


if save :
    params['l'] = {}
    params['l']['dates'] = dates
    params['l']['nom_exps'] = nom_exps
    params['l']['liste_exp'] = liste_exp
    params['l']['nb_t'] = nb_t
    params['l']['x0s'] = x0s
    params['l']['t0s'] = t0s
    params['l']['xfs'] = xfs
    params['l']['a'] = a
    params['l']['l'] = l
    params['l']['kappa'] = kappa
    params['l']['long_onde'] = long_onde
    params['l']['l_exp'] = l_exp
    params['l']['popt'] = popt

    dic.save_dico(params, 'E:\\Baptiste\\Resultats_exp\\Courbure\\Resultats\\20240515_l_lambda_a6\\' + tools.datetimenow() + '_params_l_a6' + '.pkl')

if save :
    plt.savefig(path_save + tools.datetimenow() + 'l_lambda_a6+fit' + '.pdf')
    plt.savefig(path_save + tools.datetimenow() + 'l_lambda_a6+fit' + '.png', dpi = 200)


#%% Comparaison exp : TEMPS


#trace forme du laser superposé à forme de la théorie. On prend les params de l'exp. EN TEMPS


#FORME EXP (t)

x_l2 = 737
x0 = x_l2 - lambda_exp/2 * params['mmparpixel'] / 1000
xf = x_l2 + lambda_exp/2 * params['mmparpixel'] / 1000
t0 = 86
tf = t0 + 13


forme = -data[x_l2, t0:tf]

t_plotexp = np.linspace(0, (tf-t0) / facq,tf-t0)

#FORME THEORIQUE (t)
Amp_exp_eta3 = np.max(forme) * 0.5
Amp_exp_eta2 = np.max(forme) * 1.1
Amp_exp_eta1 = np.max(forme)

nb_periodes = 1
x = lambda_exp / 2
t_exp = np.linspace(0, 2 * nb_periodes * np.pi / omega_exp, 500)
t_exp_eta_3 = np.linspace(0 + 0.142, 2 * nb_periodes * np.pi / omega_exp + 0.142, 500)


eta1 = Amp_exp_eta1 * np.sin(k_exp * x - omega_exp * t_exp)

eta1_2 = Amp_exp_eta2 * np.sin(k_exp * x - omega_exp * t_exp)
terme1_2 = Amp_exp_eta2**2 * k_exp / 4
terme2_2 = 3 * (1 / np.tanh(k_exp * H_exp))**3 - 1/ np.tanh(k_exp * H_exp)
eta2 = terme1_2 * terme2_2 * np.cos(2 * (k_exp * x - omega_exp * t_exp)) #- A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i]) #+ A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i])


eta1_3 = Amp_exp_eta3 * np.sin(k_exp * x - omega_exp * t_exp)
terme1_3 = Amp_exp_eta3**2 * k_exp / 4
terme2_3 = 3 * (1 / np.tanh(k_exp * H_exp))**3 - 1/ np.tanh(k_exp * H_exp)
eta2_3 = terme1_3 * terme2_3 * np.cos(2 * (k_exp * x - omega_exp * t_exp)) #- A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i]) #+ A[i]**2 * k[i] / 2 / np.sinh(2 * k[i] * H[i])
k_exp_prime = k_exp * (1 - Amp_exp_eta3 ** 2 * k_exp ** 2)
eta3 = (1 - 3* Amp_exp_eta3**2 * k_exp**2 / 8) * Amp_exp_eta3 * np.cos(k_exp_prime * x - omega_exp * t_exp_eta_3) + Amp_exp_eta3 **2 * k_exp * np.cos(2 * (k_exp_prime * x - omega_exp * t_exp_eta_3)) + 3 * Amp_exp_eta3**3 * k_exp ** 2 * np.cos(3 * (k_exp_prime * x - omega_exp * t_exp_eta_3)) / 8


disp.figurejolie()
disp.joliplot('t (0 à T)', 'x (m)', t_exp, eta1, exp = False, color = 4, legend = '$\eta_1$')
disp.joliplot('t (0 à T)', 'x (m)', t_exp, eta1_2 + eta2 - np.mean(eta1_2 + eta2), exp = False, color = 2, legend = '$\eta_1 + \eta_2$')
# disp.joliplot('t (0 à T)', 'x (m)', t_exp, eta1_3 + eta2_3 + eta3 - np.mean(eta1_3 + eta2_3 + eta3), exp = False, color = 12, legend = '$\eta_1 + \eta_2 + \eta_3$')
disp.joliplot('t (0 à T)', 'x (m)', t_plotexp, forme, exp = True, color = 8, legend = 'Expérience')

#%% COURBURE SEUIL

save = True
save_path = 'E:\\Baptiste\\Resultats_exp\\Courbure\\Resultats\\20240528_finalito\\'

tableau_1 = pandas.read_csv('E:\\Baptiste\\Resultats_exp\\Tableau_params\\Tableau1_Params_231117_240116\\tableau_1.txt', sep = '\t', header = 0)
tableau_1 = np.asarray(tableau_1)

a30 = False
best_a = True

#Seuils en amplitude en changeant le critère arbitraire pour le définir

seuil_lcrack_lambda = np.array([0.00819783, 0.01068348, 0.00296666, 0.0133115 , 0.0274532 , 0.00532008, 0.02435793, 0.01961565, 0.01898656, 0.00344864])
seuil_lcrack_0 = np.array([0.00687083, 0.0029307 , 0.0026463 , 0.00381471, 0.01692686, 0.00420929, 0.01304042, 0.00983684, 0.00981689, 0.00162004])
seuil_lcrack_0 = np.array([0.00721563, 0.00456913, 0.00267049, 0.0041336 , 0.01692686, 0.00427489, 0.01343315, 0.01010076, 0.01013454, 0.00162004])

exppps = np.array(['ECTDR', 'ETHYR', 'NPDPR', 'RLPYR', 'MLOCR', 'EJCJ1', 'DMLOR', 'QSC0R', 'TNB0R', 'CCM0R'])
titre = 'a_30_lcrack_0_kappa_h_'

if a30 :
    path = ['E:\\Baptiste\\Resultats_exp\\Courbure\\240109_QSC_a30\\' + '20240422_172741_params_courbure.pkl',
            'E:\\Baptiste\\Resultats_exp\\Courbure\\231120_ECTD\\' + '20240422_180550_params_courbure.pkl' ,
            'E:\\Baptiste\\Resultats_exp\\Courbure\\231122_NPDP\\' + '20240419_171920_params_courbure.pkl' , 
            'E:\\Baptiste\\Resultats_exp\\Courbure\\231124_RLPY\\' + '20240422_153853_params_courbure.pkl' ,
            'E:\\Baptiste\\Resultats_exp\\Courbure\\231129_EJCJ_a30\\' + '20240423_123023_params_courbure.pkl' ,
            'E:\\Baptiste\\Resultats_exp\\Courbure\\231129_MLO_a30\\' + '20240422_182628_params_courbure.pkl',
            'E:\\Baptiste\\Resultats_exp\\Courbure\\231130_DML_a30\\' + '20240424_190707_params_courbure.pkl',
            'E:\\Baptiste\\Resultats_exp\\Courbure\\240115_TNB_a30\\' + '20240424_123149_params_courbure.pkl',
            'E:\\Baptiste\\Resultats_exp\\Courbure\\231121_EDTH_a30\\' + '20240424_172527_params_courbure.pkl',
            'E:\\Baptiste\\Resultats_exp\\Courbure\\240116_CCM_ttpts\\' + 'CCM_nomexp_A_kappamax_kappath_a30.txt']

if best_a :
    path = ['E:\\Baptiste\\Resultats_exp\\Courbure\\240109_QSC_a50\\' + '20240422_171834_params_courbure.pkl',
            'E:\\Baptiste\\Resultats_exp\\Courbure\\231120_ECTD\\' + '20240422_180550_params_courbure.pkl' ,
            'E:\\Baptiste\\Resultats_exp\\Courbure\\231122_NPDP\\' + '20240419_171920_params_courbure.pkl' , 
            'E:\\Baptiste\\Resultats_exp\\Courbure\\231124_RLPY\\' + '20240422_153853_params_courbure.pkl' ,
            'E:\\Baptiste\\Resultats_exp\\Courbure\\231129_EJCJ_a30\\' + '20240423_123023_params_courbure.pkl' ,
            'E:\\Baptiste\\Resultats_exp\\Courbure\\231129_MLO_a50\\' + '20240422_182428_params_courbure.pkl',
            'E:\\Baptiste\\Resultats_exp\\Courbure\\231130_DML_a50\\' + '20240424_190359_params_courbure.pkl',
            'E:\\Baptiste\\Resultats_exp\\Courbure\\240115_TNB_a30\\' + '20240424_123149_params_courbure.pkl',
            'E:\\Baptiste\\Resultats_exp\\Courbure\\231121_EDTH_a30\\' + '20240424_172527_params_courbure.pkl',
            'E:\\Baptiste\\Resultats_exp\\Courbure\\240116_CCM_ttpts\\' + 'CCM_nomexp_A_kappamax_kappath_a30.txt']
    
    

index = [7, 0, 2, 3, 4, 5, 6, 8, 1, 9]
a_s = np.zeros(len(path))
k_s = np.zeros(len(path))
lambda_s = np.zeros(len(path))
L_d = np.zeros(len(path))
h = np.zeros(len(path))
D = np.zeros(len(path))
l_s = np.zeros(len(path))
l_crack = np.zeros(len(path))

l = np.array([0.03012138, 0.01904747, 0.01279402, 0.02179882, 0.01237957, 0.04493331, 0.05779294, 0.03804824, 0.02929459, 0.00936516])
l = np.array([0.02192333, 0.0162    , 0.00695415, 0.02272338, 0.0206, 0.03893331, 0.0438713 , 0.03204824, 0.02809459, 0.005135  ])





for i in range (len(path)) :
    nom_expp = tableau_1[index[i],0]
    print(nom_expp)
    if index[i] == 9 :        
        params_k = pandas.read_csv(path[i], header = None, sep = '\t')
        params_k = np.asarray(params_k)
        
        kmin_QSC = params_k[:,2]
        hmin_QSC = params_k[:,1]
        
        params = {}
        params ['nom_exp'] = 'CCM'
        params['date'] = '240116'
       
    elif index[i] == 1 or index[i] == 8 :
        params = dic.open_dico(path[i])
        date = path[i][35:41]
        kmin_QSC = params['courbure']['k_maxmax']
        # hmin_QSC = 
        hmin_QSC = params['courbure']['h_maxmax']
        
        
    else :
        params = dic.open_dico(path[i])
        kmin_QSC = params['courbure']['k_minmin']
        hmin_QSC = params['courbure']['h_mimmin']
        
    
    def fit_2(x, a, b) :
        return a * x**2 + b * x
    
    popt, pcov = curve_fit(fit_2, hmin_QSC, kmin_QSC, p0 = [10000000, 10000], bounds = [[0,0], [100000000, 10000000]]) #np.polyfit(hmin_QSC, kmin_QSC,2)
    
    h_tot = np.linspace(np.min(hmin_QSC), np.max(hmin_QSC), 100)
    
    # disp.figurejolie()
    # disp.joliplot('A (m)', r'$\kappa$ (m$^{-1}$)', hmin_QSC, kmin_QSC, color = 5)
    # disp.joliplot('A (m)', r'$\kappa$ (m$^{-1}$)', h_tot, h_tot**2 * popt[0] + h_tot * popt[1], color = 2, exp = False)
    
    L_d[i] = (tableau_1[index[i],2] / 900 / 9.81) ** 0.25
    D[i] = tableau_1[index[i],2]
    h[i] = tableau_1[index[i],4]
    l_s[i] = l[index[i]]
    l_crack[i] = np.sum(dico[dates[index[i]]][nom_exps[index[i]]]['l_cracks']) / 1000
    
    kloug = np.where(nom_expp == exppps)[0][0]
    a_s[i] = seuil_lcrack_0[kloug]        # L_crack = 0
    # a_s[i] = seuil_lcrack_lambda[kloug]   # Lcrack = lambda
    # a_s[i] = tableau_1[index[i], 9]       # L_crack = lambda / 2 
    
    k_s[i] = a_s[i]**2 * popt[0] + a_s[i] * popt[1]
    lambda_s[i] = tableau_1[index[i], 6]
    plt.plot(a_s[i], k_s[i], 'ko')
    # plt.title(str(tableau_1[i,0]))
    # if save :    
    #     plt.savefig('E:\\Baptiste\\Resultats_exp\\Courbure\\Resultats\\240429_seuil_kappa_all\\' + params['date']  + '_' + params['nom_exp'][:3] + '_' + 'fitcourbure_seuil_' + titre + tools.datetimenow() + '.pdf')#, dpi = 500)
    #     plt.savefig('E:\\Baptiste\\Resultats_exp\\Courbure\\Resultats\\240429_seuil_kappa_all\\' + params['date']  + '_' + params['nom_exp'][:3] + '_' + 'fitcourbure_seuil_png_' + titre + tools.datetimenow() + '.png', dpi = 500)



disp.figurejolie()
# disp.joliplot('$\lambda$ (m)', r'$\kappa$ (m$^{-1}$)', lambda_s, k_s, zeros = True, color = 8)
# disp.joliplot('$\lambda$ (m)', r'$\kappa$ $L_d$ ', lambda_s, k_s * L_d, zeros = True, color = 8)
# disp.joliplot('$\lambda$ (m)', r'$\kappa$ h ', lambda_s, k_s * h, zeros = True, color = 8)
# disp.joliplot('$\lambda$ (m)', r'k $L_d$', lambda_s, L_d * 2 * np.pi / lambda_s, zeros = True, color = 8)

# disp.joliplot('$\lambda$ (m)', r'$\kappa$ (m$^{-1}$)', lambda_s, 1 / (k_s**2*h**2 * lambda_s), zeros = True, color = 8)


disp.joliplot('$\lambda$ (m)', r'$\kappa_c$ (m$^{-1}$)', lambda_s, k_s, zeros = True, color = 8, log = True)
plt.axis('equal')

if save : 
    plt.savefig(save_path + 'kappac_lambda_' + tools.datetimenow() + '.pdf')
    plt.savefig(save_path + 'kappac_lambda_png_' + tools.datetimenow() + '.png', dpi = 500)





# kld = np.pi *2 / lambda_s * L_d
fits.fit_powerlaw(lambda_s, k_s, display = True, xlabel = '$\lambda$ (m)', ylabel = r'$\kappa_c$ (m$^{-1}$)')

if save : 
    plt.savefig(save_path + 'kappac_lambda_powerlaw_' + tools.datetimenow() + '.pdf')
    plt.savefig(save_path + 'kappac_lambda_powerlaw_png_' + tools.datetimenow() + '.png', dpi = 500)

# def fct(x, alpha, gc):
#     return gc / np.sqrt(np.cos(alpha * x))


# # disp.figurejolie()

# plt.plot(kld, fct(kld, 5e0, 10), 'kx')


# fits.fit_powerlaw(lambda_s, k_s * L_d, display = True, xlabel = '$\lambda$ (m)', ylabel = r'$\kappa$ $L_d$')
# fits.fit_powerlaw(lambda_s, k_s * h, display = True, xlabel = '$\lambda$ (m)', ylabel = r'$\kappa$ h')
# if save :
#     plt.savefig('E:\\Baptiste\\Resultats_exp\\Courbure\\Resultats\\240429_seuil_kappa_all\\' + 'Courbure_seuil_lambda_powerlaw_png_'+ titre + tools.datetimenow() + '.png', dpi = 500)
#     plt.savefig('E:\\Baptiste\\Resultats_exp\\Courbure\\Resultats\\240429_seuil_kappa_all\\' + 'Courbure_seuil_lambda_powerlaw_'+ titre + tools.datetimenow() + '.pdf')

disp.figurejolie()
disp.joliplot(r'$\lambda$ (m)', r'l (m)', lambda_s, l_s, color = 17)

if save : 
    plt.savefig(save_path + 'l_lambda_' + tools.datetimenow() + '.pdf')
    plt.savefig(save_path + 'l_lambda_png_' + tools.datetimenow() + '.png', dpi = 500)
    
def fct_x(x, a) :
    return a * x

popt, pcov = curve_fit(fct_x, lambda_s, l_s)

l_fit = lambda_s * popt[0]

disp.joliplot(r'$\lambda$ (m)', r'l (m)', lambda_s, l_fit, color = 8, exp = False)


if save : 
    plt.savefig(save_path + 'l_lambda_fit_' + tools.datetimenow() + '.pdf')
    plt.savefig(save_path + 'l_lambda_fit_png_' + tools.datetimenow() + '.png', dpi = 500)


disp.figurejolie()
disp.joliplot(r'$A_c$ (m)', r'$\kappa_c$ (m$^{-1}$)', a_s, k_s, color = 17, log = True)

if save : 
    plt.savefig(save_path + 'kappac_ac_' + tools.datetimenow() + '.pdf')
    plt.savefig(save_path + 'kappac_ac_png_' + tools.datetimenow() + '.png', dpi = 500)

fits.fit_powerlaw(a_s, k_s, display = True, xlabel = r'$A_c$ (m)', ylabel =  r'$\kappa_c$ (m$^{-1}$)')

if save : 
    plt.savefig(save_path + 'kappac_ac_powerlaw_' + tools.datetimenow() + '.pdf')
    plt.savefig(save_path + 'kappac_ac_powerlaw_png_' + tools.datetimenow() + '.png', dpi = 500)

l_crack[5] = 0.35

disp.figurejolie()
disp.joliplot(r'$\lambda$ (m)', r'$l_{crack}$ (m)', lambda_s, l_crack, color = 17)

# disp.figurejolie()
# disp.joliplot(r'$\lambda$ (m)', r'$\kappa_c^{2}lD / l_{crack}$ (J)', lambda_s, l_s * k_s**2 * np.mean(D) / l_crack, log = False, color = 2, zeros = True)

fits.fit_powerlaw(lambda_s, l_s * k_s**2 * np.mean(D) / np.mean(h), display = True, xlabel = '$\lambda$ (m)', ylabel = r'$\kappa_c^{2}lD / h$')

#%%Graphs finaux
save = False
#l_s redefini avec le fit linéaire
disp.figurejolie()
def fct_x(x, a) :
    return a * x

popt, pcov = curve_fit(fct_x, lambda_s, l_s)

l_fit = lambda_s * popt[0]


disp.joliplot(r'$\lambda$ (m)', r'$\kappa_c^{2}lD / h$ (J.m$^{-2}$)', lambda_s, l_s * k_s**2 * np.mean(D) /np.mean(h), log = False, color = 18, zeros = True)

plt.ylim([0,0.5])
plt.xlim([0,0.7])

if save : 
    plt.savefig(save_path + 'Gc_lambda_' + tools.datetimenow() + '.pdf')
    plt.savefig(save_path + 'Gc_lambda_png_' + tools.datetimenow() + '.png', dpi = 500)

disp.figurejolie()
disp.joliplot(r'$\lambda$ (m)', r'$\kappa_c^{2}lD / h$ (J.m$^{-2}$)', lambda_s, l_s * k_s**2 * np.mean(D) /np.mean(h), log = False, color = 18, zeros = True)

plt.ylim([0,0.5])
plt.xlim([0,0.7])

stdd = np.std(l_s * k_s**2 * np.mean(D) /np.mean(h))
yy = np.linspace(0,0,100) + np.mean(l_s * k_s**2 * np.mean(D) /np.mean(h))

xx = np.linspace(0,0.7,100)

yy_sp = np.linspace(0,0,100) + np.mean(l_s * k_s**2 * np.mean(D) /np.mean(h)) + stdd

yy_sm = np.linspace(0,0,100) + np.mean(l_s * k_s**2 * np.mean(D) /np.mean(h)) - stdd
plt.plot(xx,yy, 'k-')
plt.plot(xx,yy_sp, 'k--')
plt.plot(xx,yy_sm, 'k--')

if save : 
    plt.savefig(save_path + 'Gc_lambda_moy_' + tools.datetimenow() + '.pdf')
    plt.savefig(save_path + 'Gc_lambda_moy_png_' + tools.datetimenow() + '.png', dpi = 500)
    
disp.figurejolie()
def fct_x(x, a) :
    return a * x

popt, pcov = curve_fit(fct_x, lambda_s, l_s)

l_fit = lambda_s * popt[0]


disp.joliplot(r'$\lambda$ (m)', r'$\kappa_c^{2}lD / h$ (J.m$^{-2}$)', lambda_s, l_fit * k_s**2 * np.mean(D) /np.mean(h), log = False, color = 18, zeros = True)

plt.ylim([0,0.5])
plt.xlim([0,0.7])

if save : 
    plt.savefig(save_path + 'Gc_lambda_lfit_' + tools.datetimenow() + '.pdf')
    plt.savefig(save_path + 'Gc_lambda_lfit_png_' + tools.datetimenow() + '.png', dpi = 500)

disp.figurejolie()
disp.joliplot(r'$\lambda$ (m)', r'$\kappa_c^{2}lD / h$ (J.m$^{-2}$)', lambda_s, l_fit * k_s**2 * np.mean(D) /np.mean(h), log = False, color = 18, zeros = True)

plt.ylim([0,0.5])
plt.xlim([0,0.7])

stdd = np.std(l_fit * k_s**2 * np.mean(D) /np.mean(h))
yy = np.linspace(0,0,100) + np.mean(l_fit * k_s**2 * np.mean(D) /np.mean(h))

xx = np.linspace(0,0.7,100)

yy_sp = np.linspace(0,0,100) + np.mean(l_fit * k_s**2 * np.mean(D) /np.mean(h)) + stdd

yy_sm = np.linspace(0,0,100) + np.mean(l_fit * k_s**2 * np.mean(D) /np.mean(h)) - stdd
plt.plot(xx,yy, 'k-')
plt.plot(xx,yy_sp, 'k--')
plt.plot(xx,yy_sm, 'k--')

if save : 
    plt.savefig(save_path + 'Gc_lambda_lfit_moy_' + tools.datetimenow() + '.pdf')
    plt.savefig(save_path + 'Gc_lambda_lfit_moy_png_' + tools.datetimenow() + '.png', dpi = 500)
    

if save :
    params_tot = {}
    params_tot['L_d'] = L_d
    params_tot['D'] = D
    params_tot['l_s'] = l_s
    params_tot['a_s'] = a_s
    params_tot['k_s'] = k_s
    params_tot['h'] = h
    params_tot['lambda_s'] = lambda_s
    params_tot['l_crack'] = l_crack
    params_tot['dates'] = dates
    params_tot['path'] = path
    params_tot['nom_exps'] = nom_exps
    params_tot['nom_exps'] = nom_exps
    params_tot['nom_exps'] = nom_exps
    dic.save_dico(params_tot, path = save_path + 'params_tot.pkl')
    



