




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




import pandas as pd, numpy as np, re, matplotlib.pyplot as plt
path = "Y:\Banquise\\Baptiste\\Soutenance/seuil.txt"

df = pd.read_csv(path, sep=r'\t+', engine='python', decimal=',', encoding='utf-8')

# Normalize and find columns as before
def normalize(s):
    return re.sub(r'[^0-9a-z]', '', s.lower())

cols = {normalize(c): c for c in df.columns}
def find_column(keywords):
    nk = ''.join(re.sub(r'[^0-9a-z]', '', kw.lower()) for kw in keywords)
    for n,c in cols.items():
        if all(k in n for k in [re.sub(r'[^0-9a-z]','',kw.lower()) for kw in keywords]):
            return c
    for n,c in cols.items():
        if nk in n:
            return c
    return None

col_lambda = find_column(['lambda','glace']) or find_column(['lambda'])
col_E = find_column(['E','Pa']) or find_column(['E'])
col_errE = find_column(['err_E','Pa']) or find_column(['err_E'])
col_h = find_column(['h','m']) or find_column(['h'])
col_curv = find_column(['courbure']) or find_column(['kappa','curv'])

# Replace comma decimal with dot for numeric-like strings, but keep thousands? Here no thousands sep.
for col in [col_lambda, col_E, col_h, col_curv,col_errE ]:
    if col is not None:
        # replace comma with dot and remove spaces
        df[col] = df[col].astype(str).str.replace(' ', '').str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

print("Nombre de lignes total:", len(df))
print("Nombres non-NA pour colonnes choisies:")
for name in [col_lambda, col_E, col_h, col_curv]:
    print(name, df[name].notna().sum())

df['h_over_lambda'] = df[col_h] / df[col_lambda]
df['E_h_kappa2'] = df[col_E] * df[col_h] * (df[col_curv] ** 2)

plot_df = df.dropna(subset=['h_over_lambda','E_h_kappa2']).copy()
print("\nNombre de points valides pour tracé:", len(plot_df))

print("\nAperçu calculs :")
print(plot_df[[col_lambda, col_E, col_h, col_curv, 'h_over_lambda','E_h_kappa2']].head().to_string())



E = np.array(df['E (Pa)'])
h = np.array(df['h (m)'])
kappa = np.array(df['Courbure (m-1)'])
Ld = np.array(df['ld (m)'])
lambbda = np.array(df['lambda glace (m)'])
err_E = np.array(df['err_E'], dtype = float)
err_h = np.array(df['err_h'])
err_lambbda = np.array(df['err_lambda'])
err_kappa = np.array(df['err_kappa'])

sigma = np.array(df['sigma'])
err_sigma =np.array(df['err_sigma'])



#%%

Ehkappa_gre = []
hsurlambda_gre = []
Autre = []
hsurlambda_autre = []

for i in range (len(df['Jeu de donnée'])) :
    if df['Jeu de donnée'][i] == 'Gre25' :
        Ehkappa_gre.append(df['E_h_kappa2'][i])
        hsurlambda_gre.append(df['h_over_lambda'][i])
    else :
        Autre.append(df['E_h_kappa2'][i])
        hsurlambda_autre.append(df['h_over_lambda'][i])

plt.figure()
disp.joliplot('h sur lambda', 'Eh kappa2', hsurlambda_gre, Ehkappa_gre, cm = 2, log = True, legend = 'Gre')
disp.joliplot('h sur lambda', 'Eh kappa2', hsurlambda_autre, Autre, cm = 4, log = True, legend = 'Autre')


#%% Resultat plot

err_tot_y = np.sqrt( (err_E / E)**2 +  (err_h / h)**2 + (err_kappa / kappa)**2 + (err_sigma / sigma)**2 )

err_tot_x = np.sqrt( (err_h / h)**2 +  (err_lambbda / lambbda)**2)

Ibr = E * h  * kappa / sigma / (2 * (np.pi**2))


disp.figurejolie(width = 8.6* 4 / 5)
disp.joliplot(r'h / $\lambda$', '$I_{br}$', h / lambbda, Ibr, cm = 2, log = True)
# for i in range (len(df['Jeu de donnée'] )):
#     plt.annotate(df['Jeu de donnée'][i] , (h[i] / lambbda[i], E[i] * h[i]  * kappa[i] / sigma[i] / 2 / np.pi**2))

plt.errorbar(h / lambbda, Ibr, yerr=  err_tot_y *  Ibr, xerr = err_tot_x * h / lambbda, fmt = 'kx', ms = 0)


fits.fit_powerlaw(h / lambbda, Ibr, display=True)


#%%


disp.figurejolie(width = 8.6* 4 / 5)
disp.joliplot(r'h / $\lambda$', '$I_{br}$', lambbda, h, cm = 2, log = True)

#%% Mon seuil

D = E * h**3 / 10


fits.fit_powerlaw(h / lambbda, kappa / sigma / h**0.5 * D**0.5, display = True)













