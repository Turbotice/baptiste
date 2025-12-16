# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 15:11:57 2025

@author: Banquise
"""

import baptiste.image_processing.cracks as cr

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

import baptiste.main.Show_image_georectifiée as si


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



X = data['X'] #m 

Y = data['Y'] #m 

T = data["t"] #s 

facq = data['SCALE']['facq_x']
mparpixel = 1/data['SCALE']['facq_x']


nx = X.shape[0]
ny = Y.shape[1]
x = X[0,:]

y = Y[:,0]
x_pix = np.linspace(0, nx-1, nx, dtype=int)
y_pix = np.linspace(0, ny-1, ny, dtype=int)

x= x_pix * mparpixel
y= y_pix * mparpixel

#%%

# t = 1800
# si.plot_img(X, Y, t, renvoyer = True)

path_img = 'Y:\Banquise\\BicWin2024\\PIV_images\\0211\\Fulmar\\im_6086.tiff'

from skimage import io, color, morphology

# Charger l'image
image = io.imread(path_img)

# Convertir en niveaux de gris
gray = color.rgb2gray(image)
gray = (gray * 255).astype(np.uint8)

h, w = gray.shape
threshold_map = np.zeros((h, w), dtype=np.uint8)
min_thresh = 50 
max_thresh = 65

# Créer une map de seuils qui varie selon x et y
y_coords = np.linspace(0, 1, h).reshape(h, 1)
x_coords = np.linspace(0, 1, w).reshape(1, w)
threshold_map = min_thresh + (max_thresh - min_thresh) * (x_coords + y_coords) / 2
threshold_map = threshold_map.astype(np.uint8)
# Binarisation avec seuil variable
binary = gray < threshold_map


# Binariser (seuil automatique)
# binary = gray < 0.22 

# Squelettisation
skeleton = morphology.skeletonize(binary)

# Afficher les résultats
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(image)
ax[0].set_title("Image originale")

ax[1].imshow(binary, cmap="gray")
ax[1].set_title("Binaire")

ax[2].imshow(skeleton, cmap="gray")
ax[2].set_title("Squelette")

for a in ax:
    a.axis("off")

plt.show()

coords = np.column_stack(np.where(skeleton))

skeleton_sans_nodes = cr.RemoveNodes(skeleton)

skeleton_sans_nodes = (skeleton_sans_nodes).astype(np.uint8) * 255
skeleton = (skeleton).astype(np.uint8) * 255

liste_arg, use, NX, NY = cr.cracks(skeleton_sans_nodes, 1)

disp.figurejolie()
angles, weights = cr.analyze_cracks(liste_arg, use, plot = True, original = None)

disp.figurejolie()

plt.imshow(skeleton_sans_nodes)


#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

path_img = 'Y:\Banquise\\BicWin2024\\PIV_images\\0211\\Fulmar\\im_6086.tiff'

from skimage import io, color, morphology

# === 1. Charger et squelettiser ===
image = io.imread(path_img)

# Convertir en niveaux de gris
gray = color.rgb2gray(image)
gray = (gray * 255).astype(np.uint8)

h, w = gray.shape
threshold_map = np.zeros((h, w), dtype=np.uint8)
min_thresh = 50 
max_thresh = 65

mparpixel = 0.046901
x_pix = np.linspace(0, h-1, h, dtype=int)
y_pix = np.linspace(0, w-1, w, dtype=int)

x= x_pix * mparpixel
y= y_pix * mparpixel

# Créer une map de seuils qui varie selon x et y
y_coords = np.linspace(0, 1, h).reshape(h, 1)
x_coords = np.linspace(0, 1, w).reshape(1, w)
threshold_map = min_thresh + (max_thresh - min_thresh) * (x_coords + y_coords) / 2
threshold_map = threshold_map.astype(np.uint8)
# Binarisation avec seuil variable
binary = gray < threshold_map


skeleton = skeletonize(binary)

# === 2. Trouver les points clés (extrémités et jonctions) ===
kernel = np.ones((3, 3), np.uint8)
neighbors = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
endpoints = np.logical_and(skeleton, neighbors == 2)
junctions = np.logical_and(skeleton, neighbors >= 4)

# === 3. Lister les segments via les composantes connectées ===
skeleton_clean = skeleton.copy()
skeleton_clean[junctions] = 0  # enlever les jonctions pour séparer les branches
num_labels, labels = cv2.connectedComponents(skeleton_clean.astype(np.uint8))

segments = [np.column_stack(np.where(labels == i)) for i in range(1, num_labels)]

# === 4. Calcul longueur et angle, avec filtrage par longueur minimale ===
length_min = 20
results = []

for seg in segments:
    if len(seg) < 2:
        continue
    # longueur
    diffs = np.diff(seg, axis=0)
    length = np.sum(np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2))
    if length < length_min:
        continue
    # angle
    y1, x1 = seg[0]
    y2, x2 = seg[-1]
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    results.append((seg, length, angle))

print(f"{len(results)} segments > {length_min}px")

# === 5. Visualisation ===
disp.figurejolie(width = 8.6 * nx / nx * 4, height= 8.6 * ny / nx * 4)
plt.pcolormesh(y,x,np.flip(skeleton,1), cmap='ma_cm')
# plt.title("Segments du squelette (filtrés)")
# plt.axis('off')

for seg, length, angle in results:
    plt.plot( (nx-seg[:,1]) * mparpixel, seg[:,0] * mparpixel, linewidth=2)
    # cy, cx = np.mean(seg, axis=0)
    # plt.text(cx, cy, f"{angle:.1f}°", color='red', fontsize=8, ha='center')


plt.show()

moky = [0,20,40,60,80,100]
mokx = [0,25,50,75,100,125,150,175]

uyuy = np.zeros((len(mokx), len(moky)))

disp.figurejolie(width = 8.6 * nx / nx * 1.5, height= 8.6 * ny / nx * 1.5)
plt.pcolormesh(mokx,moky,np.rot90(uyuy))
 
#%%

angles_deg = []
length_frac = []
x_frac = []
y_frac = []
for seg, length, angle in results :
    angles_deg.append(angle)
    length_frac.append(length)
    x_frac.append(seg[0,0])
    y_frac.append(seg[0,1])

x_frac = np.array(x_frac)
y_frac = np.array(y_frac)


disp.figurejolie()
plt.hist(angles_deg, bins = 24, weights=length_frac)

disp.figurejolie()
plt.scatter(angles_deg, length_frac)

disp.figurejolie()
plt.scatter(x_frac, angles_deg)

disp.figurejolie()
plt.scatter(y_frac, angles_deg)

disp.figurejolie()
plt.scatter(x_frac**2 + y_frac**2, angles_deg, alpha= 0.3)

#%%
import numpy as np
import matplotlib.pyplot as plt

select_angle = True

if select_angle :
    theta_max = (1.1501329776618616 + -0.08955351 + np.pi/2) * 180 / np.pi
    theta_min = (0.3217505543966422 + -0.08955351 + np.pi/2) * 180 / np.pi

zone_size = 200
h, w = skeleton.shape
n_rows = h // zone_size
n_cols = w // zone_size

# Créer une liste pour accumuler angles et poids (longueur) par zone
zone_angles = [[[] for _ in range(n_cols)] for _ in range(n_rows)]
zone_weights = [[[] for _ in range(n_cols)] for _ in range(n_rows)]

# Parcourir tous les segments
for seg, length, angle in results:
    ys, xs = seg[:, 0], seg[:, 1]
    for y, x in zip(ys, xs):
        i = y // zone_size
        j = x // zone_size
        if i < n_rows and j < n_cols:
            if select_angle :
                if angle < theta_max and angle > theta_min :
                    zone_angles[i][j].append(angle)
                    zone_weights[i][j].append(length)  # pondération par longueur du segment
            else : 
                zone_angles[i][j].append(angle)
      
# Calcul de l'angle moyen pondéré pour chaque zone
angle_grid = np.full((n_rows, n_cols), np.nan)

for i in range(n_rows):
    for j in range(n_cols):
        if zone_angles[i][j]:
            angles_rad = np.radians(zone_angles[i][j])
            weights = np.array(zone_weights[i][j])
            # Moyenne circulaire pondérée
            mean_sin = np.sum(np.sin(angles_rad) * weights) / np.sum(weights)
            mean_cos = np.sum(np.cos(angles_rad) * weights) / np.sum(weights)
            mean_angle = np.arctan2(mean_sin, mean_cos)
            angle_grid[i, j] = np.degrees(mean_angle)
            

angle_grid = angle_grid - 90

disp.figurejolie(width = n_cols * 8.6 / 12*2, height=n_rows* 8.6 / 12*2)
plt.pcolormesh(np.flip(angle_grid,1), cmap='ma_cm')#, vmin=-90, vmax=90)
plt.colorbar(label="Angle moyen (°)")
# plt.clim(theta_min-90, theta_max-90)

# k_grid = 9.81 / 8.19**2 / np.cos(angle_grid / 180 * np.pi)**2
# lambda_grid =2 * np.pi / k_grid 

# disp.figurejolie(width = n_cols * 8.6 / 12*2, height=n_rows* 8.6 / 12*2)
# plt.pcolormesh(np.flip(k_grid,1), cmap='ma_cm')#, vmin=-90, vmax=90)
# plt.colorbar(label="k")
# # plt.clim(theta_min, theta_max)
# disp.figurejolie(width = n_cols * 8.6 / 12*2, height=n_rows* 8.6 / 12*2)
# plt.pcolormesh(np.flip(lambda_grid,1), cmap='ma_cm')#, vmin=-90, vmax=90)
# plt.colorbar(label="lambda")

#%%Verif angle


cx, cy = 1000, 1000   # centre du trait
angle = theta_min         # degrés
length = 500         # longueur du trait

# Calcul des extrémités
theta = np.radians(angle)
dx = (length / 2) * np.cos(theta)
dy = (length / 2) * np.sin(theta)

x_start = cx - dx
x_end   = cx + dx
y_start = cy - dy
y_end   = cy + dy

# Affichage
# plt.imshow(np.zeros((200,200)), cmap='gray')  # image fictive
plt.plot([x_start, x_end], [y_start, y_end], 'r-', linewidth=2)
# plt.scatter(cx, cy, color='blue')  # centre
plt.show()

cx, cy = 1500, 1000   # centre du trait
angle = theta_max         # degrés
length = 500         # longueur du trait

# Calcul des extrémités
theta = np.radians(angle)
dx = (length / 2) * np.cos(theta)
dy = (length / 2) * np.sin(theta)

x_start = cx - dx
x_end   = cx + dx
y_start = cy - dy
y_end   = cy + dy

# Affichage
# plt.imshow(np.zeros((200,200)), cmap='gray')  # image fictive
plt.plot([x_start, x_end], [y_start, y_end], 'b-', linewidth=2)
# plt.scatter(cx, cy, color='blue')  # centre
plt.show()
#%%

image = io.imread(path_img)
gray = color.rgb2gray(image)
gray = (gray * 255).astype(np.uint8)


# Créer une map de seuils qui varie verticalement
threshold_map = np.zeros((h, w), dtype=np.uint8)
min_thresh = 45
max_thresh = 70

# Créer une map de seuils qui varie selon x et y
y_coords = np.linspace(0, 1, h).reshape(h, 1)
x_coords = np.linspace(0, 1, w).reshape(1, w)
threshold_map = min_thresh + (max_thresh - min_thresh) * (x_coords + y_coords) / 2
threshold_map = threshold_map.astype(np.uint8)
# Binarisation avec seuil variable
binary = (gray < threshold_map).astype(np.uint8) * 255
plt.figure()
plt.imshow(binary)






