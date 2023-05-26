# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 16:48:53 2023

@author: Banquise

Sensé corriger la parallaxe, mais marche pas
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2


focal_length = 0.00000001
baseline = 2
image = cv2.imread("D:\Banquise\Baptiste\Resultats_video\d230105\d230105_MPMF3_LAS_26sur082_facq101Hz_texp8960us_Tmot003_Vmot200_Hw11cm_tacq020s\image_sequence\\Basler_a2A1920-160ucBAS__40232066__20230105_174557686_00001.tiff",cv2.IMREAD_GRAYSCALE)

plt.figure()
plt.imshow(image)

"""
Méthode 1
"""
image = image + 1

# def correct_parallax(image, baseline, focal_length):
#     # Calculer la parallaxe pour chaque pixel de l'image
#     parallax = baseline * focal_length / image

#     # Créer une image de même taille que l'image d'origine
#     corrected_image = np.zeros_like(image)

#     # Pour chaque ligne et colonne de l'image
#     for y in range(image.shape[0]):
#         for x in range(image.shape[1]):
#             # Calculez la nouvelle position du pixel en utilisant la parallaxe
#             corrected_x = x - parallax[y, x]
#             # Copiez le pixel de l'image d'origine à la nouvelle position dans l'image corrigée
#             corrected_image[y, x] = image[y, int(corrected_x)]

#     return corrected_image


    # Calculer la parallaxe pour chaque pixel de l'image
parallax = baseline * focal_length / image

    # Créer une image de même taille que l'image d'origine
corrected_image = np.zeros_like(image)

    # Pour chaque ligne et colonne de l'image
for y in range(image.shape[0]):
    for x in range(image.shape[1]):
            # Calculez la nouvelle position du pixel en utilisant la parallaxe
        corrected_x = x - parallax[y, x]
            # Copiez le pixel de l'image d'origine à la nouvelle position dans l'image corrigée
        corrected_image[y, x] = image[y, int(corrected_x)]

 

plt.figure()
plt.imshow(corrected_image)
plt.title("image corrige")

"""
Méthode 2
"""
# Convertir l'image en niveaux de gris
gray = image#cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Détecter les lignes verticales dans l'image
edges = cv2.Canny(gray, 10, 210, apertureSize=3)
lines = cv2.HoughLines(edges, 3, np.pi/180, 200)

# Initialiser les coefficients k1 et k2 à 0
k1 = 0
k2 = 0
plt.figure()


# Pour chaque ligne détectée
for line in lines:
    rho, theta = line[0]
    # Si la ligne est verticale (à +/- 10 degrés de la verticale)
    if abs(theta - np.pi/2) < np.pi/18:
        # Calculer les coordonnées des extrémités de la ligne
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        # Calculer la distance entre les deux extrémités de la ligne
        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)
        # Calculer la déformation de la ligne
        k1 += (y1 - y2) / distance
        k2 += (x1 - x2) / distance

# Calculer les coefficients k1 et k2 moyens
k1 /= len(lines)
k2 /= len(lines)

print("k1 =", k1)
print("k2 =", k2)

# k1 = 0.001
# k2 = 0.005

plt.imshow(image)


"""
Méthode 3
"""
# Créer une image de même taille que l'image d'origine
corrected_image_vert = np.zeros((image.shape[0]+400000, image.shape[1]))
image_new = np.zeros((image.shape[0]+400000, image.shape[1]))
image_new[200000:212000,:] = image

# Pour chaque ligne et colonne de l'image
for y in range(image.shape[0]):
    for x in range(image.shape[1]):
        # Calculez la nouvelle position du pixel en utilisant la formule de correction de la distorsion verticale optique
        corrected_y = y * (1 + k1 * y**2 + k2 * y**4)
        # Copiez le pixel de l'image d'origine à la nouvelle position dans l'image corrigée
        # if corrected_y <= image.shape[0] :
        corrected_image_vert[y, x] = image_new[int(corrected_y), x]



plt.figure()
plt.imshow(corrected_image_vert)
plt.title('image corrigée verticalement')