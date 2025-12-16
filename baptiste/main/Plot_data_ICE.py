# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 20:02:04 2025

@author: Banquise
"""

import pandas as pd
import gzip

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
import baptiste.tools.tools as tools

path = 'Y:\Banquise\\Baptiste\\Manuscrit_these\\Introduction\\Figure\\Genral\\'

# --- Charger PIOMAS volume mensuel ---
piomas_df = pd.read_csv(path + "PIOMAS.2sst.monthly.Current.v2.1.txt", 
                        delim_whitespace=True, header=None)
piomas_df.columns = ["year"] + [f"{m:02d}" for m in range(1, 13)]
piomas_long = piomas_df.melt(id_vars=["year"], var_name="month", value_name="piomas_volume_1000km3")
piomas_long["month"] = piomas_long["month"].astype(int)

# --- Charger PIOMAS épaisseur quotidienne ---
with gzip.open(path + "PIOMAS.thick.daily.1979.2025.Current.v2.1.dat.gz", "rt") as f:
    thick_df = pd.read_csv(f, delim_whitespace=True, header=None,
                           names=["year", "month", "day", "thickness_m"])
thick_monthly = thick_df.groupby(["year", "month"], as_index=False)["thickness_m"].mean()

# piomas_combined = pd.merge(piomas_long, thick_monthly, on=["year", "month"], how="inner")

# --- Charger NSIDC (extension mensuelle) ---
nsidc_list = []
for m in range(1, 13):
    fname = path +  f"N_{m:02d}_extent_v4.0.csv"
    df = pd.read_csv(fname, header=1, names=["year", "month", "extent_million_km2", "missing", "source"])
    df["month"] = m
    # df['year'] = df.index
    nsidc_list.append(df[["year", "month", "missing", "source"]])
nsidc_df = pd.concat(nsidc_list, ignore_index=True)

# --- Fusionner ---
# merged = pd.merge(piomas_combined, nsidc_df, on=["year", "month"], how="inner")
# merged = merged[(merged["year"] >= 1979) & (merged["year"] <= 2024)]

# --- Exporter ---
# merged.to_csv("arctic_seaice_extent_volume_thickness_1979_2024.csv", index=False)
print("CSV généré : arctic_seaice_extent_volume_thickness_1979_2024.csv")


     
#%% Volume (month)

piomas_long = np.array(piomas_long) #volume

nb_year = int(np.max(piomas_long[:,0]) - np.min(piomas_long[:,0]))
y0 = np.min(piomas_long[:,0])

year, month, volume = tools.sort_listes(piomas_long[:, 0], piomas_long[:, 1], piomas_long[:, 2], nb_list = 3)
disp.figurejolie(width = 8.6 * 7 / 5, height=8.6 * 2 / 4)


mois = ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
        "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"]

mois = ["Jan", "Fév", "Mars", "Avr", "Mai", "Juin",
        "Jui", "Août", "Sep", "Oct", "Noc", "Déc"]
# plt.xlabel(mois)

for i in range(int(nb_year)):

    
    disp.joliplot('Mois', '$V$ (km$^3$)', mois, volume[i*12:(i+1)*12], cm=int(9 - 10 * i / nb_year), exp = False, zeros  = True)




df = pd.DataFrame({"year": year, "month": month})

# Transformer en date (on choisit le premier jour du mois)
df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=12))

disp.figurejolie(width = 8.6 * 7 / 5, height=8.6 * 2 / 4)
blbl = [[np.max(year), np.max(year)],[np.max(year), np.min(year)]]

plt.pcolormesh(blbl, cmap = 'ma_cm_r')
plt.colorbar()


# disp.figurejolie()
    
# disp.joliplot('Date', '$h$ (m)', df["date"], volume, exp = False, cm = 5)
    
    
#%%Thickness (month)

thick_df = np.array(thick_df)

year = np.asarray(thick_df[1:,0], dtype = int)
day = thick_df[1:,1]
thickness = np.asarray(thick_df[1:,2], dtype = float)

nb_year = int(np.max(year) - np.min(year))
y0 = np.min(year)


dates = pd.date_range(start="1979-01-01", end="2025-08-19", freq="D")
data = pd.Series(thickness, index=dates)

# Passage en moyenne mensuelle
data_mensuelle = data.resample("M").mean()
data_10ans = data_mensuelle.resample("10A").mean()
# Affichage


    
    


disp.figurejolie(width = 8.6 * 4 / 5)

for i in range(int(nb_year)):

    
    disp.joliplot('Date', '$h$ (m)', mois, data_mensuelle.values[i*12:(i+1)*12], cm = int(9-10* i / nb_year), exp = False, zeros = True)


# disp.figurejolie(width = 8.6 * 7 / 5, height=8.6 * 2 / 4)
disp.figurejolie(width = 8.6 * 4 / 5)    

disp.joliplot('Date', '$h$ (m)', data_mensuelle.index, data_mensuelle.values, exp = False, cm = 5)# color = disp.mcolors(int(10* i / nb_year)))

plt.ylim( 0, np.nanmax(data_mensuelle.values)* 1.05)
    
#%%Surface (month)

year = np.array([])
extent = np.array([])
area = np.array([])
month = np.array([])


for i in range (12) :
    extent = np.append(extent, np.array(nsidc_list[i]['missing']))
    area = np.append(area, np.array(nsidc_list[i]['source']))
    year = np.append(year, np.array(nsidc_list[i]['year'].index))
    month = np.append(month, np.zeros(len(np.array(nsidc_list[i]['year'].index))) + i)


area[area < 0] = np.nan
extent[extent < 0] = np.nan

nb_year = int(np.max(year) - np.min(year))
y0 = np.min(year)




dates = np.array([f"{int(y)}-{int(m):02d}-01" for y, m in zip(year, month + 1)], dtype="datetime64[D]")
indices = np.argsort(dates)

# Trier les deux arrays
dates_sorted = dates[indices]
area_sorted = area[indices]


extent_sorted = extent[indices]

disp.figurejolie(width = 8.6 * 4 / 5)
# disp.figurejolie(width = 8.6 * 7 / 5, height=8.6 * 2 / 4)
    
disp.joliplot('Date', '$h$ (m)', dates_sorted, extent_sorted, exp = False, cm = 5)# color = disp.mcolors(int(10* i / nb_year)))

    
from scipy.signal import find_peaks

uuu = find_peaks(extent_sorted)[0]

www = find_peaks(-extent_sorted)[0]




jours = dates_sorted.astype('timedelta64[D]').astype(int)

a1, b1 = np.polyfit(jours[www], extent_sorted[www], 1)

a2, b2 = np.polyfit(jours[uuu], extent_sorted[uuu], 1)


disp.joliplot('Date', '$h$ (m)', dates_sorted, jours * a1 + b1, exp = False, color = 8)
disp.joliplot('Date', '$h$ (m)', dates_sorted, jours * a2 + b2, exp = False, color = 8)



disp.joliplot('Date', '$h$ (m)', dates_sorted[uuu], extent_sorted[uuu], exp = False, cm = 2)


disp.joliplot('Date', '$S$ (km$^2$)', dates_sorted[www], extent_sorted[www], exp = False, cm = 2)

plt.ylim( 0, np.nanmax(extent)* 1.05)


pourcent_an_été = np.mean(1-(jours[uuu[1:]] * a1 + b1) / (jours[uuu[:-1]] * a1 + b1))


pourcent_an_hiver = np.mean(1-(jours[www[1:]] * a2 + b2) / (jours[www[:-1]] * a2 + b2))




