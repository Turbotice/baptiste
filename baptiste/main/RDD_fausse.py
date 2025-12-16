
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




# k = np.logspace(-1,2, 1000)


# Dsurrho = 10
# h = 0.1
# rhosurrho = 1
# g = 9.81

# def fct(k, Dsurrho, h, rhosurrho, g) :
#     return np.sqrt( (9.81 * k + Dsurrho * k **5) / ( 1 + rhosurrho * h * k) )

# disp.figurejolie()
# disp.joliplot(r'k (m$^{-1}$)', r'$\omega$', k, fct(k, Dsurrho, h, rhosurrho, g), cm = 2, exp = False, log = True)


def RDD_fl (k):
    return k**2.5

def RDD_g (k):
    return k**0.5

def fct(k) :
    return k**2



k1 = np.logspace(-1,0, 100)
k2 = np.logspace(0,1, 100)
# k3 = np.logspace(1,2, 100)

disp.figurejolie(width = 8.6 * 4 / 5)
disp.joliplot(r'k (m$^{-1}$)', r'$\omega$ (s$^{-1}$)', k1, RDD_g(k1), cm = 2, exp = False, log = True, linewidth= 3)
disp.joliplot(r'k (m$^{-1}$)', r'$\omega$ (s$^{-1}$)', k2, RDD_fl(k2), cm = 2, exp = False, log = True, linewidth= 3)
# disp.joliplot(r'k (m$^{-1}$)', r'$\omega$ (s$^{-1}$)', k3, fct(k3)*np.sqrt(10), cm = 2, exp = False, log = True, linewidth= 3)