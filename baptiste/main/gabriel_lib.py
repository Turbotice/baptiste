from __future__ import division
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage import io
from skimage import transform
from skimage import filters
from skimage import color
import skimage
import pandas as pd
import os
#import trackpy as tp
from skimage.filters.rank import median
from skimage.morphology import disk
from skimage.color import rgb2gray
from scipy import *                 
from pylab import *
import statistics
from scipy.integrate import odeint
import math
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
# from skimage.feature import canny
from skimage.draw import circle_perimeter
# from skimage.util import img_as_ubyte
from pims import ImageSequence
from skimage import filters

## TAILLE DES FIGURES

def set_size(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    #golden_ratio = (5**.5 - 1) / 2
    golden_ratio = .95
    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    #fig_height_in = fig_width_in * .9 * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

## FONTS

tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 10,
    "font.size": 12,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
}
plt.rcParams.update(tex_fonts)
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [
r'\usepackage{lmodern}', #lmodern: lateX font; tgheros: helvetica font
r'\usepackage{sansmath}', # math-font matching helvetica
r'\sansmath' # actually tell tex to use it!
r'\usepackage[scientific-notation=false]{siunitx}', # micro symbols
r'\sisetup{detect-all}', # force siunitx to use the fonts
]

## CHOIX DES COULEURS
n = 7
wlq_colors = plt.cm.viridis(np.linspace(0,1,n))