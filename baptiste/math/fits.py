# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 17:54:53 2023

@author: Banquise

Pour faire des fits
"""
import numpy as np
import baptiste.display.display_lib as disp
import baptiste.tools.tools as tools


def fit_powerlaw(x,y, display = False, xlabel = '', ylabel = '', legend = ''):
    
    x_sort, y_sort = tools.sort_listes(x,y)

    logx = np.log(x_sort)
    logy = np.log(y_sort)

    popt = np.polyfit(logx, logy, 1)

    if display :
        disp.figurejolie()
        disp.joliplot(xlabel, ylabel, logx, logy, color= 13, exp = True, legend = legend, log = True)
        disp.joliplot(xlabel, ylabel, logx, popt[0] * logy  + popt[1], color = 5, exp = False, legend = 'fit, a = ' + str(round(popt[0], 3)), log = True)
    return popt