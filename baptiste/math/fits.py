# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 17:54:53 2023

@author: Banquise

Pour faire des fits
"""
import numpy as np
import baptiste.display.display_lib as disp
import baptiste.tools.tools as tools
from skimage.measure import ransac, LineModelND
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def fit_powerlaw(x,y, display = False, xlabel = '', ylabel = '', legend = ''):
    
    x_sort, y_sort = tools.sort_listes(x,y)

    logx = np.log(x_sort)
    logy = np.log(y_sort)
    
    popt = np.polyfit(logx, logy, 1)
    
    xscale = np.linspace(np.min(logx), np.max(logx), 200)

    if display :
        disp.joliplot(xlabel, ylabel, logx, logy, color= 13, exp = True, legend = legend, log = False)
        disp.joliplot(xlabel, ylabel, xscale, popt[0] * xscale  + popt[1], color = 5, exp = False, legend = 'fit, a = ' + str(round(popt[0], 3)), log = False)
    return popt


def fit_ransac (x, y, thresh = 0.1, display = True, xlabel = '', ylabel = '', newfig = False) :
    data = np.stack((x,y), axis = -1)
    model_robust, inliers = ransac(data, LineModelND, min_samples=2, residual_threshold=thresh, max_trials=2000)
    outliers = (inliers == False)
    xx = x
    yy = model_robust.predict_y(xx)
    if display :
        if newfig :
            disp.figurejolie()
        disp.joliplot(xlabel, xlabel,data[inliers, 0], data[inliers, 1], legend='Inlier data', color = 3)
        disp.joliplot(xlabel, xlabel,data[outliers, 0], data[outliers, 1], legend = 'Outlier data', color = 7)
        disp.joliplot(xlabel, xlabel, xx, yy, exp = False, color = 2)
    
        plt.annotate('Pente : ' + str(round(model_robust.params[1][1],2)),(x[-1],y[-1]))
    
        plt.legend(loc='lower right')
    return model_robust, inliers, outliers


def fit(fct, x, y, display = True, err = False, nb_param = 1, p0 = [0], bounds = False, 
        zero = False, th_params = False, xlabel = r'k (m$^{-1}$)', ylabel = r'$\omega$'):
    
    if bounds is not False :
        popt, pcov = curve_fit(fct, x, y, p0 = p0, bounds= bounds)
    else :
        popt, pcov = curve_fit(fct, x, y, p0 = p0)
    x_range = np.linspace(np.min(x), np.max(x), len(x))
    if zero :
        x_range = np.linspace(0, np.max(x), len(x))
    if display :
        if nb_param == 1 :
            disp.joliplot(xlabel, ylabel, x, y, color= 13, exp = True, legend = r'Experimental Data')
            disp.joliplot(xlabel, ylabel, x_range, fct(x_range, popt[0]), color= 5, exp = False, legend = r'fit : $\delta\rho * h$ = ' + str(round(popt[0],4)))
            if th_params is not False :
                disp.joliplot(xlabel, ylabel, x_range, fct(x_range, th_params), exp = False, legend = r'Theoretical result', color = 3,zeros = True)
            if err :
                plt.fill_between(x_range, fct(x_range, popt[0] + np.sqrt(pcov[0])), fct(x_range, popt[0] - np.sqrt(pcov[0])), color = disp.vcolors(2))
        if nb_param == 2 :
            disp.joliplot(xlabel, ylabel, x, y, color= 13, exp = True, legend = r'Experimental Data')
            disp.joliplot(xlabel, ylabel, x_range, fct(x_range, popt[0], popt[1]), color= 5, exp = False, legend = r'fit : param 1 = ' + str(round(popt[0],4)) + ' param 2 = ' + str(round(popt[1],4)))
            if th_params is not False :
                disp.joliplot(xlabel, ylabel, x_range, fct(x_range, th_params[0], th_params[1]), exp = False, legend = r'Theoretical result', color = 3,zeros = True)
            if err :
                plt.fill_between(x_range, fct(x_range, popt[0] + np.sqrt(np.diag(pcov))[0], popt[1] + np.sqrt(np.diag(pcov))[1]),
                                 fct(x_range, popt[0] - np.sqrt(np.diag(pcov))[0], popt[1] - np.sqrt(np.diag(pcov))[1]), color = disp.vcolors(2))
                plt.fill_between(x_range, fct(x_range, popt[0] + np.sqrt(np.diag(pcov))[0], popt[1] - np.sqrt(np.diag(pcov))[1]),
                                 fct(x_range, popt[0] - np.sqrt(np.diag(pcov))[0], popt[1] + np.sqrt(np.diag(pcov))[1]), color = disp.vcolors(2))


    
    
    
    return popt, pcov