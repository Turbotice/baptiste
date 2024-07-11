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
from math import floor

def fit_powerlaw(x,y, display = False, xlabel = '', ylabel = '', legend = '', new_fig = True, fit = 'poly', color = False):
    
    x_sort, y_sort = tools.sort_listes(x,y)
    xscale = np.linspace(np.min(x_sort), np.max(x_sort), 200)

    logx = np.log(x_sort)
    logy = np.log(y_sort)
    if fit == 'poly' :

        def fct(x, a, b) : 
            return a*x + b
        popt, pcov = curve_fit(fct, logx, logy)


        if display :
            if new_fig :
                disp.figurejolie()
            
            if color != False :
                disp.joliplot(xlabel, ylabel,x_sort, y_sort, legend=legend, color = color, log = True)
                disp.joliplot(xlabel, ylabel, xscale, np.exp(popt[1]) * xscale ** popt[0], exp = False, color = color, legend = 'Slope : ' + str(round(popt[0],2)), log = True)
            else : 
                disp.joliplot(xlabel, ylabel,x_sort, y_sort, legend=legend, color = 2, log = True)
                disp.joliplot(xlabel, ylabel, xscale, np.exp(popt[1]) * xscale ** popt[0], exp = False, color = 8, legend = 'Slope : ' + str(round(popt[0],2)), log = True)
                
                
            plt.xlim( (10**(floor(np.min(logx)/ np.log(10)))) , 10**(floor(np.max(logx)/ np.log(10)) + 1 ) )
            plt.ylim( (10**(floor(np.min(logy)/ np.log(10)))) , 10**(floor(np.max(logy)/ np.log(10)) + 1) )
        
        return popt, pcov


    if fit == 'ransac' :
        model_robust, inliers, outliers = fit_ransac(logx, logy, display = False)
        if display :
            if new_fig :
                disp.figurejolie()
            disp.joliplot(xlabel, ylabel,x_sort[inliers], y_sort[inliers], legend=legend, color = 4, log = False)
            if np.mean(outliers) != 0.0 :
                disp.joliplot(xlabel, ylabel, x_sort[outliers], y_sort[outliers], legend = 'Outlier data', color = 7, log = True)
            disp.joliplot(xlabel, ylabel, xscale, np.exp(model_robust.predict_y(np.log(xscale))), exp = False, color = 2, legend = 'Pente : ' + str(round(model_robust.params[1][1],2)), log = False)
        
        return model_robust, inliers, outliers

def fit_ransac (x, y, thresh = 0.5, display = True, xlabel = '', ylabel = '', newfig = True) :
    data = np.stack((x,y), axis = -1)
    model_robust, inliers = ransac(data, LineModelND, min_samples=5, residual_threshold=thresh, max_trials=2000)
    outliers = (inliers == False)
    xx = x
    yy = model_robust.predict_y(xx)
    if display :
        if newfig :
            disp.figurejolie()
        disp.joliplot(xlabel, ylabel,data[inliers, 0], data[inliers, 1], legend='Inlier data', color = 3)
        disp.joliplot(xlabel, xlabel,data[outliers, 0], data[outliers, 1], legend = 'Outlier data', color = 7)
        disp.joliplot(xlabel, xlabel, xx, yy, exp = False, color = 2, legend = 'Pente : ' + str(round(model_robust.params[1][1],2)))
    
        # plt.annotate('Pente : ' + str(round(model_robust.params[1][1],2)),(x[-1],y[-1]))
    
        plt.legend(loc='lower right')
    return model_robust, inliers, outliers


def fit(fct, x, y, display = True, err = False, nb_param = 1, p0 = [0], bounds = False, 
        zero = False, th_params = False, xlabel = r'k (m$^{-1}$)', ylabel = r'$\omega$', legend_data = r'Experimental Data', 
        legend_fit = 'h = ', log = False):
    
    if bounds is not False :
        popt, pcov = curve_fit(fct, x, y, p0 = p0, bounds= bounds)
    else :
        popt, pcov = curve_fit(fct, x, y, p0 = p0)
    x_range = np.linspace(np.min(x), np.max(x), len(x))
    if zero :
        x_range = np.linspace(0, np.max(x), len(x))
    if display :
        if nb_param == 1 :
            disp.joliplot(xlabel, ylabel, x, y, color= 13, exp = True, legend = legend_data)
            disp.joliplot(xlabel, ylabel, x_range, fct(x_range, popt[0]), color= 5, exp = False, legend = legend_fit + str(popt) + ' (fit)') #round(popt[0],2)) + ' (fit)')
            if th_params is not False :
                disp.joliplot(xlabel, ylabel, x_range, fct(x_range, th_params), exp = False, legend = r'Theoretical curve', color = 3, zeros = True)
            if err :
                plt.fill_between(x_range, fct(x_range, popt[0] + np.sqrt(pcov[0])), fct(x_range, popt[0] - np.sqrt(pcov[0])), color = disp.vcolors(2))
            if log :
                plt.xscale('log')
                plt.yscale('log')
        if nb_param == 2 :
            disp.joliplot(xlabel, ylabel, x, y, color= 13, exp = True, legend = legend_data)
            disp.joliplot(xlabel, ylabel, x_range, fct(x_range, popt[0], popt[1]), color= 5, exp = False, legend = r'fit : param 1 = ' + str(round(popt[0],4)) + ' param 2 = ' + str(round(popt[1],4)))
            if th_params is not False :
                disp.joliplot(xlabel, ylabel, x_range, fct(x_range, th_params[0], th_params[1]), exp = False, legend = r'Theoretical curve', color = 3,zeros = True)
            if err :
                plt.fill_between(x_range, fct(x_range, popt[0] + np.sqrt(np.diag(pcov))[0], popt[1] + np.sqrt(np.diag(pcov))[1]),
                                 fct(x_range, popt[0] - np.sqrt(np.diag(pcov))[0], popt[1] - np.sqrt(np.diag(pcov))[1]), color = disp.vcolors(2))
                plt.fill_between(x_range, fct(x_range, popt[0] + np.sqrt(np.diag(pcov))[0], popt[1] - np.sqrt(np.diag(pcov))[1]),
                                 fct(x_range, popt[0] - np.sqrt(np.diag(pcov))[0], popt[1] + np.sqrt(np.diag(pcov))[1]), color = disp.vcolors(2))
    return popt, pcov


    
    
    
    return popt, pcov