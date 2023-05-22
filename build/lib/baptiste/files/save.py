# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 14:19:46 2023

@author: Banquise

Pour save proprement tout type de données

"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.io import savemat

import baptiste.files.dictionaries as dic
import baptiste.tools.tools as tools



def data_to_dict(labels, axis, data, num_fig) :
    #Pour mettreles datas en format dict pour les save facilement en nc ou pkl
    #attention ne met que 2 axes mais tt les datas
    
    # >>> even better : (['time', 'y', 'x'], ux.astype(float32))),

    data_dict = {str(num_fig) : 
             dict(
    data = dict(data = (labels, data)),
    
    coords = dict(
    x = (labels[0], np.asarray(axis[0])) , 
    y = (labels[1], np.asarray(axis[1])) ) )}

    return data_dict
    


def save_graph (path, nom_fig, params = False, data_dict = False, num_fig = False, nc = False, pkl = True) :
    
    #met la date dans le nom et va dans le folder resultats
    date_time = tools.datetimenow()
    nom_fig = nom_fig + '_' + date_time
    path = path + "\\resultats\\"
    
    #par défaut save la derniere figure faite
    if type(params) != bool :
        if type(num_fig) == bool :
            num_fig = params['num_fig'][-1]
    #save la figure
    plt.savefig(path + nom_fig + "_" + num_fig + ".pdf", dpi = 1)
    
    
    print( 'figure ' + str(num_fig) + " saved, with name " + nom_fig)
    
    if nc :
        #MARCHE PAS
        if type(data_dict) == bool or type(params) == bool :
            print('Spécifier data et metadata')
        else :
            #save en .nc
            data = data_dict[str(num_fig)]['data']
            coords = data_dict[str(num_fig)]['coords']
            
            save_nc(path + "data_" + nom_fig + "_" + num_fig, data, coords, params)
    
    if pkl :
        #save tout en .pkl
        if type(data_dict) != bool :
            dic.save_dico(data_dict, path + "data_" + nom_fig + "_" + num_fig + ".pkl")
            
        if params != {} :
            dic.save_dico(params, path + "params_" + nom_fig + "_" + num_fig + ".pkl")
        
        

def save_image(image):
    return image

def save_signal(signal):
    """
    en format Baptiste
    """
    return signal

def save_mat(champ, path, title = 'data'):
    
    mdic = {"data": champ, "label": "experiment"}
    savemat(path + title + ".mat", mdic)
    
def save_nc(path, data, coords, params) :
    """
    MARCHE PAS

    Parameters
    ----------
    path : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.
    coords : TYPE
        DESCRIPTION.
    params : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # create your Xarray dataset 
    ds = xr.Dataset(data_vars=data, coords=coords, attrs=params)
    # encoding for compression (optional)
    encoding = {var: dict(zlib=True, complevel=5) for var in ds.data_vars}
    # saving as a netcdf file
    ds.to_netcdf(path + ".nc", encoding=encoding)
    


