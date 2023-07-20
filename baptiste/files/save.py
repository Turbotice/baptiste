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
import os

import baptiste.files.dictionaries as dic
import baptiste.tools.tools as tools
import baptiste.display.display_lib as disp



def data_to_dict(labels, axis, data) :
    #Pour mettreles datas en format dict pour les save facilement en nc ou pkl
    #attention ne met que 2 axes mais tt les datas
    
    # >>> even better : (['time', 'y', 'x'], ux.astype(float32))),

    data_dict = dict(
    data = dict(data = (labels, data)),
    
    coords = dict(
    x = (labels[0], np.asarray(axis[0])) , 
    y = (labels[1], np.asarray(axis[1])) ) )

    return data_dict
    


def save_graph (path, nom_fig, params = False, num_fig = False, nc = False, pkl = True, pdf = True, data = False) :
    
    #met la date dans le nom et va dans le folder resultats
    date_time = tools.datetimenow()
    nom_fig = date_time + '_' + nom_fig   
    
    #par défaut save la derniere figure faite
    if type(num_fig) == bool :
        if type(params) != bool :
            num_fig = params['num_fig'][-1]
            disp.figurejolie(num_fig = num_fig)
        else :
            print('Reinseigner params ou num_fig, paramètres non sauvegardés')
            num_fig = 44
    else : 
        disp.figurejolie(num_fig = num_fig)
    num_fig = str(num_fig)
    path = path + "\\resultats\\" + nom_fig + "\\"
    if not os.path.exists(path):
        os.makedirs(path)
    
    #save la figure
    if pdf :
        plt.savefig(path + nom_fig + "_" + str(num_fig) + ".pdf", dpi = 1)
    else :
        plt.savefig(path + nom_fig + "_" + str(num_fig) + ".png", dpi = 1)
    
    print( 'figure ' + str(num_fig) + " saved with the name " + nom_fig)
    
    if nc :
        #MARCHE PAS
        if type(params) == bool :
            print('Spécifier data et metadata')
        else :
            #save en .nc
            data = params[str(num_fig)]['data']['data']
            coords = params[str(num_fig)]['data']['coords']
            
            save_nc(path + "data_" + nom_fig + "_" + num_fig, data, coords, params)
    
    if pkl :
        #save tout en .pkl
        if type(params) != bool :
            
            if 'data' in params[str(num_fig)].keys() :
                dic.save_dico(params[num_fig]['data'], path + "data_" + nom_fig + "_" + num_fig + ".pkl")
            full_params = {}
            for i in params.keys() :
                if True : #not i.isdigit() :
                    full_params[i] = params[i]
                
            dic.save_dico(params, path + "params_" + nom_fig + "_" + num_fig + ".pkl")
        elif type(data) == dict :
            dic.save_dico(data, path + "data_" + nom_fig + "_" + num_fig + ".pkl")
            

def save_all_figs(path, params) :
    # nom_figs = []
    # for i in params['num_fig'] :
    #     if str(params[str(i)]['nom_fig']) not in nom_figs :
    #         save_graph(path,str(params[str(i)]['nom_fig']), params = params, num_fig = i)
    #     nom_figs.append(str(params[str(i)]['nom_fig']))
    for i in params['num_fig'] :
        save_graph(path,str(params[str(i)]['nom_fig']), params = params, num_fig = i)
    print('All figures saved')
        

def save_image(image):
    #marche avec save graph si image dans figure
    return image

def save_mat(champ, path, title = 'data'):
    data = dict(data = champ)
    
    # mdic = {"data": champ, "label": "experiment"}
    # savemat(path + 'resultats\\' + title + ".mat", mdic)
    savemat(path + 'resultats\\' + title + '.mat', data)
    
    
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
    


