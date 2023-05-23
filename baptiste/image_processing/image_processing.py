# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 14:01:28 2023

@author: Banquise

Pour tout ce qui est modification, amélioration et convolution d'images
"""


from __future__ import annotations
from scipy.ndimage.filters import convolve
# from colour.hints import ArrayLike, Literal, NDArray, Union
# from colour.utilities import as_float_array, tstack
# from colour_demosaicing.bayer import masks_CFA_Bayer
import numpy as np
from scipy.signal import convolve2d
from skimage.restoration import inpaint
from scipy.interpolate import griddata
from scipy.interpolate import interpn



def conv2(x, y, mode='same'):
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)

def scale_256(img, scale = 256):
    return (img * scale).astype(np.uint8)

def substract_mean(data, space = False, temp = True):
    '''
    ATTENTION : data(x,y,t) avec temporel à la fin sinon marche pas
    temp = True : Eneleve la moyenne temporelle de chaque pixel (si derniere dimension de data est le temps), pour les ondes
    space = True : enleve la moyenne spatiale à chaque temps
    '''
    
    dim = len(data.shape)
    if space :
        if dim == 1 :
            temp = True
        else :   
            for i in range (data.shape[-1]):
                if dim == 2 :
                    data[:,i] = data[:,i] - np.nanmean(data[:,i])
                if dim == 3 :
                    data[:,:,i] = data[:,:,i] - np.nanmean(data[:,:,i])
    if temp :
        data = data - np.nanmean(data, axis = -1)
    
    return data

def interp_3D_inpaint(data):
    print('LENT ET PAS PERFORMANT')
    for i in range(data.shape[-1]):
        missing_data = np.isnan(data[:,:,i])
        data[:,:,i] = inpaint.inpaint_biharmonic(data[:,:,i], missing_data)
        if np.mod(i,3)==0:
            print('iteration ' + str(i) + ' sur ' + str(data.shape[-1]))
            
            
def interp_3D_scipy(data) : 
    print('MARCHE PAS')
    [nx,ny,nt] = data.shape
    points = np.asarray(np.meshgrid(np.linspace(0, nx, nx, dtype = 'int'), np.linspace(0, ny, ny, dtype = 'int'), np.linspace(0, nt, nt, dtype = 'int'), indexing='ij'))
    values = data
    point = np.array(np.where(np.isnan(data)))
        
    data_int = interpn(points, values, point)
    data = data_int 
        
    grid_x, grid_y = np.meshgrid(np.linspace(0, nx, nx, dtype = 'int'), np.linspace(0, ny, ny, dtype = 'int'), indexing='ij')     
    for i in range (nt) :
        data_tofit =np.array(np.where(np.isfinite(data[:,:,i])))
        data_index = np.transpose(data_tofit)
        # data_missing = np.where(np.isnan(data))
        data[:,:,i] = griddata(data_index, data[data_index[:,0],data_index[:,1],i], (grid_x, grid_y), method = 'linear')
        if np.mod(i,100)==0:
            print('iteration ' + str(i) + ' sur ' + str(nt))
    return data




# def demosaicing_CFA_Bayer_bilinear(
#     CFA: ArrayLike,
#     pattern: Union[Literal["RGGB", "BGGR", "GRBG", "GBRG"], str] = "RGGB",
# ) -> NDArray:
#     """
#     __author__ = "Colour Developers"
#     __copyright__ = "Copyright 2015 Colour Developers"
#     __license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
#     __maintainer__ = "Colour Developers"
#     __email__ = "colour-developers@colour-science.org"
#     __status__ = "Production"

#     __all__ = [
#         "demosaicing_CFA_Bayer_bilinear",
#     ]
    
#     Return the demosaiced *RGB* colourspace array from given *Bayer* CFA using
#     bilinear interpolation.

#     Parameters
#     ----------
#     CFA
#         *Bayer* CFA.
#     pattern
#         Arrangement of the colour filters on the pixel array.

#     Returns
#     -------
#     :class:`numpy.ndarray`
#         *RGB* colourspace array.

#     Notes
#     -----
#     -   The definition output is not clipped in range [0, 1] : this allows for
#         direct HDRI image generation on *Bayer* CFA data and post
#         demosaicing of the high dynamic range data as showcased in this
#         `Jupyter Notebook <https://github.com/colour-science/colour-hdri/\
# blob/develop/colour_hdri/examples/\
# examples_merge_from_raw_files_with_post_demosaicing.ipynb>`__.

#     References
#     ----------
#     :cite:`Losson2010c`

#     Examples
#     --------
#     >>> import numpy as np
#     >>> CFA = np.array(
#     ...     [
#     ...         [0.30980393, 0.36078432, 0.30588236, 0.3764706],
#     ...         [0.35686275, 0.39607844, 0.36078432, 0.40000001],
#     ...     ]
#     ... )
#     >>> demosaicing_CFA_Bayer_bilinear(CFA)
#     array([[[ 0.69705884,  0.17941177,  0.09901961],
#             [ 0.46176472,  0.4509804 ,  0.19803922],
#             [ 0.45882354,  0.27450981,  0.19901961],
#             [ 0.22941177,  0.5647059 ,  0.30000001]],
#     <BLANKLINE>
#            [[ 0.23235295,  0.53529412,  0.29705883],
#             [ 0.15392157,  0.26960785,  0.59411766],
#             [ 0.15294118,  0.4509804 ,  0.59705884],
#             [ 0.07647059,  0.18431373,  0.90000002]]])
#     >>> CFA = np.array(
#     ...     [
#     ...         [0.3764706, 0.360784320, 0.40784314, 0.3764706],
#     ...         [0.35686275, 0.30980393, 0.36078432, 0.29803923],
#     ...     ]
#     ... )
#     >>> demosaicing_CFA_Bayer_bilinear(CFA, "BGGR")
#     array([[[ 0.07745098,  0.17941177,  0.84705885],
#             [ 0.15490197,  0.4509804 ,  0.5882353 ],
#             [ 0.15196079,  0.27450981,  0.61176471],
#             [ 0.22352942,  0.5647059 ,  0.30588235]],
#     <BLANKLINE>
#            [[ 0.23235295,  0.53529412,  0.28235295],
#             [ 0.4647059 ,  0.26960785,  0.19607843],
#             [ 0.45588237,  0.4509804 ,  0.20392157],
#             [ 0.67058827,  0.18431373,  0.10196078]]])
#     """

#     CFA = as_float_array(CFA)
#     R_m, G_m, B_m = masks_CFA_Bayer(CFA.shape, pattern)

#     H_G = (
#         as_float_array(
#             [
#                 [0, 1, 0],
#                 [1, 4, 1],
#                 [0, 1, 0],
#             ]
#         )
#         / 4
#     )

#     H_RB = (
#         as_float_array(
#             [
#                 [1, 2, 1],
#                 [2, 4, 2],
#                 [1, 2, 1],
#             ]
#         )
#         / 4
#     )

#     R = convolve(CFA * R_m, H_RB)
#     G = convolve(CFA * G_m, H_G)
#     B = convolve(CFA * B_m, H_RB)

#     del R_m, G_m, B_m, H_RB, H_G

#     return tstack([R, G, B])


