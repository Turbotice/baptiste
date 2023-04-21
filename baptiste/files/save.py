# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 14:19:46 2023

@author: Banquise

Pour save proprement tout type de données

"""

import matplotlib.pyplot as plt
import numpy as np

def save_graph (graph) :
    plt.savefig(graph)
    return graph

def save_image(image):
    return image

def save_signal(signal):
    """
    en format Baptiste
    """
    return signal



