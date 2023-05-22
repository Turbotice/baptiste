# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 13:51:31 2023

@author: Banquise
"""

'''
Fonctions pour le dico
'''
import pickle

def open_dico():
    
    a_file = open("D:\Banquise\Baptiste\Resultats\\Dictionnaire.pkl", "rb")
    
    dico = pickle.load(a_file)
    
    a_file.close
    
    return dico
    
def save_dico(dico, path = "D:\Banquise\Baptiste\Resultats\\Dictionnaire.pkl"):
    

    a_file = open(path, "wb")
    
    pickle.dump(dico, a_file)
    
    a_file.close()
    

def add_dico(dico, date, nom_exp, name, value):
    dico[date][nom_exp][name] = value
    return dico
    
    

def remove_dico(dico, date, nom_exp, name):
    del dico[date][nom_exp][name]
    
def rename_variable_dico (dico,date, nom_exp, old_name, new_name):
    value = dico[date][nom_exp][old_name]
    dico = add_dico(dico, date, nom_exp, new_name, value)
    remove_dico(dico,date,nom_exp,old_name)
    return dico