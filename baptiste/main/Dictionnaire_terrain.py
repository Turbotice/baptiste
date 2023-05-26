# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 12:20:47 2023

@author: Banquise
"""

'''
CREATION D'UN DICTIONNAIRE POUR TRIER LES DONNEES DE TERRAIN
'''
#test
dictionnaire = {
    'fruit': ['pomme', 'banane', 'orange'],
    'legume': ['tomate', 'carotte', 'poivron'],
    'viande': ['poulet', 'boeuf', 'porc']
}

# Mot-clé recherché
mot_cle = 'pomme'

# Recherche de données associées au mot-clé
donnees_associees = []
for categorie, elements in dictionnaire.items():
    if mot_cle in elements:
        donnees_associees.append((categorie, elements))
        
# Affichage des données associées
if donnees_associees:
    print(f"Données associées au mot-clé '{mot_cle}':")
    for categorie, elements in donnees_associees:
        print(f"{categorie}: {', '.join(elements)}")
else:
    print(f"Aucune donnée associée au mot-clé '{mot_cle}'")
    
#%% Fonctions pour dictionnaire :

    
def open_dico():
    a_file = open("W:\Banquise\\Rimouski_2023\\Dictionnaire_terrain.pkl", "rb")
    dico = pickle.load(a_file)
    a_file.close
    return dico
    
def save_dico(dico):
    a_file = open("W:\Banquise\\Rimouski_2023\\Dictionnaire_terrain.pkl", "wb")
    pickle.dump(dico, a_file)
    a_file.close()
    

def add_dico(dico, date, info, param, value):
    dico[date][info][param] = value
    return dico
    
    

def remove_dico(dico, date, info, param):
    del dico[date][info][param]
    
def rename_variable_dico (dico,date, info, old_name, new_name):
    value = dico[date][info][old_name]
    dico = add_dico(dico, date, info, new_name, value)
    remove_dico(dico,date,info,old_name)
    return dico




    
#%% Création dictionnaire 

"""
On crée un dictionnaire, on met dedans les expériences qu'on trie par ???

qualité : A on est sur, B possible mais ca va etre chaud, C miracle necessaire
"""
import numpy as np
import pickle


dic_t = {'230306': {'instruments': {'geophones': {'objectifs': {'1','2'},'lien' : 'W:\Banquise\Rimouski_2023\Data\\Geophones\\20230306', 'mesures': {'E' : 3.9e9} },
                                   'buoys' : {'objectifs': {'1','2'},'lien' : 'W:\Banquise\Rimouski_2023\Data\\buoys\\20230306', 'mesures': {'k' : 2}},
                                   'geonics' : {'objectifs': {'1'}},'lien' : 'W:\Banquise\Rimouski_2023\\Data\\GEONICS_20230306',
                                   'gps' : {'objectifs': {'0'}},'lien' : 'W:\Banquise\Rimouski_2023\Data\\GPS\\20230306',
                                   'foreuse': {'objectifs': {'1'}},'lien' : ''},
                    'lieu' : 'Marina'},
         '230307': {'instruments': {'geophones': {'objectifs': {'1','2'}, 'qualité' : {'1' : 'A', '2' : 'C'}},'lien' : 'W:\Banquise\Rimouski_2023\Data\\Geophones\\20230307', 
                                    'stereo': {'objectifs': {'1','2'}, 'qualité' : {'1' : 'A', '2' : 'C'}},'lien' : 'W:\Banquise\Rimouski_2023\Data\\Stereo\\d230307',
                                    'buoys': {'objectifs': {'1', '2'}, 'qualité' : {'1' : 'A', '2' : 'C'}},'lien' : 'W:\Banquise\Rimouski_2023\Data\\buoys\\20230307',
                                    'drone': {'objectifs': {'0','3'}},'lien' : 'W:\Banquise\Rimouski_2023\Data\\drone\\20230307',
                                    'foreuse': {'objectifs': {'1'}},'lien' : '',
                                    'gps' : {'objectifs': {'0'}}},
                    'lieu' : 'Marina'} ,
         '230308': {'instruments': {'drone': {'objectifs': {'0','3'}},
                                    'gps' : {'objectifs': {'0'}}},
                    'lieu' : 'Ha!Ha!'},
         '230309': {'instruments': {'geophones': {'objectifs': {'1','2'}, 'qualité' : {'1' : 'A', '2' : 'C'}}, 
                                    'stereo': {'objectifs': {'1','2'}, 'qualité' : {'1' : 'C', '2' : 'C'}},
                                    'drone': {'objectifs': {'0'}},
                                    'geonics': {'objectifs': {'1'}},
                                    'foreuse': {'objectifs': {'1'}},
                                    'gps' : {'objectifs': {'0'}}},
                    'lieu' : 'Rimouski'},
         '230310': {'instruments': {'geophones': {'objectifs': {'3'}},
                                    'buoys': {'objectifs': {'3'}},
                                    'drone': {'objectifs': {'0','3'}},
                                    'stereo': {'objectifs': {'3'}},
                                    'foreuse': {'objectifs': {'1'}},
                                    'gps' : {'objectifs': {'0'}}},
                    'lieu' : 'Ha!Ha!'},
         '230313': {'instruments': {'geophones': {'objectifs': {'1','2','3'}, 'qualité' : {'1' : 'A', '2' : 'A', '3' : 'C'}},
                                    'buoys': {'objectifs': {'1', '2'}},
                                    'drone': {'objectifs': {'0','1', '2', '3'}},
                                    'stereo': {'objectifs': {'1','2'}},
                                    'foreuse': {'objectifs': {'1'}},
                                    'gps' : {'objectifs': {'0'}}},
                    'lieu' : 'Rimouski'}}


#%% Recherche d'un mot clé

# Mot-clé recherché
mot_cle = '1'
info_recherchees = ['lieu'] #False

# Recherche de données associées au mot-clé
chemins = []
branches = []
dico_temporaire1 = [[dic_t], [['data']]] #dico, chemin dico
dico_temporaire2 = [[],[]]

pos_dico = []

cherche = True
u = 0
while cherche :
    i = 0
    for dico in np.asarray(dico_temporaire1)[0,:] :
        for dessus, dessous in dico.items():
            pos_dico = dessus
            print(dessus)
            if type(dessous) is str:
                if mot_cle in dessous :
                    chemins.append(pos_dico)                
            elif type(dessous) is dict :
                dico_temporaire2[0].append(dico[dessus])
                dico_temporaire2[1].append([dico_temporaire1[1][i],pos_dico])
                if mot_cle in dessous:
                    chemins.append(dico_temporaire2[1])
                
            elif type(dessous) is set:
                # print(dessous)
                if mot_cle in dessous:
                    chemins.append(pos_dico)            
            else :
                if mot_cle == dessous :
                    chemins.append(pos_dico)
                print(dessous)
        i+=1
    dico_temporaire1 = dico_temporaire2.copy()
    dico_temporaire2 = [[],[]]
    u += 1
    if u > 5 :
        cherche = False
        
    
       
    
    
        
# Affichage des données associées
if not cherche:
    print(f"{chemins}")
else:
    print(f"Aucune donnée associée au mot-clé '{mot_cle}'")