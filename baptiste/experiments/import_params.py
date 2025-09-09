# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 13:52:05 2023

@author: Banquise

Toutes les fonctions qi permettent d'importer les parmaètres expérimentaux des expériences depuis les titres et autre'
"""

import matplotlib.pyplot as plt
import baptiste.files.file_management as fm
import numpy as np
import os

import baptiste.files.dictionaries as dic
import baptiste.display.display_lib as disp

def import_images(loc, nom_exp, exp_type, nom_fich = "\image_sequence\\"):
    fichiers = []                             
    liste_images = []

    fichiers = os.listdir(loc)        

    for j in range (len (fichiers)):
        if nom_exp == fichiers[j][8:13] :
            if exp_type in fichiers[j]:
                titre_exp = fichiers[j]
                path_images = str(loc + fichiers [j] + nom_fich)

    liste_images = os.listdir(path_images)

    print (path_images)
    return path_images, liste_images, titre_exp

def import_param (titre_exp,date, exp_type = "TT"):
    """
    Trouve les paramètres à partir du nom du fichiers d'une expérience'

    Parameters
    ----------
    titre_exp : TYPE
        DESCRIPTION.
    date : TYPE
        DESCRIPTION.
    exp_type : TYPE, optional, LAS or IND or FSD or PIV...
        DESCRIPTION. The default is "TT".

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if exp_type == "TT":
    #"\d" + date + "_vagues_facq" + str(facq) + "Hz_texp" + str(texp) + "us_fmotor"+ str(fpot) + "_Amplitude" +str(Apot) + "_Hw" + str(Hw) + "cm_Pasdamier" + str(pasdamier) + "mm"
        if float(date) >= 221007 :  
            facq = float ( titre_exp[titre_exp.index("facq") + 4:titre_exp.index("facq") + 7]) # fréquence d'acquisition (Hz)
        else :
            facq = float ( titre_exp[titre_exp.index("facq") + 4:titre_exp.index("facq") + 6]) # fréquence d'acquisition (Hz)
        texp = float ( titre_exp[titre_exp.index("texp") + 4:titre_exp.index("texp") + 8])# temps d'exposition (muS)
        Tmot = float ( titre_exp[titre_exp.index("Tmot") + 4:titre_exp.index("Tmot") + 7])# tps entre mvt moteur (ms)
        Vmot = float ( titre_exp[titre_exp.index("Vmot") + 4:titre_exp.index("Vmot") + 7])# vitesse du moteur (0 - 255)
        Hw = float ( titre_exp[titre_exp.index("Hw") + 2:titre_exp.index("Hw") + 4])# hauteur eau (cm)
        if titre_exp[titre_exp.index("sur") + 5].isdigit() :
            Long_ice = float ( titre_exp[titre_exp.index("sur") + 3:titre_exp.index("sur") + 6])#longeur glace (cm)
        else :
            Long_ice = float ( titre_exp[titre_exp.index("sur") + 3:titre_exp.index("sur") + 5])#longeur glace (cm)
        Larg_ice = float ( titre_exp[titre_exp.index("sur") - 2:titre_exp.index("sur")])#largeur glace (cm)
        
        tacq =  float ( titre_exp[titre_exp.index("tacq") + 4:titre_exp.index("tacq") + 7])#temps total acquisition (s)

        return facq, texp, Tmot, Vmot, Hw, Larg_ice, Long_ice, tacq
    if exp_type == "IND":
        h = float ( titre_exp[titre_exp.index("ha") + 2:titre_exp.index("ha") + 4])
        return h




def import_angle (date, nom_exp, loc, display = False):
    """
    Trouve les paramètres de la nappe laser associée à une expérience depuis le fichier texte rempli avec ces paramètres. 2 colonnes : 1) distance à la nappe 2) hauteur de la nappe

    Parameters
    ----------
    date : TYPE
        DESCRIPTION.
    nom_exp : TYPE
        DESCRIPTION.
    loc : racine du dossier des données.
    display : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    grossissement : TYPE
        DESCRIPTION.
    er_grossissement : TYPE
        DESCRIPTION.
    angle : angle entre la nappe et la surface du vernis
    er_angle : TYPE
        DESCRIPTION.

    """

    path, liste, titre = fm.import_images(loc,nom_exp,"LAS",nom_fich = '\\references')

    dx_dh = np.loadtxt(path + '\\angle_laser.txt')

    dx = dx_dh[:,0]
    dh = dx_dh[:,1]        

    hprime = np.polyfit(dx,dh,1)

    dh_prime = dh - hprime[1]
    xxx = np.linspace(0,max(dx),200)
    
    if display :
        
        disp.figurejolie(width = 6)
        disp.joliplot("Distance horizontale au laser (mm)", "Hauteur laser (mm)", dx,dh, color = 18)
        disp.joliplot("Distance horizontale au laser (mm)", "Hauteur laser (mm)", xxx, hprime[0] * xxx + hprime[1], color = 8, exp = False)
    
    tab_angles = np.arctan(dh_prime/dx) * 180 / np.pi

    angle = np.mean(tab_angles)
    xxx = np.linspace(1,len(dh_prime) + 1, len(dh_prime))
    er_angle = (max(tab_angles) - min(tab_angles))/2
    if display :
        disp.figurejolie()
        disp.joliplot("Distance horizontale au laser (mm)", "Angle (en radian)", dx, tab_angles, color = 3)
        print('angle moyen : ', angle, '+-', er_angle )
    grossissement = 1 / np.tan(angle* np.pi / 180)
    grossissements = 1 / np.tan(tab_angles* np.pi / 180)
    er_grossissement = (max(grossissements) - min (grossissements))/2
    if display :
        print('grossissement :',grossissement, "+- ", er_grossissement)
    
    return grossissement, er_grossissement, angle, er_angle



def import_calibration(titre_exp, date):
    """
    Une date et un nom d'exp donne une calibration en mm par pixel
    
    """
    
    if "LAS" in titre_exp:
        if float(date) >= 220412:
            mmparpixelx = 0.0974506899508849 #profondeur
            mmparpixely = 0.0390800554936788 #vertical
            mmparpixelz = 0.0421864387474003 #horizontal
            mmparpixel = 1
            if float(date) >= 220512:
                mmparpixelx = 0.150384343422359 #profondeur
                mmparpixely = 0.043816976811791 #vertical
                mmparpixelz = 0.045110366648328 #horizontal
                mmparpixel = 1
                if float(date) >= 220523:
                    mmparpixelx = 0.150811071469620 #profondeur
                    mmparpixely = 0.042888820001944 #vertical
                    mmparpixelz = 0.044245121848119 #horizontal
                    mmparpixel = 1
                    if float(date) >= 220607:
                        mmparpixelz = 0.045008078623617
                        mmparpixely = 0.047095115932238
                        mmparpixel = 1
                        if float(date) >= 221006:
                            mmparpixel = 0.28
                            mmparpixelz = 0.28
                            mmparpixely = 0.28
                            if float(date) >= 221011:
                                mmparpixel = 0.29003161344586
                                mmparpixelz = 0.29003161344586
                                mmparpixely = 0.29003161344586
                                if float(date) >= 221012:
                                    mmparpixel = 0.2001
                                    mmparpixelz = 0.2001
                                    mmparpixely = 0.2001
                                    if float(date) >= 221024:
                                        mmparpixel = 0.19434
                                        mmparpixelz = 0.19434
                                        mmparpixely = 0.19434
                                        if float(date) >= 221116:
                                            mmparpixel = 0.18763
                                            mmparpixelz = 0.18763
                                            mmparpixely = 0.18763
                                            if float(date) >= 221128:
                                                mmparpixel = 0.1803036313
                                                mmparpixelz = 0.1803036313
                                                mmparpixely = 0.1803036313
                                                if float(date) >= 221214:
                                                    mmparpixel = 0.1823586
                                                    mmparpixelz = 0.1823586
                                                    mmparpixely = 0.1823586
                                                    if float(date) >= 230103:
                                                        mmparpixel = 0.43071886979
                                                        mmparpixelz = 0.43071886979
                                                        mmparpixely = 0.43071886979
                                                        if float(date) >= 230105:
                                                            mmparpixel = 0.42824718
                                                            mmparpixelz = 0.42824718
                                                            mmparpixely = 0.42824718
                                                            if float(date) >= 230110:
                                                                mmparpixel = 0.431276146
                                                                mmparpixelz = 0.431276146
                                                                mmparpixely = 0.431276146
                                                                if float(date) >= 230117:
                                                                    mmparpixel = 0.429258241758
                                                                    mmparpixelz = 0.429258241758
                                                                    mmparpixely = 0.429258241758
                                                                    if float(date) >= 230220:
                                                                        mmparpixel = 0.41809515845806505
                                                                        mmparpixelz = 0.41809515845806505
                                                                        mmparpixely = 0.41809515845806505
                                                                        if float(date) >= 231115:
                                                                            mmparpixel = 0.252947
                                                                            mmparpixelz = 0.252947
                                                                            mmparpixely = 0.252947
                                                                            if float(date) >= 231124:
                                                                                mmparpixel = 0.25134469411350724
                                                                                mmparpixelz = 0.25134469411350724
                                                                                mmparpixely = 0.25134469411350724
                                                                                if float(date) >= 231129:
                                                                                    mmparpixel = 0.3976933784052496
                                                                                    mmparpixelz = 0.3976933784052496
                                                                                    mmparpixely = 0.3976933784052496
                                                                                    if float(date) >= 240109:
                                                                                        mmparpixel = 0.2516229681445322
                                                                                        mmparpixelz = 0.2516229681445322
                                                                                        mmparpixely = 0.2516229681445322
                                                                                    
                                                                                    
                                                                                    
                                          
                                                            
                                                            
                                                        
                                                       
                                                    
                                                
                                               
                                        
                                


                        
        # angle_cam_LAS = np.arccos(mmparpixely/mmparpixelz) * 180 / np.pi
        return mmparpixelx, mmparpixely, mmparpixelz, mmparpixel

    if "FSD" in titre_exp or "PIV" in titre_exp :

        if float(date) >= 220405:
            mmparpixel = 0.04                   #220405 f = 50mm
            if float(date) >= 220407:
                mmparpixel = 0.2196122526067974     #220407 f = 12mm salle 235
                if float(date) >= 220512:
                    mmparpixel = 0.230629922620838      #220512 f = 12mm salle 223
                    if float(date) >= 220516:
                        mmparpixel = 0.230578001345791      #220516 f = 12mm salle 223
                        if float(date) >= 220523:
                            mmparpixel = 0.230433333578193      #220523 f = 12mm salle 223
                            if float(date) >= 220607:
                                mmparpixel = 0.230340502325192
                                if float(date) >= 221006:
                                    mmparpixel = 0.28
                                    if float(date) >= 221011:
                                        mmparpixel = 0.29003161344586
                                        if float(date) >= 221012:
                                            mmparpixel = 0.2001
                                            if float(date) >= 221024:
                                                mmparpixel = 0.19434
                                                if float(date) >= 221116:
                                                    mmparpixel = 0.18763
                                                    if float(date) >= 221128:
                                                        mmparpixel = 0.1803036313
                                                        if float(date) >= 221214:
                                                            mmparpixel = 0.1823586
                                                            if float(date) >= 230103:
                                                                mmparpixel = 0.43071886979
                                                                if float(date) >= 230105:
                                                                    mmparpixel = 0.42824718
                                                                    if float(date) >= 230110:
                                                                        mmparpixel = 0.431276146
                                                                        if float(date) >= 230110:
                                                                            mmparpixel = 0.431276146
                                                                            if float(date) >= 230117:
                                                                                mmparpixel = 0.429258241758
                                                                                if float(date) >= 230220:
                                                                                    mmparpixel =  0.41809515845806505
                                                                                    if float(date) >= 231115:
                                                                                        mmparpixel = 0.252947
                                                                                        if float(date) >= 231124:
                                                                                            mmparpixel = 0.25134469411350724
                                                                                            if float(date) >= 231129:
                                                                                                mmparpixel = 0.3976933784052496
                                                                                                if float(date) >= 240109:
                                                                                                    mmparpixel = 0.2516229681445322
                                                                                            
                                                                                  
                                                                                
                                                                                
                                                                                
                                                                               
                                                                        
                                                                        
                                                                       
                                                            
                                                            
                                                    
        

        return mmparpixel
    
    if  "IND" in titre_exp:
        if float(date) >= 220701 : #TIPP1
             mmparpixely = (0.03305009402751750828731107740002 + 0.03298185668063997994703113817089) /2 #horizontal
             mmparpixelz = (4.175626168548983268432967670966E-2 + 0.0416406412658754944826150322715) / 2  #vertical
             if float(date) >= 220708:
                 if "IJSP2" in titre_exp :
                     mmparpixely = 0.0316139
                     mmparpixelz = (0.0428535 + 0.04309175 + 0.0432496) / 3
                 else :
                     mmparpixely = 0.031963
                     mmparpixelz = 0.042547
                     
                    
            
        
        
        
        
        
        return mmparpixely, mmparpixelz
        
def initialisation(date, nom_exp = '', exp = False, exp_type = "LAS", display = False, dico = 'f', loc = 'f'):
    #à mettre au debut d'un code.
    #cree les parametres, cree fichier resultats, trouve les images, calcul angle laser si besoin
    
    if loc == 'f' :
        if float(date) > 231116 :
            loc = "E:\Baptiste\Resultats_video\d" + date + "\\"
        else :    
            loc = "D:\Banquise\Baptiste\Resultats_video\d" + date + "\\"
            
    # loc = 'W:\\Banquise\\Baptiste\\Resultats_video\\d' + date + "\\"
    
    if dico == 'f' :
        dico = dic.open_dico()
    params = {}
    params['date'] = date
    
    if exp_type == 'IND':
        pass
    
    elif exp :
        params['exp_type'] = exp_type
        params['nom_exp'] = nom_exp

        params.update({'path_images':fm.import_images(loc, nom_exp, exp_type)[0],  'liste_images': fm.import_images(loc, nom_exp, exp_type)[1], 
                       'titre_exp':fm.import_images(loc, nom_exp, exp_type)[2]})
        
        print (params['path_images'])
        
        params.update({'mmparpixelx': import_calibration(params['titre_exp'], date)[0], 'mmparpixely':import_calibration(params['titre_exp'], date)[1], 
                       'mmparpixelz': import_calibration(params['titre_exp'], date)[2], 'mmparpixel': import_calibration(params['titre_exp'], date)[3]})

        
        params.update({'facq': import_param(params['titre_exp'], date)[0], 'texp':import_param(params['titre_exp'], date)[1] ,
               'Tmot': import_param(params['titre_exp'], date)[2], 'Vmot': import_param(params['titre_exp'], date)[3], 
               'Hw' : import_param(params['titre_exp'], date)[4], 'Larg_ice' : import_param(params['titre_exp'], date)[5], 
               'Long_ice': import_param(params['titre_exp'], date)[6], 'tacq': import_param(params['titre_exp'], date)[7]})
        
        params.update({'grossissement' : import_angle(date, nom_exp, loc, display = False)[0], 'er_grossissement':import_angle(date, nom_exp, loc, display = False)[1] , 
               'angle': import_angle(date, nom_exp, loc, display = False)[2], 'er_angle': import_angle(date, nom_exp, loc, display = False)[3]}) 
        
        params['nb_frames'] = len(params['liste_images'])
        
        if display : 
            import_angle(date, nom_exp, loc, display = True)
            
        try:
            os.mkdir(params['path_images'][:-15] + 'resultats')
        except:
            pass
        
        if float(date) >= 231115 :
            params['fexc'] = params['Tmot']/10
         
        if not(date in dico.keys()): 
            dico[date] = {}
            
        if not(nom_exp in dico[date].keys()):
            dico[date][nom_exp] = {}
            dico[date][nom_exp] = params
            
            dic.save_dico(dico)
            
        params['save_folder_exp'] = "E:\\Baptiste\\Resultats_exp\\d" + date + "\\" + nom_exp
        
        try:
            os.mkdir(params['path_images'][:-15] + 'resultats')
        except:
            pass
        
    return dico, params, loc
        


