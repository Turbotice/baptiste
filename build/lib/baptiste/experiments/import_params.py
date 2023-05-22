# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 13:52:05 2023

@author: Banquise

Toutes les fonctions qi permettent d'importer les parmaètres expérimentaux des expériences depuis les titres et autre'
"""

import matplotlib.pyplot as plt
import baptiste.files.file_management as fm
import numpy as np
import baptiste.files.dictionaries as dic




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
        if float(date) >= 221020 :  
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
        type_exp =  str (titre_exp[14:17])    #FSD pour cassage glace, FCD pour mesure damier, LAS pour profil laser
        return facq, texp, Tmot, Vmot, Hw, Larg_ice, Long_ice, tacq, type_exp
    if exp_type == "IND":
        h = float ( titre_exp[titre_exp.index("ha") + 2:titre_exp.index("ha") + 4])
        return h




def import_angle (date, nom_exp, loc, display = False):
    """
    Trouve les paramètres de la nappe laser associée à une expérience depuis le fichier texte rempli avec ces paramètres

    Parameters
    ----------
    date : TYPE
        DESCRIPTION.
    nom_exp : TYPE
        DESCRIPTION.
    loc : racine des données.
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
    
    if display :
        plt.figure()
        plt.plot(dx,dh,'ko')
        

    hprime = np.polyfit(dx,dh,1)

    dh_prime = dh - hprime[1]
    xxx = np.linspace(0,max(dx),200)
    
    if display :
        plt.plot(xxx, hprime[0] * xxx + hprime[1])
        plt.xlabel("Distance laser")
        plt.ylabel("h mesuré")
    
    tab_angles = np.arctan(dh_prime/dx) * 180 / np.pi

    angle = np.mean(tab_angles)
    xxx = np.linspace(1,len(dh_prime) + 1, len(dh_prime))
    er_angle = (max(tab_angles) - min(tab_angles))/2
    if display :
        plt.figure()
        plt.plot(dx, dh_prime, 'mx')
        plt.figure()
        plt.plot(xxx, tab_angles, 'ko')
        plt.xlabel("Mesure")
        plt.ylabel("Angle (en radian)")
        plt.title('Angles')
        print('angle moyen : ', angle, 'erreur', er_angle )
        plt.figure()
        plt.plot(dx, tab_angles, 'ko')
        plt.xlabel("distance laser")
        plt.ylabel("Angle (en radian)")
    grossissement = 1 / np.tan(angle* np.pi / 180)
    grossissements = 1 / np.tan(tab_angles* np.pi / 180)
    er_grossissement = (max(grossissements) - min (grossissements))/2
    if display :
        print('grossissement :',grossissement, "erreur ", er_grossissement)
    
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
        
def initialisation(exp = False, titre_exp = '',date = '', exp_type = "TT", loc = '', nom_exp = ''):
    dico = dic.open_dico()
    params = {}
    if exp :
        params['facq, texp, Tmot, Vmot, Hw, Larg_ice, Long_ice, tacq, type_exp'] = import_param(titre_exp, date, exp_type)
        params['mmparpixelx, mmparpixely, mmparpixelz, mmparpixel'] = import_angle(date, nom_exp, loc, display = False)
        
    return dico, params
        


