# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 16:48:59 2023

@author: Banquise
"""

#%% detecton contours
frac_auto = False

if frac_auto :
    n_fracs = 5 #nombre de fractures à detecter
    
    fracs = np.zeros((n_fracs,2))
    
    #trouve les contours et garde les n_fracs plus gros (en aire)
    
    contours, hierarchy = cv2.findContours(erod_G_2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_0 = contours # if imutils.is_cv2() else contours[1]  
    cntsSorted = sorted(contours_0, key=lambda x: cv2.contourArea(x))
    
    big_contours = cntsSorted[-n_fracs:]
    
    img_contours_binaire = np.zeros(np.append(data.shape, 3), np.uint8)
    
    cv2.drawContours(img_contours_binaire,big_contours,-1, color = (255,255,255), thickness = 4)
    
    figurejolie()
    plt.pcolormesh(img_contours_binaire[:,:,0],shading='auto')
    
    #trouve le debut en t et le milieu en x des ces contours ce qui donne t_0 et x_0 frac
    for i in range (len(big_contours)):
        # dans fracs on a les t_0 et x_0 des fractures
        fracs[i,0] = np.min(big_contours[i][:,:,0])
        fracs[i,1] = np.mean(big_contours[i][:,:,1])
else :
    n_fracs = 6
    fracs = np.zeros((n_fracs,2))
    fracs[0,0] = 3270 # t_0_frac 
    fracs[0,1] = 526 # x_0_frac !!!!!!! ATTENTION x imageJ = nx- x pour python !!!!!!!
    fracs[1,0] = 3750
    fracs[1,1] = 640
    fracs[2,0] = 3730
    fracs[2,1] = 848
    fracs[3,0] = 3700
    fracs[3,1] = 950
    fracs[4,0] = 5700
    fracs[4,1] = 1271
    fracs[5,0] = 5450
    fracs[5,1] = 1574
    
    
    

#%% Amp moyenne avant frac
n_periodes = 10
padding = 12
padpad = True
n_periodesavantfracture = 0

save_fig = False

liste_noeuds = [0,2]
liste_frac = [0,1,2,3,4,5]


fexc = dico[date][nom_exp]['fexc'] # Hz
facq = dico[date][nom_exp]['facq'] # Hz
longueur_donde = int(dico[date][nom_exp]['lambda'] / mmparpixel * 1000 / 2) # pixel
t_nperiodes = (int(facq/fexc) + 1) * n_periodes #durée regardée

amp_fracs_complet = np.zeros((n_fracs,2,longueur_donde,2 ), dtype = np.ndarray)

amp_fracs_fft = np.zeros((n_fracs,8)) #pour sauvegarder amp(f_0), idem avec onde autre sens, t_0, et x(amp_max), et amp_RMS


for j in range(n_fracs) :#fracs.shape[0]) :
    
    #AMP AVEC FFT
    if padpad :
        t_nperiodes = (int(facq/fexc) + 1) * n_periodes
    
    x_0_frac = int(fracs[j,1] - longueur_donde/2)
    x_f_frac = int(fracs[j,1] + longueur_donde/2)
    t_n_frac = int(fracs[j,0] - t_nperiodes - int(t_nperiodes/n_periodes * n_periodesavantfracture))
    t_0_frac = int(fracs[j,0] - int(t_nperiodes/n_periodes * n_periodesavantfracture) )
    
    amp_fracs_complet = np.zeros((2,x_f_frac - x_0_frac,2 ), dtype = np.ndarray)
    
    figurejolie()
    sum_real = []
    sum_FFT = []
    
    
    if j in liste_noeuds :
        i = int(fracs[j,1])
        meann = np.mean(data[i,t_n_frac:t_0_frac])
        
        data_used = data[i,t_n_frac:t_0_frac] - meann #data qui nous interesse
        
        Y1 = fft.fft(data_used)
        # Y_12 = Y1.copy()
        if padpad :
            data_padding = np.append(data_used, np.zeros(2**padding - (t_0_frac-t_n_frac)))
            Y1_padding = fft.fft(data_padding)
            Y1 = Y1_padding.copy()

        
        Y_t = Y1[:int(len(Y1)/2)] #sens t
        Y_moinst = Y1[int(len(Y1)/2):] #sens moins t
            
        #normalisation
        P2_t = np.abs(Y_t)#/ (t_0_frac-t_n_frac))  
        P2_moinst = np.abs(Y_moinst)
        
        P2 = np.append(P2_t,P2_moinst) #abs(FFT)
        if padpad :
            t_nperiodes = 2**padding
            
        f = facq * np.arange(0,int(t_nperiodes)) / t_nperiodes    
                
        peaks, _ = find_peaks(P2,prominence = np.max(P2)/4, width=(1,50))
    
        peaks_select = np.append(peaks[:1], peaks[-1:]) #que le fondamental et le retour
            
        # peaks_select = np.append(np.argmax(P2_t), np.argmax(P2_moinst))
        
        amp_fracs_complet[0,i - x_0_frac, 0] = peaks_select[0]
        amp_fracs_complet[0,i - x_0_frac, 1] = peaks_select[1] #+ int(len(Y1)/2)
        amp_fracs_complet[1,i - x_0_frac, 0] = P2[peaks_select][0] #P2_t[peaks_select][0]
        amp_fracs_complet[1,i - x_0_frac, 1] = P2[peaks_select][0] #P2_moinst[peaks_select][1]  

        dt = 1/ facq
        sum_amp_reelle = np.sum ( np.abs(data_used) **2) / (t_0_frac-t_n_frac)
        sum_real.append(sum_amp_reelle)
        
        # df = facq / t_nperiodes
        sum_amp_FFT = np.sum(np.abs(P2 /np.sqrt(t_0_frac-t_n_frac) )**2 ) / t_nperiodes
        sum_FFT.append(sum_amp_FFT)
        
        joliplot("f (Hz)", "Amplitude (m)", f, P2 / (t_0_frac-t_n_frac), exp = False)
        plt.plot(amp_fracs_complet[0,int(fracs[j,1]) - x_0_frac] * facq /t_nperiodes, amp_fracs_complet[1,int(fracs[j,1]) - x_0_frac] / (t_0_frac-t_n_frac), 'ro', label = "pics trouvés")

        
        
    else :
        for i in range (x_0_frac, x_f_frac):
            meann = np.mean(data[i,t_n_frac:t_0_frac])
            
            data_used = data[i,t_n_frac:t_0_frac] - meann #data qui nous interesse
            
            Y1 = fft.fft(data_used)
            # Y_12 = Y1.copy()
            if padpad :
                data_padding = np.append(data_used, np.zeros(2**padding - (t_0_frac-t_n_frac)))
                Y1_padding = fft.fft(data_padding)
                Y1 = Y1_padding.copy()
    
            
            Y_t = Y1[:int(len(Y1)/2)] #sens t
            Y_moinst = Y1[int(len(Y1)/2):] #sens moins t
                
            #normalisation
            P2_t = np.abs(Y_t)#/ (t_0_frac-t_n_frac))  
            P2_moinst = np.abs(Y_moinst)
            
            P2 = np.append(P2_t,P2_moinst) #abs(FFT)
            if padpad :
                t_nperiodes = 2**padding
                
            f = facq * np.arange(0,int(t_nperiodes)) / t_nperiodes    
                    
            peaks, _ = find_peaks(P2,prominence = np.max(P2)/4, width=(1,50))
        
            peaks_select = np.append(peaks[:1], peaks[-1:]) #que le fondamental et le retour
                
            # peaks_select = np.append(np.argmax(P2_t), np.argmax(P2_moinst))
            
            amp_fracs_complet[0,i - x_0_frac, 0] = peaks_select[0]
            amp_fracs_complet[0,i - x_0_frac, 1] = peaks_select[1] #+ int(len(Y1)/2)
            amp_fracs_complet[1,i - x_0_frac, 0] = P2[peaks_select][0] #P2_t[peaks_select][0]
            amp_fracs_complet[1,i - x_0_frac, 1] = P2[peaks_select][0] #P2_moinst[peaks_select][1]  
    
            dt = 1/ facq
            sum_amp_reelle = np.sum ( np.abs(data_used) **2) / (t_0_frac-t_n_frac)
            sum_real.append(sum_amp_reelle)
            
            # df = facq / t_nperiodes
            sum_amp_FFT = np.sum(np.abs(P2 /np.sqrt(t_0_frac-t_n_frac) )**2 ) / t_nperiodes
            sum_FFT.append(sum_amp_FFT)
            
                
                
            if np.mod(i,20)==0:
                joliplot("f (Hz)", "Amplitude (m)", f, P2 / (t_0_frac-t_n_frac), exp = False)
                plt.plot(amp_fracs_complet[0,i - x_0_frac] * facq /t_nperiodes, amp_fracs_complet[1,i - x_0_frac] / (t_0_frac-t_n_frac), 'ro', label = "pics trouvés")
    
    amp_fracs_fft[j,0] = t_0_frac #t_0_frac
    amp_fracs_fft[j,1] = np.argmax(amp_fracs_complet[1,:,0]) + x_0_frac #là où on trouve le max, donc x_0 frac (en supposant que ca casse sur le max)
    amp_fracs_fft[j,2] = np.max(amp_fracs_complet[1,:,0]) / (t_0_frac-t_n_frac) #amp(f_0)
    amp_fracs_fft[j,3] = np.max(amp_fracs_complet[1,:,1]) / (t_0_frac-t_n_frac) #amp(f_i)
    if j in liste_noeuds :
        amp_fracs_fft[j,4] = np.sqrt(np.max(sum_real)) #amp RMS là où le pic FFT est le plus fort
    else :
        amp_fracs_fft[j,4] = np.sqrt(sum_real[int(amp_fracs_fft[j,1] - x_0_frac)]) #amp RMS là où le pic FFT est le plus fort
    amp_fracs_fft[j,5] = amp_fracs_fft[j,2]/ amp_fracs_fft[j,4] #part d'energie dans le pic 
    
    amp_fracs_fft[j,6] = False #si ca casse sur cette fracture
    amp_fracs_fft[j,7] = False #si c'est peut être sur un noeud
    if j in liste_frac :
        amp_fracs_fft[j,6] = True #si ca casse sur cette fracture
    if j in liste_noeuds :
        amp_fracs_fft[j,7] = True #si c'est peut être sur un noeud


    plt.plot(amp_fracs_complet[0,i - x_0_frac] * facq /t_nperiodes, np.append(amp_fracs_fft[j,2], amp_fracs_fft[j,3]), 'mx', label = "maxs")
    if save_fig :
        plt.savefig(path_images[:-15] + "resultats" + "/" + "FFT_avec_pics_trouves_fracture_" + str(j) + ".pdf", dpi = 1)
        
    figurejolie()
        
    if padpad :
        t_nperiodes = (int(facq/fexc) + 1) * n_periodes
    
    x_amp = np.linspace(0, mmparpixel * (x_f_frac - x_0_frac) / 1000, x_f_frac - x_0_frac)
    if j in liste_noeuds :
        joliplot('X (m)','Amplitude RMS (m)',amp_fracs_fft[j,1] * mmparpixel / 1000, np.sqrt(np.max(sum_real)), legend = 'amp RMS(x) frac numero ' + str(j), exp = False, color = 3 )
    else :
        joliplot('X (m)','Amplitude RMS (m)',x_amp, np.sqrt(sum_real), legend = 'amp RMS(x) frac numero ' + str(j), exp = False, color = 3 )
        
    # plt.plot( (amp_fracs_fft[j,1] - x_0_frac) * mmparpixel / 1000, amp_fracs_fft[j,2], 'ko')
    plt.plot( (amp_fracs_fft[j,1] - x_0_frac) * mmparpixel / 1000, amp_fracs_fft[j,3], 'ro', label = "pic max FFT")
    plt.plot( (amp_fracs_fft[j,1] - x_0_frac) * mmparpixel / 1000, amp_fracs_fft[j,4], 'go', label = "pic max RMS")
    plt.legend()
     
    if save_fig :
        plt.savefig(path_images[:-15] + "resultats" + "/" + "Amplitude RMS (x)_fracture_" + str(j) + ".pdf", dpi = 1)
    figurejolie()
    t_pixel_max = np.linspace(amp_fracs_fft[j,0] - t_nperiodes, amp_fracs_fft[j,0], t_nperiodes)
    x_pixel_max = int(amp_fracs_fft[j,1])
    meann = np.mean(data[x_pixel_max,t_n_frac:t_0_frac])
    data_used = data[x_pixel_max,t_n_frac:t_0_frac] - meann
    joliplot("t","x", t_pixel_max, data_used, exp = False, color = 3, legend = 'Y(t) sur la crete, de t = ' + str(round(t_pixel_max[0])) + " à t = " + str(round(t_pixel_max[-1])) + " pour x = " + str(x_pixel_max))
    if save_fig :
        plt.savefig(path_images[:-15] + "resultats" + "/" + "amp(t)_x" + str(int(amp_fracs_fft[j,1])) + "_fracture_" + str(j) + ".pdf", dpi = 1)
    print("Parseval = ",np.mean(np.array(sum_FFT)/np.array(sum_real)))
    # figurejolie()
    # plt.plot(sum_FFT, label = 'sum FFT')
    # plt.plot(sum_real, label = 'sum reelle')
    # plt.legend()
    
    
print(amp_fracs_fft) #t_0, x_0, pic 1, pic 2, amp RMS pour chaque fracture
