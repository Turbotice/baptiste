# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 14:32:41 2020

@author: alice
"""

import math
import matplotlib.pyplot as plt
import numpy as np
 

from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
# from skimage.feature import canny
from skimage.draw import circle_perimeter
# from skimage.util import img_as_ubyte
import datetime
from pims import ImageSequence
from skimage import filters


display=True
radius_analysis=True
histogram=True
save= False#Pour la sauvegarde des données
plot=True
edges_detection='Scharr' # ou canny


echelles={'60microns':1/0.646} #key is the particle diameter, associated with the number of pxl/micron


################################################################
D='60microns'# microns
# D='BS572 300'# microns

thresh=0.1 # For the edge filter, lower if you want more signal
r_min,r_max=5,50 # pxl
#r_min,r_max=30,80 # pxl
hough_radii = np.array(list((np.arange(r_min,r_max,1))))
hough_thresh=0.5 # In Hough space, lower if you want to be more permissive
min_distance=20 # min distance between two particles

################################################################
##################   IMAGES OF PARTICLES    ####################
################################################################

# path=r'E:/Qlice/Granulometrie/PMMA Arkema 3 mars/PMMA Arkema 2mars2021/{}microns/Petites/'.format(D) # path where picture are stored
# path=r'E:/Qlice/Granulometrie/PMMA Arkema 3 mars/PMMA Arkema 2mars2021/{}microns/Moyennes/'.format(D) # path where picture are stored
# path=r'E:/Qlice/Granulometrie/PMMA Arkema 3 mars/PMMA Arkema 2mars2021/{}microns/Grosses/'.format(D) # path where picture are stored
path=r'./Volumes/Expansion/PhD_Gabriel/images_particules_piv03112021/'

images = ImageSequence('Image1.tif') # name of the image sequence



def groups(a,b,d):
    '''
    makes groups of close doublets such that the distance 
    separating them is below d (pxl distance)
    a,b are the X,Y coordinates
    '''  
    doublets=[] #list of close doublets
    index=[] #indices already places in different neighboor groups        
    group=[] #indices of points in the same group (list of lists)    
    for k in range(len(a)):
        for l in range(len(a[k:])):
            # print((k,l))
            if (a[k]-a[l+k])**2+(b[k]-b[l+k])**2<=d**2:
                doublets.append((k,l+k))

        for (k,l) in doublets:
            if k not in index:
                group.append([k])
                index.append(k)
            elif k in index and l not in index:
                for i in range(len(group)):
                    if k in group[i]:
                        group[i].append(l)
                        index.append(l)
    # return a,b, doublets,index, group
    return group

        
def filter_Hough(x,y,r,accums,group):
    '''
    Given the indices of points belonging to the same group,
    compute the averange position x_final, y_final, average accumulator 
    and average radius of the differents groups
    '''
    r_final,x_final,y_final,accums_final=[],[],[],[]
    for list_k in group: #In a given group
        radius_k,x_k,y_k,accums_k=0,0,0,0 #Variables to store the group properties
        for k in list_k: # for all the elements of the group
#            print(list_k,k)
            radius_k+=(r[k])
            x_k+=(x[k])
            y_k+=(y[k])
            accums_k+=(accums[k])
        radius_k=radius_k/len(list_k) #Makes the averages of the group properties
        x_k=x_k/len(list_k)
        y_k=y_k/len(list_k)
        accums_k=accums_k/len(list_k)
        
        r_final.append(radius_k)
        x_final.append(x_k)
        y_final.append(y_k)
        accums_final.append(accums_k)       
    return x_final,y_final,r_final,accums_final
        
if radius_analysis:
    Hough_radii={}
    for i in range(0,1):
        print('Analyzing frame '+str(i))
        # Loading of the image to analyze
        image_gray = images[i]  #if pictures are already in gray-level  
        I_min,I_max=np.min(image_gray),np.max(image_gray)
        image_gray=(image_gray-I_min)/(I_max-I_min) #Contrast enhancment
#        image_gray=image_gray[0:750,:700]   #If you want to crop pictures
        if display:
            fig, ([[ax0,ax1],[ax2,ax3]]) = plt.subplots(ncols=2, nrows=2, figsize=(8, 8))
            ax0.imshow(image_gray, cmap=plt.cm.gray)
            ax0.set_title('Frame i={}/{}'.format(str(i),str(len(images)-1)))       
 
        # Edges of particles --> parameter thresh
        if edges_detection=='Scharr' or edges_detection=='scharr':
            edges=filters.scharr(image_gray)>thresh
        elif edges_detection=='canny':
            edges = canny(image_gray, sigma=sigma_thresh,low_threshold=low_thresh, high_threshold=high_thresh)>thresh
        else:
            print('Unknown edges detection method')
        if display :
            ax1.imshow(edges, cmap=plt.cm.gray)
            ax1.set_title('Edges detection = '+edges_detection+'\n Threshold='+str(thresh))
            fig.show()
        
        if radius_analysis:        
            print('Performing Hough transform')
            hough_res = hough_circle(edges, hough_radii)
            print('Transposition in real space')
            accums, cx, cy, radii = hough_circle_peaks(hough_res,hough_radii,
                                                       min_xdistance=20, min_ydistance=20,
                                                       threshold=hough_thresh) 
            print('Making groups of redondant centers')
            redondant=groups(cx,cy,min_distance) # last parameter is the max-distance in pxl 
            print('Removing duplicates')
            cx,cy,radii,accums=filter_Hough(cx,cy,radii,accums,redondant)
            
            image = np.ones(np.shape(image_gray)) #Picture where detected particles will be drawn
            
            N=0 #Number of particles detected
            for center_y, center_x, radius in zip(cy, cx, radii):
                N+=1
                #Plot of the detected particle
                circy, circx = circle_perimeter(int(center_y), int(center_x), int(radius),
                                                shape=image.shape)
                image[circy, circx] = 0
            if display:
                ax2.imshow(image+image_gray/5, cmap=plt.cm.viridis)
                ax2.set_title('N={} particles detected \n Threshold in dual space = {}'.format(N, hough_thresh))
            Hough_radii[i]=radii
        L=[]
        for i in sorted(Hough_radii):
            L+=list(Hough_radii[i])
        ax3.hist(L,bins=100)
        R=np.mean(L)/echelles[D]*2
        ax3.axvline(R*echelles[D]/2,color='r',linestyle='--')
        ax3.set_title('{} microns, N={}, \n Average diameter={:.2f} microns,\n Standart deviation={:.2f}'.format(D, len(L),R,np.std(L)))
        ax3.set_xlabel('Radius [pxl]')
        fig.tight_layout()
        fig.savefig('\Hough_analysis_{}.png'.format(i),dpi=200)
        plt.show()

if save:
    x = datetime.datetime.now()
   
    file = open("Parameters_Hough_analysis.txt", "a")
    file.write('\n ### Date : '+str(x))
    file.write('\n'+path+'\n') 
    file.write('Edges : scharr threshold={}\n'.format(thresh)) 
#    file.write('Hough transform : [r_min:r_max]=[{}:{}] pxl, hough threshold={}\n'.format(r_min,r_max,hough_thresh)) 
    file.close()     
    np.savetxt(str(D)+'microns_Hough_rmin{}_rmax{}pxl.txt'.format(r_min,r_max),L)
    print('Saving successed !')
    
 
plt.show()
