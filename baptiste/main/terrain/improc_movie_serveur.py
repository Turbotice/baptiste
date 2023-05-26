import os
import glob
import matplotlib.pyplot as plt
import imageio
import numpy as np

from PIL import Image

datafolder = '/volume3/labshared2/Banquise/Rimouski_2023/Data/drone/'

file_mov = glob.glob(datafolder+'*/contexte/video/*.MOV')
file_mp4 = glob.glob(datafolder+'*/contexte/video/*.MP4')

filelist = file_mov+file_mp4
print(len(filelist))


for i,filename in enumerate(file_mov+file_mp4):
    print(i,filename)


for i,filename in []:#enumerate(filelist):
    vid = imageio.get_reader(filename,  'ffmpeg')
    nums = [0]
    
    directory = os.path.dirname(filename)
    print(directory)
    print(i,os.path.basename(filename))
    savefolder = directory+'/'+os.path.basename(filename).split('.')[0]+'_images/'
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
        
    
    for num in nums:
        data = vid.get_data(num)
        print(data.shape)
        #im = Image.fromarray(data)
        savename = savefolder+'im_'+str(num)+'.png'
        #im.save(savename)
        #np.save(savename,image)
        #fig = plt.figure()
        #fig.suptitle('image #{}'.format(num), fontsize=20)
        #plt.imshow(image)
        #plt.savefig(savename,format='png')
#    pylab.show()
