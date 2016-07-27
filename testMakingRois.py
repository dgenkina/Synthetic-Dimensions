# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 11:22:06 2016

@author: dng5
"""

import numpy as np
import readIgor
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def getRoi(array, image, xCent,yCent,r=10,eps=5):
    ylen=array.shape[0]
    xlen=array.shape[1]
    bbox=(xCent-r,yCent-np.int(np.sqrt(r*r-eps*eps)),xCent+r,yCent+np.int(np.sqrt(r*r-eps*eps)))
    
    y,x = np.ogrid[-yCent:ylen-yCent, -xCent:xlen-xCent]
    mask = np.sqrt((x-eps)**2.0 + y*y) +np.sqrt((x+eps)**2.0 + y*y) <= 2.0*r

    counts=np.sum(array[mask])
    draw = ImageDraw.Draw(image)
    
    draw.ellipse(bbox,outline=(255,0,0))
    return counts
    
roi=np.array([400, 640, 400, 660])
filestart=325
filestop=208
fileroot = 'Y:/Data/2016/March/15/PIXIS_15Mar2016'  

dict1 =readIgor.processIBW(fileroot+"_"+ str(filestart).zfill(4) + ".ibw", angle=-37)

odRoi=dict1['rotOD'][roi[0]:roi[1],roi[2]:roi[3]]  
#yCent, xCent =200,180

norm=mpl.colors.Normalize(vmin=-0.15,vmax=0.3)
im = Image.fromarray(np.uint8(plt.cm.jet(norm(odRoi))*255))

counts00= getRoi(odRoi, im, xCent,yCent)
counts0p= getRoi(odRoi, im, xCent,yCent+69)
counts0m= getRoi(odRoi, im, xCent,yCent-69)
countsp0= getRoi(odRoi, im, xCent-58,yCent+49)
countspp= getRoi(odRoi, im, xCent-58,yCent+119)
countspm= getRoi(odRoi, im, xCent-58,yCent-21)
countsm0= getRoi(odRoi, im, xCent+57,yCent-47)
countsmp= getRoi(odRoi, im, xCent+57,yCent+21)
countsmm= getRoi(odRoi, im, xCent+57,yCent-115)
imshow(im)
    
im.save('local.png')


#draw.line((0, 0) + im.size, fill=128)
#draw.line((0, im.size[1], im.size[0], 0), fill=128)
#del draw 