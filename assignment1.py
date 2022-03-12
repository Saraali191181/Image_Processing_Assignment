# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 21:55:47 2022

@author: hp
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import skimage.color
import skimage.io

img = cv.imread('Desktop\Image_processor_assignment\img_1.PNG',0)

# display the image
fig, ax = plt.subplots()
plt.imshow(img)
plt.show()

hist,bins = np.histogram(img.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()
plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')

cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')

img2 = cdf[img]

img = cv.imread('Desktop\Image_processor_assignment\img_1.PNG',0)
equ = cv.equalizeHist(img)
res = np.hstack((img,equ)) #stacking images side-by-side
cv.imwrite('res.png',res)

# create a CLAHE object (Arguments are optional).
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)
cv.imwrite('Desktop\Image_processor_assignment\img_1.PNG',cl1)

# read the image  as grayscale from the outset
image = skimage.io.imread(fname='Desktop\Image_processor_assignment\img_1.PNG', as_gray=True)

#display the image
fig, ax = plt.subplots()
plt.imshow(image, cmap='gray')
plt.show()
