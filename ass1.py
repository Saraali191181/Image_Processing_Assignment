# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import skimage.color
import skimage.io
import cv2 

#read
image = cv2.imread('Desktop\Image_processor_assignment\img_1.PNG')
 

# reads an input image
image = cv2.imread('Desktop\Image_processor_assignment\img_1.PNG',0)
  
#show
print(image)

image.shape
image.max()
image.min()

import matplotlib.pyplot as plt
plt.figure(figsize=(10,10)) # 10, 10: are the width and the height of the 
                            # figure containing the image.
plt.imshow(image) # Show the image in the figure above.
plt.show() # Show the figure with the image.


# read the image  as grayscale from the outset
image = skimage.io.imread(fname='Desktop\Image_processor_assignment\img_1.PNG', as_gray=True)

#display the image
fig, ax = plt.subplots()
plt.imshow(image, cmap='gray')
plt.show()



# find frequency of pixels in range 0-255
histr = cv2.calcHist([image],[0],None,[256],[0,256])

# show the plotting graph of an image
plt.plot(histr)
plt.show()


# create the histogram
histogram, bin_edges = np.histogram(image, bins=256, range=(0, 1))

# configure and draw the histogram figure
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixel count")
plt.xlim([0.0, 1.0])  # <- named arguments do not work here

# read original image, in full color
image = skimage.io.imread('Desktop\Image_processor_assignment\img_1.PNG')

# display the image
fig, ax = plt.subplots()
plt.imshow(image)
plt.show()


# alternative way to find histogram of an image
plt.hist(image.ravel(),256,[0,256])
plt.show()
