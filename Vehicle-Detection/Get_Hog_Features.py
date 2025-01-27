# -*- coding: utf-8 -*-
"""
Created on Sat May 12 23:48:58 2018

@author: Vamshidhar P
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from skimage.feature import hog

# Read in our vehicles
car_images = glob.glob('vehicles/GTI_Far/*.png') + glob.glob('vehicles/GTI_Left/*.png') + \
             glob.glob('vehicles/GTI_MiddleClose/*.png') + glob.glob('vehicles/GTI_Right/*.png') + \
             glob.glob('vehicles/KITTI_extracted/*.png')
        
# Define a function to return HOG features and visualization
# Features will always be the first element of the return
# Image data will be returned as the second element if visualize= True
# Otherwise there is no second return element

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis = True, 
                     feature_vec=True):
                         
    # TODO: Complete the function body and returns
    if vis:        
        hog_features, hog_image = hog(img, orientations = orient,
                                      pixels_per_cell = (pix_per_cell, pix_per_cell), 
                                      cells_per_block = (cell_per_block, cell_per_block), 
                                      visualise = vis, feature_vector = feature_vec,
                                      block_norm="L2-Hys",
                                      transform_sqrt=True)
        return hog_features, hog_image
    else:
        hog_features = hog(img, orientations = orient,
                                      pixels_per_cell = (pix_per_cell, pix_per_cell), 
                                      cells_per_block = (cell_per_block, cell_per_block), 
                                      visualise = vis, feature_vector = feature_vec,
                                      block_norm="L2-Hys",
                                      transform_sqrt=True)
        return hog_features




# Generate a random index to look at a car image
ind = np.random.randint(0, len(car_images))
# Read in the image
image = mpimg.imread(car_images[ind])
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Call our function with vis=True to see an image output
features, hog_image = get_hog_features(gray, orient = 9, 
                                       pix_per_cell = 8, cell_per_block = 2, 
                                       vis = True, feature_vec = True)
print(features.shape)

# Plot the examples
fig = plt.figure()
plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title('Example Car Image')
plt.subplot(122)
plt.imshow(hog_image, cmap='gray')
plt.title('HOG Visualization')