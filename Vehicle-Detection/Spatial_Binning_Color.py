# -*- coding: utf-8 -*-
"""
Created on Sat May 12 22:27:42 2018

@author: Vamshidhar P
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in an image
# You can also read cutout2, 3, 4 etc. to see other examples
image = mpimg.imread('cutout1.jpg')

# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(image[:,:,0], bins=32, range=(0, 256))
    ghist = np.histogram(image[:,:,1], bins=32, range=(0, 256))
    bhist = np.histogram(image[:,:,2], bins=32, range=(0, 256))
    
    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    
    # Concatenate RGB Histograms
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return rhist, ghist, bhist, bin_centers, hist_features


# Pass the color_space flag as 3-letter all caps string
# like 'HSV' or 'LUV' etc.
# KEEP IN MIND IF YOU DECIDE TO USE THIS FUNCTION LATER
# IN YOUR PROJECT THAT IF YOU READ THE IMAGE WITH 
# cv2.imread() INSTEAD YOU START WITH BGR COLOR!
def bin_spatial(img, color_space='RGB', size=(32, 32)):
    # Convert image to new color space (if specified)
    # Use cv2.resize().ravel() to create the feature vector
    
    if color_space != 'RGB':
        if color_space == 'HSV':
            new_color_space = cv2.COLOR_RGB2HSV
        elif color_space == 'HLS':
            new_color_space = cv2.COLOR_RGB2HLS
        elif color_space == 'LUV':
            new_color_space = cv2.COLOR_RGB2LUV
        elif color_space == 'YUV':
            new_color_space = cv2.COLOR_RGB2YUV
        elif color_space == 'LUV':
            new_color_space = cv2.COLOR_RGB2YCrCb      
        img_converted = cv2.cvtColor(img, new_color_space)
    else:
        img_converted = np.copy(img)
        
    
    img_small = cv2.resize(img_converted, size)
    features = img_small.ravel() 
    # Return the feature vector
    return features
    
feature_vec = bin_spatial(image, color_space='HSV', size=(32, 32))

# Plot features
plt.plot(feature_vec)
plt.title('Spatially Binned Features')