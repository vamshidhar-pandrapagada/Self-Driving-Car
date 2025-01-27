# -*- coding: utf-8 -*-
"""
Created on Sun May 13 20:57:19 2018

@author: Vamshidhar P
"""

import numpy as np
import cv2
from skimage.feature import hog


def convert_color(img, cspace = 'RGB'):
    if cspace != 'RGB':
        if cspace == 'HSV':
            new_color_space = cv2.COLOR_RGB2HSV
        elif cspace == 'HLS':
            new_color_space = cv2.COLOR_RGB2HLS
        elif cspace == 'LUV':
            new_color_space = cv2.COLOR_RGB2LUV
        elif cspace == 'YUV':
            new_color_space = cv2.COLOR_RGB2YUV
        elif cspace == 'HLS':
            new_color_space = cv2.COLOR_RGB2HLS   
        elif cspace == 'RGB2YCrCb':
            new_color_space = cv2.COLOR_RGB2YCrCb
        elif cspace == 'BGR2YCrCb':
            new_color_space = cv2.COLOR_BGR2YCrCb
        elif cspace == 'BGR2LUV':
            new_color_space = cv2.COLOR_BGR2LUV
        img_converted = cv2.cvtColor(img, new_color_space)
    else:
        img_converted = np.copy(img)   
    return img_converted

# Define a function to compute binned color features  
def bin_spatial(img, size = (32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features  
def color_hist(img, nbins = 32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis = False, 
                     feature_vec=True, hog_channel =  'ALL'):
                         
    # TODO: Complete the function body and returns
    if hog_channel == 'ALL':
            hog_features = []
            for channel in range(img.shape[2]):               
                hog_features_channel = hog(img[:,:,channel], orientations = orient,
                                           pixels_per_cell = (pix_per_cell, pix_per_cell), 
                                           cells_per_block = (cell_per_block, cell_per_block), 
                                           visualise = vis, feature_vector = feature_vec,
                                           block_norm="L2-Hys",
                                           transform_sqrt=False)
                hog_features.append(hog_features_channel)
            hog_features = np.ravel(hog_features)
    else:
        if vis:        
            hog_features, hog_image = hog(img, orientations = orient,
                                          pixels_per_cell = (pix_per_cell, pix_per_cell), 
                                          cells_per_block = (cell_per_block, cell_per_block), 
                                          visualise = vis, feature_vector = feature_vec,
                                          block_norm="L2-Hys",
                                          transform_sqrt=False)
            
            return hog_features, hog_image
                
            
        else:
            hog_features = hog(img, orientations = orient,
                                          pixels_per_cell = (pix_per_cell, pix_per_cell), 
                                          cells_per_block = (cell_per_block, cell_per_block), 
                                          visualise = vis, feature_vector = feature_vec,
                                          block_norm="L2-Hys",
                                          transform_sqrt=False)
                
    return hog_features  

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy
    
    
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    # Compute the span of the region to be searched    
    # Compute the number of pixels per step in x/y
    # Compute the number of windows in x/y
    # Initialize a list to append window positions to
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


