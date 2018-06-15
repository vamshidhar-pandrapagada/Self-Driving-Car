# -*- coding: utf-8 -*-
"""
Created on Sat May 12 23:58:22 2018

@author: Vamshidhar P
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
import glob
from sklearn.model_selection import train_test_split
import time
from sklearn.svm import LinearSVC
from skimage.feature import hog

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

###### TODO ###########
# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace ='RGB', spatial_size = (32, 32),
                        hist_bins = 32, hist_range = (0, 256)):
    # Create a list to append feature vectors to
    features = []    
    
    # Iterate through the list of images
        # Read in each one by one
        # apply color conversion if other than 'RGB'
        # Apply bin_spatial() to get spatial color features
        # Apply color_hist() to get color histogram features
        # Append the new feature vector to the features list
    # Return list of feature vectors
    for img_path in imgs:
        img = mpimg.imread(img_path)
        if cspace != 'RGB':
            if cspace == 'HSV':
                new_color_space = cv2.COLOR_RGB2HSV
            elif cspace == 'HLS':
                new_color_space = cv2.COLOR_RGB2HLS
            elif cspace == 'LUV':
                new_color_space = cv2.COLOR_RGB2LUV
            elif cspace == 'YUV':
                new_color_space = cv2.COLOR_RGB2YUV
            elif cspace == 'LUV':
                new_color_space = cv2.COLOR_RGB2YCrCb      
            img_converted = cv2.cvtColor(img, new_color_space)
        else:
            img_converted = np.copy(img)          
        spatial_features = bin_spatial(img = img_converted, size = spatial_size)
        color_hist_features = color_hist(img = img_converted, nbins = hist_bins, bins_range = hist_range)                      
        features.append(np.concatenate((spatial_features, color_hist_features)))     
    return features

def extract_features_hog(imgs, cspace = 'RGB', orient = 9, 
                        pix_per_cell = 8, cell_per_block = 2, 
                        hog_channel = 0):    
    features = []
    for img_path in imgs:
        img = mpimg.imread(img_path)
        if cspace != 'RGB':
            if cspace == 'HSV':
                new_color_space = cv2.COLOR_RGB2HSV
            elif cspace == 'HLS':
                new_color_space = cv2.COLOR_RGB2HLS
            elif cspace == 'LUV':
                new_color_space = cv2.COLOR_RGB2LUV
            elif cspace == 'YUV':
                new_color_space = cv2.COLOR_RGB2YUV
            elif cspace == 'LUV':
                new_color_space = cv2.COLOR_RGB2YCrCb      
            img_converted = cv2.cvtColor(img, new_color_space)
        else:
            img_converted = np.copy(img) 
        
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(img_converted.shape[2]):
                hog_features_channel = get_hog_features(img_converted[:,:,channel], orient, pix_per_cell, 
                                                        cell_per_block, vis = False, feature_vec = True)
                hog_features.append(hog_features_channel)
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(img_converted[:,:,hog_channel], orient, pix_per_cell, 
                                            cell_per_block, vis = False, feature_vec = True)
        features.append(hog_features)
    return features
                

cars = glob.glob('vehicles/GTI_Far/*.png') + glob.glob('vehicles/GTI_Left/*.png') + \
       glob.glob('vehicles/GTI_MiddleClose/*.png') + glob.glob('vehicles/GTI_Right/*.png') + \
       glob.glob('vehicles/KITTI_extracted/*.png')

notcars = glob.glob('non-vehicles/Extras/*.png') + glob.glob('non-vehicles/GTI/*.png')

spatial = 32
histbin = 32       

car_features = extract_features(cars, cspace = 'HSV', spatial_size = (spatial, spatial),
                                hist_bins = histbin, hist_range=(0, 256))
notcar_features = extract_features(notcars, cspace = 'HSV', spatial_size = (spatial, spatial),
                                   hist_bins = histbin, hist_range = (0, 256))

car_features_hog = extract_features_hog(cars, cspace = 'HSV', orient = 9, 
                        pix_per_cell = 8, cell_per_block = 2, 
                        hog_channel = 0)
notcar_features_hog = extract_features_hog(notcars, cspace = 'HSV', orient = 9, 
                        pix_per_cell = 8, cell_per_block = 2, 
                        hog_channel = 0)

if len(car_features) > 0:
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    car_ind = np.random.randint(0, len(cars))
    # Plot an example of raw and scaled features
    fig = plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.imshow(mpimg.imread(cars[car_ind]))
    plt.title('Original Image')
    plt.subplot(132)
    plt.plot(X[car_ind])
    plt.title('Raw Features')
    plt.subplot(133)
    plt.plot(scaled_X[car_ind])
    plt.title('Normalized Features')
    fig.tight_layout()
else: 
    print('Your function only returns empty feature vectors...')
    
    
# Build the Model
# Create an array stack of feature vectors
X = np.vstack((car_features_hog, notcar_features_hog)).astype(np.float64)   
# Define the labels vector
y = np.hstack((np.ones(len(car_features_hog)), np.zeros(len(notcar_features_hog))))  

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = rand_state)
    
# Fit a per-column scaler only on the training data
X_scaler = StandardScaler().fit(X_train)
# Apply the scaler to X_train and X_test
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)
    
print('Using spatial binning of: ',spatial, ' and ', histbin,' histogram bins')
print('Feature vector length:', len(X_train[0]))

svc = LinearSVC()
# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t = time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
    
    