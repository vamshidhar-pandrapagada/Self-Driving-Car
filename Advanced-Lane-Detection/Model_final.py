# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 05:10:53 2018

@author: Vamshidhar P
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 22:13:42 2018

@author: Vamshidhar P
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May 13 20:57:01 2018

@author: Vamshidhar P
"""
from Utils import bin_spatial, color_hist, get_hog_features, slide_window, draw_boxes, convert_color
import numpy as np
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import glob
from sklearn.model_selection import train_test_split
import time
from sklearn.svm import LinearSVC
#from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import SGDClassifier
#from sklearn.model_selection import GridSearchCV
#from sklearn.pipeline import Pipeline
import pickle
from Evaluate_Model import evaluate_model
import tensorflow as tf
#from itertools import chain
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from PIL import Image
from collections import deque
import random

from Neural_Network import train_Neuralnet

import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)


save_model_path = './model_save/'
tf_save_model_path = './model_save/tf_model'

# Set Image Paramters to Tune while building Features
spatial_size = (32, 32)
hist_bins = 32
hist_bins_range = (0,256)
orient = 11
pix_per_cell = 16
cell_per_block = 2
color_space = 'LUV'
spatial_feat = True
hist_feat = True
hog_feat = True
hog_channel = 'ALL'
plot_detections = False

annotated_features = False
model_name = 'Neural_Network'
model_train_YN = True
train_feature_extract = True 


# test image parameters
x_start_stop=[None, None]
#y_start_stop_list = [[400,470],[400,550],[420,580], [400,656],[500,580], [410,550], [500,656], [390,585]]
y_start_stop_list = [[400,470], [400,470],[400,550],[420,580], [400,656],[500,656], [410,550], [500,656],[500,656]]
#scales = [1.0, 1.5, 1.5, 1.5, 1.0, 2.0, 2.0, 1.5]
scales = [0.5, 1.0, 1.5, 1.5, 1.5, 1.0, 2.0, 2.0, 1.5]
   
def extract_test_features_boxes(img, scale, y_start_stop):          
    
    #Our training was done on PNG images and mpgimg.imread function reads PNG images on a scale of 0 to 1
    #Our testing images are JPG images and mpgimg.imread function reads JPG images on a scale of 0 to 255
    #Hence Scale images to convert the range between 0 and 1 
    img = img.astype(np.float32)/255
    
    y_start = y_start_stop[0]
    y_stop =  y_start_stop[1]
    
    img_tosearch = img[y_start:y_stop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, cspace = color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    #nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False, hog_channel =  0)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False, hog_channel =  1)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False, hog_channel =  2)
    
                  
    spatial_features = []
    hist_features = []
    hog_features = []
    all_test_features = []
    boxes = []
      
            
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            
            
            if hog_feat:
            # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                            
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            if spatial_feat:
                spatial_features = bin_spatial(subimg, size = spatial_size)
            if hist_feat:
                hist_features = color_hist(subimg, nbins = hist_bins, bins_range=hist_bins_range)
            #test_features = np.hstack([spatial_features, hist_features, hog_features]).reshape(1, -1) 
            test_features = np.hstack([spatial_features, hist_features, hog_features])
            
            """
            test_features = X_scaler.transform(test_features)
            
            test_prediction = model_predict(test_features, model_name)
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+y_start),(xbox_left+win_draw,ytop_draw+win_draw+y_start),(0,0,255),6) 
            """
            
            all_test_features.append(test_features)            
            
            xbox_left = np.int(xleft * scale)
            ytop_draw = np.int(ytop * scale)
            win_draw = np.int(window * scale)
            box = ((xbox_left, ytop_draw + y_start), (xbox_left + win_draw,ytop_draw + win_draw + y_start))
            boxes.append(box)    
            
    #return draw_img
    return all_test_features, boxes

def extract_train_features(annotated_features = False):
    
    def extract_annotated_features():
        annotated_cars = []
        annotated_not_cars = []
        path = 'Udacity Dataset/self-driving-car-master/annotations/object-detection-crowdai.tar/'
        read_coords = pd.read_csv(path + 'labels_crowdai.csv')
        for index, row in read_coords.iterrows():
            image = mpimg.imread(path + row['Frame'])
            try:
                image = cv2.resize(image[ row['ymin']:row['ymax'], row['xmin']:row['xmax'], ], (64,64))
                #Our training was done on PNG images and mpgimg.imread function reads PNG images on a scale of 0 to 1
                #Our testing images are JPG images and mpgimg.imread function reads JPG images on a scale of 0 to 255
                #Hence Scale images to convert the range between 0 and 1 
                image = image.astype(np.float32)/255
                if row['Label'] == 'Car':
                    annotated_cars.append(image)
                else:
                    annotated_not_cars.append(image)
            except:
                print('Exception at index' +str(index))
        return annotated_cars, annotated_not_cars
    
    
    def extract_features(imgs,
                     spatial_feat  = True, 
                     hist_feat = True, 
                     hog_feat = True, 
                     hog_channel = 'ALL',
                     cspace = color_space):
        # Create a list to append feature vectors to
        features = []        
        # Iterate through the list of images
            # Read in each one by one
            # apply color conversion if other than 'RGB'
            # Apply bin_spatial() to get spatial color features
            # Apply color_hist() to get color histogram features
            # Append the new feature vector to the features list
        # Return list of feature vectors
        spatial_features = []
        color_hist_features = []
        hog_features = []
        for img in imgs:             
            img_converted = convert_color(img, cspace = color_space)
            if spatial_feat:
                spatial_features = bin_spatial(img_converted, spatial_size)
                
            if hist_feat:
                color_hist_features = color_hist(img_converted, nbins = hist_bins, bins_range=hist_bins_range)        
                
            if hog_feat:
                hog_features = get_hog_features(img_converted, orient, pix_per_cell, cell_per_block, vis = False, 
                                                feature_vec=True, hog_channel =  'ALL') 
                            
            features.append(np.concatenate([spatial_features, color_hist_features, hog_features]))  
        return features
    
    
    print('Reading Image Paths')
    cars = glob.glob('vehicles/GTI_Far/*.png') + glob.glob('vehicles/GTI_Left/*.png') + \
           glob.glob('vehicles/GTI_MiddleClose/*.png') + glob.glob('vehicles/GTI_Right/*.png') + \
           glob.glob('vehicles/KITTI_extracted/*.png')

    not_cars = glob.glob('non-vehicles/Extras/*.png') + glob.glob('non-vehicles/GTI/*.png')
    
   
    # Get Train features and labels
    print('Extracting Features from images')   
    
    car_images= []
    not_car_images= []
    for img_path in cars:
        image = mpimg.imread(img_path)
        car_images.append(image)
    for img_path in not_cars:
        image = mpimg.imread(img_path)
        not_car_images.append(image)
        
    # Augment Not Car Images        
    print('Augmenting Not Car Images')
    augmented_images = []
    for idx, img in enumerate(not_car_images):            
            ##Randomly Adjust Brightness of images 
            new_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            brightness_level = (0.2 * np.random.uniform()) + 0.4
            new_img[:,:,2] = new_img[:,:,2] * brightness_level
            new_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2RGB)
            
            # Randomly shift the image virtially and horizontally           
            x_shift = 100 * (np.random.rand() - 0.6)
            y_shift = 20 * (np.random.rand() - 0.4)
            transition_matrix = np.float32([[1, 0, x_shift],[0, 1, y_shift]])
            height, width = new_img.shape[:2]
            new_img = cv2.warpAffine(new_img, transition_matrix, (width, height))
            
            augmented_images.append(new_img)
            
                        
    #Randomly select images and Flip them and append to main Set
    num_imgs = len(augmented_images)
    random_flip_idx = random.sample(range(num_imgs), num_imgs//2)   
    for idx in random_flip_idx:
         new_img = np.fliplr(augmented_images[idx]) 
         augmented_images.append(new_img)              
    
    not_car_images =  np.vstack((augmented_images, not_car_images))           
    
    if annotated_features:
        annotated_cars, annotated_not_cars = extract_annotated_features()
        print('Length of Annotated Car Images:' +str(len(annotated_cars)))
        print('Length of Annotated Non Car Images:' +str(len(annotated_not_cars)))
    
        car_images =  np.vstack((annotated_cars, car_images))
        not_car_images =  np.vstack((annotated_not_cars, not_car_images))  
    
    
    print('Total Number of Car Images: ' +str(len(car_images)))
    print('Total Number of Non Car Images: ' +str(len(not_car_images)))
    
    
    train_car_features =  extract_features(car_images, spatial_feat  = spatial_feat, 
                                           hist_feat = hist_feat, 
                                           hog_feat = hog_feat, 
                                           hog_channel = hog_channel,
                                           cspace = color_space)

    train_not_car_features =  extract_features(not_car_images, spatial_feat  = spatial_feat, 
                                               hist_feat = hist_feat, 
                                               hog_feat = hog_feat, 
                                               hog_channel = 'ALL',
                                               cspace = color_space) 
    # Create an array stack of feature vectors
    X = np.vstack((train_car_features, train_not_car_features)).astype(np.float64)            
    # Define the labels vector
    y = np.hstack((np.ones(len(train_car_features)), np.zeros(len(train_not_car_features))))  
    
    #Shuffle the data set well enough
    for i in range(5):
        X, y = shuffle(X,y)
    
    return X, y


def model_train(X, y):    
    
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    print('Splitting Features into Train and test Setsl')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = rand_state)
        
      
    # Fit a per-column scaler only on the training data
    # Save Sclaer to disk
    with open(save_model_path + 'scale_features_X.pkl', 'wb') as output:
        X_scaler = StandardScaler().fit(X_train)
        pickle.dump(X_scaler, output, pickle.HIGHEST_PROTOCOL)
            
   
    
    print('Train, Test Shapes: ', X_train.shape, X_test.shape)
    
    # Train The Model
    print('Training the Model')
    
    if model_name =='SVM':
        # Apply the scaler to X_train and X_test
        X_train = X_scaler.transform(X_train)
        X_test = X_scaler.transform(X_test)
        
        t = time.time()
        clf = LinearSVC(random_state = 1)
        with open(save_model_path + 'Best SVC_Classifier' + '.pkl', 'wb') as output:  
                clf.fit(X_train, y_train)
                print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))
                pickle.dump(clf, output, pickle.HIGHEST_PROTOCOL) 
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
        
        
        
        #y_pred_proba = clf.predict_proba(X_test[:, 1])
        y_pred_proba = None
        y_pred = clf.predict(X_test)     
        
        evaluate_model(y_pred, y_test, 'SVM Classifier',  pred_proba = y_pred_proba, ROC_Curve  = False)  
        
    elif model_name == 'SGD':
        X_train = X_scaler.transform(X_train)
        X_test = X_scaler.transform(X_test)
        t = time.time()
        clf = SGDClassifier(random_state = 1, loss = 'log', penalty ='elasticnet', alpha = 0.001, max_iter = 500)
        
        with open(save_model_path + 'SGD_Classifier' + '.pkl', 'wb') as output:  
                clf.fit(X_train, y_train)
                print('Test Accuracy of SGD = ', round(clf.score(X_test, y_test), 4))
                pickle.dump(clf, output, pickle.HIGHEST_PROTOCOL) 
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SGD...')
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)     
        
        evaluate_model(y_pred, y_test, 'SGD Classifier',  pred_proba = y_pred_proba, ROC_Curve  = True) 
        
    elif model_name == 'Neural_Network':
        t = time.time()
        X_train = X_scaler.transform(X_train)
        train_Neuralnet(X_train, X_test, y_train, y_test, X_scaler,
                        batch_size = 256, epochs = 15 , lr = 0.001, dropout_prob = 0.7) 
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train Neural Network...')


def  model_predict(test_image, model_name, X_scaler, scales, y_start_stop_list, 
                   classfier = None, sess = None,
                   loaded_x = None, loaded_logits = None, loaded_keep_probability = None):
           
    car_detected_boxes = []
    draw_image = np.copy(test_image)
    for scale, y_start_stop in zip(scales, y_start_stop_list):    
        all_features, boxes = extract_test_features_boxes(test_image, scale, y_start_stop)
        all_features = X_scaler.transform(all_features)
    
        if model_name == 'Neural_Network':
            test_model_feed_dict = {loaded_x: all_features, loaded_keep_probability: 1.0}
            #t2 = time.time()
            #print(round(t2-t, 2), 'Seconds to Retrieve Model')
            softmax_predictions = sess.run(tf.nn.softmax(loaded_logits),feed_dict = test_model_feed_dict)
            test_predictions = []
            for i in range(len(softmax_predictions)):
                if softmax_predictions[i][1] >= 0.5:
                    test_predictions.append(1)
                else:
                    test_predictions.append(0)
            test_predictions = np.array(test_predictions)
        else:
            test_predictions = classfier.predict(all_features)
        for i, pred in enumerate(test_predictions):
            if pred == 1:
                cv2.rectangle(draw_image , boxes[i][0], boxes[i][1], (255,0,0), 6) 
                car_detected_boxes.append(boxes[i])
    return car_detected_boxes, draw_image

def model_test(test_image):
    
    def add_heat(heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    
        # Return updated heatmap
        return heatmap
    
    def apply_threshold(heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap
    
    def draw_labeled_bboxes(img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (255,0,0), 6)
        # Return the image
        return img 
        
    
    test_image = np.array(test_image)      
    with open(save_model_path + 'scale_features_X.pkl', 'rb') as inputs:
            X_scaler = pickle.load(inputs)          
           
    if model_name == 'SVM':
        with open(save_model_path + 'Best SVC_Classifier' + '.pkl', 'rb') as inputs:
            clf = pickle.load(inputs)
            car_detected_boxes, draw_image = model_predict(test_image, model_name, X_scaler, scales, y_start_stop_list, 
                                                           classfier = clf)
    elif model_name == 'SGD':
        with open(save_model_path + 'SGD_Classifier' + '.pkl', 'rb') as inputs:
            clf = pickle.load(inputs)      
            car_detected_boxes, draw_image = model_predict(test_image, model_name, X_scaler, scales, y_start_stop_list, 
                                                           classfier = clf)
        
    elif model_name == 'Neural_Network':
        loaded_graph = tf.Graph()        
        with tf.Session(graph=loaded_graph) as sess:            
            # Load model
            
            #t = time.time()
            loader = tf.train.import_meta_graph(tf_save_model_path + '.meta')
            loader.restore(sess, tf_save_model_path)

            # Get Tensors from loaded model
            loaded_x = loaded_graph.get_tensor_by_name('x:0')            
            loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
            loaded_keep_probability = loaded_graph.get_tensor_by_name('kp:0') 
            car_detected_boxes, draw_image = model_predict(test_image, model_name, X_scaler, scales, y_start_stop_list, 
                                                           classfier = None, sess = sess,
                                                           loaded_x = loaded_x, loaded_logits = loaded_logits, 
                                                           loaded_keep_probability = loaded_keep_probability)
        
        
    heat  = np.zeros_like(test_image[:,:,0]).astype(np.float)
    heatmap = add_heat(heat, car_detected_boxes)
    heatmap = apply_threshold(heatmap, 2)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    if plot_detections:
        print(labels[1], 'cars found')
    final_image = draw_labeled_bboxes(np.copy(test_image), labels)   
    
    if plot_detections:        
        plt.figure(figsize=(15,15))
        plt.subplot(131)
        plt.imshow(draw_image)
        plt.title('All Detections')
        plt.subplot(132)
        plt.imshow(heatmap, cmap='hot')
        plt.title('All Detections heat')
        plt.subplot(133)
        plt.imshow(final_image)
        plt.title('Final Detection')
    
    
    return final_image          
    
"""  

if model_train_YN:    
    if train_feature_extract:    
        X, y = extract_train_features(annotated_features)
        np.save(save_model_path + 'features_X.npy', X)       
        np.save(save_model_path + 'labels_y.npy', y)   
    else:
        print('Reading features from disk')
        X = np.load(save_model_path + 'features_X.npy')
        y = np.load(save_model_path + 'labels_y.npy')
    print('Features Shape: ' , X.shape , y.shape)             
    model_train(X, y) 
    

test_images_paths  = glob.glob('test_images/*.jpg') 
def register_extension(id, extension): Image.EXTENSION[extension.lower()] = id.upper()
Image.register_extension = register_extension
def register_extensions(id, extensions): 
  for extension in extensions: register_extension(id, extension)
Image.register_extensions = register_extensions


plot_detections = True
for path in test_images_paths:
    test_image = mpimg.imread(path)
    final_image = model_test(test_image)
  
"""    
    
#path  = glob.glob('test_images/test6.jpg') 
#test_image = cv2.imread(path[0])
#test_features, boxes = find_cars(test_image, [400,656] )    
""" 
test_movie_output =  'Videos/test_video_output.mp4'   
clip = VideoFileClip('Videos/test_video.mp4')
image_clip = clip.fl_image(model_test)
%time image_clip.write_videofile(test_movie_output, audio=False)

 
project_video_output =  'Videos/project_video_output.mp4'   
clip3 = VideoFileClip('Videos/project_video.mp4')
image_clip = clip3.fl_image(model_test)
%time image_clip.write_videofile(project_video_output, audio=False)
"""
