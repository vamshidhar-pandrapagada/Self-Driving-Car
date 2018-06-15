  # -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 15:09:40 2018

@author: vpandrap
"""

from Utils import slide_window, draw_boxes
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import glob
from sklearn.model_selection import train_test_split
import time
from Evaluate_Model import evaluate_model
import tensorflow as tf
#from itertools import chain
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from PIL import Image
import random
from itertools import chain

from Convolution_Network import train_Convnet

import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)


save_model_path = './model_save/'
tf_save_model_path = './model_save/tf_model_convnet'


# test image parameters
x_start_stop=[None, None]
#y_start_stop=[400, 707]

y_start_stop_list = [[400,470], [400,470], [400,550], [400,550],[420,580], [400,656],[400,656]]
xy_windows = [[48, 48],[64, 64],[64, 64], [128,128],[128,128], [128,128],[256,256]]
xy_overlaps = [[0.85, 0.85],[0.8, 0.8],[0.8, 0.8],[0.8, 0.8],[0.8, 0.8],[0.8, 0.8],[0.8, 0.8]]

annotated_features = True
plot_output = True
model_train_YN = False
train_feature_extract = False  


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
        
  
    # Create an array stack of all images
    X = np.vstack((car_images, not_car_images))       
    # Define the labels vector
    y = np.hstack((np.ones(len(car_images)), np.zeros(len(not_car_images))))  
    
    #Shuffle the data set well enough
    for i in range(5):
        X, y = shuffle(X,y)
    
    return X, y


def model_train(X, y):   
    
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    print('Splitting Features into Train and test Setsl')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = rand_state)     
    
    print('Train, Test Shapes: ', X_train.shape, X_test.shape)
    
    # Train The Model
    print('Training the Model')     
    
    t = time.time()
    image_shape = X_train[0].shape
    train_Convnet(X_train, X_test, y_train, y_test, image_shape, 
                  batch_size = 256, epochs = 15 , lr = 0.001, dropout_prob = 0.7) 
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train Neural Network...')
    
    
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
    
    
    all_windows = []
    car_detected_boxes = []
    slide_window_images = []
    draw_image = np.copy(test_image)
    
    for xy_window, xy_overlap, y_start_stop in zip(xy_windows, xy_overlaps, y_start_stop_list):  
        windows = slide_window(test_image, x_start_stop = x_start_stop, y_start_stop = y_start_stop, 
                                xy_window = xy_window, xy_overlap = xy_overlap)
        
        all_windows.append(windows)
    all_windows = list(chain(*all_windows))        
    for window in all_windows:
        resized_image = cv2.resize(test_image[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
    
        #Our training was done on PNG images and mpgimg.imread function reads PNG images on a scale of 0 to 1
        #Our testing images are JPG images and mpgimg.imread function reads JPG images on a scale of 0 to 255
        #Hence Scale images to convert the range between 0 and 1 
        resized_image = resized_image.astype(np.float32)/255
        slide_window_images.append(resized_image)   
    
   
    loaded_graph = tf.Graph()        
    with tf.Session(graph=loaded_graph) as sess:            
        # Load model
        loader = tf.train.import_meta_graph(tf_save_model_path + '.meta')
        loader.restore(sess, tf_save_model_path)

        # Get Tensors from loaded model
        loaded_x = loaded_graph.get_tensor_by_name('x:0')            
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        loaded_keep_probability = loaded_graph.get_tensor_by_name('kp:0') 
        test_model_feed_dict = {loaded_x: slide_window_images, loaded_keep_probability: 1.0}            
        softmax_predictions = sess.run(tf.nn.softmax(loaded_logits),feed_dict = test_model_feed_dict)
        test_predictions = []
        for i in range(len(softmax_predictions)):
            if softmax_predictions[i][1] >= 0.5:
                test_predictions.append(1)
            else:
                test_predictions.append(0)
        test_predictions = np.array(test_predictions)
        
             
    for i, pred in enumerate(test_predictions):
        if pred == 1:
            cv2.rectangle(draw_image , all_windows[i][0], all_windows[i][1], (255,0,0), 6) 
            car_detected_boxes.append(all_windows[i])           
    
   
         
    heat  = np.zeros_like(test_image[:,:,0]).astype(np.float)
    heatmap = add_heat(heat, car_detected_boxes)
    heatmap = apply_threshold(heatmap, 2)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    
    if plot_output:
        print(labels[1], 'cars found')
    final_image = draw_labeled_bboxes(np.copy(test_image), labels)
    
    if plot_output: 
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



if model_train_YN:    
    if train_feature_extract:    
        X, y = extract_train_features(annotated_features)
        np.save(save_model_path + 'Conv_features_X.npy', X)       
        np.save(save_model_path + 'Conv_labels_y.npy', y)   
    else:
        print('Reading features from disk')
        X = np.load(save_model_path + 'Conv_features_X.npy')
        y = np.load(save_model_path + 'Conv_labels_y .npy')
    print('Features Shape: ' , X.shape , y.shape)             
    model_train(X, y) 
    

test_images_paths  = glob.glob('test_images/*.jpg') 
def register_extension(id, extension): Image.EXTENSION[extension.lower()] = id.upper()
Image.register_extension = register_extension
def register_extensions(id, extensions): 
  for extension in extensions: register_extension(id, extension)
Image.register_extensions = register_extensions



for path in test_images_paths:
    test_image = mpimg.imread(path)
    final_image = model_test(test_image)
  
    
    
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