# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 00:41:43 2017

@author: Vamshidhar P
"""

import csv
import cv2
import numpy as np
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense, Lambda, Cropping2D, Activation
from keras.layers.advanced_activations import ELU
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.callbacks import ModelCheckpoint  
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import math
from keras.layers.core import SpatialDropout2D


class BehaviorCloning(object):
    def __init__(self, epochs, learning_rate, batch_size, path):
        self.epochs = epochs        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.path = path
        self.model = Sequential()
        print('...Reading Data Set...')
        self.driving_data = self.read_dataset()        
        
        
    def read_dataset(self):
        
        driving_data = []
        
        with open(self.path, 'r') as csvfile:
            data = csv.reader(csvfile)
            for row in data:
                driving_data.append(row)
               
        return driving_data    
           
    def generator(self, samples, batch_size):    
        
        """
        Training the neural network with large number of images loaded into memory may slow down the entire process. 
        Data generator functions in python are used to mitigate this problem by reading the required set of images 
        in chunks using the batch size.
       
        Args:
         samples: Image Samples read from Driving Log 
         batch_size: batch size
        Return:
         generator Yield:   Images and Labels   
        
        Step 1: Read Image         
        Step 2: Calculate the Steer angle using the correction factor
        Step 3: Invoke data_augment function
       
        """
        
        
        # Only full batches
        n_batches = len(samples)//batch_size
        samples = samples[:n_batches*batch_size]
        num_samples = len(samples)
        
        while 1:
            shuffle(samples)
            for idx in range(0, num_samples, batch_size):
                images = []
                labels = []
                for row in samples[idx: idx + batch_size]:
                     for camera in range(3):
                         img_read = cv2.imread(row[camera])
                                                  
                         #Crop Image to get rid of SKY and Car Hood
                         #img_shape = img_read.shape                         
                         #top_crop = math.floor(img_shape[0]/5)
                         #bottom_crop = img_shape[0]-22
                         #img_read = img_read[top_crop:bottom_crop, 0:img_shape[1]]
                                                  
                         
                         images.append(img_read)
                         angle = float(row[3])
                         steer_angle = angle if camera == 0 else (angle + 0.2) if camera == 1 else (angle - 0.2)
                         labels.append(float(steer_angle)) 
                
                images = np.array(images)
                labels = np.array(labels)
                                
                images, labels = self.data_augment(images, labels)
                                
                yield shuffle(images, labels)           
               
    
    def data_augment(self, images, labels):  
        """
        Deep artificial neural networks require a large corpus of training data in order to effectively learn, 
        where collection of such training data is often expensive and laborious. 
        Data augmentation overcomes this issue by artificially inflating the training set with label 
        preserving transformations. Recently there has been extensive use of generic data augmentation 
        to improve Convolutional Neural Network (CNN) task performance.
       
        Args:
         images: list of images
         labels: list of labels
        Return:
         Augmented Images and Labels 
        
        Step 1: Randomly Adjust Brightness of images using random brightness value
                Generator function uses CV2 to read images in BGR format 
                Convert images to HSV(Hue-Saturation-Value), randomly alter V value and convert back to RGB
                Drive.py gets the images from simulator using PIL image and is also read in RGB format
        Step 2: Randomly shift the image virtially and horizontally and adjust the steeing angle using correction factor
        Step 3: Randomly select images and Flip them and append to main Set
        Step 4: Return Augmented Images and Labels
       
        """
        
        
        augmented_images = []
        augmented_labels = []
        
        for idx, img in enumerate(images):
            
            ##Randomly Adjust Brightness of images 
            #
            new_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            brightness_level = (0.2 * np.random.uniform()) + 0.4
            new_img[:,:,2] = new_img[:,:,2] * brightness_level
            new_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2RGB)
            
            # Randomly shift the image virtially and horizontally           
            x_shift = 100 * (np.random.rand() - 0.6)
            y_shift = 20 * (np.random.rand() - 0.4)
            new_steer_angle = labels[idx] + x_shift * 0.002
            transition_matrix = np.float32([[1, 0, x_shift],[0, 1, y_shift]])
            height, width = new_img.shape[:2]
            new_img = cv2.warpAffine(new_img, transition_matrix, (width, height))
            
            augmented_images.append(new_img)
            augmented_labels.append(new_steer_angle)  
            
            
        #Randomly select images and Flip them and append to main Set
        num_imgs = len(augmented_images)
        random_flip_idx = random.sample(range(num_imgs), num_imgs//2)   
        for idx in random_flip_idx:
             new_img = np.fliplr(augmented_images[idx]) 
             new_steer_angle = -augmented_labels[idx]
             augmented_images.append(new_img)
             augmented_labels.append(new_steer_angle)  
        
           
        images = np.array(augmented_images)    
        labels = np.array(augmented_labels)    
          
        return images, labels
    
       
    def model_pipeline(self, input_shape): 
        
        """
        This architecture used here is published by autonomous vehicle team in NVIDIA.   
        Hyper Parameters:
                The number of epochs used: 35
                Learning Rate: 0.01.
                Batch size : 32
                Momentum
        Weights updated using back propagation and stochastic gradient descent optimizer. 
        Learning rate exponential decay was applied with global_step value computed as (learning_rate / epochs).
        When training a model, it is often recommended to lower the learning rate 
        as the training progresses, which helps the model converge and reach global minimum.
       
        Args:
         input_shape: shape of the input image    
         
        Step 1: Normalize the pixel values to a range between -1 and 1
        Step 2: Crop Image: If you observe the images plotted , almost 1/5th of the image from the top is the sky 
        and around 20 pixels from the bottom is the hood of the car. These pixels provide no added value to the 
        neural network. 
        Cropping the image to get rid of these pixels will help the neural network look only at the road as the car moves.
        Step 3: Build Model Pipleline        
        
        The model follows The All Convolutional Net achitecture.
        Max-pooling layers are simply replaced by a convolutional layer with increased stride without loss in accuracy. 
        This yielded competitive or state of the art performance on several object 
        recognition datasets (CIFAR-10, CIFAR-100, ImageNet).

        After several attempts, Spatial Dropout regularization on third and fourth convolutions followed by regular dropout 
        on fisth convolution provided least loss on validation set.

        Our network is fully convolutional and images exhibit strong spatial correlation, the feature map activations 
        are also strongly correlated. In the standard dropout implementation, network activations are "dropped-out" 
        during training with independent probability without considering the spatial correlation.

        On the other hand Spatial dropout extends the dropout value across the entire feature map. 
        Therefore, adjacent pixels in the dropped-out feature map are either all 0 (dropped-out) or all active. 
        This technique proved to be very effective and improves performance

        Maxpool layer is used only on the last convolution layer with regular drop out.
        
        """
        
        self.model = Sequential()
        
        #As for any data-set, image data has been normalized so that the numerical rangeof the pixels is between -1 and 1.
        self.model.add(Lambda(lambda x: x/127.5 - 1.0,  input_shape = input_shape))
        self.model.add(Cropping2D(cropping=((50,22), (0,0))))
               
        self.model.add(Conv2D(filters = 24, kernel_size = (5,5), padding = 'same', 
                              kernel_initializer= 'truncated_normal'))        
        self.model.add(Activation('elu'))
        self.model.add(Conv2D(filters = 24, kernel_size = (5,5), strides = (2,2), padding = 'same', 
                              kernel_initializer= 'truncated_normal'))   
        
        
        self.model.add(Conv2D(filters = 36, kernel_size = (5,5), padding = 'same', 
                              kernel_initializer= 'truncated_normal'))        
        self.model.add(Activation('elu'))
        self.model.add(Conv2D(filters = 36, kernel_size = (5,5), strides = (2,2), padding = 'same', 
                              kernel_initializer= 'truncated_normal'))   
        
        
        self.model.add(Conv2D(filters = 48, kernel_size = (5,5), padding = 'same', 
                              kernel_initializer= 'truncated_normal'))
        
        self.model.add(Activation('elu'))
        self.model.add(Conv2D(filters = 48, kernel_size = (5,5), strides = (2,2), padding = 'same', 
                              kernel_initializer= 'truncated_normal'))           
        self.model.add(SpatialDropout2D(0.3))
        
        self.model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', 
                              kernel_initializer= 'truncated_normal'))        
        self.model.add(Activation('elu'))                
        self.model.add(SpatialDropout2D(0.3))
        
        self.model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', 
                              kernel_initializer= 'truncated_normal'))
        
        self.model.add(Activation('elu'))
        self.model.add(MaxPooling2D(pool_size=(1,1)))
        self.model.add(Dropout(0.3))
        
        
        self.model.add(Flatten())
        self.model.add(Activation('elu'))
        self.model.add(Dropout(0.3))

        
        self.model.add(Dense(100,kernel_initializer= 'truncated_normal'))
        self.model.add(Activation('elu'))
        

        self.model.add(Dense(50, kernel_initializer= 'truncated_normal'))
        self.model.add(Activation('elu'))
        
        
        self.model.add(Dense(10, kernel_initializer= 'truncated_normal'))
        self.model.add(Activation('elu'))
        

        self.model.add(Dense(1))
              
        self.model.summary()
        
        momentum = 0.9
        decay_rate = self.learning_rate / self.epochs
        sgd_opt = optimizers.SGD(lr = self.learning_rate, momentum = momentum, decay = decay_rate, nesterov=False)
        #adam_opt = optimizers.Adam(lr = self.learning_rate)
        self.model.compile(optimizer = sgd_opt, loss='mean_squared_error')
    
    def train(self):        
        
        """       
        Train the Network
        Args:
         None
        Return:
         None
         
        Step 1: Split the samples into Train and Validation sets.
        Step 2: Invoke the train and validation generator functions created above.
        Step 3: Calculate Train and Validation sample lengths
        Step 4: Trigger Keras model.fit_generator to initiate training
        Step 5: Print Model Stats (Training and Validation Loss)
       
        """
        
        train_samples, validation_samples = train_test_split(self.driving_data, test_size=0.25, shuffle = True)
                
        train_generator = self.generator(train_samples, self.batch_size)
        validation_generator = self.generator(validation_samples, self.batch_size)
        
        for i ,j in train_generator:
            input_shape = i.shape[1:]
            train_sample_length = i.shape[0] * (len(train_samples)//self.batch_size)
            break
        
        for i ,j in validation_generator:
            valid_sample_length = i.shape[0] * (len(validation_samples)//self.batch_size)
            break
        
        print(train_sample_length,valid_sample_length, input_shape)
        
        checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', save_best_only=True)
        print('...Constructing Model Pipeline...')
        self.model_pipeline(input_shape)
        
        print('...Training...')
               
        with tf.device('/gpu:0'):
            model_stats = self.model.fit_generator(train_generator,
                                                   steps_per_epoch=train_sample_length//self.batch_size,
                                                   epochs = self.epochs, verbose=2, callbacks=[checkpointer],
                                                   validation_data = validation_generator,
                                                   validation_steps=valid_sample_length//self.batch_size)
        
        self.model.save('saved_models/model.h5')
        
        print(model_stats.history.keys())
        plt.plot(model_stats.history['loss'])
        plt.plot(model_stats.history['val_loss'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        plt.show()
    
    

        
            
                
        