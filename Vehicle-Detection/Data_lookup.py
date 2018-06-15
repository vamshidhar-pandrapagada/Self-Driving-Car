# -*- coding: utf-8 -*-
"""
Created on Sat May 12 23:09:26 2018

@author: Vamshidhar P
"""
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from random import randint
import numpy as np
import cv2
import glob

car_images = glob.glob('vehicles/GTI_Far/*.png') + glob.glob('vehicles/GTI_Left/*.png') + \
             glob.glob('vehicles/GTI_MiddleClose/*.png') + glob.glob('vehicles/GTI_Right/*.png') + \
             glob.glob('vehicles/KITTI_extracted/*.png')

len(car_images)

non_car_images = glob.glob('non-vehicles/Extras/*.png') + glob.glob('non-vehicles/GTI/*.png')

len(non_car_images)

def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict['n_cars'] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict['n_notcars'] = len(notcar_list)
    # Read in a test image, either car or notcar
    img = mpimg.imread(car_list[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict['image_shape'] = img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict['data_type'] =  img.dtype
    # Return data_dict
    return data_dict

data_info = data_look(car_images, non_car_images)

print('Your function returned a count of', data_info["n_cars"], ' cars and', data_info["n_notcars"], ' non-cars')
print('of size: ',data_info["image_shape"], ' and data type:', data_info["data_type"])

# Just for fun choose random car / not-car indices and plot example images   
car_ind = np.random.randint(0, len(car_images))
notcar_ind = np.random.randint(0, len(non_car_images))
    
# Read in car / not-car images
car_image = mpimg.imread(car_images[car_ind])
notcar_image = mpimg.imread(non_car_images[notcar_ind])
# Plot the examples
fig = plt.figure()
plt.subplot(121)
plt.imshow(car_image)
plt.title('Example Car Image')
plt.subplot(122)
plt.imshow(notcar_image)
plt.title('Example Not-car Image')