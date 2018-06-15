# -*- coding: utf-8 -*-
"""
Created on Sat May 12 22:25:25 2018

@author: Vamshidhar P
"""

import cv2
import matplotlib.image as mpimg

image = mpimg.imread('test_img.jpg')
small_img = cv2.resize(image, (32, 32))
print(small_img.shape)



feature_vec = small_img.ravel()
print(feature_vec.shape)