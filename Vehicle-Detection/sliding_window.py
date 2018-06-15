# -*- coding: utf-8 -*-
"""
Created on Sun May 13 01:09:54 2018

@author: Vamshidhar P
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('test_image_hog.jpg')

# Here is your draw_boxes function from the previous exercise
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
    if x_start_stop[0] == None and x_start_stop[1] == None:
        x_start_stop = [0, img.shape[1]]
    if y_start_stop[0] == None and y_start_stop[1] == None:
        y_start_stop = [0, img.shape[0]]
    
    #image_to_span = img[y_start_stop[0]:y_start_stop[1], x_start_stop[0]:x_start_stop[1]]
    
    image_span_width, image_span_height = x_start_stop[1] - x_start_stop[0], y_start_stop[1] - y_start_stop[0]
    window_width, window_height = xy_window[0], xy_window[1]
    windows_x = np.int(1 + (image_span_width - window_width)/(window_width * xy_overlap[0]))
    windows_y = np.int(1 + (image_span_height - window_height)/(window_height * xy_overlap[1]))
    num_windows = np.int(windows_x * windows_y)
    
    pixels_per_step_x = np.int(xy_window[0] * (1- xy_overlap[0]))
    pixels_per_step_y = np.int(xy_window[1] * (1 -xy_overlap[1]))
    window_list = []
    # Loop through finding x and y window positions
    #     Note: you could vectorize this step, but in practice
    #     you'll be considering windows one by one with your
    #     classifier, so looping makes sense
        # Calculate each window position
        # Append window position to list
    # Return the list of windows
    max_x_lim = x_start_stop[1]
    max_y_lim = y_start_stop[1]
    
    intial_x = x_start_stop[0]
    intial_y = y_start_stop[0]
    for w_s in range(num_windows):
        box = ((intial_x, intial_y), (intial_x + window_width, intial_y + window_height))
        window_list.append(box)
        intial_x += pixels_per_step_x
        if intial_x + window_width >  max_x_lim:
            box = ((intial_x, intial_y), (max_x_lim, intial_y + window_height))
            window_list.append(box)
            intial_x = 0
            intial_y += pixels_per_step_y   
    print(intial_y)    
    if intial_y <  max_y_lim:
        intial_x = 0  
        for w in range(windows_x):
            box = ((intial_x, intial_y), (intial_x + window_width, max_y_lim))
            window_list.append(box)
            intial_x += pixels_per_step_x
            if intial_x + window_width >  max_x_lim:
                box = ((intial_x, intial_y), (max_x_lim, max_y_lim))
                window_list.append(box)          
    
    return window_list


def slide_window_alternate(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
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


x_start_stop=[None, None]
y_start_stop_list = [[400,550]]
xy_windows = [[96, 96]]
xy_overlaps = [[0.5, 0.5]]


for xy_window, xy_overlap, y_start_stop in zip(xy_windows, xy_overlaps, y_start_stop_list):  
    plt.figure()
    #print(xy_window, y_start_stop)
    windows = slide_window_alternate(image, x_start_stop = x_start_stop, y_start_stop = y_start_stop, 
                            xy_window = xy_window, xy_overlap = xy_overlap)
    draw_image = np.copy(image)
    window_img = draw_boxes(draw_image, windows, color=(255, 0, 0), thick=6)                    
    plt.imshow(window_img)
    
"""
windows = slide_window_alternate(image, x_start_stop=[270, None], y_start_stop=[400 , 656], 
                    xy_window=(96, 96), xy_overlap=(0.75, 0.75))
                       
draw_image = np.copy(image)
window_img = draw_boxes(draw_image, windows, color=(255, 0, 0), thick=6)                    
plt.imshow(window_img)
"""

