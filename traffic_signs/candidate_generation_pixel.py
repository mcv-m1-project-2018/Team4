#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import cv2
from skimage import color

def morphological_operators(mask_final):
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(mask_final, kernel, iterations =1)
    dilated = cv2.dilate(erosion, kernel, iterations=1)
    dilated = cv2.erode(dilated, kernel, iterations =1)
    return dilated

def connected_labels(dilated):
    # Connected components
    ret, labels, stats, centroid = cv2.connectedComponentsWithStats(dilated, connectivity=4)
    sizes = stats[1:, -1]
    ret = ret - 1

    min_size = 200

    mask_removeSmall = labels.copy()

    for i in range(0, ret):
        if sizes[i] <= min_size:
            mask_removeSmall[labels == i + 1] = 0

    label_hue = np.uint8(179 * mask_removeSmall / np.max(mask_removeSmall))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)
    ret, labeled_img = cv2.threshold(labeled_img, 0, 255, cv2.THRESH_BINARY)

    return labeled_img

def candidate_generation_pixel_normrgb(im):
    # convert input image to the normRGB color space

    normrgb_im = np.zeros(im.shape)
    eps_val = 0.00001
    norm_factor_matrix = im[:,:,0] + im[:,:,1] + im[:,:,2] + eps_val

    normrgb_im[:,:,0] = im[:,:,0] / norm_factor_matrix
    normrgb_im[:,:,1] = im[:,:,1] / norm_factor_matrix
    normrgb_im[:,:,2] = im[:,:,2] / norm_factor_matrix
    
    # Develop your method here:
    # Example:
    pixel_candidates = normrgb_im[:,:,1]>100;

    return pixel_candidates
 
def candidate_generation_pixel_hsv(im):
    # convert input image to HSV color space
    hsv_im = color.rgb2hsv(im)
    
    # Develop your method here:
    # Example:
    pixel_candidates = hsv_im[:,:,1] > 0.4;

    return pixel_candidates

def candidate_generation_pixel_hsv_manual(img):
    # convert input image to HSV color space
    imgHSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
   
    maskHSV_red1 = cv2.inRange(imgHSV, np.array([0, 70, 0]), np.array([10, 255, 255]))
    maskHSV_red2 = cv2.inRange(imgHSV, np.array([160, 70, 0]), np.array([179, 255, 255]))
    maskHSV_blue = cv2.inRange(imgHSV, np.array([100, 70, 0]), np.array([140, 255, 255]))
    # maskHSV_blue[maskHSV_blue == 255] = 127
    
    maskHSV_red = cv2.bitwise_or(maskHSV_red1, maskHSV_red2)
    mask_final = cv2.bitwise_or(maskHSV_blue, maskHSV_red)

    pixel_candidates = mask_final

    return pixel_candidates


def candidate_generation_pixel_hsv_hist(img):
    # convert input image to HSV color space
    imgHSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
   
    maskHSV_red1 = cv2.inRange(imgHSV, np.array([0, 45, 45]), np.array([17, 255, 255]))
    maskHSV_red2 = cv2.inRange(imgHSV, np.array([165, 45, 45]), np.array([255, 255, 255]))
    maskHSV_blue = cv2.inRange(imgHSV, np.array([84, 45, 45]), np.array([115, 255, 255]))
    # maskHSV_blue[maskHSV_blue == 255] = 127
    
    maskHSV_red = cv2.bitwise_or(maskHSV_red1, maskHSV_red2)
    mask_final = cv2.bitwise_or(maskHSV_blue, maskHSV_red)

    pixel_candidates = mask_final

    return pixel_candidates

def candidate_generation_pixel_hsv_hist_equal(img):
    # apply adaptive equilization of separate channel histograms
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
    eqR = clahe.apply(img[:,:,0])
    eqG = clahe.apply(img[:,:,1])
    eqB = clahe.apply(img[:,:,2])
    img[:,:,0] = eqR
    img[:,:,1] = eqG
    img[:,:,2] = eqB

    # convert input image to HSV color space
    imgHSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
   
    maskHSV_red1 = cv2.inRange(imgHSV, np.array([0, 70, 0]), np.array([10, 255, 255]))
    maskHSV_red2 = cv2.inRange(imgHSV, np.array([160, 70, 0]), np.array([179, 255, 255]))
    maskHSV_blue = cv2.inRange(imgHSV, np.array([100, 70, 0]), np.array([140, 255, 255]))
    # maskHSV_blue[maskHSV_blue == 255] = 127
    
    maskHSV_red = cv2.bitwise_or(maskHSV_red1, maskHSV_red2)
    mask_final = cv2.bitwise_or(maskHSV_blue, maskHSV_red)

    pixel_candidates = mask_final

    return pixel_candidates
 
# Create your own candidate_generation_pixel_xxx functions for other color spaces/methods
# Add them to the switcher dictionary in the switch_color_space() function
# These functions should take an image as input and output the pixel_candidates mask image
 
def switch_color_space(im, color_space):
    switcher = {
        'normrgb': candidate_generation_pixel_normrgb,
        'hsv_manual': candidate_generation_pixel_hsv_manual,
        'hsv_hist': candidate_generation_pixel_hsv_hist,
        'hsv_hist_equal': candidate_generation_pixel_hsv_hist_equal
        #'lab'    : candidate_generation_pixel_lab,
    }
    # Get the function from switcher dictionary
    func = switcher.get(color_space, lambda: "Invalid color space")
    func = switcher[color_space]


    # Execute the function
    pixel_candidates =  func(im)

    return pixel_candidates


def candidate_generation_pixel(im, color_space):

    pixel_candidates = switch_color_space(im, color_space)

    return pixel_candidates

    
if __name__ == '__main__':
    im = None
    pixel_candidates1 = candidate_generation_pixel(im, 'hsv_manual')
    pixel_candidates2 = candidate_generation_pixel(im, 'hsv_hist')
    pixel_candidates3 = candidate_generation_pixel(im, 'hsv_hist_equal')

    
