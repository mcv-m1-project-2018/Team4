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
