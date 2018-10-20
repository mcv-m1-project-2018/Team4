#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import cv2
from morphological_operators import morphological_operators
from connected_labels import connected_labels
# from sliding_window import union, intersection

def template_matching(im, pixel_candidates):

    mask = morphological_operators(pixel_candidates)
    mask = connected_labels(mask)
    mask = mask.astype(np.uint8)
    # im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    # im_gray = np.multiply(im_gray, mask)
    # matches_by_type = []
    threshold = 0.7
    # padding_ratio = 0.1
    size_range = [64, 96, 128, 192, 256]
    final_mask = np.zeros(pixel_candidates.shape)
    for sign_type in range(0,6):
        template = cv2.imread("templates/"+str(sign_type)+"bw.jpg",0)
        # template = cv2.normalize(template, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        for size in size_range:
            scaled_template = cv2.resize(template,(size, size))
            w, h = scaled_template.shape[::-1]         
            ret, template_mask = cv2.threshold(scaled_template, 1, 255, cv2.THRESH_BINARY)
            # padding = 2*int(template_mask.shape[1]*padding_ratio)
            # padded_mask = np.zeros((int(template_mask.shape[1]+padding),int(template_mask.shape[0]+padding)), dtype=np.uint8)


            # padded_mask [int(padding*0.5):template_mask.shape[1]+int(padding*0.5), int(padding*0.5):template_mask.shape[1]+int(padding*0.5)] = template_mask
            # cv2.imshow("mask", padded_mask)
            # cv2.waitKey()
            # result = cv2.matchTemplate(im_gray,scaled_template, cv2.TM_CCOEFF_NORMED,template_mask)
            result = cv2.matchTemplate(mask, template_mask, cv2.TM_CCOEFF_NORMED)
            # minVal, maxVal, loc1, loc2 = cv2.minMaxLoc(result)
            # print("min: "+str(minVal)+" max: "+ str(maxVal))
            loc = np.where( result >= threshold)

            for pt in zip(*loc[::-1]):
                # cv2.rectangle(im, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
                final_mask[pt[1]:pt[1]+h,pt[0]:pt[0]+w]=1
            # matches_by_type[sign_type] = match
            # cv2.rectangle(im_gray, maxLoc, (maxLoc[0]+w, maxLoc[1]+h),(100,255,200),5)
    ret,thresh1 = cv2.threshold(final_mask,0,255,cv2.THRESH_BINARY)
    im2, contours, hierarchy = cv2.findContours(thresh1.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    window_candidates = []
    # tmp = im.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        window_candidates.append([y,x,y+h,x+w])
        # cv2.rectangle(tmp,(x,y),(x+w,y+h),(0,40,255),3)
    # cv2.imshow('contoured_bb',tmp)
    # cv2.waitKey(100)
    # cv2.imshow("image", im)
        # cv2.imshow("template", template)
    # cv2.waitKey()

    return window_candidates

