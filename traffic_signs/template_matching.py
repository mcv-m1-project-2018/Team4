#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import cv2
from morphological_operators import morphological_operators
from connected_labels import connected_labels
# from sliding_window import union, intersection

def template_matching(im, pixel_candidates):
    # cv2.imshow("image", pixel_candidates)
    # cv2.waitKey()
    mask = morphological_operators(pixel_candidates)
    # cv2.imshow("image", mask)
    # cv2.waitKey()
    mask = connected_labels(mask)
    mask = mask.astype(np.uint8)
    cv2.imshow("image", mask)
    cv2.waitKey()
    im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    im_gray = np.multiply(im_gray, mask)
    matches_by_type = []
    threshold = 0.75
    padding = 0.1
    size_range = [64, 96, 128, 192, 256]
    for sign_type in range(0,5):
        template = cv2.imread("templates/"+str(sign_type)+".jpg",0)
        # template = cv2.normalize(template, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        for size in size_range:
            scaled_template = cv2.resize(template,(size, size))
            w, h = scaled_template.shape[::-1]         
            ret, template_mask = cv2.threshold(scaled_template, 1, 255, cv2.THRESH_BINARY)
            padded_mask = np.zeros((int(template_mask.shape[1]+template_mask.shape[1]*padding),int(template_mask.shape[0]+template_mask.shape[0]*padding)))
            print(padded_mask.shape[1])
            print(padded_mask.shape[0])

            padded_mask [int(template_mask.shape[1]*padding*0.5):template_mask.shape[1], int(template_mask.shape[1]*padding*0.5):template_mask.shape[1]] = template_mask
            cv2.imshow("mask", template_mask)
            cv2.waitKey()
            # result = cv2.matchTemplate(im_gray,scaled_template, cv2.TM_CCOEFF_NORMED,template_mask)
            result = cv2.matchTemplate(mask,template_mask, cv2.TM_CCOEFF_NORMED)
            minVal, maxVal, loc1, loc2 = cv2.minMaxLoc(result)
            print("min: "+str(minVal)+" max: "+ str(maxVal))
            loc = np.where( result >= threshold)

            for pt in zip(*loc[::-1]):
                cv2.rectangle(im, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
            # matches_by_type[sign_type] = match
            # cv2.rectangle(im_gray, maxLoc, (maxLoc[0]+w, maxLoc[1]+h),(100,255,200),5)
        cv2.imshow("image", im)
        cv2.imshow("template", template)
        cv2.waitKey()

    # best_match = matches_by_type.index(min(matches_by_type[:,0]))
    # match_location = matches_by_type[best_match,1]
    # window_candidate = [match_location.x, match_location.y, template.cols, template.rows]



    window_candidates = [[17.0, 12.0, 49.0, 44.0], [60.0,90.0,100.0,130.0]]

    return window_candidates

