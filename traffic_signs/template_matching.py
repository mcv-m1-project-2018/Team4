#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import cv2

def template_matching(im, pixel_candidates):
    im_gray = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
    im_gray = np.multiply(im_gray, pixel_candidates)
    matches_by_type = []
    threshold = 0.6
    size_range = [32, 64, 128, 192, 256]
    for sign_type in range(0,6):
        template = cv2.imread("templates/"+str(sign_type)+".jpg",0)
        for size in size_range:
            scaled_template = cv2.resize(template,(size, size))
            w, h = scaled_template.shape[::-1]         
            result = cv2.matchTemplate(im_gray,scaled_template,5)
            print(result)
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

