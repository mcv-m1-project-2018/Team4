#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import cv2
from skimage import color

def connected_labels(dilated):
    # Connected components
    ret, labels, stats, centroid = cv2.connectedComponentsWithStats(dilated, connectivity=4)
    sizes = stats[1:, -1]
    ret = ret - 1

    min_size = 460
    max_size = 52700

    mask_removeSmall = labels.copy()

    for i in range(0, ret):
        if sizes[i] <= min_size:
            mask_removeSmall[labels == i + 1] = 0

        if sizes[i] >= max_size:
            mask_removeSmall[labels == i + 1] = 0


    label_hue = np.uint8(179 * mask_removeSmall / np.max(mask_removeSmall))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    labeled_img_gray = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)
    toDrawContours = labeled_img.copy()


    ret, labeled_img_bin = cv2.threshold(labeled_img_gray, 0, 255, cv2.THRESH_BINARY)
    copy_labeled_img_bin = labeled_img_bin.copy()

    im2, contours, hierarchy = cv2.findContours(labeled_img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    factor = 7

    form_factor_array = [1.064, 1.024, 0.950, 0.975, 0.943, 0.827]
    form_factor_sd_array = [0.074, 0.144, 0.107, 0.085, 0.104, 0.177]

    filling_ratio_array = [0.501, 0.496, 0.784, 0.779, 0.785, 1.0]
    filling_ratio_sd_array =  [0.004, 0.004, 0.006, 0.055, 0.005, 0.0]


    final_mask = np.zeros(labeled_img_bin.shape)

    # cv2.imshow('bounding_boxes', copy_labeled_img_bin)
    # cv2.waitKey()
    #
    cv2.drawContours(copy_labeled_img_bin, contours, -1, 255, -1)
    #
    # cv2.imshow('bounding_boxes', copy_labeled_img_bin)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # cv2.imshow('img', copy_labeled_img_bin)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        crop = copy_labeled_img_bin[y:y+h, x:x+w]



        size = cv2.countNonZero(crop)
        form_factor = crop.shape[1] / crop.shape[0]
        filling_ratio = size / (crop.shape[1] * crop.shape[0])

        form_factor_bool = False
        filling_ratio_bool = False


        # for idx, ff in enumerate(form_factor_array):

        #     if form_factor >= (ff-factor*form_factor_sd_array[idx]) and form_factor <= (ff+factor*form_factor_sd_array[idx]):
        #         form_factor_bool = True

        for idx,fr in enumerate(filling_ratio_array):
            if filling_ratio >= (fr - factor*filling_ratio_sd_array[idx]) and filling_ratio <= (fr+factor*filling_ratio_sd_array[idx]):
                filling_ratio_bool = True

        if filling_ratio_bool:
            final_mask[y:y+h, x:x+w] = crop

    # cv2.imshow('mask', final_mask)
    # cv2.waitKey()
    # cv2.destroyAllWindows()


        # if form_factor_bool == True:
        #     final_mask[y:y+h, x:x+w] = crop


    ret, labeled_img = cv2.threshold(labeled_img, 0, 255, cv2.THRESH_BINARY)
    ret, final_mask = cv2.threshold(final_mask, 0, 255, cv2.THRESH_BINARY)

    return final_mask
