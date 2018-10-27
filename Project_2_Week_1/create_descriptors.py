
import numpy as np
import cv2
import sys
import os
from matplotlib import pyplot as plt
from compare import compare, compare_3channel, compare_block

def read_set(dataset_path, block_color_space):
    dataset = []

    for filename in sorted(os.listdir(dataset_path)):
        img = cv2.imread(os.path.join(dataset_path, filename))
        dataset.append(img)

    return dataset, create_descriptors(dataset, block_color_space)

def read_query_image(image_path, block_color_space):

    img = cv2.imread(image_path, 1)

    return img, create_descriptors(img, block_color_space)

def create_descriptors(dataset, block_color_space):
    gray_hists_dataset = []
    rgb_hists_dataset = []
    lab_hists_dataset = []
    ycb_hists_dataset = []
    hsv_hists_dataset = []

    block_hists_dataset = []

    pyramid_hists_dataset = []

    full_set_representation=[]

    block_factor = 8

    for idx, im in enumerate(dataset):
        gray_hists_ind, rgb_hists_ind, lab_hists_ind, ycb_hists_ind, hsv_hists_ind = global_representation(im)

        gray_hists_dataset.append(gray_hists_ind)
        rgb_hists_dataset.append(rgb_hists_ind)
        lab_hists_dataset.append(lab_hists_ind)
        ycb_hists_dataset.append(ycb_hists_ind)
        hsv_hists_dataset.append(hsv_hists_ind)

        block_hist_ind = block_representation(im, block_factor, block_color_space)
        block_hists_dataset.append(block_hist_ind)

        pyramid_levels = pyramid_representation(im, block_factor, block_color_space)
        if block_color_space == 0:
            pyramid_levels.append(gray_hists_dataset[idx])
        elif block_color_space == 1:
            pyramid_levels.append(rgb_hists_dataset[idx])
        elif block_color_space == 2:
            pyramid_levels.append(lab_hists_dataset[idx])
        elif block_color_space == 3:
            pyramid_levels.append(ycb_hists_dataset[idx])
        elif block_color_space == 4:
            pyramid_levels.append(hsv_hists_dataset[idx])

        pyramid_hists_dataset.append(pyramid_levels)

        full_representation = []
        full_representation.append(gray_hists_ind)
        full_representation.append(rgb_hists_ind)
        full_representation.append(lab_hists_ind)
        full_representation.append(ycb_hists_ind)
        full_representation.append(hsv_hists_ind)
        full_representation.append(block_hist_ind)
        full_representation.append(pyramid_levels)

        full_set_representation.append(full_representation)

    return gray_hists_dataset, rgb_hists_dataset, lab_hists_dataset, ycb_hists_dataset, hsv_hists_dataset, block_hists_dataset, pyramid_hists_dataset, full_set_representation
    

def global_representation(im):
    gray_hists = []
    rgb_hists = []
    lab_hists = []
    ycb_hists = []
    hsv_hists = []

    """Compute gray histogram"""
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    hist_0 = cv2.calcHist([im_gray], [0], None, [256], [0, 256])
    gray_hists.append(hist_0)
    """Compute RGB histograms"""
    # plt.plot(b_hist, color='b')
    # plt.show()
    hist_0 = cv2.calcHist([im], [0], None, [256], [0, 256])
    hist_1 = cv2.calcHist([im], [1], None, [256], [0, 256])
    hist_2 = cv2.calcHist([im], [2], None, [256], [0, 256])
    cv2.normalize(hist_0, hist_0, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_1, hist_1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_2, hist_2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    rgb_hists.append(hist_0)
    rgb_hists.append(hist_1)
    rgb_hists.append(hist_2)
    """Compute lab histograms"""
    im_lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    hist_0 = cv2.calcHist([im_lab], [0], None, [256], [0, 256])
    hist_1 = cv2.calcHist([im_lab], [1], None, [256], [0, 256])
    hist_2 = cv2.calcHist([im_lab], [2], None, [256], [0, 256])
    cv2.normalize(hist_0, hist_0, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_1, hist_1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_2, hist_2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    lab_hists.append(hist_0)
    lab_hists.append(hist_1)
    lab_hists.append(hist_2)
    """Compute YCB histograms"""
    im_ycb = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
    hist_0 = cv2.calcHist([im_ycb], [0], None, [256], [0, 256])
    hist_1 = cv2.calcHist([im_ycb], [1], None, [256], [0, 256])
    hist_2 = cv2.calcHist([im_ycb], [2], None, [256], [0, 256])
    cv2.normalize(hist_0, hist_0, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_1, hist_1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_2, hist_2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    ycb_hists.append(hist_0)
    ycb_hists.append(hist_1)
    ycb_hists.append(hist_2)
    """Compute HSV histograms"""
    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    hist_0 = cv2.calcHist([im_hsv], [0], None, [256], [0, 256])
    hist_1 = cv2.calcHist([im_hsv], [1], None, [256], [0, 256])
    hist_2 = cv2.calcHist([im_hsv], [2], None, [256], [0, 256])
    cv2.normalize(hist_0, hist_0, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_1, hist_1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_2, hist_2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    hsv_hists.append(hist_0)
    hsv_hists.append(hist_1)
    hsv_hists.append(hist_2)
    return gray_hists, rgb_hists, lab_hists, ycb_hists, hsv_hists

def block_representation(im, block_factor, color_space=0):


    """BLOCK REPRESENTATION"""
    # Shape = rows and columns
    remainder_rows = im.shape[0] % block_factor
    remainder_cols = im.shape[1] % block_factor

    im_block = cv2.copyMakeBorder(im, block_factor - remainder_rows, 0, block_factor - remainder_cols, 0,
                                  cv2.BORDER_CONSTANT)

    windowsize_r = int(im_block.shape[0] / block_factor)
    windowsize_c = int(im_block.shape[1] / block_factor)

    # print(im_block.shape)
    # print(str(windowsize_r)+' '+str(windowsize_c))
    # cv2.imshow("fullImg", im_block)

    hist_crops = []
    for r in range(0, im_block.shape[0], windowsize_r):
        for c in range(0, im_block.shape[1], windowsize_c):
            window = im_block[r:r + windowsize_r, c:c + windowsize_c]
            if color_space == 0:
                window_gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
                hist_crop = cv2.calcHist([window_gray], [0], None, [256], [0, 256])
                cv2.normalize(hist_crop, hist_crop, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                hist_crops.append(hist_crop)
            elif color_space == 1:
                hist_crop = cv2.calcHist([window], [0], None, [256], [0, 256])
                cv2.normalize(hist_crop, hist_crop, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                hist_crops.append(hist_crop)
                hist_crop = cv2.calcHist([window], [1], None, [256], [0, 256])
                cv2.normalize(hist_crop, hist_crop, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                hist_crops.append(hist_crop)
                hist_crop = cv2.calcHist([window], [2], None, [256], [0, 256])
                cv2.normalize(hist_crop, hist_crop, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                hist_crops.append(hist_crop)
            elif color_space == 2:
                window= cv2.cvtColor(window, cv2.COLOR_BGR2LAB)
                hist_crop = cv2.calcHist([window], [0], None, [256], [0, 256])
                cv2.normalize(hist_crop, hist_crop, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                hist_crops.append(hist_crop)
                hist_crop = cv2.calcHist([window], [1], None, [256], [0, 256])
                cv2.normalize(hist_crop, hist_crop, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                hist_crops.append(hist_crop)
                hist_crop = cv2.calcHist([window], [2], None, [256], [0, 256])
                cv2.normalize(hist_crop, hist_crop, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                hist_crops.append(hist_crop)
            elif color_space == 3:
                window = cv2.cvtColor(window, cv2.COLOR_BGR2YCrCb)
                hist_crop = cv2.calcHist([window], [0], None, [256], [0, 256])
                cv2.normalize(hist_crop, hist_crop, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                hist_crops.append(hist_crop)
                hist_crop = cv2.calcHist([window], [1], None, [256], [0, 256])
                cv2.normalize(hist_crop, hist_crop, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                hist_crops.append(hist_crop)
                hist_crop = cv2.calcHist([window], [2], None, [256], [0, 256])
                cv2.normalize(hist_crop, hist_crop, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                hist_crops.append(hist_crop)
            elif color_space == 4:
                window = cv2.cvtColor(window, cv2.COLOR_BGR2HSV)
                hist_crop = cv2.calcHist([window], [0], None, [256], [0, 256])
                cv2.normalize(hist_crop, hist_crop, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                hist_crops.append(hist_crop)
                hist_crop = cv2.calcHist([window], [1], None, [256], [0, 256])
                cv2.normalize(hist_crop, hist_crop, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                hist_crops.append(hist_crop)
                hist_crop = cv2.calcHist([window], [2], None, [256], [0, 256])
                cv2.normalize(hist_crop, hist_crop, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                hist_crops.append(hist_crop)

    return hist_crops

def pyramid_representation(im, block_factor, color_space=0):
    """SPATIAL PYRAMID REPRESENTATION"""
    reduced_block_factor = block_factor;
    pyramid_levels = []
    while reduced_block_factor % 2 == 0:

        if reduced_block_factor == 1:
            break

        remainder_rows = im.shape[0] % reduced_block_factor
        remainder_cols = im.shape[1] % reduced_block_factor

        im_block = cv2.copyMakeBorder(im, reduced_block_factor - remainder_rows, 0, reduced_block_factor - remainder_cols, 0, cv2.BORDER_CONSTANT)

        windowsize_r = int(im_block.shape[0] / reduced_block_factor)
        windowsize_c = int(im_block.shape[1] / reduced_block_factor)

        hist_crops = []
        for r in range(0, im_block.shape[0], windowsize_r):
            for c in range(0, im_block.shape[1], windowsize_c):
                window = im_block[r:r + windowsize_r, c:c + windowsize_c]
                if color_space == 0:
                    window_gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
                    hist_crop = cv2.calcHist([window_gray], [0], None, [256], [0, 256])
                    cv2.normalize(hist_crop, hist_crop, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                    hist_crops.append(hist_crop)
                elif color_space == 1:
                    hist_crop = cv2.calcHist([window], [0], None, [256], [0, 256])
                    cv2.normalize(hist_crop, hist_crop, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                    hist_crops.append(hist_crop)
                    hist_crop = cv2.calcHist([window], [1], None, [256], [0, 256])
                    cv2.normalize(hist_crop, hist_crop, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                    hist_crops.append(hist_crop)
                    hist_crop = cv2.calcHist([window], [2], None, [256], [0, 256])
                    cv2.normalize(hist_crop, hist_crop, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                    hist_crops.append(hist_crop)
                elif color_space == 2:
                    window = cv2.cvtColor(window, cv2.COLOR_BGR2LAB)
                    hist_crop = cv2.calcHist([window], [0], None, [256], [0, 256])
                    cv2.normalize(hist_crop, hist_crop, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                    hist_crops.append(hist_crop)
                    hist_crop = cv2.calcHist([window], [1], None, [256], [0, 256])
                    cv2.normalize(hist_crop, hist_crop, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                    hist_crops.append(hist_crop)
                    hist_crop = cv2.calcHist([window], [2], None, [256], [0, 256])
                    cv2.normalize(hist_crop, hist_crop, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                    hist_crops.append(hist_crop)
                elif color_space == 3:
                    window = cv2.cvtColor(window, cv2.COLOR_BGR2YCrCb)
                    hist_crop = cv2.calcHist([window], [0], None, [256], [0, 256])
                    cv2.normalize(hist_crop, hist_crop, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                    hist_crops.append(hist_crop)
                    hist_crop = cv2.calcHist([window], [1], None, [256], [0, 256])
                    cv2.normalize(hist_crop, hist_crop, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                    hist_crops.append(hist_crop)
                    hist_crop = cv2.calcHist([window], [2], None, [256], [0, 256])
                    cv2.normalize(hist_crop, hist_crop, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                    hist_crops.append(hist_crop)
                elif color_space == 4:
                    window = cv2.cvtColor(window, cv2.COLOR_BGR2HSV)
                    hist_crop = cv2.calcHist([window], [0], None, [256], [0, 256])
                    cv2.normalize(hist_crop, hist_crop, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                    hist_crops.append(hist_crop)
                    hist_crop = cv2.calcHist([window], [1], None, [256], [0, 256])
                    cv2.normalize(hist_crop, hist_crop, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                    hist_crops.append(hist_crop)
                    hist_crop = cv2.calcHist([window], [2], None, [256], [0, 256])
                    cv2.normalize(hist_crop, hist_crop, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                    hist_crops.append(hist_crop)
        pyramid_levels.append(hist_crops)

        reduced_block_factor = int(reduced_block_factor / 2)

    return pyramid_levels

