# """
# Usage:
#   task.py <museum_set_path> <query_set_path>
# Options:
# """


import numpy as np
import cv2
import sys
import os
from matplotlib import pyplot as plt

def read_sets(museum_path, query_path):
    museum_set = []
    query_set = []

    for filename in os.listdir(museum_path):
        img = cv2.imread(os.path.join(museum_path,filename))
        museum_set.append(img)

    for filename in os.listdir(query_path):
        img = cv2.imread(os.path.join(query_path, filename))
        query_set.append(img)

    create_descriptors(museum_set, query_set)

def create_descriptors(museum_set, query_set):
    rgb_hists_dataset = []
    lab_hists_dataset = []
    ycb_hists_dataset = []
    hsv_hists_dataset = []

    block_hists_dataset = []

    pyramid_hists_dataset = []

    block_factor = 4

    for idx, im in enumerate(museum_set):
        rgb_hists_ind, lab_hists_ind, ycb_hists_ind, hsv_hists_ind = global_representation(im)

        rgb_hists_dataset.append(rgb_hists_ind)
        lab_hists_dataset.append(lab_hists_ind)
        ycb_hists_dataset.append(ycb_hists_ind)
        hsv_hists_dataset.append(hsv_hists_ind)

        block_hist_ind = block_representation(im, block_factor)
        block_hists_dataset.append(block_hist_ind)

        pyramid_levels = pyramid_representation(im, block_factor)
        pyramid_levels.insert(0, block_hists_dataset[idx])
        pyramid_hists_dataset.append(pyramid_levels)
    

def global_representation(im):
    rgb_hists = []
    lab_hists = []
    ycb_hists = []
    hsv_hists = []
    """Compute RGB histograms"""
    # plt.plot(b_hist, color='b')
    # plt.show()
    hist_0 = cv2.calcHist([im], [0], None, [256], [0, 256])
    hist_1 = cv2.calcHist([im], [1], None, [256], [0, 256])
    hist_2 = cv2.calcHist([im], [2], None, [256], [0, 256])
    rgb_hists.append(hist_0)
    rgb_hists.append(hist_1)
    rgb_hists.append(hist_2)
    """Compute lab histograms"""
    im_lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    hist_0 = cv2.calcHist([im_lab], [0], None, [256], [0, 256])
    hist_1 = cv2.calcHist([im_lab], [1], None, [256], [0, 256])
    hist_2 = cv2.calcHist([im_lab], [2], None, [256], [0, 256])
    lab_hists.append(hist_0)
    lab_hists.append(hist_1)
    lab_hists.append(hist_2)
    """Compute YCB histograms"""
    im_ycb = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
    hist_0 = cv2.calcHist([im_ycb], [0], None, [256], [0, 256])
    hist_1 = cv2.calcHist([im_ycb], [1], None, [256], [0, 256])
    hist_2 = cv2.calcHist([im_ycb], [2], None, [256], [0, 256])
    ycb_hists.append(hist_0)
    ycb_hists.append(hist_1)
    ycb_hists.append(hist_2)
    """Compute HSV histograms"""
    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    hist_0 = cv2.calcHist([im_hsv], [0], None, [256], [0, 256])
    hist_1 = cv2.calcHist([im_hsv], [1], None, [256], [0, 256])
    hist_2 = cv2.calcHist([im_hsv], [2], None, [256], [0, 256])
    hsv_hists.append(hist_0)
    hsv_hists.append(hist_1)
    hsv_hists.append(hist_2)
    return rgb_hists, lab_hists, ycb_hists, hsv_hists


def block_representation(im, block_factor):
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
            window_gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
            hist_crop = cv2.calcHist([window_gray], [0], None, [256], [0, 256])
            hist_crops.append(hist_crop)

    return hist_crops

def pyramid_representation(im, block_factor):
    """SPATIAL PYRAMID REPRESENTATION"""
    reduced_block_factor = block_factor;
    pyramid_levels = []
    while reduced_block_factor % 2 == 0:
        reduced_block_factor = int(reduced_block_factor / 2)

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
                window_gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
                hist_crop = cv2.calcHist([window_gray], [0], None, [256], [0, 256])
                hist_crops.append(hist_crop)

        pyramid_levels.append(hist_crops)

    return pyramid_levels


if __name__ == "__main__":
    if len(sys.argv) > 1:
        museum_path = sys.argv[1]
        query_path = sys.argv[2]
        read_sets(museum_path, query_path)

    else:
        print("ARGUMENTS NEEDED!")
        quit();