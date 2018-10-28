#!/usr/bin/python
import cv2

# methods: 1=chi-square, 2=histogram-intersection 3=Hellinger-distance
def compare(hist1, hist2, method=1):
    score = cv2.compareHist(hist1[0], hist2[0], method=method)
    return score

def compare_3channel(hist1, hist2, block_color_space, method=1):
    """
    methods: 1=chi-square, 2=histogram-intersection 3=Hellinger-distance
    """
    if block_color_space != 0:
        score1 = cv2.compareHist(hist1[0], hist2[0], method=method)
        score2 = cv2.compareHist(hist1[1], hist2[1], method=method)
        score3 = cv2.compareHist(hist1[2], hist2[2], method=method)
        score = score1 + score2 + score3
    else:
        score = cv2.compareHist(hist1[0], hist2[0], method=method)
    return score

def compare_block(hist1, hist2, method=1):

    overall_score = 0
    for r in range(0,len(hist1)):
        score = cv2.compareHist(hist1[r], hist2[r], method=method)
        overall_score += score

    return overall_score

def compare_pyramid(hist1, hist2, block_color_space, block_factor, method=1):

    overall_score = 0
    whole_image_index = len(hist1)-1
    overall_score = compare_3channel(hist1[whole_image_index], hist2[whole_image_index], block_color_space, method=method)
    
    for r in range(0, len(hist1)-1):
        score = compare_block(hist1[r], hist2[r], method=method)
        overall_score += score
    return overall_score

def compare_pyramid_weights(hist1, hist2, block_color_space, block_factor, method=1):

    overall_score = 0
    whole_image_index = len(hist1)-1
    overall_score = compare_3channel(hist1[whole_image_index], hist2[whole_image_index], block_color_space, method=method)
    ratio = block_factor*block_factor
    
    for r in range(0, len(hist1)-1):
        score = compare_block(hist1[r], hist2[r], method=method)
        score = score / ratio
        ratio = ratio / 4
        overall_score += score
    return overall_score

def compare_full(hist1, hist2, block_color_space, method=1):

    overall_score = 0
    for r in range(0, len(hist1)-1):
        if r==0:
            overall_score += compare(hist1[r], hist2[r], method=method)
        elif r==1 or r==2 or r==3 or r==4:
            overall_score += compare_3channel(hist1[r], hist2[r], block_color_space, method=method)
        elif r==5:
            score = compare_block(hist1[r], hist2[r], method=1)
            overall_score += score
        elif r==6:
            score = compare_pyramid(hist1[r], hist2[r], method=1)
            overall_score += score
    return overall_score