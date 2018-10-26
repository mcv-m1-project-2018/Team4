#!/usr/bin/python
import cv2

# methods: 1=chi-square, 2=histogram-intersection 3=Hellinger-distance
def compare(hist1, hist2, method=1):
    score = cv2.compareHist(hist1, hist2, method=method)
    return score

def compare_3channel(hist1, hist2, method=1):
    """
    methods: 1=chi-square, 2=histogram-intersection 3=Hellinger-distance
    """
    score1 = cv2.compareHist(hist1[0], hist2[0], method=method)
    score2 = cv2.compareHist(hist1[1], hist2[1], method=method)
    score3 = cv2.compareHist(hist1[2], hist2[2], method=method)
    score = score1 + score2 + score3
    return score

def compare_block(hist1, hist2, method=1):

    overall_score = 0
    for r in range(0,len(hist1)):
            score = cv2.compareHist(hist1[r], hist2[r], method=method)
            overall_score += score

    return overall_score

def compare_pyramid(hist1, hist2, method=1):

    overall_score = 0
    whole_image_index = len(hist1)-1
    overall_score = cv2.compareHist(hist1[whole_image_index], hist2[whole_image_index], method=method)
    
    for r in range(0,len(hist1)-1):
            score = cv2.compareHist(hist1[r], hist2[r], method=method)
            overall_score += score

    return overall_score    