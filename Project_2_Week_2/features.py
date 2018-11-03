import numpy as np
import cv2
import os

from matplotlib import pyplot as plt

def read_set_features(dataset_path, feature_type='ORB'):
    dataset = []
    set_names = []
    dataset_features = []
    for filename in sorted(os.listdir(dataset_path)):
        img = cv2.imread(os.path.join(dataset_path, filename))
        dataset.append(img)
        set_names.append(filename)
        dataset_features.append(find_features(img,feature_type))

    return dataset, dataset_features, set_names

def find_features(img, feature_type='ORB'):
    """
    Available descriptors: 'ORB', 'FAST','SIFT', 'SURF' 'KAZE', 'AKAZE', 'BRISK', 'MSER'
    """
    features_dataset = []

    if feature_type=='ORB':
        features = cv2.ORB_create()
    elif feature_type=='FAST':
        features = cv2.FastFeatureDetector_create()
    elif feature_type=='SIFT':
        features = cv2.SIFT_create()
    elif feature_type=='SURF':
        features = cv2.SURF_create()
    elif feature_type=='KAZE':
        features = cv2.KAZE_create()
    elif feature_type=='AKAZE':
        features = cv2.AKAZE_create()
    elif feature_type=='BRISK':
        features = cv2.BRISK_create()
    elif feature_type=='MSER':
        features = cv2.MSER_create()
    
    kp, des = features.detectAndCompute(img,None)
    # features_dataset.append([kp,des])
    features_dataset.append(des)

    return des
    
def match_features(des1, des2, matcher_type='BF', matching_method='KNN', threshold=0.85, norm_type='NORM_HAMMING', cross_check=False):
    """
    Available matchers: 'BF', 'FLANN'
    
    Parameters:	

    normType – One of NORM_L1, NORM_L2, NORM_HAMMING, NORM_HAMMING2. 
    L1 and L2 norms are preferable choices for SIFT and SURF descriptors, 
    NORM_HAMMING should be used with ORB, BRISK and BRIEF, 
    NORM_HAMMING2 should be used with ORB when WTA_K==3 or 4 (see ORB::ORB constructor description).
    
    crossCheck : If it is false, this is will be default BFMatcher behaviour when it finds the k nearest neighbors for each query descriptor. 
    If crossCheck==true, then the knnMatch() method with k=1 will only return pairs (i,j) such that
    for i-th query descriptor the j-th descriptor in the matcher’s collection is the nearest and vice versa,
    i.e. the BFMatcher will only return consistent pairs. 
    Such technique usually produces best results with minimal number of outliers when there are enough matches. 
    This is alternative to the ratio test, used by D. Lowe in SIFT paper.

    """


    if matcher_type=='ORB':
        matcher = cv2.BFMatcher_create(norm_type, cross_check)
    elif matcher_type=='FAST':
        matcher = cv2.FlannBasedMatcher_create()

    if matching_method=='KNN':
        matches = matcher.knnMatch(des1,des2, k=2)
    elif matching_method=='MATCH':
        matcher = matcher.match()
    elif matching_method=='RADIUS':
        matcher = matcher.radiusMatch()


    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < threshold*n.distance:
            good.append([m])
    
    return len(good)
