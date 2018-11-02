import numpy as np
import cv2
import os

from matplotlib import pyplot as plt


def match_features(des1, des2, threshold):
    # img1 = cv2.imread('BBDD_W4/ima_000047.jpg',0)          # queryImage
    # img2 = cv2.imread('query_devel_W4/ima_000025.jpg',0) # trainImage

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, 2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < threshold*n.distance:
            good.append([m])
    return len(good)



def read_set_features(dataset_path, feature_type='ORB'):
    dataset = []
    set_names = []
    dataset_features = []
    for filename in sorted(os.listdir(dataset_path)):
        img = cv2.imread(os.path.join(dataset_path, filename))
        dataset.append(img)
        set_names.append(filename)
        dataset_features.append(find_features(img,'ORB'))

    return dataset, dataset_features, set_names


def find_features(img, feature_type):
    features_dataset = []

    features = cv2.ORB_create()

    if feature_type=='ORB':
        features = cv2.ORB_create()
    elif feature_type=='FAST':
        features = cv2.FastFeatureDetector_create()

    kp, des = features.detectAndCompute(img,None)
    # features_dataset.append([kp,des])
    features_dataset.append(des)

    return des
    