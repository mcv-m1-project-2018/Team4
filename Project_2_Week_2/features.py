import numpy as np
import cv2
import os
from resize import image_resize
import cropRotate
import numpy as np

from matplotlib import pyplot as plt

def read_set_features(dataset_path, feature_type='ORB', crop_rotate=False, mask_museum=False):
    dataset = []
    set_names = []
    dataset_features = []
    bboxes = []
    text_bboxes =  cropRotate.load_annotations("./text.pkl")
    for idx,filename in enumerate(sorted(os.listdir(dataset_path))):
        img = cv2.imread(os.path.join(dataset_path, filename),0)
        if (crop_rotate):
            bboxes, img = cropRotate.crop_rotate_manu(img)
        if (mask_museum):
            mask = mask_text(img, text_bboxes[idx])
            dataset_features.append(find_features(img, feature_type, mask))
        else:
            dataset_features.append(find_features(img,feature_type))
        dataset.append(img)
        set_names.append(filename)

    return dataset, dataset_features, set_names, bboxes

def find_features(img, feature_type='ORB', mask=None):
    """
    Available descriptors: 'ORB', 'FAST','SIFT', 'SURF' 'KAZE', 'AKAZE', 'BRISK', 'MSER'
    """
    features_dataset = []

    features = cv2.ORB_create()
    
    if feature_type=='ORB':
        features = cv2.ORB_create()
    elif feature_type=='FAST':
        features = cv2.FastFeatureDetector_create()
    elif feature_type=='SIFT':
        features = cv2.xfeatures2d.SIFT_create()
    elif feature_type=='SURF':
        features = cv2.xfeatures2d.SURF_create(hessianThreshold=3000, upright=True, extended=False)
        # features = cv2.xfeatures2d.SURF_create(hessianThreshold=3000, upright=False, extended=False)
    elif feature_type=='KAZE':
        features = cv2.KAZE_create()
    elif feature_type=='AKAZE':
        features = cv2.AKAZE_create()
    elif feature_type=='BRISK':
        features = cv2.BRISK_create()
    elif feature_type=='MSER':
        features = cv2.MSER_create()

    kp, des = features.detectAndCompute(img,mask)
    # img_print = cv2.drawKeypoints(img, kp, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow("kp",img_print)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # features_dataset.append([kp,des])
    # features_dataset.append(des)

    # return [kp, des]
    return des
    
def match_features(des1, des2, matcher_type='BF', matching_method='KNN', threshold=0.75, norm_type='NORM_HAMMING', cross_check=False, swap_check=False):
    """
    Available matchers: 'BF', 'FLANN'
    Available matching methods: 'KNN', 'MATCHER', 'RADIUS'
    
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

    matcher = cv2.BFMatcher_create()
    if matcher_type=='BF':
        if norm_type=='L1':
            matcher = cv2.BFMatcher_create(cv2.NORM_L1)
        if norm_type=='L2':
            matcher = cv2.BFMatcher_create(cv2.NORM_L2)
        if norm_type=='NORM_HAMMING':
            matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING)
        if norm_type=='NORM_HAMMING2':
            matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING2)
    elif matcher_type=='FLANN':
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        matcher = cv2.FlannBasedMatcher(index_params,search_params)
   
    
    k = 2
    matches = []
    if (des1 is not None and des2 is not None):
        if(len(des1)>2 and len(des2)>2):
            matches = matcher.knnMatch(des1,des2, k=k)
            if matching_method=='KNN':
                matches = matcher.knnMatch(des1,des2, k=k)
            elif matching_method=='MATCH':
                matches = matcher.match(des1,des2)
            elif matching_method=='RADIUS':
                matches = matcher.radiusMatch(des1,des2)

    # Apply ratio test
    good = []
    # print(len(matches))
    # print(matches)
    if len(matches): 
        if (len(matches[0])>1):
            for m,n in matches:
                if m.distance < threshold*n.distance:
                    good.append([m])

    if (swap_check):
        matches = []
        if (des1 is not None and des2 is not None):
            if(len(des1)>2 and len(des2)>2):
                matches = matcher.knnMatch(des2,des1, k=k)
                if matching_method=='KNN':
                    matches = matcher.knnMatch(des2,des1, k=k)
                elif matching_method=='MATCH':
                    matches = matcher.match(des2,des1)
                elif matching_method=='RADIUS':
                    matches = matcher.radiusMatch(des2,des1)

        # Apply ratio test
        good_swap = []
        # print(len(matches))
        # print(matches)
        if len(matches): 
            if (len(matches[0])>1):
                for m,n in matches:
                    if m.distance < threshold*n.distance:
                        good_swap.append([m])
        return(min(len(good), len(good_swap)))

    for m,n in matches:
        if m.distance < threshold*n.distance:
            good.append([m])

    return len(good)


def mask_text (img, bboxes):
    mask = np.zeros((img.shape[0], img.shape[1], 1),dtype=np.uint8)
    mask[bboxes[1]:bboxes[3], bboxes[0]:bboxes[2]] = 255
    mask = cv2.bitwise_not(mask)
    return mask


# def delete_keypoints_in_ROI (dataset_features, bboxes):
#     # for keypoints, bbox in zip(dataset_features, bboxes):
#     for i in range(len(dataset_features)):
#         good_keypoints = []
#         for keypoint in dataset_features[0][i]:
#             if keypoint.pt[0]>bboxes[i][0] and keypoint.pt[0]<bboxes[i][2] and keypoint.pt[1]>bboxes[i][1] and keypoint.pt[1]<bboxes[i][3]:
#                 continue
#             else:
#                 good_keypoints.append(keypoint)
#         dataset_features[0].insert(i, good_keypoints)
#     return dataset_features
