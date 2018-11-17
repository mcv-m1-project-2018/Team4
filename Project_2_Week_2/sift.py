import numpy as np
import cv2
from matplotlib import pyplot as plt

img2 = cv2.imread('BBDD_W4/ima_000001.jpg',0)          # queryImage
img1 = cv2.imread('query_devel_W4/ima_000000.jpg',0) # trainImage

# img2 = cv2.imread('BBDD_W4/ima_000014.jpg',0)          # queryImage
# img1 = cv2.imread('query_devel_W4/ima_000003.jpg',0) # trainImage

width = 1000
aspect_ratio1 = img1.shape[0]/img1.shape[1]
new_height1 = int(width*aspect_ratio1)
img1=cv2.resize(img1,(width, new_height1))
aspect_ratio2 = img2.shape[0]/img2.shape[1]
new_height2 = int(width*aspect_ratio2)
img2=cv2.resize(img2,(width, new_height2))

# img1 = cv2.imread('query_devel_W4/ima_000119.jpg',0) # trainImage

# img2 = cv2.imread('BBDD_W4/ima_000014.jpg',0)          # queryImage
# img1 = cv2.imread('query_devel_W4/ima_000001.jpg',0) # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SURF_create(hessianThreshold=3000,upright=True)

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

bf = cv2.BFMatcher(crossCheck=False)
# FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks=50)   # or pass empty dictionary
# bf = cv2.FlannBasedMatcher(index_params,search_params)


matches = bf.knnMatch(des2,des1, k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
print(len(good))

# BFMatcher with default params
matches = bf.knnMatch(des1,des2, k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
print(len(good))
# print(len(matches))
# # cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,img1,flags=4)
# img3 = cv2.drawMatches(img1,kp1,img2,kp2,img1, matches, flags=2)

plt.imshow(img3),plt.show()