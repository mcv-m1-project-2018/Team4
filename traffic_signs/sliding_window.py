import cv2
import numpy as np
import connected_labels_pixel_cand

def union(a,b):
  x = min(a[0], b[0])
  y = min(a[1], b[1])
  w = max(a[0]+a[2], b[0]+b[2]) - x
  h = max(a[1]+a[3], b[1]+b[3]) - y
  return (x, y, w, h)

def intersection(a,b):
  x = max(a[0], b[0])
  y = max(a[1], b[1])
  w = min(a[0]+a[2], b[0]+b[2]) - x
  h = min(a[1]+a[3], b[1]+b[3]) - y
  if w<0 or h<0: return (0,0,0,0) # or (0,0,0,0) ?
  return (x, y, w, h)

def slide_window(image, image_mask, ratio=0.55,h_ratio=0.7, window_size = [30,60,150,200]):
    image_mask = connected_labels_pixel_cand.connected_labels_pixel_cand(image, image_mask)
    tmp=image.copy()

    stepSize = 5
    bbox_list = []
    zeros_mask = np.zeros(image_mask.shape)

    for size in range(0, len(window_size)):

        w_width = window_size[size]
        w_height = window_size[size]
        total_count = w_width*w_height
        # print(window_size[size])
        for x in range(0, image_mask.shape[1] - w_width , stepSize):
            for y in range(0, image_mask.shape[0] - w_height, stepSize):
                window = image_mask[y:y + w_width, x:x + w_height]
                pixels_count = cv2.countNonZero(window)
                #checking based on ratio
                if((pixels_count/total_count>ratio)and(pixels_count/total_count<h_ratio)):
                    # print('found:')
                    # print(pixels_count)
                    # print(total_count)
                    # print(x)
                    # print(y)
                    # cv2.rectangle(tmp, (x, y), (x + w_width, y + w_height), (255, 255, 255), 1) # draw rectangle on image
                    zeros_mask[y:y+w_height,x:x+w_width] = 1
                    bbox_list.append([x,y, w_width, w_height])
                    

    ret,thresh1 = cv2.threshold(zeros_mask,0,255,cv2.THRESH_BINARY)
    # cv2.imshow('thres',thresh1)
    # cv2.waitKey()
    # print(zeros_mask.shape)
    # print(thresh1.shape)
    # print(zeros_mask.dtype)
    # print(thresh1.dtype)
    im2, contours, hierarchy = cv2.findContours(thresh1.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    bbox_list_nonoverlap = []
    form_factor_array = [1.064, 1.024, 0.950, 0.975, 0.943, 0.827]
    form_factor_sd_array = [0.074, 0.144, 0.107, 0.085, 0.104, 0.177]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bbox_list_nonoverlap.append([y,x,y+h,x+w])

        cv2.rectangle(tmp,(x,y),(x+w,y+h),(255,255,255),3)
    cv2.imshow('imgg',tmp)
    cv2.waitKey(100)
    return(bbox_list_nonoverlap)
