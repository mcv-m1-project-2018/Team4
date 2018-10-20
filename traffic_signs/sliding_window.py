import cv2
import matplotlib.pyplot as plt
import numpy as np

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

def slide_window(image, image_mask, ratio=0.4,h_ratio=0.95, window_size = [30,60,90,150,200,250]):

    tmp=image_mask.copy()

    stepSize = 1
    bbox_list = []
    zeros_mask = np.zeros(image_mask.shape)

    for size in range(0, len(window_size)):

        w_width = window_size[size]
        w_height = window_size[size]
        total_count = w_width*w_height
        print(window_size[size])
        for x in range(0, image_mask.shape[1] - w_width , stepSize):
            for y in range(0, image_mask.shape[0] - w_height, stepSize):
                window = image_mask[y:y + w_width, x:x + w_height]
                pixels_count = cv2.countNonZero(window)
                #checking based on ratio
                if((pixels_count/total_count>ratio)and(pixels_count/total_count<h_ratio)):
                    print(pixels_count)
                    print(total_count)
                    print(x)
                    print(y)
                    # print(window)
                    # cv2.rectangle(tmp, (x, y), (x + w_width, y + w_height), (255, 255, 255), 1) # draw rectangle on image
                    zeros_mask[y:y+w_height,x:x+w_width] = 1
                    bbox_list.append([x,y, w_width, w_height])
                    
                    # cv2.namedWindow('winidows', cv2.WINDOW_NORMAL)
                    # cv2.imshow('winidows',tmp)
                    # cv2.waitKey()
                    print('found:')
    # plt.imshow(np.array(tmp).astype('uint8'))
    # cv2.waitKey()
    # Check if multiple windows detect same object

    # overlap_mask = zeros_mask[np.where(zeros_mask>1)]
    # print(overlap_mask)
    # 
    ret,thresh1 = cv2.threshold(zeros_mask,0,255,cv2.THRESH_BINARY)
    cv2.imshow('thres',thresh1)
    cv2.waitKey()
    print(zeros_mask.shape)
    print(thresh1.shape)
    print(zeros_mask.dtype)
    print(thresh1.dtype)
    # overlap_mask = overlap_mask.astype(int)
    im2, contours, hierarchy = cv2.findContours(thresh1.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(tmp, contours, -1, (255,255,255), 3)
    # cv2.imshow('img',tmp)
    # cv2.waitKey()
    bbox_list_nonoverlap = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bbox_list_nonoverlap.append([x,y,w,h])
        cv2.rectangle(tmp,(x,y),(x+w,y+h),(2555,255,255),3)
    cv2.imshow('imgg',tmp)
    cv2.waitKey()
    return(bbox_list_nonoverlap)

    
    # if(len(bbox_list)>0):
    # # Add final element to the bbox list equal to the first one
    #     bbox_list.append(bbox_list[0])

    # all_overlapping_idx = []
    # for bbox_idx in range(0, len(bbox_list)):
        # if(bbox_idx not in all_overlapping_idx):
    #         overlapping_inx = []
    #         for bbox_idx2 in range(0,len(bbox_list)):
    #             bbox_intersect = intersection(bbox_list[bbox_idx],bbox_list[bbox_idx2])
    #             if (bbox_intersect[2]*bbox_intersect[3]>ratio*total_count):
    #                 overlapping_inx.append(bbox_idx2)
    #                 all_overlapping_idx.append(bbox_idx2)
    #         if(len(overlapping_inx)>0):
    #             for idx_to_unite in range(0, len(overlapping_inx)):
    #                 print('uniting')
    #                 bbox_list[bbox_idx]=union(bbox_list[bbox_idx], bbox_list[idx_to_unite])
    #         # else:
    #         bbox_list_nonoverlap.append(bbox_list[bbox_idx])
    # for good_idx in range(0, len(bbox_list_nonoverlap)):
    #     cv2.rectangle(tmp,(bbox_list_nonoverlap[good_idx][0],bbox_list_nonoverlap[good_idx][1]),(bbox_list_nonoverlap[good_idx][0]+bbox_list_nonoverlap[good_idx][2],bbox_list_nonoverlap[good_idx][1]+bbox_list_nonoverlap[good_idx][3]) , (255, 255, 255), 6) # draw rectangle on image
    
    # cv2.imshow('windows',tmp)
    # cv2.waitKey()


