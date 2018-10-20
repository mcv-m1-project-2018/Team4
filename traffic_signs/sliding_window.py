import cv2
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

def slide_window(image, image_mask, ratio=0.4, window_size = [50,50]):

    tmp=image_mask.copy()
    stepSize = 1
    w_width = window_size[0]
    w_height = window_size[1]
    total_count = w_width*w_height
    bbox_list = []

    for x in range(0, image_mask.shape[1] - w_width , stepSize):
        for y in range(0, image_mask.shape[0] - w_height, stepSize):
            window = image_mask[y:y + w_width, x:x + w_height]
            pixels_count = cv2.countNonZero(window)
            #checking based on ratio
            if(pixels_count/total_count>ratio):
                print(pixels_count)
                print(total_count)
                print(x)
                print(y)
                print(window)
                cv2.rectangle(tmp, (x, y), (x + w_width, y + w_height), (255, 255, 255), 1) # draw rectangle on image
                bbox_list.append([x,y, w_width, w_height])
                # cv2.namedWindow('winidows', cv2.WINDOW_NORMAL)
                # cv2.imshow('winidows',tmp)
                # cv2.waitKey()
                print('found:')
    # plt.imshow(np.array(tmp).astype('uint8'))
    # cv2.waitKey()
    # Check if multiple windows detect same object
    bbox_list_nonoverlap = []
    if(len(bbox_list)>0):
    # Add final element to the bbox list equal to the first one
        bbox_list.append(bbox_list[0])

    for bbox_idx in range(0, len(bbox_list)-1):
        bbox_intersect = intersection(bbox_list[bbox_idx],bbox_list[bbox_idx+1])
        bbox_sum = union(bbox_list[bbox_idx],bbox_list[bbox_idx+1])
        # If there's overlapping, I sum the bboxes and process them iteratively.
        # Only when the next bbox is not overlapping I'm exporting the current bbox.
        if (bbox_intersect[2]*bbox_intersect[3]>ratio*total_count):
            bbox_list[bbox_idx+1] = bbox_sum
            print('overlapped')
        else:
            bbox_list_nonoverlap.append(bbox_list[bbox_idx])
            cv2.rectangle(tmp,(bbox_list[bbox_idx][0],bbox_list[bbox_idx][1]),(bbox_list[bbox_idx][0]+bbox_list[bbox_idx][2],bbox_list[bbox_idx][1]+bbox_list[bbox_idx][3]) , (0, 255, 0), 2) # draw rectangle on image
    cv2.imshow('windows',tmp)
    cv2.waitKey()


    return(bbox_list_nonoverlap)
