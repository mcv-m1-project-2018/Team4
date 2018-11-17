"""
Usage:
  cropRotate.py <museum_set_path> <query_set_path>
  cropRotate.py -h | --help
Options:
  <museum_set_path>
  <query_set_path>
"""
import sys
import os
from docopt import docopt
import cv2
import pickle
import time
import numpy as np
import math


def read_set(dataset_path):
    dataset = []
    set_names = []
    dataset_features = []
    for filename in sorted(os.listdir(dataset_path)):
        img = cv2.imread(os.path.join(dataset_path, filename))
        dataset.append(img)
        set_names.append(filename)
        # dataset_features.append(find_features(img,feature_type))

    return dataset, set_names


def find_if_close(cnt1,cnt2,im_width):
    #row1,row2 = cnt1.shape[0],cnt2.shape[0]
    row1 = cnt1[1] + cnt1[3]
    row2 = cnt2[1] + cnt2[3]
    right = []
    left = []
    if cnt1[0] > cnt2[0]:
        right = cnt1
        left = cnt2
    else:
        right = cnt2
        left = cnt1

    clos1= [left[0] + left[2],left[1] + left[3]]
    clos2 = [right[0], right[1] + right[3]]
    cv2.waitKey()

    if row2 >= row1-10:
        if row2 <= row1+10:
            dist = clos1[0] - clos2[0]
            if abs(dist) < 0.15*im_width:
                return True
            else:
                # print("else1")
                return False
        else:
            # print("else2")
            return False
    else:
        # print("else2")
        return False
    # for i in range(row1):
    #     for j in range(row2):
    #         dist = np.linalg.norm(cnt1[i]-cnt2[j])
    #         if abs(dist) < 0.15*im_width:
    #             return True
    #         elif i==row1-1 and j==row2-1:
    #             return False


def find_text(current_set):
    bboxes = []

    for n,im in enumerate(current_set):
        # if n!=184:
        #     continue
        print(n)
        h, w, _ = im.shape
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        retval = cv2.xphoto.createSimpleWB()
        im_balanced = retval.balanceWhite(im_gray)

        # im_balanced = cv2.GaussianBlur(im_balanced, (5, 5), 0)

        laplacian = cv2.Laplacian(im_balanced, cv2.CV_8U)
        #
        # cv2.namedWindow("lap", cv2.WINDOW_NORMAL)
        # cv2.imshow("lap", laplacian)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # adaptive = cv2.adaptiveThreshold(im_balanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
        # open_mask = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, (11, 11))

        _,im_gray_thresh_w = cv2.threshold(laplacian, 200, 255,cv2.THRESH_BINARY)
        _,im_gray_thresh_b = cv2.threshold(laplacian, 60, 255, cv2.THRESH_BINARY)
        #
        # im_gray_thresh_w = cv2.morphologyEx(im_gray_thresh_w, cv2.MORPH_OPEN, (3, 3))
        # im_gray_thresh_b = cv2.morphologyEx(im_gray_thresh_b, cv2.MORPH_OPEN, (3, 3))

        open_mask = im_gray_thresh_w | im_gray_thresh_b
        # open_mask = cv2.morphologyEx(open_mask,cv2.MORPH_CLOSE, (7,7))
        # open_mask = cv2.dilate(open_mask, (3, 3), iterations=2)
        # open_mask = cv2.erode(open_mask, (3,3), iterations=1)

        new_mask = np.zeros(open_mask.shape, dtype=np.uint8)

        """Contornos para rellenar las letras y asi poder eliminar los blobs pequeños de tamaño y que no se lleve las letras por delante"""
        im2, contours, hierarchy = cv2.findContours(open_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(new_mask, contours, -1, color=(255, 0, 0), thickness=cv2.FILLED)

        new_mask = cv2.erode(new_mask, (5,5), iterations=1)

        ret, labels, stats, centroid = cv2.connectedComponentsWithStats(new_mask, connectivity=4)
        sizes = stats[1:, -1]
        ret = ret - 1

        min_size = 150

        mask_remove = new_mask.copy()

        for i in range(0, ret):
            if sizes[i] <= min_size:
                mask_remove[labels == i + 1] = 0

            # if sizes[i] >= max_size:
            #     mask_removeSmall[labels == i + 1] = 0

        """Como en todas las imagenes el texto está o por encima de la mitad o por debajo, veo donde hay mas pixeles
        blancos y si es debajo (quiere decir que el texto está ahí) pues me cargo la parte de arriba que ahi solo va a
        haber basura y viceversa"""
        top_mask = np.zeros(im_gray.shape, dtype=np.uint8)
        top_mask[0:int(top_mask.shape[0]/2.5), :] = 255
        bottom_mask = np.zeros(im_gray.shape, dtype=np.uint8)
        bottom_mask[int(top_mask.shape[0]/1.5):top_mask.shape[0], :] = 255

        check_top = cv2.bitwise_and(mask_remove, top_mask)
        check_bottom = cv2.bitwise_and(mask_remove, bottom_mask)

        if cv2.countNonZero(check_top) <= cv2.countNonZero(check_bottom):
            mask_remove = check_bottom
        else:
            mask_remove = check_top

        im2, contours, hierarchy = cv2.findContours(mask_remove.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # mask_remove = cv2.dilate(mask_remove,(7,7),iterations=1)
        mask_remove = cv2.morphologyEx(mask_remove, cv2.MORPH_DILATE, (20,1),iterations=3)
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            if h/w>10 and h<15 and w<15:
                #print(h/w)
                mask_remove[y:y + h, x:x + w] = 0

                #cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # cv2.namedWindow("maskt", cv2.WINDOW_NORMAL)
        # cv2.imshow("maskt", top_mask)
        # cv2.namedWindow("maskb", cv2.WINDOW_NORMAL)
        # cv2.imshow("maskb", bottom_mask)
        # cv2.waitKey()

        """CLOSING TOCHO PARA UNIR LETRAS EN HORIZONTAL"""

        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (70, 1))
        mask_remove = cv2.morphologyEx(mask_remove, cv2.MORPH_CLOSE, rect_kernel)

        """FIND CONTOURS, BUSCAR EL MAS GRANDES"""

        im2, contours, hierarchy = cv2.findContours(mask_remove.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            #print(h / w)
            if h/w > 3.5:
                mask_remove[y:y + h, x:x + w] = 0

        # cv2.namedWindow("mask_dil", cv2.WINDOW_NORMAL)
        # cv2.imshow("mask_dil", mask_remove)
        # cv2.waitKey()

        """Solucionar problema Can - Framis"""
        im2, contours, hierarchy = cv2.findContours(mask_remove.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        LENGTH = len(contours)
        status = np.zeros((LENGTH, 1))

        for i, cnt1 in enumerate(contours):
            x = i
            if i != LENGTH - 1:
                for j, cnt2 in enumerate(contours[i + 1:]):
                    x = x + 1
                    dist = find_if_close(cv2.boundingRect(cnt1), cv2.boundingRect(cnt2), im.shape[0])
                    if dist:
                        val = min(status[i], status[x])
                        status[x] = status[i] = val
                    else:
                        if status[x] == status[i]:
                            status[x] = i + 1
        unified = []
        maximum = int(status.max()) + 1
        for i in range(maximum):
            pos = np.where(status == i)[0]
            if pos.size != 0:
                cont = np.vstack(contours[i] for i in pos)
                hull = cv2.convexHull(cont)
                unified.append(hull)



        if len(unified) != 0:
            c = max(unified, key=cv2.contourArea)

            x, y, w, h = cv2.boundingRect(c)
            # draw the book contour (in green)
            if h/w < 1:
                x= x-15
                y = y-10
                w = w+30
                h=h+20
                #print("entra")
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
            bb = (x, y, x+w, y+h)
            bboxes.append(bb)



        # cv2.namedWindow("mask_fina", cv2.WINDOW_NORMAL)
        # cv2.imshow("mask_fina", mask_remove)
        # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        # cv2.imshow("img", im)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

    return bboxes


def crop_rotate_manu(im):
    # output_bboxes = []
    # output_crops = []

    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    retval = cv2.xphoto.createSimpleWB()
    im_balanced = retval.balanceWhite(im_gray)
    im_balanced = cv2.medianBlur(im_balanced,5)

    canny = cv2.Canny(im_balanced,10, 240, apertureSize=3, L2gradient=False)

    _, canny_thresh = cv2.threshold(canny, 10, 255, cv2.THRESH_BINARY)

    thresh_hor = cv2.morphologyEx(canny_thresh, cv2.MORPH_DILATE, (30,3),iterations=10)
    thresh_ver = cv2.morphologyEx(thresh_hor, cv2.MORPH_DILATE, (3,30),iterations=10)

    im2, contours, hierarchy = cv2.findContours(thresh_ver.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(thresh_ver, contours, -1, color=(255, 0, 0), thickness=cv2.FILLED)

    ret, labels, stats, centroid = cv2.connectedComponentsWithStats(thresh_ver, connectivity=4)
    sizes = stats[1:, -1]
    ret = ret - 1

    min_size = 0.001*(im_gray.shape[0]*im_gray.shape[1])



    mask_remove = thresh_ver.copy()

    for i in range(0, ret):
        if sizes[i] <= min_size:
            mask_remove[labels == i + 1] = 0

    im2, contours, hierarchy = cv2.findContours(mask_remove.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maximum=0
    idx_maximum = 0
    bboxes = []
    rects = []
    bbox = []
    for idx, cnt in enumerate(contours):
        rect_final_box = cv2.minAreaRect(cnt)
        """( center (x,y), (width, height), angle of rotation )"""
        rects.append(rect_final_box)
        area = rect_final_box[1][0]*rect_final_box[1][1]
        box = cv2.boxPoints(rect_final_box) # cv2.boxPoints(rect) for OpenCV 3.x
        box = np.int0(box)
        bboxes.append(box)
        if area>maximum:
            maximum = area
            idx_maximum = idx

    angle = int(rects[idx_maximum][2])
    final_bbox = bboxes[idx_maximum]

    """Top left, top_right, bottom_right, bottom_left----->(x,y)"""
    rect_final_box = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = final_bbox.sum(axis=1)
    rect_final_box[0] = final_bbox[np.argmin(s)]
    rect_final_box[2] = final_bbox[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(final_bbox, axis=1)
    rect_final_box[1] = final_bbox[np.argmin(diff)]
    rect_final_box[3] = final_bbox[np.argmax(diff)]


    """
    if the long side is in the left of the bottom Point, 
    the angle value is between long side and Y+ axis, but if the long side is in the right of the bottom Point,
    the angle value is between long side and X+ axis
    """

    if rects[idx_maximum][1][0] > rects[idx_maximum][1][1]:
        angle = int(90 - angle)

    else:
        angle = int(-angle)

    forHough = np.zeros(im_gray.shape, dtype=np.uint8)
    cv2.drawContours(forHough, [bboxes[idx_maximum]],0,(255,0,0),thickness=4)
    toShow1 = im.copy()
    cv2.drawContours(toShow1, [bboxes[idx_maximum]],0,(0,0,255),thickness=2)

    # rect_final_box_rotated = rect_final_box

    sigma = 10
    img_rotate = im_gray.copy()
    forHough_rotate = forHough.copy()
    if angle != 0 and angle != 90:
        if angle < 90:
            if angle >= 45:
                angle = 90-angle
                img_rotate = rotateImage(im_gray, angle)
                forHough_rotate = rotateImage(forHough, angle)
            else:
                angle = -angle
                img_rotate = rotateImage(im_gray, angle)
                forHough_rotate = rotateImage(forHough, angle)
        elif angle > 90:
            if angle >= 135:
                angle = 180-angle
                img_rotate = rotateImage(im_gray, angle)
                forHough_rotate = rotateImage(forHough, angle)
            else:
                angle = -(angle - 90)
                img_rotate = rotateImage(im_gray, angle)
                forHough_rotate = rotateImage(forHough, angle)

    fill_contours = np.zeros(im_gray.shape, dtype=np.uint8)
    im2, contours, hierarchy = cv2.findContours(forHough_rotate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for cnt in contours:
        rect = cv2.boundingRect(cnt)
        rects.append(rect)
    crop = img_rotate[rects[0][1]:rects[0][3]+rects[0][1], rects[0][0]:rects[0][0]+rects[0][2]]
    cv2.drawContours(fill_contours, contours, -1, color=(255, 0, 0), thickness=cv2.FILLED)


    bbox = [(rect_final_box[0][0], rect_final_box[0][1]), (rect_final_box[2][0], rect_final_box[2][1]),
            (rect_final_box[2][0], rect_final_box[2][1]), (rect_final_box[3][0], rect_final_box[3][1])]
    results = [angle, bbox]
    # output_bboxes.append(results)
    # output_crops.append(crop)

    return results, crop


def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


def bbox_iou(bboxA, bboxB):
    # compute the intersection over union of two bboxes

    # Format of the bboxes is [tly, tlx, bry, brx, ...], where tl and br
    # indicate top-left and bottom-right corners of the bbox respectively.

    # determine the coordinates of the intersection rectangle
    # print(bboxA)
    # print(bboxB)
    xA = max(bboxA[1], bboxB[1])
    yA = max(bboxA[0], bboxB[0])
    xB = min(bboxA[3], bboxB[3])
    yB = min(bboxA[2], bboxB[2])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both bboxes
    bboxAArea = (bboxA[2] - bboxA[0] + 1) * (bboxA[3] - bboxA[1] + 1)
    bboxBArea = (bboxB[2] - bboxB[0] + 1) * (bboxB[3] - bboxB[1] + 1)

    iou = interArea / (0.000000001 + float(bboxAArea + bboxBArea - interArea))

    # return the intersection over union value
    return iou


def performance_accumulation_window(detections, annotations):
    """
    performance_accumulation_window()

    Function to compute different performance indicators (True Positive,
    False Positive, False Negative) at the object level.

    Objects are defined by means of rectangular windows circumscribing them.
    Window format is [ struct(x,y,w,h)  struct(x,y,w,h)  ... ] in both
    detections and annotations.

    An object is considered to be detected correctly if detection and annotation
    windows overlap by more of 50%

       function [TP,FN,FP] = PerformanceAccumulationWindow(detections, annotations)

       Parameter name      Value
       --------------      -----
       'detections'        List of windows marking the candidate detections
       'annotations'       List of windows with the ground truth positions of the objects

    The function returns the number of True Positive (TP), False Positive (FP),
    False Negative (FN) objects
    """

    detections_used = np.zeros(len(detections))
    annotations_used = np.zeros(len(annotations))
    TP = 0
    for ii in range(len(annotations)):
        for jj in range(len(detections)):
            if (detections_used[jj] == 0) & (bbox_iou(annotations[ii], detections[jj]) > 0.5):
                TP = TP + 1
                detections_used[jj] = 1
                annotations_used[ii] = 1

    FN = np.sum(annotations_used == 0)
    FP = np.sum(detections_used == 0)

    return [TP, FN, FP]


def performance_evaluation_window(TP, FN, FP):
    """
    performance_evaluation_window()

    Function to compute different performance indicators (Precision, accuracy,
    sensitivity/recall) at the object level

    [precision, sensitivity, accuracy] = PerformanceEvaluationPixel(TP, FN, FP)

       Parameter name      Value
       --------------      -----
       'TP'                Number of True  Positive objects
       'FN'                Number of False Negative objects
       'FP'                Number of False Positive objects

    The function returns the precision, accuracy and sensitivity
    """

    precision = float(TP) / float(TP + FP);  # Q: What if i do not have TN?
    sensitivity = float(TP) / float(TP + FN)
    accuracy = float(TP) / float(TP + FN + FP)
    F1 = 0

    if (precision + sensitivity) != 0:
        F1 = 2 * ((precision * sensitivity) / (precision + sensitivity))

    return [precision, sensitivity, accuracy, F1]


def load_annotations(pkl_file):
    # Annotations are stored in text files containing
    # the coordinates of the corners (top-left and bottom-right) of
    # the bounding box plus an alfanumeric code indicating the signal type:
    # tly, tlx, bry,brx, code
    annotations = []
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    for query in data:
        annotations.append(query)
    return annotations


def save_results(directory, task, result):
    #  TODO
    directory = directory
    print(directory)
    base = task
    out_list_name = '{}/{}.pkl'.format(directory, base)
    if not os.path.exists(directory):
        os.makedirs(directory)

    pickle.dump(result, open(out_list_name, "wb"))


if __name__ == "__main__":
    if len(sys.argv) > 1:

        args = docopt(__doc__)
        museum_path = args['<museum_set_path>']
        query_path = args['<query_set_path>']
        # museum_dataset, museum_dataset_names = read_set(museum_path)
        query_dataset, query_dataset_names = read_set(query_path)

        annotations = load_annotations("results/text.pkl")

        # gt = load_annotations("./w5_text_bbox_list.pkl")
        # #
        # #
        # # text_bboxes = find_text(museum_dataset)
        # # directory = "results/"
        # # save_results(directory, "text", text_bboxes)
        # # groundtruth_bbox = text_bboxes
        # #
        # #
        # TP, FN, FP = performance_accumulation_window(annotations, gt)
        # print(TP)
        # print(FN)
        # print(FP)
        # precision, sensitivity, accuracy, F1 = performance_evaluation_window(TP, FN, FP)
        # print(precision)
        # print(F1)

        """MIRA MI CODIGO!!!! LAS HOUGH LINES FUNCIONAN CASI EN TODAS BIEN MENOS EN LA PRIMERA Y EN OTRA
        QUE APARECEN COMO ROTAS. SI TAL PUEDES SEGUIR DESDE AHI CON LO DEL ANGULO AHORA. En algunas la sombra aparece
        como linea pero yo creo que para la hora de girar el cuadro tampoco pasa nada, además el angulo se puede
        calcular perfectamente que siempre es paralela a uno de los lados y no influye"""
        # rotated_images = []

        # directory = "results/"
        # save_results(directory, "result", rotated_images)
        """"###########################################################################"""

        # for line in open('output.txt').read().splitlines():
        #
        #     # annot_values = line.split(",")
        #     # print(annot_values)
        #     annot_values = [0, 0, 0 ,0]
        #     annot_values_aux = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        #     annot_values_aux = [x.strip() for x in annot_values_aux]
        #     annot_values[0] = int(annot_values_aux[1])
        #     annot_values[1] = int(annot_values_aux[0])
        #     annot_values[2] = int(annot_values_aux[3])
        #     annot_values[3] = int(annot_values_aux[2])
        #     # for ii in range(4):
        #     #     annot_values[ii] = int(annot_values[ii])
        #     annot_values = tuple(annot_values)
        #     groundtruth_bbox.append(annot_values)

        # with open('output.txt', 'w') as f:
        #     for item in text_bboxes:
        #         f.write("{}\n".format(item))
