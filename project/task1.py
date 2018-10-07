import cv2
import os
from pathlib import Path
import numpy as np
import argparse
<<<<<<< HEAD
# from matplotlib import pyplot as plt

=======
>>>>>>> 303e7c32116274a68ac2af2d2db96f9d83170295


IMG_EXTENSIONS = [
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.tif', '.tiff'
]

#Functions statement

def is_image_file(filename):
    return any(filename.lower().endswith(extension)
               for extension in IMG_EXTENSIONS)


def read_mask(dataset_path, filename):
    return cv2.imread(str(Path(dataset_path) / Path("mask") / Path("mask." + filename.stem + ".png")))


def read_gt(dataset_path, filename):
    bounding_boxes = []
    with open(str(dataset_path / "gt" / Path("gt." + filename.stem + ".txt")), 'r') as file:
        bounding_boxes = [line.split() for line in file]
    return bounding_boxes


# Calculate the size by counting the amount of white pixels
def calc_size(crop):
    return cv2.countNonZero(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))



def calculate_characteristics(dataset_path, classes_array=["A", "B", "C", "D", "E", "F"]):
    """
    Determine the characteristics of the signals in the training set: max and min
    size, form factor, filling ratio of each type of signal, frequency of appearance (using
    text annotations and ground-truth masks).

    :return:
        output: array of arrays. [frequencies, form_factor_avg, filling_ratio_avg, max_size, min_size]
                Each element of the output array has values splitted by classes in the following order:
                 ["A", "B", "C", "D", "E", "F"]

    """
    computed_values = []
    dataset = []

    # initiating dataset grouped in classes, containing separate datasets for each class
    datasetA =[]
    datasetB =[]
    datasetC =[]
    datasetD =[]
    datasetE =[]
    datasetF =[]
    dataset_grouped = [datasetA,datasetB,datasetC,datasetD,datasetE,datasetF]
    
    # This 'for' iterates within the train dataset

    for filename in os.listdir(dataset_path):

        if is_image_file(filename):
            # Read the image, mask and gt corresponding to certain filename
            img = cv2.imread(str(Path(dataset_path) / Path(filename)))
            img_out = np.zeros(img.shape, dtype=img.dtype)
            clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))

            #uncomment to apply equilization of histograms
            # eqR = clahe.apply(img[:,:,0])
            # eqG = clahe.apply(img[:,:,1])
            # eqB = clahe.apply(img[:,:,2])
            # img_out[:,:,0] = eqR
            # img_out[:,:,1] = eqG
            # img_out[:,:,2] = eqB
            # img = img_out
            mask = read_mask(dataset_path, Path(filename))
            bounding_boxes = read_gt(Path(dataset_path), Path(filename))

            # Compute binary mask
            ret, bin_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

            # Uncomment to see the patch from the image corresponding to the no zero values in the mask
            # final_mask = cv2.bitwise_and(img, bin_mask)
            # cv2.imshow('Final mask', bin_mask)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # For each signal in images, compute size, form_factor and filling ratio

            for bounding_box in bounding_boxes:
                coordinates = list(map(float, bounding_box[0:4]))
                crop = bin_mask[int(coordinates[0]):int(coordinates[2]), int(coordinates[1]):int(coordinates[3])]

                # Uncomment to see the crop used to compute values
                # cv2.imshow('crop', crop)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                size = calc_size(crop)
                form_factor = crop.shape[1]/crop.shape[0]
                filling_ratio = size/(crop.shape[1]*crop.shape[0])

                computed_values.append([bounding_box[4], size, form_factor, filling_ratio])
                dataset.append([bounding_box[4],img,mask,coordinates])


    # computed_values has the following structure: [class_id, size, form _factor, filling_ratio]

    computed_values = np.asarray(computed_values)
    frequencies = []
    form_factor_avg = []
    filling_ratio_avg = []
    max_size = []
    min_size = []

    # Compute the final values for each class and fill the output with them
    for class_id in classes_array:
        split_by_classes = computed_values[(computed_values[:, 0] == class_id)]

        frequencies.append(len(split_by_classes))

        split_by_classes_float = np.array(split_by_classes[:, 1: 4], dtype=float)
        form_factor_avg.append(np.mean(split_by_classes_float[:, 1]))
        filling_ratio_avg.append(np.mean(split_by_classes_float[:, 2]))

        max_size.append(np.amax(split_by_classes_float[:, 0]))
        min_size.append(np.amin(split_by_classes_float[:, 0]))

    # Split the whole dataset array into classes and put them into one big array
    for index in range(0, len(dataset)):
        for class_id in classes_array:
            if (dataset[index][0] == class_id):
                dataset_grouped[classes_array.index(class_id)].append(dataset[index])

    output = [frequencies, form_factor_avg, filling_ratio_avg, max_size, min_size]
    print(output)

    return dataset_grouped, frequencies

def task2 (dataset_grouped, frequencies):
    """
    This function splits dataset into training and validation dataset in proportion 7:3
    proportionally with regard to image shapes and colors (i.e. classes).

    :return:
        output: dataset_train and dataset_valid, Python lists that contain 6 rows with columns of images, masks and data
    """

    train_set_ratio = 0.7
    frequencies_train = np.asarray(frequencies) * train_set_ratio

    # initiate dataset train, containing separate datasets for each class
    dataset_trainA =[]
    dataset_trainB =[]
    dataset_trainC =[]
    dataset_trainD =[]
    dataset_trainE =[]
    dataset_trainF =[]
    dataset_train = [dataset_trainA,dataset_trainB,dataset_trainC,dataset_trainD,dataset_trainE,dataset_trainF]
    
    # initiate dataset valid, containing separate datasets for each class
    dataset_validA =[]
    dataset_validB =[]
    dataset_validC =[]
    dataset_validD =[]
    dataset_validE =[]
    dataset_validF =[]
    dataset_valid = [dataset_validA,dataset_validB,dataset_validC,dataset_validD,dataset_validE,dataset_validF]

    # Here we split the datasets for each class in proportions 7:3
    for class_id in range(0, len(dataset_grouped)):
        for index in range (0, len(dataset_grouped[class_id])):
            if (index < frequencies_train[class_id]):
                dataset_train[class_id].append(dataset_grouped[class_id][index])
            else:
                dataset_valid[class_id].append(dataset_grouped[class_id][index])
    return dataset_train, dataset_valid

    # uncomment to print number of signals for train and validation from 
    # class A, and to print frequency of A class signals in the whole dataset
    # print(len(dataset_train[0]))
    # print(len(dataset_valid[0]))
    # print(frequencies[0])ç

<<<<<<< HEAD
def save_dataset(dataset, directory):
    index = 0
    for class_id in range(len(dataset_grouped)):
        for element in dataset_grouped[class_id]:
            filename = directory + "/00." + str(index) + ".jpg"
            filename_mask = directory + "/mask/mask.00." + str(index) + ".png"
            cv2.imwrite(filename, element[1])
            mask = cv2.cvtColor(element[2], cv2.COLOR_BGR2GRAY)
            cv2.imwrite(filename_mask, mask)
            index+=1
=======

def task_3(dataset_train, dataset_valid):
    output_final= []
    benchmark = []

    for class_id in range(6):
        data = dataset_valid[class_id][:]
        for idx, n in enumerate(data):
            img = n[1]
            gt_mask = n[2]
            gt_bounding_boxes = n[3]
            # cv2.imshow(' mask', gt_mask)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            maskHSV_red1 = cv2.inRange(imgHSV, np.array([0, 70, 0]), np.array([10, 255, 255]))
            maskHSV_red2 = cv2.inRange(imgHSV, np.array([160, 70, 0]), np.array([179, 255, 255]))

            maskHSV_blue = cv2.inRange(imgHSV, np.array([100, 70, 0]), np.array([140, 255, 255]))

            # maskHSV_blue[maskHSV_blue == 255] = 127

            maskHSV_red = cv2.bitwise_or(maskHSV_red1, maskHSV_red2)
            mask_final = cv2.bitwise_or(maskHSV_blue, maskHSV_red)
            kernel = np.ones((3, 3), np.uint8)
            erosion = cv2.erode(mask_final, kernel, iterations=3)
            dilated = cv2.dilate(erosion, kernel, iterations=2)

            # Connected components
            ret, labels, stats, centroid = cv2.connectedComponentsWithStats(dilated, connectivity=4)
            sizes = stats[1:, -1]
            ret = ret-1

            min_size = 200

            img_noSmall = labels.copy()

            for i in range(0, ret):
                if sizes[i] <= min_size:
                    img_noSmall[labels == i + 1] = 0

            label_hue = np.uint8(179 * img_noSmall / np.max(img_noSmall))
            blank_ch = 255 * np.ones_like(label_hue)
            labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

            # cvt to BGR for display
            labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

            # set bg label to black
            labeled_img[label_hue == 0] = 0

            # cv2.imshow('labeled.png', labeled_img)
            # cv2.waitKey()

            """finish connected components"""
            output = [img, gt_mask, labeled_img, gt_bounding_boxes]

            output_final.append(output)

            gt_mask =cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)

            [pixelTP, pixelFP, pixelFN, pixelTN] = performance_accumulation_pixel(dilated, gt_mask)
            [pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity] = performance_evaluation_pixel(pixelTP, pixelFP, pixelFN, pixelTN)
            #print(pixel_precision)

            benchmark.append([pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity])

            # cv2.imshow('Final mask',labeled_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
    benchmark = np.asarray(benchmark)
    print((benchmark[:][0]).mean())

    return output_final


def performance_accumulation_pixel(pixel_candidates, pixel_annotation):
    """
    performance_accumulation_pixel()

    Function to compute different performance indicators
    (True Positive, False Positive, False Negative, True Negative)
    at the pixel level

    [pixelTP, pixelFP, pixelFN, pixelTN] = performance_accumulation_pixel(pixel_candidates, pixel_annotation)

    Parameter name      Value
    --------------      -----
    'pixel_candidates'   Binary image marking the detected areas
    'pixel_annotation'   Binary image containing ground truth

    The function returns the number of True Positive (pixelTP), False Positive (pixelFP),
    False Negative (pixelFN) and True Negative (pixelTN) pixels in the image pixel_candidates
    """

    pixel_candidates = np.uint64(pixel_candidates > 0)
    pixel_annotation = np.uint64(pixel_annotation > 0)

    pixelTP = np.sum(pixel_candidates & pixel_annotation)
    pixelFP = np.sum(pixel_candidates & (pixel_annotation == 0))
    pixelFN = np.sum((pixel_candidates == 0) & pixel_annotation)
    pixelTN = np.sum((pixel_candidates == 0) & (pixel_annotation == 0))

    return [pixelTP, pixelFP, pixelFN, pixelTN]


def performance_evaluation_pixel(pixelTP, pixelFP, pixelFN, pixelTN):
    """
    performance_evaluation_pixel()

    Function to compute different performance indicators (Precision, accuracy,
    specificity, sensitivity) at the pixel level

    [pixelPrecision, pixelAccuracy, pixelSpecificity, pixelSensitivity] = PerformanceEvaluationPixel(pixelTP, pixelFP, pixelFN, pixelTN)

       Parameter name      Value
       --------------      -----
       'pixelTP'           Number of True  Positive pixels
       'pixelFP'           Number of False Positive pixels
       'pixelFN'           Number of False Negative pixels
       'pixelTN'           Number of True  Negative pixels

    The function returns the precision, accuracy, specificity and sensitivity
    """
    if (pixelTP + pixelFP) == 0 :
        pixel_precision = 0
    else:
        pixel_precision = float(pixelTP) / float(pixelTP + pixelFP)

    if (pixelTP + pixelFP + pixelFN + pixelTN) == 0:
        pixel_accuracy = 0
    else:
        pixel_accuracy = float(pixelTP + pixelTN) / float(pixelTP + pixelFP + pixelFN + pixelTN)

    if(pixelTN + pixelFP) == 0:
        pixel_specificity = 0
    else:
        pixel_specificity = float(pixelTN) / float(pixelTN + pixelFP)

    if (pixelTP + pixelFN) == 0:
        pixel_sensitivity =0
    else:
        pixel_sensitivity = float(pixelTP) / float(pixelTP + pixelFN)
>>>>>>> 303e7c32116274a68ac2af2d2db96f9d83170295

    return [pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity]

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-dp", "--dataset_path", required=True,
                    help="Dataset path, it should be at the same level as task1.py")

    args = vars(ap.parse_args())

    file_path = Path(__file__).parent.absolute()
    dataset_path = str(file_path / Path(args["dataset_path"]))
    # executing tasks:
    dataset_grouped, frequencies = calculate_characteristics(dataset_path)
    dataset_train, dataset_valid = task2(dataset_grouped, frequencies)
    save_dataset(dataset_valid, "valid")
    # compute_color_spaces_avg(dataset_train)
    task_3(dataset_train, dataset_valid)
    # traffic_sign_detection()

