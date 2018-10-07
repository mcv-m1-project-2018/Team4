"""
To execute this script you should do the following:

python task1.py --dataset_path train

Taking into account that dir train must be at the same level as task1.py
"""

import argparse
import cv2
import os
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

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


def task_1(dataset_path):
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
    for class_id in ["A", "B", "C", "D", "E", "F"]:
        split_by_classes = computed_values[(computed_values[:, 0] == class_id)]

        frequencies.append(len(split_by_classes))

        split_by_classes_float = np.array(split_by_classes[:, 1: 4], dtype=float)
        form_factor_avg.append(np.mean(split_by_classes_float[:, 1]))
        filling_ratio_avg.append(np.mean(split_by_classes_float[:, 2]))

        max_size.append(np.amax(split_by_classes_float[:, 0]))
        min_size.append(np.amin(split_by_classes_float[:, 0]))

    # Split the whole dataset array into classes and put them into one big array
    for index in range(0, len(dataset)):
        for class_id in ["A", "B", "C", "D", "E", "F"]:
            if (dataset[index][0] == class_id):
                dataset_grouped[["A", "B", "C", "D", "E", "F"].index(class_id)].append(dataset[index])

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


def task_3(dataset_train, dataset_valid):

    for class_id in range(6):
        data = dataset_valid[class_id][:]
        for n in data:
            img = n[1]
            gt_mask = n[2]
            gt_bounding_boxes = n[3]
            cv2.imshow(' mask', gt_mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            maskHSV_red1 = cv2.inRange(imgHSV, np.array([0, 70, 0]), np.array([10, 255, 255]))
            maskHSV_red2 = cv2.inRange(imgHSV, np.array([160, 70, 0]), np.array([179, 255, 255]))

            maskHSV_blue = cv2.inRange(imgHSV, np.array([100, 70, 0]), np.array([140, 255, 255]))

            maskHSV_blue[maskHSV_blue == 255] = 127

            maskHSV_red = cv2.bitwise_or(maskHSV_red1, maskHSV_red2)
            mask_final = cv2.bitwise_or(maskHSV_blue, maskHSV_red)
            kernel = np.ones((3, 3), np.uint8)
            erosion = cv2.erode(mask_final, kernel, iterations=2)
            dilated = cv2.dilate(erosion, kernel, iterations=1)

            cv2.imshow('Final mask', dilated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-dp", "--dataset_path", required=True,
                    help="Dataset path, it should be at the same level as task1.py")

    args = vars(ap.parse_args())

    file_path = Path(__file__).parent.absolute()
    dataset_path = str(file_path / Path(args["dataset_path"]))
    # executing tasks:
    dataset_grouped, frequencies = task_1(dataset_path)
    dataset_train, dataset_valid = task2(dataset_grouped, frequencies)
    # compute_color_spaces_avg(dataset_train)
    task_3(dataset_train, dataset_valid)
