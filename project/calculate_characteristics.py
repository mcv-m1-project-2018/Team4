import cv2
import os
from pathlib import Path
import numpy as np
# import matplotlib.pyplot as plt

IMG_EXTENSIONS = [
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.tif', '.tiff'
]


# Functions statement
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

def plot_std_dev(x, y, e, metric):
    plt.errorbar(x, y, e, linestyle='None', marker='o')
    plt.title(metric)
    plt.ylabel(metric)
    plt.xlabel('Class')

    plt.show()


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
    form_factor_sd = []
    filling_ratio_sd = []


    # Compute the final values for each class and fill the output with them
    for class_id in classes_array:
        split_by_classes = computed_values[(computed_values[:, 0] == class_id)]

        frequencies.append(len(split_by_classes))

        split_by_classes_float = np.array(split_by_classes[:, 1: 4], dtype=float)
        # means
        form_factor_avg.append(np.mean(split_by_classes_float[:, 1]))
        filling_ratio_avg.append(np.mean(split_by_classes_float[:, 2]))
        # max, min sizes
        max_size.append(np.amax(split_by_classes_float[:, 0]))
        min_size.append(np.amin(split_by_classes_float[:, 0]))
        #standard deviation
        form_factor_sd.append(np.std(split_by_classes_float[:, 1]))
        filling_ratio_sd.append(np.std(split_by_classes_float[:, 2]))



    # Split the whole dataset array into classes and put them into one big array
    for index in range(0, len(dataset)):
        for class_id in classes_array:
            if dataset[index][0] == class_id:
                dataset_grouped[classes_array.index(class_id)].append(dataset[index])

    output = {"Frequecie": frequencies, "Form factor": form_factor_avg, "Filling ratio": filling_ratio_avg,
              "Max size": max_size, "Mins size": min_size, "Form Factor Standard Deviation": form_factor_sd,
              "Filling ratio Standard Deviation": filling_ratio_sd}
    print(output)
    # plot_std_dev(classes_array, form_factor_avg, form_factor_sd, "Form Factor")
    # plot_std_dev(classes_array, filling_ratio_avg, filling_ratio_sd, "Filling ratio")

    return dataset_grouped, frequencies


