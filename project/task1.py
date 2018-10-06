import cv2
import os
from pathlib import Path
import numpy as np



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
            # cv2.imshow('Final mask', final_mask)
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
                dataset.append([bounding_box[4],img,mask,bounding_boxes])


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

    return dataset_grouped, dataset
