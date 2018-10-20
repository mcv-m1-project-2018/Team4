import numpy as np
import cv2
from sklearn.model_selection import train_test_split

def dataset_split(dataset_grouped, test_set_ratio=0.3):
    """
    This function splits dataset into training and validation dataset in proportion 7:3
    proportionally with regard to image shapes and colors (i.e. classes).

    :return:
        output: dataset_train and dataset_valid, Python lists that contain 6 rows with columns of images, masks and data
    """
    # initiate dataset train, containing separate datasets for each class
    dataset_trainA =[]
    dataset_trainB =[]
    dataset_trainC =[]
    dataset_trainD =[]
    dataset_trainE =[]
    dataset_trainF =[]
    dataset_train = [dataset_trainA, dataset_trainB, dataset_trainC, dataset_trainD, dataset_trainE, dataset_trainF]
    
    # initiate dataset valid, containing separate datasets for each class
    dataset_validA =[]
    dataset_validB =[]
    dataset_validC =[]
    dataset_validD =[]
    dataset_validE =[]
    dataset_validF =[]
    dataset_valid = [dataset_validA, dataset_validB, dataset_validC, dataset_validD, dataset_validE, dataset_validF] 

    # Here we split the datasets for each class in requested proportions
    for class_id in range(0, len(dataset_grouped)):
        dataset_train[class_id], dataset_valid[class_id] = train_test_split(dataset_grouped[class_id], test_size=test_set_ratio, random_state=42)
    
    return dataset_train, dataset_valid

def save_dataset(dataset_grouped, directory):
    index = 0
    for class_id in range(len(dataset_grouped)):
        for element in dataset_grouped[class_id]:
            filename = directory + "/00." + str(index) + ".jpg"
            filename_mask = directory + "/mask/mask.00." + str(index) + ".png"
            filename_gt = directory + "/gt/gt.00." + str(index) + ".txt"
            cv2.imwrite(filename, element[1])
            mask = cv2.cvtColor(element[2], cv2.COLOR_BGR2GRAY)
            cv2.imwrite(filename_mask, mask)
            # print(element[3])
            np.savetxt(filename_gt,element[3], delimiter=' ', newline=' ', fmt='%f')
            index+=1

    # uncomment to print number of signals for train and validation from 
    # class A, and to print frequency of A class signals in the whole dataset
    # print(len(dataset_train[0]))
    # print(len(dataset_valid[0]))
    # print(frequencies[0])
