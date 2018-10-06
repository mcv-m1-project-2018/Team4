import numpy as np
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

    # uncomment to print number of signals for train and validation from 
    # class A, and to print frequency of A class signals in the whole dataset
    # print(len(dataset_train[0]))
    # print(len(dataset_valid[0]))
    # print(frequencies[0])