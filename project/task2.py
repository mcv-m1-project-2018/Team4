import numpy as np

def split_proportionally (dataset_grouped, frequencies, train_set_ratio=0.7):
    """
    This function splits dataset into training and validation dataset in proportion 7:3
    proportionally with regard to image shapes and colors (i.e. classes).

    :return:
        output: dataset_train and dataset_valid, Python lists that contain 6 rows with columns of images, masks and data
    """
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
    # print(frequencies[0])