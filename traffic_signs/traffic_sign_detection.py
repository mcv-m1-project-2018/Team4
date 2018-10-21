#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Usage:
  traffic_sign_detection.py <dirName> <outPath> <pixelMethod> [--windowMethod=<wm>] [--calculateMetrics] 
  traffic_sign_detection.py -h | --help
Options:
  <pixelMethod>              Method for selecting pixel candidates  ('hsv_manual', 'hsv_manual_improved', 'hsv_hist', 'hsv_hist_equal')        
  --windowMethod=<wm>        Method for selecting window candidates       [default: None]
  --calculateMetrics         Turn on calculating the metrics  (Leave false when generating the masks for test set)              [default: False]
"""


import fnmatch
import os
import sys
import pickle
import time

import numpy as np
import imageio
from docopt import docopt

from candidate_generation_pixel import candidate_generation_pixel
from candidate_generation_window import candidate_generation_window
from evaluation.load_annotations import load_annotations
import evaluation.evaluation_funcs as evalf
from connected_labels_pixel_cand import connected_labels_pixel_cand
from morphological_operators import morphological_operators

def traffic_sign_detection(directory, output_dir, pixel_method, window_method, calculate_metrics):

    pixelTP  = 0
    pixelFN  = 0
    pixelFP  = 0
    pixelTN  = 0

    pixel_F1  = 0

    windowTP = 0
    windowFN = 0
    windowFP = 0

    window_precision = 0
    window_accuracy  = 0
    window_F1 = 0

    counter = 0

    # Load image names in the given directory
    file_names = sorted(fnmatch.filter(os.listdir(directory), '*.jpg'))

    
    for name in file_names:
        counter += 1
        base, extension = os.path.splitext(name)

        # Read file
        im = imageio.imread('{}/{}'.format(directory,name))
        #print ('{}/{}'.format(directory,name))

        # Candidate Generation (pixel) ######################################
        pixel_candidates = candidate_generation_pixel(im, pixel_method)
        # pixel_candidates = morphological_operators(pixel_candidates)
        pixel_candidates = connected_labels_pixel_cand(im, pixel_candidates)
        
        fd = '{}/{}_{}'.format(output_dir, pixel_method, window_method)
        if not os.path.exists(fd):
            os.makedirs(fd)
        
        out_mask_name = '{}/{}.png'.format(fd, base)

        
        if window_method != 'None':

            window_candidates = candidate_generation_window(im, pixel_candidates, window_method) 
            window_mask = np.zeros(pixel_candidates.shape)
            for window_candidate in window_candidates:
                window_mask[window_candidate[0]:window_candidate[2],window_candidate[1]:window_candidate[3]]=pixel_candidates[window_candidate[0]:window_candidate[2],window_candidate[1]:window_candidate[3]]
            out_list_name = '{}/{}.pkl'.format(fd, base)
            pixel_candidates=window_mask
            with open(out_list_name, "wb") as fp:   #Pickling
                pickle.dump(window_candidates, fp)

        imageio.imwrite (out_mask_name, np.uint8(np.round(pixel_candidates)))

                      
        pixel_precision = 0
        pixel_accuracy = 0
        pixel_specificity = 0
        pixel_sensitivity = 0
        window_precision = 0
        window_accuracy = 0

        if (calculate_metrics):
            # Accumulate pixel performance of the current image #################
            pixel_annotation = imageio.imread('{}/mask/mask.{}.png'.format(directory,base)) > 0

            [localPixelTP, localPixelFP, localPixelFN, localPixelTN] = evalf.performance_accumulation_pixel(pixel_candidates, pixel_annotation)
            pixelTP = pixelTP + localPixelTP
            pixelFP = pixelFP + localPixelFP
            pixelFN = pixelFN + localPixelFN
            pixelTN = pixelTN + localPixelTN
            
            # [pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity, pixel_F1] = evalf.performance_evaluation_pixel(pixelTP, pixelFP, pixelFN, pixelTN)
            
            if window_method != 'None':
                # Accumulate object performance of the current image ################
                window_annotationss = load_annotations('{}/gt/gt.{}.txt'.format(directory, base))
                [localWindowTP, localWindowFN, localWindowFP] = evalf.performance_accumulation_window(window_candidates, window_annotationss)

                windowTP = windowTP + localWindowTP
                windowFN = windowFN + localWindowFN
                windowFP = windowFP + localWindowFP


                # Plot performance evaluation
                # [window_precision, window_sensitivity, window_accuracy, window_F1] = evalf.performance_evaluation_window(windowTP, windowFN, windowFP)
    
    if (calculate_metrics):
        [pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity, pixel_F1] = evalf.performance_evaluation_pixel(pixelTP, pixelFP, pixelFN, pixelTN)
        print("Pixel precision: "+ str(pixel_precision))
        print("Pixel accuracy: "+ str(pixel_accuracy))
        print("Pixel recall: "+ str(pixel_sensitivity))
        print("Pixel F1-measure: "+ str(pixel_F1))
        print("Pixel TP: "+ str(pixelTP))
        print("Pixel FP: "+ str(pixelFP))
        print("Pixel FN: "+ str(pixelFN))
        print("Pixel TN: "+ str(pixelTN))

        if window_method != 'None':
            [window_precision, window_sensitivity, window_accuracy, window_F1] = evalf.performance_evaluation_window(windowTP, windowFN, windowFP)
            print("Window precision: "+ str(window_precision))
            print("Window accuracy: "+ str(window_accuracy))
            print("Window F1-measure: "+ str(window_F1))
    
    return [pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity, pixel_F1, window_precision, window_accuracy, window_F1, counter]





                
if __name__ == '__main__':
    # read arguments
    args = docopt(__doc__)

    images_dir = args['<dirName>']          # Directory with input images and annotations
                                            # For instance, '../../DataSetDelivered/test'
    output_dir = args['<outPath>']          # Directory where to store output masks, etc. For instance '~/m1-results/week1/test'

    pixel_method = args['<pixelMethod>']
    window_method = args['--windowMethod']
    calculate_metrics = args['--calculateMetrics']

    print ("Computing masks for pixel method: "+pixel_method+" and window method: "+ window_method + " Calculate metrics: " + str(calculate_metrics))
    start_time = time.time()
    pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity, pixel_F1, window_precision, window_accuracy, window_F1, counter = traffic_sign_detection(images_dir, output_dir, pixel_method, window_method, calculate_metrics)
    total_time = time.time() - start_time
    per_frame_time = 0
    if counter != 0:
        per_frame_time = total_time/counter

    # print(f"Processed {counter:d} images in {total_time:.2f} seconds.")
    # print(f"Time per frame: {per_frame_time:.2f} seconds.")

   
