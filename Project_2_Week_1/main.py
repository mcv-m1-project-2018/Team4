#!/usr/bin/python
"""
Usage:
  main.py <museum_set_path> <query_set_path> [--colorSpace=<cs>] [--featureType=<pm>] [--blockFactor=<bf>] [--compareMethod=<cm>] 
  main.py -h | --help
Options:
  <museum_set_path>
  <query_set_path>
  [--colorSpace=<cs>]        Choose one of: gray, RGB, Lab, YCrCb, HSV     [default: gray]
  [--featureType=<pm>]       Choose one of: block, pyramid, whole [default: block]
  [--blockFactor=<bf>]       Number of blocks in a row [default: '1']
  [--compareMethod=<cm>]     Choose one of: chi-square, intersection, Hellinger [default: Hellinger]
"""

import sys
import os
from compare import compare, compare_3channel, compare_block, compare_pyramid, compare_full, compare_pyramid_weights
from create_descriptors import read_set
from operator import itemgetter
from ml_metrics import mapk
from docopt import docopt
import cv2
import pickle
import time


def get_param_from_arg(color_space_name, feature_type_name, compare_method_name):
    color_space = 0
    hist_type = 0
    compare_method = 0

    if (color_space_name == 'gray'):
        color_space = 0
    elif (color_space_name == 'RGB'):
        color_space = 1
    elif (color_space_name == 'Lab'):
        color_space = 2
    elif (color_space_name == 'YCrCb'):
        color_space = 3
    elif (color_space_name == 'HSV'):
        color_space = 4
    else:
        print('Wrong colorspace name. Using gray as default.')
        color_space = 0
    
    
    if (feature_type_name == 'block'):
        hist_type = 5
    elif (feature_type_name == 'pyramid'):
        hist_type = 6
    elif (feature_type_name == 'pyramid_weights'):
        hist_type = 8
    elif (feature_type_name == 'all_spaces'):
        hist_type = 7
    else:
        hist_type = color_space
    
    if (compare_method_name == 'chi-square'):
        compare_method = 1
    elif (compare_method_name == 'intersection'):
        compare_method = 2
    elif (compare_method_name == 'Hellinger'):
        compare_method = 3
    else:
        print('Wrong compare method name. Using chi-square as default.')
        compare_method = 1

    return hist_type, color_space, compare_method

def save_results(directory, result, color_space, hist_type, block_factor, compare_method):
        color_space_name = ' '
        if (color_space == 0):
            color_space_name = 'gray'
        elif (color_space == 1):
            color_space_name = 'RGB'
        elif (color_space == 2):
            color_space_name = 'Lab'
        elif (color_space == 3):
            color_space_name = 'YCrCb'
        elif (color_space == 4):
            color_space_name = 'HSV'
        
        block_factor_name = str(block_factor)+'x'+str(block_factor)
        hist_method_name = ' '
        if (hist_type == 5):
            hist_method_name = 'block_'+block_factor_name
        elif (hist_type == 6):
            hist_method_name = 'pyramid_'+block_factor_name
        elif (hist_type == 8):
            hist_method_name = 'pyramid_weights'+block_factor_name
        elif (hist_type == 7):
            hist_method_name = 'all_spaces_'+block_factor_name
        else:
            hist_method_name = 'whole'
        
        compare_method_name = ' '
        if (compare_method == 1):
            compare_method_name = 'chi-square'
        elif (compare_method == 2):
            compare_method_name = 'intersection'
        elif (compare_method == 3):
            compare_method_name = 'Hellinger'
        else:
            compare_method_name = 'other'

        directory = directory +'/method_'+color_space_name+'_'+hist_method_name+'_'+compare_method_name
        print(directory)
        base = 'result'
        out_list_name = '{}/{}.pkl'.format(directory, base)
        if not os.path.exists(directory):
            os.makedirs(directory)

        pickle.dump(result, open(out_list_name, "wb"))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        args = docopt(__doc__)

        museum_path = args['<museum_set_path>']
        query_path = args['<query_set_path>']          
        color_space = args['--colorSpace']
        feature_type = args['--featureType']
        block_factor = int(args['--blockFactor'])
        compare_method_name = args['--compareMethod']
        hist_type, block_color_space, compare_method = get_param_from_arg(color_space, feature_type, compare_method_name)

        museum_set, museum_histograms_by_type, museum_set_names = read_set(museum_path, block_color_space, block_factor)
        start_time = time.time()        
        query_set, query_histograms_by_type, query_set_names = read_set(query_path, block_color_space, block_factor)
        groundtruth_names=[]
        grndtrth_lines = []
        grndtrth = open( "groundtruth.txt", "r" )
        grndtrth_lines = grndtrth.read().splitlines()
        for line in grndtrth_lines:
            groundtruth_names.append([line])

        actual_query = groundtruth_names
        K = 10
        predicted_query = []

        for idx_q, query_histogram in enumerate (query_histograms_by_type[hist_type]):
            scores = []
            for idx, img_histogram in enumerate (museum_histograms_by_type[hist_type]):
                if (hist_type < 5):
                    score = compare_3channel(img_histogram, query_histogram, block_color_space, compare_method)
                elif (hist_type == 5):
                    score = compare_block(img_histogram, query_histogram, compare_method)
                elif (hist_type == 6):
                    score = compare_pyramid(img_histogram, query_histogram, block_color_space, block_factor, compare_method)
                elif (hist_type == 7):
                    score = compare_full(img_histogram, query_histogram, block_color_space, compare_method)
                elif (hist_type == 8):
                    score = compare_pyramid_weights(img_histogram, query_histogram, block_color_space, block_factor, compare_method)
                scores.append([score, idx])
    

            # print(f"Processed {counter:d} images in {total_time:.2f} seconds.")
            # print(f"Time per frame: {per_frame_time:.2f} seconds.")
        
            scores.sort(key=itemgetter(0))
            # cv2.imshow("query", query_set[idx_q])
            predicted_query_single =[]
            for idx in range (0,K):
                # cv2.imshow("matched", museum_set[scores[idx][1]])
                predicted_query_single.append(museum_set_names[scores[idx][1]])
                # if (idx == 0):
                #     predicted_query_single.append(museum_set_names[scores[idx][1]])

                #     cv2.waitKey()
                # cv2.waitKey(500)
            predicted_query.append(predicted_query_single)
        # print(groundtruth_names)
        # print(predicted_query)
        total_time = time.time() - start_time
        per_frame_time = 0
        if K != 0:
            per_frame_time = total_time/len(query_set)

        print('Processed ' + str(len(query_set)) + ' queries in '+ str(total_time) + ' seconds.')
        print('Time per query: '+ str(per_frame_time)+ ' seconds.')

        mapk_score = mapk(actual_query,predicted_query,K)
        print('MAP@K SCORE:')
        print(mapk_score)
        # print(scores)

        max_index = scores.index(min(scores))
        # print(max_index)
        # print(max_score)
        result = score
        # print(predicted_query)
        save_results('results', predicted_query, block_color_space, hist_type, block_factor, compare_method)
        
        

    else:
        print("ARGUMENTS NEEDED!")
        quit()
