#!/usr/bin/python
"""
Usage:
  main.py <museum_set_path> <query_set_path>   [--featureType=<ft>] [--matcherType=<mt>] [--matchingMethod=<mm>] [--distanceThreshold=<dt>] [--normType=<nt>] [--crossCheck=<cc>] 
  main.py -h | --help
Options:
  <museum_set_path>
  <query_set_path>
  --featureType=<ft>          Available features: 'ORB', 'FAST','SIFT', 'SURF', 'KAZE', 'AKAZE', 'BRISK', 'MSER'         [default: ORB]
  --matcherType=<mt>          Available matchers: 'BF', 'FLANN'                                                          [default: BF]
  --matchingMethod=<mm>       Available matching methods: 'KNN', 'MATCHER', 'RADIUS'                                     [default: KNN]
  --distanceThreshold=<dt>    The maximum distance between matched keypoints (lower value -> less weak matches)          [default: 0.75]
  --normType=<nt>             L1, L2 norms preferable choices for SIFT, SURF. NORM_HAMMING for ORB, BRISK                [default: NORM_HAMMING]
  --crossCheck=<cc>           optional bool parameter for BF matcher                                                     [default: False]
"""

import sys
import os
from features import read_set_features, match_features
from operator import itemgetter
from ml_metrics import mapk
from docopt import docopt
import cv2
import pickle
import time
import numpy as np


def save_results(directory, result, color_space, hist_type, block_factor, compare_method):
        
    #  TODO
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
        feature_type = args['--featureType']
        matcher_type = args['--matcherType']
        matching_method = args['--matchingMethod']
        distance_threshold = float(args['--distanceThreshold'])
        norm_type = args['--normType']
        cross_check = args['--crossCheck']

        museum_set, museum_set_features, museum_set_names = read_set_features(museum_path,feature_type)

        start_time = time.time()        
        query_set, query_set_features, query_set_names = read_set_features(query_path, feature_type)
        # groundtruth_names=[]
        # grndtrth_lines = []
        # grndtrth = open( "w4_query_devel.pkl", "r" )
        # grndtrth_lines = grndtrth.read().splitlines()
        # for line in grndtrth_lines:
        #     groundtruth_names.append([line])

        # actual_query = groundtruth_names
        K = 5
        predicted_query = []

        for idx_q, query_features in enumerate (query_set_features):
            scores = []
            for idx, museum_features in enumerate (museum_set_features):
                
                score = match_features(query_features, museum_features, matcher_type, matching_method, distance_threshold, norm_type, cross_check)
                if score > 57:
                    scores.append([score, idx])

            cv2.namedWindow("query",cv2.WINDOW_NORMAL)
            cv2.imshow("query", query_set[idx_q])

            if (len(scores)):
                scores.sort(key=itemgetter(0), reverse=True)
                cv2.namedWindow("matched",cv2.WINDOW_NORMAL)
                predicted_query_single =[]
                for idx in range (0,len(scores)):
                    cv2.imshow("matched", museum_set[scores[idx][1]])
                    print(scores[idx][0])
                    predicted_query_single.append(museum_set_names[scores[idx][1]])
                    if (idx == 0):
                        predicted_query_single.append(museum_set_names[scores[idx][1]])

                        cv2.waitKey()
                    cv2.waitKey(500)
                predicted_query.append(predicted_query_single)
            else:
                print("No match for the picture")
                no_match_img = np.zeros((500, 500,3), dtype=np.uint8)
                cv2.putText(no_match_img,'PICTURE NOT FOUND',(80, 270),2,1,(100,40,255),4)
                cv2.imshow("matched", no_match_img)
                cv2.waitKey()

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
