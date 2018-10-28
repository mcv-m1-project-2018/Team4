"""
Usage:
  main.py <museum_set_path> <query_set_path> <color_space_block=0(gray), 1(rgb), 2(lab), 3(ycb), 4(hsv)>
Options:
"""""
import sys
import os
from compare import compare, compare_3channel, compare_block, compare_pyramid, compare_full
from create_descriptors import read_set
from operator import itemgetter
from ml_metrics import mapk
import cv2
import pickle

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
        base = 'result'
        out_list_name = '{}/{}.pkl'.format(directory, base)
        if not os.path.exists(directory):
            os.makedirs(directory)

        pickle.dump(result, open(out_list_name, "wb"))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        museum_path = sys.argv[1]
        query_path = sys.argv[2]
        block_color_space = int(sys.argv[3])

        museum_set, museum_histograms_by_type, museum_set_names = read_set(museum_path, block_color_space, block_factor)
        query_set, query_histograms_by_type, query_set_names = read_set(query_path, block_color_space, block_factor)
        groundtruth_names=[]
        grndtrth_lines = []
        grndtrth = open( "groundtruth.txt", "r" )
        grndtrth_lines = grndtrth.read().splitlines()
        for line in grndtrth_lines:
            groundtruth_names.append([line])


        block_factor = int(sys.argv[4])

        # query_histogram = query_histograms_by_type[3][15]
        # print(len(museum_histograms_by_type[6][0][2]))
        # print(len(query_histograms_by_type[6][0]))
        color_space = block_color_space
        hist_type = 6
        compare_method = 3

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
                    score = compare_pyramid(img_histogram, query_histogram, block_color_space, compare_method)
                elif (hist_type == 7):
                    score = compare_full(img_histogram, query_histogram, block_color_space, compare_method)
                scores.append([score, idx])
        
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
        print(groundtruth_names)
        print(predicted_query)
        mapk_score = mapk(actual_query,predicted_query,K)
        print('MAP@K SCORE:')
        print(mapk_score)
        # print(scores)

        max_index = scores.index(min(scores))
        print(max_index)
        # print(max_score)
        result = score
        save_results('results', result, color_space, hist_type, block_factor, compare_method)
        
        

    else:
        print("ARGUMENTS NEEDED!")
        quit()
