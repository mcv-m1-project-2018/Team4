"""
Usage:
  main.py <museum_set_path> <query_set_path> <color_space_block=0(gray), 1(rgb), 2(lab), 3(ycb), 4(hsv)>
Options:
"""""
import sys
from compare import compare, compare_3channel, compare_block, compare_pyramid, compare_full
from create_descriptors import read_set
from operator import itemgetter
from ml_metrics import mapk
import cv2

if __name__ == "__main__":
    if len(sys.argv) > 1:
        museum_path = sys.argv[1]
        query_path = sys.argv[2]
        block_color_space = int(sys.argv[3])
        museum_set, museum_histograms_by_type = read_set(museum_path, block_color_space)
        query_set, query_histograms_by_type = read_set(query_path, block_color_space)
        # query_histogram = query_histograms_by_type[3][15]
        # print(len(museum_histograms_by_type[6][0][2]))
        # print(len(query_histograms_by_type[6][0]))

        hist_type = 6
        compare_method = 4

        actual_query = query_set
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
            cv2.imshow("query", query_set[idx_q])
            predicted_query_single =[]
            for idx in range (0,K):
                cv2.imshow("matched", museum_set[scores[idx][1]])
                predicted_query_single.append(museum_set[scores[idx][1]])
                if (idx == 0):
                    cv2.waitKey()
                cv2.waitKey(100)
            predicted_query.append(predicted_query_single)
        mapk_score = mapk(actual_query,predicted_query,K)
        print('MAP@K SCORE:')
        print(mapk_score)
        print(scores)

        max_index = scores.index(min(scores))
        print(max_index)
        # print(max_score)
        
        

    else:
        print("ARGUMENTS NEEDED!")
        quit()
