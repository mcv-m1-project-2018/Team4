"""
Usage:
  main.py <museum_set_path> <query_set_path>
Options:
"""
import sys
from compare import compare, compare_3channel, compare_block
from create_descriptors import read_set
from operator import itemgetter
import cv2

if __name__ == "__main__":
    if len(sys.argv) > 1:
        museum_path = sys.argv[1]
        query_path = sys.argv[2]
        block_color_space = int(sys.argv[3])
        museum_set, museum_histograms_by_type = read_set(museum_path, block_color_space)
        query_set, query_histograms_by_type = read_set(query_path, block_color_space)
        # query_histogram = query_histograms_by_type[3][15]
        print(len(museum_histograms_by_type[4][0]))

        hist_type = 3
        for idx_q, query_histogram in enumerate (query_histograms_by_type[hist_type]):
            scores = []

            for idx, img_histogram in enumerate (museum_histograms_by_type[hist_type]):
                # score = compare_3channel(img_histogram, query_histogram,1)
                score = compare_block(img_histogram, query_histogram, 3)
    
                scores.append([score, idx])
        
            scores.sort(key=itemgetter(0))
            cv2.imshow("query", query_set[idx_q])
            for idx in range (0,10):
                cv2.imshow("matched", museum_set[scores[idx][1]])
                if (idx == 0):
                    cv2.waitKey()
                cv2.waitKey(500)

        print(scores)

        max_index = scores.index(min(scores))
        print(max_index)
        # print(max_score)
        
        

    else:
        print("ARGUMENTS NEEDED!")
        quit()
