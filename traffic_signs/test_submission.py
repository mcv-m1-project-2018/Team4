import fnmatch
import os
import pickle
import imageio
from evaluation.load_annotations import load_annotations
import evaluation.evaluation_funcs as evalf


# Use this script to test your submission. Do not look at the results, as they are computed with fake annotations and masks.
# This is just to see if there's any problem with files, paths, permissions, etc. *'

# Configure this
team              = 0  # Your team number
week              = 1  # Week number
window_evaluation = 1  # Whether to perform or not window evaluation: 0 for week 1, 1 for week 2


#Do not make changes below this point ---------------------------------
# If you find a bug, please report it to ramon.morros@upc.edu ----------


# This folder contains fake masks and text annotations. Do not change this.
test_dir   = '/home/mcv00/DataSet1/fake_test/'  

# This folder contains your results: mask imaged and window list pkl files. Do not change this.
results_dir = '/home/mcv{:02d}/m1-results/week{}/test'.format(team, week)

test_dir = '../../../../prova'
results_dir = './aaa'

# Load image names in the given directory
test_files = sorted(fnmatch.filter(os.listdir(test_dir), '*.jpg'))

test_files_num = len(test_files)

print ('TP01')

pixelTP  = 0
pixelFN  = 0
pixelFP  = 0
pixelTN  = 0
windowTP = 0
windowFN = 0
windowFP = 0 

# List all folders (corresponding to the different methods) in the results directory
methods = next(os.walk(results_dir))[1]


for method in methods:
    print ('Method: {}\n'.format(method))

    result_files = sorted(fnmatch.filter(os.listdir('{}/{}'.format(results_dir, method)), '*.png'))

    result_files_num = len(result_files)

    if result_files_num != test_files_num:
        print ('Method {} : {} result files found but there are {} test files'.format(method, result_files_num, test_files_num)) 


    for ii in range(len(result_files)):

        # Read mask file
        candidate_masks_name = '{}/{}/{}'.format(results_dir, method, result_files[ii])
        print ('File: {}'.format(candidate_masks_name))
        
        pixelCandidates = imageio.imread(candidate_masks_name)>0
        
        # Accumulate pixel performance of the current image %%%%%%%%%%%%%%%%%
        name, ext = os.path.splitext(test_files[ii])
        gt_mask_name = '{}/mask/mask.{}.png'.format(test_dir, name)

        pixelAnnotation = imageio.imread(gt_mask_name)>0
        [localPixelTP, localPixelFP, localPixelFN, localPixelTN] = evalf.performance_accumulation_pixel(pixelCandidates, pixelAnnotation)
        pixelTP = pixelTP + localPixelTP
        pixelFP = pixelFP + localPixelFP
        pixelFN = pixelFN + localPixelFN
        pixelTN = pixelTN + localPixelTN

        if window_evaluation == 1:
            # Read .pkl file
            
            name_r, ext_r = os.path.splitext(result_files[ii])
            pkl_name      = '{}/{}/{}.pkl'.format(results_dir, method, name_r)


            with open(pkl_name, "rb") as fp:   # Unpickling
                windowCandidates = pickle.load(fp)

            gt_annotations_name = '{}/gt/gt.{}.txt'.format(test_dir, name)
            windowAnnotations = load_annotations(gt_annotations_name)

            [localWindowTP, localWindowFN, localWindowFP] = evalf.performance_accumulation_window(windowCandidates, windowAnnotations)
            windowTP = windowTP + localWindowTP
            windowFN = windowFN + localWindowFN
            windowFP = windowFP + localWindowFP

    # Plot performance evaluation
    [pixelPrecision, pixelAccuracy, pixelSpecificity, pixelSensitivity] = evalf.performance_evaluation_pixel(pixelTP, pixelFP, pixelFN, pixelTN)
    pixelF1 = 2*((pixelPrecision*pixelSensitivity)/(pixelPrecision + pixelSensitivity))

    print ('Team {:02d} pixel, method {} : {:.2f}, {:.2f}, {:.2f}\n'.format(team, method, pixelPrecision, pixelSensitivity, pixelF1))      

    if window_evaluation == 1:
        [windowPrecision, windowSensitivity, windowAccuracy] = evalf.performance_evaluation_window(windowTP, windowFN, windowFP) # (Needed after Week 3)
        windowF1 = 0

        print ('Team {:02d} window, method {} : {:.2f}, {:.2f}, {:.2f}\n'.format(team, method, windowPrecision, windowSensitivity, windowF1)) 
