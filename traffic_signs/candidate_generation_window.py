#!/usr/bin/python
# -*- coding: utf-8 -*-

def candidate_generation_window_example1(im, pixel_candidates):
    window_candidates = [[17.0, 12.0, 49.0, 44.0], [60.0,90.0,100.0,130.0]]

    return window_candidates
 
def candidate_generation_window_example2(im, pixel_candidates):
    window_candidates = [[21.0, 14.0, 54.0, 47.0], [63.0,92.0,103.0,132.0],[200.0,200.0,250.0,250.0]]

    return window_candidates
 
# Create your own candidate_generation_window_xxx functions for other methods
# Add them to the switcher dictionary in the switch_method() function
# These functions should take an image, a pixel_candidates mask (and perhaps other parameters) as input and output the window_candidates list.
 
def switch_method(im, pixel_candidates, method):
    switcher = {
        'example1': candidate_generation_window_example1,
        'example2': candidate_generation_window_example2
    }
    # Get the function from switcher dictionary
    func = switcher.get(method, lambda: "Invalid method")

    # Execute the function
    window_candidates = func(im, pixel_candidates)

    return window_candidates

def candidate_generation_window(im, pixel_candidates, method):

    window_candidates = switch_method(im, pixel_candidates, method)

    return window_candidates

    
if __name__ == '__main__':
    window_candidates1 = candidate_generation_window(im, pixel_candidates, 'example1')
    window_candidates2 = candidate_generation_window(im, pixel_candidates, 'example2')

    
