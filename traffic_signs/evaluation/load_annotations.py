def load_annotations(annot_file):
    # Annotations are stored in text files containing
    # the coordinates of the corners (top-left and bottom-right) of
    # the bounding box plus an alfanumeric code indicating the signal type:
    # tly, tlx, bry,brx, code

    annotations = []
    signs       = [] 

    for line in open(annot_file).read().splitlines():

        annot_values = line.split()
        annot_values = [x.strip() for x in annot_values]
        for ii in range(4):
            annot_values[ii] = float(annot_values[ii])
        annotations.append(annot_values)
        
    return annotations
