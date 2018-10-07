import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def analyse_single_class(dataset, filename="hist.png"):
    zoom=1
    B = np.array([])
    G = np.array([])
    R = np.array([])
    H = np.array([])
    S = np.array([])
    V = np.array([])
    Y = np.array([])
    Cr = np.array([])
    Cb = np.array([])
    LL = np.array([])
    LA = np.array([])
    LB = np.array([])

    for img in dataset:
            # Read the image, mask and gt corresponding to certain filename
            image = img[1]
            mask = img[2]
            bounding_boxes=img[3]
            # Compute binary mask
            ret, bin_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
            final_mask = cv2.bitwise_and(image, bin_mask)
            # debugging:
            # print(len(bounding_boxes))
            for bounding_box in bounding_boxes:
                coordinates = list(map(float, bounding_box[0:4]))
                crop = final_mask[int(coordinates[0]):int(coordinates[2]), int(coordinates[1]):int(coordinates[3])]
                # cv2.imshow('hey', crop)
                b = crop[:,:,0]
                b = b.reshape(b.shape[0]*b.shape[1])
                g = crop[:,:,1]
                g = g.reshape(g.shape[0]*g.shape[1])
                r = crop[:,:,2]
                r = r.reshape(r.shape[0]*r.shape[1])
                B = np.append(B,b)
                G = np.append(G,g)
                R = np.append(R,r)
                # HSV
                hsv = cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
                h = hsv[:,:,0]
                h = h.reshape(h.shape[0]*h.shape[1])
                s = hsv[:,:,1]
                s = s.reshape(s.shape[0]*s.shape[1])
                v = hsv[:,:,2]
                v = v.reshape(v.shape[0]*v.shape[1])
                H = np.append(H,h)
                S = np.append(S,s)
                V = np.append(V,v)
                # YCrCb
                ycb = cv2.cvtColor(crop,cv2.COLOR_BGR2YCrCb)
                y = ycb[:,:,0]
                y = y.reshape(y.shape[0]*y.shape[1])
                cr = ycb[:,:,1]
                cr = cr.reshape(cr.shape[0]*cr.shape[1])
                cb = ycb[:,:,2]
                cb = cb.reshape(cb.shape[0]*cb.shape[1])
                Y = np.append(Y,y)
                Cr = np.append(Cr,cr)
                Cb = np.append(Cb,cb)
                # Lab
                lab = cv2.cvtColor(crop,cv2.COLOR_BGR2LAB)
                ll = lab[:,:,0]
                ll = ll.reshape(ll.shape[0]*ll.shape[1])
                la = lab[:,:,1]
                la = la.reshape(la.shape[0]*la.shape[1])
                lb = lab[:,:,2]
                lb = lb.reshape(lb.shape[0]*lb.shape[1])
                LL = np.append(LL,ll)
                LA = np.append(LA,la)
                LB = np.append(LB,lb)

            try:
                nbins = 10
                plt.figure(figsize=[20,10])
                plt.subplot(2,3,1)
                plt.hist2d(B, G, bins=nbins, norm=LogNorm())
                plt.xlabel('B')
                plt.ylabel('G')
                plt.title('RGB')
                if not zoom:
                    plt.xlim([0,255])
                    plt.ylim([0,255])
                plt.colorbar()
                plt.subplot(2,3,2)
                plt.hist2d(B, R, bins=nbins, norm=LogNorm())
                plt.colorbar()
                plt.xlabel('B')
                plt.ylabel('R')
                plt.title('RGB')
                if not zoom:
                    plt.xlim([0,255])
                    plt.ylim([0,255])
                plt.subplot(2,3,3)
                plt.hist2d(R, G, bins=nbins, norm=LogNorm())
                plt.colorbar()
                plt.xlabel('R')
                plt.ylabel('G')
                plt.title('RGB')
                if not zoom:
                    plt.xlim([0,255])
                    plt.ylim([0,255])
                
                Hmax=H.mean()
                # print(Hmax)
                # print(H.shape())
                # print(H)
                plt.subplot(2,3,4)
                plt.hist2d(H, S, bins=nbins, norm=LogNorm())
                plt.colorbar()
                plt.xlabel('H')
                plt.ylabel('S')
                plt.title('HSV')
                if not zoom:
                    plt.xlim([0,180])
                    plt.ylim([0,255])
                plt.subplot(2,3,5)
                plt.hist2d(Cr, Cb, bins=nbins, norm=LogNorm())
                plt.colorbar()
                plt.xlabel('Cr')
                plt.ylabel('Cb')
                plt.title('YCrCb')
                if not zoom:
                    plt.xlim([0,255])
                    plt.ylim([0,255])
                plt.subplot(2,3,6)
                plt.hist2d(H, V, bins=nbins, norm=LogNorm())
                plt.colorbar()
                plt.xlabel('H')
                plt.ylabel('V')
                plt.title('HSV')
                if not zoom:
                    plt.xlim([0,255])
                    plt.ylim([0,255])
                plt.savefig(filename,bbox_inches='tight')

                return plt
            except:
                print("sorry, wont happen")
                return 0


def analyse_obj_hist(dataset, is_grouped = 0):
    """
    Determine and plot histograms of the signs in the training set for different color spaces in each class
    or in the whole training set regardless of the class (is_grouped = 0)

    :return:
        output: image files with plotted histograms for each class

    """

    if (is_grouped):
        for class_id in range(0, len(dataset)):
            plt=analyse_single_class(dataset[class_id],'color' + str(class_id)+'.png')

    else:
        plt=analyse_single_class(dataset,'whole-dataset-original.png')
