def compute_color_spaces_avg(dataset_train):
    """

    :param dataset_train:
    :return:
    """
    # Uncomment to see the crop used to compute values
    # cv2.imshow('crop', dataset_train[0][0][1])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    total_mean_BGR = []
    total_mean_LAB = []
    total_mean_HSV = []
    total_mean_YCB = []

    for class_id in range(6):
        data = dataset_train[class_id][:]

        rgb_pixels = [0, 0, 0]
        lab_pixels = [0, 0, 0]
        ycb_pixels = [0, 0, 0]
        hsv_pixels = [0, 0, 0]

        for n in data:
            img = n[1]
            mask = n[2]
            bounding_boxes = n[3]

            # Compute binary mask
            ret, bin_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
            final_mask = cv2.bitwise_and(img, bin_mask)

            # Uncomment to see the patch from the image corresponding to the no zero values in the mask

            # cv2.imshow('Final mask', final_mask)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            cropBGR = final_mask[int(bounding_boxes[0]):int(bounding_boxes[2]), int(bounding_boxes[1]):int(bounding_boxes[3])]
            cropLAB = cv2.cvtColor(cropBGR, cv2.COLOR_BGR2LAB)
            cropYCB = cv2.cvtColor(cropBGR, cv2.COLOR_BGR2YCrCb)
            # cv2.imshow('crop', cropYCB)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            cropHSV = cv2.cvtColor(cropBGR, cv2.COLOR_BGR2HSV)

            # Compute avg thresholds in different color spaces
            for i in range(img.shape[-1]):
                channel_BGR = cropBGR[:, :, i]
                channel_YCB = cropYCB[:, :, i]
                channel_HSV = cropHSV[:, :, i]
                channel_LAB = cropLAB[:, :, i]

                rgb_pixels[i] += np.mean(channel_BGR[channel_BGR > 0])
                lab_pixels[i] += np.mean(channel_LAB[channel_BGR > 0])
                hsv_pixels[i] += np.mean(channel_HSV[channel_BGR > 0])
                ycb_pixels[i] += np.mean(channel_YCB[channel_BGR > 0])

        mean_BGR = [(rgb_pixels[0]/len(data)), (rgb_pixels[1]/len(data)),
                       (rgb_pixels[2]/len(data))]
        mean_LAB = [(lab_pixels[0] / len(data)), (lab_pixels[1] / len(data)),
                    (lab_pixels[2] / len(data))]
        mean_HSV = [(hsv_pixels[0] / len(data)), (hsv_pixels[1] / len(data)),
                    (hsv_pixels[2] / len(data))]
        mean_YCB = [(ycb_pixels[0] / len(data)), (ycb_pixels[1] / len(data)),
                    (ycb_pixels[2] / len(data))]

        total_mean_BGR.append(mean_BGR)
        total_mean_LAB.append(mean_LAB)
        total_mean_HSV.append(mean_HSV)
        total_mean_YCB.append(mean_YCB)
        total_mean = [total_mean_BGR, total_mean_LAB, total_mean_HSV, total_mean_YCB]
    #
    # print("\n")
    # print(total_mean[0])
    # print("\n")
    # print(total_mean[1])
    # print("\n")
    # print(total_mean[2])
    # print("\n")
    # print(total_mean[3])

    def compute_k_means(dataset_train, dataset_valid):
        all_data = dataset_train[:]
        features = []

        for data in all_data:
            for sample in data:
                image = sample[1]
                # bounding_boxes = sample[3]
                cv2.imshow('img', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                # imgHSV = img[:,:,0]

                #Z = imgHSV.reshape(imgHSV.shape[0]*imgHSV.shape[1])
                Z = img.reshape((-1, 3))
                Z = np.float32(Z)
                # define criteria, number of clusters(K) and apply kmeans()
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                K = 40
                ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                # Now convert back into uint8, and make original image
                center = np.uint8(center)
                res = center[label.flatten()]
                res2 = res.reshape((img.shape))

                # Now convert back into uint8, and make original image
                # center = np.uint8(center)
                # res = center[label.flatten()]
                # res2 = np.zeros((imgHSV.shape[0], imgHSV.shape[1], 3))
                # res2[:,:,0] = res.reshape(imgHSV.shape)
                # res2[:,:,1] = img[:,:,1]
                # res2[:,:,2] = img[:,:,2]
                # res2 = np.uint8(res2)
                res2 = cv2.cvtColor(res2, cv2.COLOR_HSV2BGR)
                cv2.imshow('res2', res2)
                cv2.waitKey(0)
                cv2.destroyAllWindows()