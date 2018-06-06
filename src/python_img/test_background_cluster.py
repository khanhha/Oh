import cv2 as cv
import os
import matplotlib.pyplot as plt
import skimage.color as skcolor
import numpy as np
from common import util
from common import prob_map
from common import background_subtractor as bgr_sub

DIR = "D:\Projects\Oh\data\images\silhouette\images\\view_178"
PROB_MAP_DIR = 'D:\Projects\Oh\data\images\silhouette\\avg_silhouette_map'
img_label_mapping = 'D:\Projects\Oh\data\images\silhouette\mapping_image_avg_silhouette.txt'

for file_name in os.listdir(DIR):
    if file_name not in 'image178_laxsquadT_lBlue_LL_0427.jpg':
        continue

    prb_label = prob_map.find_mean_silhouette_label(img_label_mapping, file_name)
    prb_img = prob_map.load_mean_silhouette(PROB_MAP_DIR, prb_label)
    fg_mask, bg_mask, mask = prob_map.build_fg_bg_masks(prb_img, False, fg_threshold = 0.8)

    img = cv.imread(f'{DIR}/{file_name}')
    img = cv.medianBlur(img, 5)
    img = cv.resize(img, (prb_img.shape[1], prb_img.shape[0]), cv.INTER_CUBIC)

    TEST_DIR = 'D:\Projects\Oh\data\images\crf_test'
    test_img = np.zeros_like(img)
    test_img[fg_mask] = (255,0,0)
    test_img[bg_mask] = (0,0,255)
    cv.imwrite(f'{TEST_DIR}\\anno4.png', test_img)
    cv.imwrite(f'{TEST_DIR}\\im4.png', img)

    # plt.subplot(1,3,1); plt.imshow(fg_mask)
    # plt.subplot(1,3,2); plt.imshow(bg_mask)
    # plt.subplot(1,3,3); plt.imshow(mask)
    # plt.show()
    # labels = prob_map.cluster_GMM(img, bg_mask, 9,'full')
    # labels_color = skcolor.label2rgb(labels)
    # plt.subplot(1,2,1)
    # plt.imshow(img[:, :, ::-1])
    # plt.subplot(1,2,2)
    # plt.imshow(img[:, :, ::-1])
    # plt.imshow(labels_color, alpha=0.4)
    # plt.show()
