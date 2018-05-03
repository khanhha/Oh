import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.data import astronaut
from skimage.util import img_as_float

img_lables = 'D:\Projects\Oh\data\images\prid_450s\cam_b_classify_result.txt'
file = open(img_lables)
label_imgs = {}
for line in file.readlines():
    substr = line.split()
    label = int(substr[1])
    if label not in label_imgs:
        label_imgs[label] = list()
    label_imgs[label].append(substr[0])

prob_map_dir = 'D:\Projects\Oh\data\images\prid_450s\\avg_silhouette_map'
prob_imgs = {}
for file_name  in os.listdir(prob_map_dir):
    label = int(os.path.splitext(file_name)[0])
    prob_img = cv.imread(os.path.join(prob_map_dir, file_name))
    prob_img = cv.cvtColor(prob_img, cv.COLOR_BGR2GRAY)
    prob_img = np.float32(prob_img)/prob_img.max()
    prob_imgs[label] = prob_img

out_dir = 'D:\Projects\Oh\data\images\prid_450s\grab_cut_based_on_pose'
for label, img_paths in label_imgs.items():
    prob_map = prob_imgs[label]
    dir = os.path.join(out_dir, str(label))
    try:
        os.stat(dir)
    except:
        os.mkdir(dir)
    for img_path in img_paths:
        img = cv.imread(img_path)
        img = cv.resize(img, (prob_img.shape[1], prob_img.shape[0]))
        ret, fg_mask = cv.threshold(prob_map, 0.95, maxval=1, type = cv.THRESH_BINARY)
        ret, bg_mask = cv.threshold(prob_map, 0.05, maxval=1, type = cv.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        bg_mask = cv.morphologyEx(bg_mask, cv.MORPH_OPEN, kernel) #remove noise
        kernel = np.ones((10, 10), np.uint8)
        bg_mask = cv.dilate(bg_mask, kernel)
        bg_mask = 1 - bg_mask
        bg_mask = bg_mask.astype(bool)
        fg_mask = fg_mask.astype(bool)

        img2 = img.copy()

        bgdmodel = np.zeros((1, 65), np.float64)
        fgdmodel = np.zeros((1, 65), np.float64)
        mask = np.zeros(img2.shape[:2], np.uint8)
        mask[:,:] = cv.GC_PR_FGD
        mask[fg_mask[:,:]==True] = cv.GC_FGD
        mask[bg_mask[:,:]==True] = cv.GC_BGD
        for i in range(1):
            cv.grabCut(img2, mask, None, bgdmodel, fgdmodel, 1, cv.GC_INIT_WITH_MASK)

        mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
        output = cv.bitwise_and(img2, img2, mask=mask2)
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        plt.subplot(1,3,1)
        plt.imshow(prob_map)
        plt.subplot(1,3,2)
        plt.imshow(img)
        plt.imshow(fg_mask,alpha=0.2, cmap='BuGn')
        plt.imshow(bg_mask,alpha=0.2, cmap='gray')
        plt.subplot(1,3,3)
        plt.imshow(output)
        #plt.show()

        file_name = os.path.basename(img_path)
        plt.savefig(os.path.join(dir, file_name))
        plt.close()
