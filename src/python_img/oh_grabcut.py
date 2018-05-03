import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.data import astronaut
from skimage.util import img_as_float
from skimage.measure import compare_ssim
from skimage import data, img_as_float

def color_constancy(img):
    n_pixels = np.prod(img.shape[:2]).astype(np.float)
    img = img.astype(np.float)
    color_means = np.array([0,0,0],dtype=np.float)
    color_scales = np.array([0,0,0],dtype=np.float)
    for i in range(3):
        color_means[i] = np.mean(img[:,:,i])
    color_scales = np.mean(color_means) /color_means

    for i in range(3):
        img[:,:,i] = color_scales[i] * img[:,:,i]
    img = np.clip(img, 0,255)
    return img.astype(np.uint8)

def remove_light_tubes(img):
    pixel_idx = img[:,:,:] > 220
    img[pixel_idx] = 135
    return img

def create_or_open_dir(dir):
    try:
        os.stat(dir)
    except:
        os.mkdir(dir)

def subtract_background_skimage(dir, bg_img_path):
    bg_img = cv.imread(bg_img_path)
    bg_img = cv.GaussianBlur(bg_img,(5,5),0)
    bg_img = img_as_float(bg_img)
    #bg_img = cv.cvtColor(bg_img, cv.COLOR_BGR2GRAY)
    fg_scales = {}
    for file_name in os.listdir(dir):
        file_path = os.path.join(dir, file_name)
        if os.path.isfile(file_path) and '.jpg' in file_name:
            img = cv.imread(file_path)
            img = cv.GaussianBlur(img, (5,5),0)
            img = img_as_float(img)
            score, img_dif = compare_ssim(bg_img, img, multichannel=True, data_range=img.max() - img.min(), full=True)
            img_dif = 1.0 - img_dif
            fg_scales[file_path] = img_dif
            #img_dif = (img_dif * 255).astype("uint8")

    return fg_scales

def subtract_background_rick(dir, bg_img_path):
    bg_img = cv.imread(bg_img_path)
    bg_img = cv.GaussianBlur(bg_img,(5,5),0)
    #bg_img = cv.cvtColor(bg_img, cv.COLOR_BGR2GRAY)
    fg_scales = {}
    for file_name in os.listdir(dir):
        file_path = os.path.join(dir, file_name)
        if os.path.isfile(file_path) and '.jpg' in file_name:
            img = cv.imread(file_path)
            img = cv.GaussianBlur(img, (5,5),0)
            #img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = cv.resize(img, bg_img.shape[0:2][::-1])
            img_dif = cv.absdiff(bg_img, img)
            img_dif = img_dif / bg_img
            #mx = np.max(img_dif)
            #mi = np.min(img_dif)
            #img_dif = (img_dif - mi)/(mx-mi)
            fg_scales[file_path] = img_dif

    return fg_scales


def subtract_background(dir):
    fgbg = cv.createBackgroundSubtractorKNN()
    for file_name in os.listdir(dir):
        file_path = os.path.join(dir, file_name)
        if os.path.isfile(file_path) and '.jpg' in file_name:
            img = cv.imread(file_path)
            img = cv.GaussianBlur(img, (3, 3), 0)
            fgbg.apply(img)

    fg_masks = {}
    for file_name in os.listdir(dir):
        file_path = os.path.join(dir, file_name)
        if os.path.isfile(file_path) and '.jpg' in file_name:
            img = cv.imread(file_path)
            img = cv.GaussianBlur(img, (3, 3), 0)
            fgmask = fgbg.apply(img)
            fg_masks[file_path] = fgmask

    return fg_masks

#test color constancy
# dir = "D:\Projects\Oh\data\images\\abcAntony"
# for file_name  in os.listdir(dir):
#     fpath = os.path.join(dir, file_name)
#     if(os.path.isfile(fpath) and 'jpg' in fpath):
#         img = cv.imread(fpath)
#         img_test = img.copy()
#         img = color_constancy(img)
#         plt.subplot(1,2,1); plt.imshow(cv.cvtColor(img_test,cv.COLOR_BGRA2RGB))
#         plt.subplot(1,2,2); plt.imshow(cv.cvtColor(img,cv.COLOR_BGRA2RGB))
#         plt.show()
# exit(1)

dir = "D:\Projects\Oh\data\images\silhouette\images\\view_178"
bg_img_path = "D:\Projects\Oh\data\images\\trainim_LL_0428\\test\\image178_empty_LL_0321.jpg"
#fg_masks = subtract_background(dir)
fg_masks = subtract_background_rick(dir, bg_img_path)
#fg_masks = subtract_background_skimage(dir, bg_img_path)

# for img_path, img in fg_masks.items():
#     plt.imshow(img)
#     plt.show()
# exit(1)

img_lables = 'D:\Projects\Oh\data\images\silhouette\mapping_image_avg_silhouette.txt'
file = open(img_lables)
label_imgs = {}
for line in file.readlines():
    substr = line.split()
    label = substr[1]
    if label not in label_imgs:
        label_imgs[label] = list()
    label_imgs[label].append(substr[0])

prob_map_dir = 'D:\Projects\Oh\data\images\silhouette\\avg_silhouette_map'
prob_imgs = {}
for file_name  in os.listdir(prob_map_dir):
    label = os.path.splitext(file_name)[0] #extract label file name
    label = os.path.splitext(label)[0] #remove extension .png, .jpg
    prob_img = cv.imread(os.path.join(prob_map_dir, file_name))
    prob_img = cv.cvtColor(prob_img, cv.COLOR_BGR2GRAY)
    prob_img = np.float32(prob_img)/prob_img.max()
    prob_imgs[label] = prob_img

out_dir = 'D:\Projects\Oh\data\images\silhouette\grabcut_result'
for label, img_paths in label_imgs.items():
    prob_map = prob_imgs[label]
    dir = os.path.join(out_dir, str(label))
    create_or_open_dir(dir)
    for img_path in img_paths:
        file_name = os.path.basename(img_path)
        img = cv.imread(img_path)
        img = cv.resize(img, (prob_img.shape[1], prob_img.shape[0]))
        org_img = img.copy()
        img = remove_light_tubes(img)
        img = color_constancy(img)
        img = cv.GaussianBlur(img, (3,3),0)

        if img_path in fg_masks:
            fg_scale = fg_masks[img_path]
            fg_scale = cv.resize(fg_scale, prob_img.shape[0:2][::-1])
            # fg_mask = np.dstack((fg_mask, fg_mask, fg_mask))
            # for i in range(img.shape[0]):
            #     for j in range(img.shape[1]):
            #         img[i,j,:] =  fg_mask[i,j] * img[i,j,:]
            img = (fg_scale * img).astype(np.uint8)
            #plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
            #plt.imshow(cv.cvtColor(fg_mask, cv.COLOR_BGR2RGB))
            #plt.show()

        cv.imwrite(os.path.join("D:\Projects\Oh\data\images\silhouette\grabcut_result\\view_178\\rick_multiplifier", file_name),img)

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
        for i in range(2):
            cv.grabCut(img2, mask, None, bgdmodel, fgdmodel, 1, cv.GC_INIT_WITH_MASK)

        mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
        output = cv.bitwise_and(org_img, org_img, mask=mask2)

        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        plt.subplot(1,3,1)
        plt.imshow(prob_map)
        plt.subplot(1,3,2)
        plt.imshow(img)
        plt.imshow(fg_mask,alpha=0.3, cmap='cool')
        plt.imshow(bg_mask,alpha=0.3, cmap='Wistia')
        plt.subplot(1,3,3)
        plt.imshow(output)
        #plt.show()

        file_name = file_name.rpartition('.')[-3] #get rid of extension

        img_file_name = "{0}.png".format(file_name)
        fig_file_name = "{0}_figure.png".format(file_name)

        cv.imwrite(os.path.join(dir, img_file_name),output)

        plt.savefig(os.path.join(dir, fig_file_name))
        plt.close()
