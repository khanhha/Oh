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
from skimage import exposure
from scipy.ndimage.filters import generic_filter

def scale_image(img, scale):
    size = img.shape[:2]
    size = (int(size[1]*scale), int(size[0]*scale))
    img = cv.resize(img, size, interpolation=cv.INTER_CUBIC)
    return img

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
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    val,mask = cv.threshold(gray, 190, 255, cv.THRESH_BINARY)
    mask = cv.dilate(mask, np.ones((3,3),np.uint8))
    img = cv.inpaint(img, mask,10, cv.INPAINT_TELEA)
    return img

def create_or_open_dir(dir):
    try:
        os.stat(dir)
    except:
        os.mkdir(dir)

def subtract_background_skimage(dir, bg_img_path):
    size = (512,512)
    bg_img = cv.imread(bg_img_path)
    #bg_img = cv.GaussianBlur(bg_img,(5,5),0)
    bg_img = (cv.resize(bg_img, size, cv.INTER_AREA)*255).astype(np.uint8)
    #bg_img = remove_light_tubes(bg_img)
    bg_img = img_as_float(bg_img)
    #bg_img = cv.cvtColor(bg_img, cv.COLOR_BGR2GRAY)
    fg_scales = {}
    for file_name in os.listdir(dir):
        file_path = os.path.join(dir, file_name)
        if os.path.isfile(file_path) and '.jpg' in file_name:
            img = cv.imread(file_path)
            img = (cv.resize(img, size, cv.INTER_AREA) * 255).astype(np.uint8)
            #img = remove_light_tubes(img)
            #img = cv.GaussianBlur(img, (5,5),0)
            img = img_as_float(img)
            score, img_dif = compare_ssim(bg_img, img, multichannel=True, data_range=img.max() - img.min(), full=True)
            img_dif = 1.0 - img_dif
            fg_scales[file_path] = img_dif
            img_dif = (img_dif * 255).astype("uint8")
            plt.imshow(img_dif)
            plt.show()

    return fg_scales

def subtract_background_rick(dir, bg_img_path):
    bg_img = cv.imread(bg_img_path)
    bg_img = remove_light_tubes(bg_img)
    bg_img = cv.GaussianBlur(bg_img,(5,5),0)
    cv.imwrite(os.path.join("D:\Projects\Oh\data\images\silhouette\grabcut_result\\view_178\\rick_multiplifier", "background.png"), bg_img)
    fg_scales = {}
    for file_name in os.listdir(dir):
        file_path = os.path.join(dir, file_name)
        if os.path.isfile(file_path) and '.jpg' in file_name:
            img = cv.imread(file_path)
            img = remove_light_tubes(img)
            img = cv.GaussianBlur(img, (5,5),0)
            img = cv.resize(img, bg_img.shape[0:2][::-1])
            img_dif = cv.absdiff(bg_img, img)
            img_dif = img_dif / bg_img

            #plt.subplot(1,3,1); plt.imshow(bg_img)
            #plt.subplot(1,3,2); plt.imshow(img)
            #plt.subplot(1,3,3); plt.imshow(img_dif)
            #plt.savefig(os.path.join("D:\Projects\Oh\data\images\silhouette\grabcut_result\\view_178\\rick_multiplifier", file_name))
            #cv.imwrite(os.path.join("D:\Projects\Oh\data\images\silhouette\grabcut_result\\view_178\\rick_multiplifier", file_name), img_dif)
            fg_scales[file_path] = img_dif

    return fg_scales

def subtract_background(dir, bg_img_path):
    size = (512,512)
    bg_img = cv.imread(bg_img_path)
    #bg_img = cv.cvtColor(bg_img, cv.COLOR_BGR2GRAY)
    bg_img = cv.resize(bg_img, size, cv.INTER_AREA)
    bg_img = cv.GaussianBlur(bg_img, (9,9),0)

    #fgbg = cv.createBackgroundSubtractorKNN(history=100, dist2Threshold=100, detectShadows=True)
    fgbg = cv.createBackgroundSubtractorMOG2(history=100, detectShadows=True)
    #fgbg = cv.bgsegm.createBackgroundSubtractorCNT(minPixelStability = 10, useHistory= True, maxPixelStability=200)
    fgbg.apply(bg_img)

    for i in range(99):
        img_noise = np.random.randint(0,4, bg_img.shape)
        img_noise = (img_noise + bg_img).astype(np.uint8)
        fgbg.apply(img_noise)

    dir_out = "D:\Projects\Oh\data\images\silhouette\opencv_background_subtractor"
    fg_masks = {}
    for file_name in os.listdir(dir):
        file_path = os.path.join(dir, file_name)
        if os.path.isfile(file_path) and '.jpg' in file_name and file_path not in bg_img_path:
            img = cv.imread(file_path)
            img = cv.resize(img, size, cv.INTER_AREA)
            #img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = cv.GaussianBlur(img, (9, 9), 0)
            fgmask = fgbg.apply(img)
            val, fgmask = cv.threshold(fgmask,  190, 255, cv.THRESH_BINARY)
            #plt.subplot(1,2,1); plt.imshow(img, cmap='gray')
            #plt.subplot(1,2,2); plt.imshow(fgmask, cmap='gray')
            #plt.imshow(bg_img)
            plt.subplot(1,2,1);plt.imshow(img)
            plt.subplot(1,2,2);plt.imshow(fgmask, cmap='gray')
            plt.savefig(os.path.join(dir_out, file_name))
            plt.close()
            #cv.imwrite(os.path.join(dir_out, file_name), fgmask)
            fg_masks[file_path] = fgmask

    return fg_masks

def edge_sharpen(img):
    # Create the identity filter, but with the 1 shifted to the right!
    kernel = np.zeros((9, 9), np.float32)
    kernel[4, 4] = 2.0  # Identity, times two!
    # Create a box filter:
    boxFilter = np.ones((9, 9), np.float32) / 81.0
    # Subtract the two:
    kernel = kernel - boxFilter

    # Note that we are subject to overflow and underflow here...but I believe that
    # filter2D clips top and bottom ranges on the output, plus you'd need a
    # very bright or very dark pixel surrounded by the opposite type.
    img = cv.filter2D(img, -1, kernel)
    #plt.subplot(1,2,1); plt.imshow(img)
    #plt.subplot(1,2,2); plt.imshow(img1)
    #plt.show()
    return img


def build_graph_cau_fg_bg_masks(prob_map, bg_rect = False):
    ret, fg_mask = cv.threshold(prob_map, 0.95, maxval=1, type=cv.THRESH_BINARY)
    ret, bg_mask = cv.threshold(prob_map, 0.05, maxval=1, type=cv.THRESH_BINARY)
    fg_mask = (fg_mask * 255).astype(np.uint8)
    bg_mask = (bg_mask * 255).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    kernel_bg_expand = np.ones((10, 10), np.uint8)

    bg_mask = cv.morphologyEx(bg_mask, cv.MORPH_OPEN, kernel)  # remove noise
    bg_mask = cv.dilate(bg_mask, kernel_bg_expand)
    bg_mask = cv.morphologyEx(bg_mask, cv.MORPH_CLOSE, kernel)  # remove noise

    fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_CLOSE, kernel)  # remove noise
    fg_mask = cv.erode(fg_mask, kernel)

    if bg_rect:
        cnts = cv.findContours(bg_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv.boundingRect(cnts[0])
        cv.rectangle(bg_mask, (x,y), (x+w, y+h), 255, cv.FILLED)
        #plt.imshow(bg_mask, plt.cm.binary)
        #plt.show()

    bg_mask = 255 - bg_mask
    bg_mask = bg_mask.astype(bool)
    fg_mask = fg_mask.astype(bool)

    mask = np.zeros(prob_map.shape[:2], np.uint8)
    mask[:, :] = cv.GC_PR_FGD
    mask[fg_mask[:, :] == True] = cv.GC_FGD
    mask[bg_mask[:, :] == True] = cv.GC_BGD

    return fg_mask, bg_mask, mask

def pre_process_image(img):
    img = remove_light_tubes(img)
    #img = color_constancy(img)

    width, height  = img.shape[0], img.shape[1]
    #scale = 0.4
    #win_size = (int(scale * width), int(scale * height))
    #img = exposure.equalize_adapthist(img, kernel_size=win_size, clip_limit=0.03)
    #img = (img * 255).astype(np.uint8)

    return img


dir = "D:\Projects\Oh\data\images\silhouette\images\\view_178"
bg_img_path = "D:\Projects\Oh\data\images\\trainim_LL_0428\\test\\image178_empty_LL_0321.jpg"
#fg_scales = subtract_background(dir, bg_img_path)
#fg_scales = subtract_background_rick(dir, bg_img_path)
#fg_scales = subtract_background_skimage(dir, bg_img_path)

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
    prob_img = scale_image(prob_img, 0.6)
    prob_imgs[label] = prob_img

out_dir = 'D:\Projects\Oh\data\images\silhouette\grabcut_result'
for label, img_paths in label_imgs.items():
    prob_map = prob_imgs[label]
    dir = os.path.join(out_dir, str(label))
    print("processing file {0}".format(dir))
    create_or_open_dir(dir)
    for img_path in img_paths:
        file_name = os.path.basename(img_path)
        img = cv.imread(img_path)
        img = cv.resize(img, (prob_img.shape[1], prob_img.shape[0]), cv.INTER_AREA)
        img = pre_process_image(img)
        org_img = img.copy()
        #if img_path in fg_scales:
        #    fg_scale = fg_scales[img_path]
        #    fg_scale = cv.resize(fg_scale, prob_img.shape[0:2][::-1])
        #    img = (fg_scale * img).astype(np.uint8)
            #plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
            #plt.imshow(cv.cvtColor(fg_mask, cv.COLOR_BGR2RGB))
            #plt.show()

        fg_mask, bg_mask, mask = build_graph_cau_fg_bg_masks(prob_map, False)

        img2 = img.copy()
        for i in range(2):
            bgdmodel = np.zeros((1, 65), np.float64)
            fgdmodel = np.zeros((1, 65), np.float64)
            cv.grabCut(img2, mask, None, bgdmodel, fgdmodel, 1, cv.GC_INIT_WITH_MASK)

        mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
        output = cv.bitwise_and(org_img, org_img, mask=mask2)

        plt.subplot(1,4,1); plt.imshow(prob_map)
        plt.subplot(1,4,2); plt.imshow(img)
        plt.subplot(1,4,3); plt.imshow(img)
        plt.imshow(fg_mask,alpha=0.3, cmap='cool')
        plt.imshow(bg_mask,alpha=0.3, cmap='Wistia')
        plt.subplot(1,4,4); plt.imshow(output)
        #plt.show()

        file_name = file_name.rpartition('.')[-3] #get rid of extension

        img_file_name = "{0}.png".format(file_name)
        fig_file_name = "{0}_figure.png".format(file_name)

        plt.savefig(os.path.join(dir, fig_file_name), dpi=1000)
        plt.close()

        plt.imshow(org_img)
        plt.imshow(mask2,alpha=0.4, cmap='gray')
        plt.savefig(os.path.join(dir, img_file_name), dpi=1000)
        plt.close()
        #cv.imwrite(os.path.join(dir, img_file_name),output)


