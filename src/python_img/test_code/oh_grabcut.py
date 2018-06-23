import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import mixture

from skimage import img_as_float
from skimage.measure import compare_ssim
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.data import astronaut
from skimage.util import img_as_float
from skimage.util import img_as_ubyte
import  skimage.util as skutil
from skimage.measure import compare_ssim
from skimage import data, img_as_float
from skimage.future import graph
from skimage import exposure
from skimage.morphology import *

import skimage.feature as feature
import skimage.morphology as morphology
import skimage.draw as skdraw

import skimage as ski
import skimage.filters as filters
from scipy.ndimage.filters import generic_filter
from scipy import ndimage as ndi

from common import prob_map
from common import background_subtractor
from common import util

DIR = "D:\Projects\Oh\data\images\silhouette\images\\view_178"
PROB_MAP_DIR = 'D:\Projects\Oh\data\images\silhouette\\avg_silhouette_map'
OUT_DIR  = 'D:\Projects\Oh\data\images\silhouette\grabcut_result'
OUT_DIR_1  = 'D:\Projects\Oh\data\images\silhouette\local_grab_cut'

def create_or_open_dir(dir):
    try:
        os.stat(dir)
    except:
        os.mkdir(dir)

def mean_shift(img):
    img = cv.pyrMeanShiftFiltering(img, 50, 50)
    return img

def pre_process_image(img):
    img = util.remove_light_tubes(img, True)
    return img

def draw_split_lines(img, xs, ys):
    w,h = img.shape[:2]
    for x in xs:
        rr, cc, val = skdraw.line_aa(x, 0, x, h-1)
        skdraw.set_color(img, (rr, cc), (255,255,255))

    for y in ys:
        rr, cc, val = skdraw.line_aa(0, y, w-1, y)
        skdraw.set_color(img, (rr, cc), (255,255,255))

    return img

def do_grab_cut(img, fg_mask, bg_mask):
    bg_mask = bg_mask.astype(np.uint8)

    bg_mask_eroded = cv.erode(bg_mask, cv.getStructuringElement(cv.MORPH_RECT, (50, 50)))
    bg_mask_band    = bg_mask - bg_mask_eroded

    bg_mask_eroded = bg_mask_eroded.astype(np.bool)
    bg_mask_band   = bg_mask_band.astype(np.bool)
    bg_mask     = bg_mask.astype(np.bool)

    # plt.subplot(1,2,1); plt.imshow(bg_mask, cmap='gray')
    # plt.subplot(1,2,2); plt.imshow(bg_mask_band, cmap='gray')
    # plt.show()
    
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[:, :] = cv.GC_PR_FGD
    mask[fg_mask] = cv.GC_FGD
    mask[bg_mask] = cv.GC_BGD

    # plt.figure()
    # plt.imshow(img)
    # plt.imshow(fg_mask, alpha = 0.4,  cmap='cool')
    # plt.imshow(bg_mask, alpha = 0.4, cmap='Wistia')
    # plt.show()

    img_masked = img.copy()
    # img_masked[bg_mask_eroded] = (240,240,240)
    # plt.figure()
    # plt.imshow(img_masked)
    # plt.show()

    for i in range(4):
        bgdmodel = np.zeros((1, 65), np.float64)
        fgdmodel = np.zeros((1, 65), np.float64)
        cv.grabCut(img_masked, mask, None, bgdmodel, fgdmodel,2, cv.GC_INIT_WITH_MASK)

    mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
    img_out = cv.bitwise_and(img, img, mask=mask2)
    return img_out

def block_graph_cut(img, pmap_img, img_name):
    fg_mask, bg_mask, mask = prob_map.build_fg_bg_masks(pmap_img, fg_threshold = 0.8)

    test = fg_mask == True
    masked_img = img.copy()
    skdraw.set_color(masked_img, np.where(fg_mask ==True), color=(255,255,0), alpha=0.5)
    skdraw.set_color(masked_img, np.where(bg_mask ==True), color=(0,255,255), alpha=0.5)

    h = img.shape[0]
    w = img.shape[1]
    x = [0, 0.4*h, 0.55*h, 0.7*h, 0.85*h, h]
    y = [0, 0.55*w, w]
    x = [int(x[i]) for i in range(len(x))]
    y = [int(y[i]) for i in range(len(y))]
    blocks = []
    blocks_fg_mask = []
    blocks_bg_mask = []
    for i in range(len(x)-1):
        for j in range(len(y) - 1):
            blk = img[x[i]:x[i+1], y[j]:y[j+1]]
            blk_fg = fg_mask[x[i]:x[i+1], y[j]:y[j+1]]
            blk_bg = bg_mask[x[i]:x[i+1], y[j]:y[j+1]]
            blocks.append(blk)
            blocks_fg_mask.append(blk_fg)
            blocks_bg_mask.append(blk_bg)

    cut_blocks = []
    for i in range(len(blocks)):
        cut_ret = do_grab_cut(blocks[i], blocks_fg_mask[i], blocks_bg_mask[i])
        cut_blocks.append(cut_ret)

    blocks = cut_blocks
    # plt.close()
    # plt.imshow(img)
    # plt.imshow(fg_mask, alpha=0.4, cmap='cool')
    # plt.imshow(bg_mask, alpha=0.4, cmap='Wistia')
    # plt.show()
    ny = len(y); nx = len(x)
    tblocks = []
    for i in range(nx-1):
        tmp = [blocks[i*(ny-1)+j] for j in range(ny-1)]
        tblocks.append(tmp)
    xblocks = []

    for lst in tblocks:
        xblocks.append(np.concatenate(lst,axis=1))

    masked_img = np.concatenate([lst for lst in xblocks], axis = 0)
    masked_img = draw_split_lines(masked_img, x, y)

    plt.subplot(1,2,1)
    plt.imshow(draw_split_lines(img, x, y))
    plt.imshow(fg_mask,alpha=0.2, cmap='cool')
    plt.imshow(bg_mask,alpha=0.2, cmap='Wistia')
    plt.subplot(1,2,2)
    plt.imshow(masked_img)
    plt.savefig(f'{OUT_DIR_1}\{img_name}', dpi=1000)
    plt.close()

bg_img_path = "D:\Projects\Oh\data\images\\trainim_LL_0428\\test\\image178_empty_LL_0321.jpg"
# fg_scales = background_subtractor.subtract_background_gmm2(DIR, bg_img_path)
# fg_scales = subtract_background_rick(dir, bg_img_path)
# fg_scales = subtract_background_skimage(dir, bg_img_path)

img_lables = 'D:\Projects\Oh\data\images\silhouette\mapping_image_avg_silhouette.txt'
file = open(img_lables)
label_imgs = {}
for line in file.readlines():
    substr = line.split()
    label = substr[1]
    if label not in label_imgs:
        label_imgs[label] = list()
    label_imgs[label].append(substr[0])

prob_imgs = {}
for file_name in os.listdir(PROB_MAP_DIR):
    label = os.path.splitext(file_name)[0]  # extract label file name
    label = os.path.splitext(label)[0]  # remove extension .png, .jpg
    prob_img = cv.imread(os.path.join(PROB_MAP_DIR, file_name))
    prob_img = cv.cvtColor(prob_img, cv.COLOR_BGR2GRAY)
    prob_img = np.float32(prob_img) / prob_img.max()
    #prob_img = scale_image(prob_img, 0.5)
    prob_imgs[label] = prob_img

# for label, img_paths in label_imgs.items():
#     prb_img = prob_imgs[label]
#     for img_path in img_paths:
#         file_name = os.path.basename(img_path)
#         img = cv.imread(img_path)
#         img = cv.medianBlur(img, 5)
#         img = cv.resize(img, (prb_img.shape[1], prb_img.shape[0]), cv.INTER_CUBIC)
#         block_graph_cut(img, prb_img, file_name)
# exit(1)

for label, img_paths in label_imgs.items():
    prob_img = prob_imgs[label]
    dir_1 = os.path.join(OUT_DIR, str(label))
    create_or_open_dir(dir_1)
    for img_path in img_paths:
        file_name = os.path.basename(img_path)
        #if file_name not in 'image178_laxsquadT_lBlue_LL_0427.jpg':
        #    continue
        print("processing file {0}".format(img_path))
        img = cv.imread(img_path)
        img = cv.medianBlur(img, 5)
        img = cv.resize(img, (prob_img.shape[1], prob_img.shape[0]), cv.INTER_CUBIC)
        img = pre_process_image(img)
        org_img = img.copy()

        fg_mask, bg_mask, mask = prob_map.build_fg_bg_masks(prob_img, False, th)

        sk_gmm(img, fg_mask, bg_mask, file_name)
        continue

        img2 = img.copy()
        for i in range(2):
            bgdmodel = np.zeros((1, 65), np.float64)
            fgdmodel = np.zeros((1, 65), np.float64)
            cv.grabCut(img2, mask, None, bgdmodel, fgdmodel, 1, cv.GC_INIT_WITH_MASK)

        mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
        output = cv.bitwise_and(org_img, org_img, mask=mask2)

        plt.subplot(1, 4, 1);
        plt.imshow(prob_img)
        plt.subplot(1, 4, 2);
        plt.imshow(img)
        plt.subplot(1, 4, 3);
        plt.imshow(img)
        plt.imshow(fg_mask, alpha=0.3, cmap='cool')
        plt.imshow(bg_mask, alpha=0.3, cmap='Wistia')
        plt.subplot(1, 4, 4);
        plt.imshow(output)
        # plt.show()

        file_name = file_name.rpartition('.')[-3]  # get rid of extension

        img_file_name = "{0}.png".format(file_name)
        fig_file_name = "{0}_figure.png".format(file_name)

        plt.savefig(os.path.join(DIR, fig_file_name), dpi=1000)
        plt.close()

        plt.imshow(org_img)
        plt.imshow(mask2, alpha=0.4, cmap='gray')
        plt.savefig(os.path.join(DIR, img_file_name), dpi=1000)
        plt.close()
        # cv.imwrite(os.path.join(dir, img_file_name),output)
