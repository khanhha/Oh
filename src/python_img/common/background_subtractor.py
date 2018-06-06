import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim
from skimage import data, img_as_float
from . import util

def subtract_background_skimage(dir, bg_img_path):
    size = (512, 512)
    bg_img = cv.imread(bg_img_path)
    # bg_img = cv.GaussianBlur(bg_img,(5,5),0)
    bg_img = (cv.resize(bg_img, size, cv.INTER_AREA) * 255).astype(np.uint8)
    # bg_img = remove_light_tubes(bg_img)
    bg_img = img_as_float(bg_img)
    # bg_img = cv.cvtColor(bg_img, cv.COLOR_BGR2GRAY)
    fg_scales = {}
    for file_name in os.listdir(dir):
        file_path = os.path.join(dir, file_name)
        if os.path.isfile(file_path) and '.jpg' in file_name:
            img = cv.imread(file_path)
            img = (cv.resize(img, size, cv.INTER_AREA) * 255).astype(np.uint8)
            # img = remove_light_tubes(img)
            # img = cv.GaussianBlur(img, (5,5),0)
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
    bg_img = util.remove_light_tubes(bg_img)
    bg_img = cv.GaussianBlur(bg_img, (5, 5), 0)
    cv.imwrite(os.path.join("D:\Projects\Oh\data\images\silhouette\grabcut_result\\view_178\\rick_multiplifier",
                            "background.png"), bg_img)
    fg_scales = {}
    for file_name in os.listdir(dir):
        file_path = os.path.join(dir, file_name)
        if os.path.isfile(file_path) and '.jpg' in file_name:
            img = cv.imread(file_path)
            img = util.remove_light_tubes(img)
            img = cv.GaussianBlur(img, (5, 5), 0)
            img = cv.resize(img, bg_img.shape[0:2][::-1])
            img_dif = cv.absdiff(bg_img, img)
            img_dif = img_dif / bg_img

            # plt.subplot(1,3,1); plt.imshow(bg_img)
            # plt.subplot(1,3,2); plt.imshow(img)
            # plt.subplot(1,3,3); plt.imshow(img_dif)
            # plt.savefig(os.path.join("D:\Projects\Oh\data\images\silhouette\grabcut_result\\view_178\\rick_multiplifier", file_name))
            # cv.imwrite(os.path.join("D:\Projects\Oh\data\images\silhouette\grabcut_result\\view_178\\rick_multiplifier", file_name), img_dif)
            fg_scales[file_path] = img_dif

    return fg_scales

def build_background_model(bg_img):
    bg_img = bg_img.astype(np.float32)
    rows,cols = bg_img.shape[:2]

    range_x = 4; range_y = 4
    n_noise_samples = 1
    n_bg_samples = (2*range_x ) * (2*range_y) * n_noise_samples
    learning_rate = 1./n_bg_samples

    fgbg = cv.createBackgroundSubtractorMOG2(history=n_bg_samples, varThreshold=6, detectShadows=True)
    fgbg.apply(bg_img, 1./n_bg_samples)

    Mat = np.float32([[1,0,1],[0,1,1]])
    for x in range(-range_x, range_x):
        for y in range(-range_y, range_y):
            #img_noise = np.random.randint(0, 4, bg_img.shape)
            #img_noise = (img_noise + bg_img).astype(np.uint8)
            Mat[0,2] = x
            Mat[1,2] = y
            moved_img = cv.warpAffine(bg_img, Mat, (cols, rows), borderMode=cv.BORDER_REFLECT)
            fgbg.apply(moved_img, learning_rate)
    return fgbg

def extract_foreground_mask(bg_model, img):
    img = img.astype(np.float32)
    fgmask_0 = bg_model.apply(img, learningRate=0)

    val, fgmask = cv.threshold(fgmask_0, 200, 255, cv.THRESH_BINARY)

    strel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    morph_fg_mask = cv.morphologyEx(fgmask, cv.MORPH_CLOSE, strel)

    strel = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))
    morph_fg_mask = cv.morphologyEx(morph_fg_mask, cv.MORPH_OPEN, strel)

    largest_contour = util.extract_largest_contour(morph_fg_mask)
    morph_fg_mask = cv.drawContours(morph_fg_mask, [largest_contour], 0, (255, 255, 255), 2, cv.FILLED)

    strel = cv.getStructuringElement(cv.MORPH_RECT, (9, 9))
    morph_fg_mask = cv.morphologyEx(morph_fg_mask, cv.MORPH_CLOSE, strel)

    largest_contour = util.extract_largest_contour(morph_fg_mask)
    largest_contour = util.smooth_contour(largest_contour, 3)

    out_mask = np.zeros_like(fgmask, dtype=np.uint8)
    out_mask = cv.drawContours(out_mask, [largest_contour], 0, (255, 255, 255), cv.FILLED)

    return out_mask

def subtract_background_gmm2(dir, bg_img_path):
    dir_out = "D:\Projects\Oh\data\images\silhouette\\background_subtractor"
    gau_filter_size = (3,3)
    scale_img_val = 0.5
    bg_img = cv.imread(bg_img_path)
    bg_img = cv.medianBlur(bg_img, 5)
    bg_img = util.scale_image(bg_img, scale_img_val)
    bg_img = bg_img.astype(np.float32)
    rows,cols = bg_img.shape[:2]

    plt.clf()
    plt.subplot(1, 4, 1)
    plt.imshow(cv.cvtColor(bg_img.astype(np.uint8), cv.COLOR_BGR2RGB), label='input')
    plt.subplot(1, 4, 2)
    plt.imshow(cv.cvtColor(bg_img.astype(np.uint8), cv.COLOR_BGR2RGB), label='cv background subtractor')
    plt.subplot(1, 4, 3)
    plt.imshow(cv.cvtColor(bg_img.astype(np.uint8), cv.COLOR_BGR2RGB), label='morphology open')
    plt.subplot(1, 4, 4)
    plt.imshow(cv.cvtColor(bg_img.astype(np.uint8), cv.COLOR_BGR2RGB), label='contour with largest area')
    plt.savefig(os.path.join(dir_out, 'background_figure.png'), dpi=1000, format='png')
    plt.close()

    range_x = 4
    range_y = 4
    variance = 0.5
    n_noise_samples = 1
    n_bg_samples = (2*range_x ) * (2*range_y) * n_noise_samples
    learning_rate = 1./n_bg_samples

    # fgbg = cv.createBackgroundSubtractorKNN(history=100, dist2Threshold=100, detectShadows=True)
    fgbg = cv.createBackgroundSubtractorMOG2(history=n_bg_samples, varThreshold=6, detectShadows=True)
    # fgbg = cv.bgsegm.createBackgroundSubtractorCNT(minPixelStability = 10, useHistory= True, maxPixelStability=200)
    fgbg.apply(bg_img, 1./n_bg_samples)

    Mat = np.float32([[1,0,1],[0,1,1]])
    noise = (variance * np.random.randn(n_noise_samples))
    for x in range(-range_x, range_x):
        for y in range(-range_y, range_y):
            #img_noise = np.random.randint(0, 4, bg_img.shape)
            #img_noise = (img_noise + bg_img).astype(np.uint8)
            Mat[0,2] = x
            Mat[1,2] = y
            moved_img = cv.warpAffine(bg_img, Mat, (cols, rows), borderMode=cv.BORDER_REFLECT)
            fgbg.apply(moved_img, learning_rate)

    fg_masks = {}
    for file_name in os.listdir(dir):
        file_path = os.path.join(dir, file_name)
        if os.path.isfile(file_path) and '.jpg' in file_name and file_path not in bg_img_path:
            img = cv.imread(file_path)
            img = cv.medianBlur(img, 5)
            img = util.scale_image(img, scale_img_val)
            img = img.astype(np.float32)
            # img = remove_light_tubes(img)
            # img = cv.resize(img, size, cv.INTER_AREA)
            # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # img = cv.GaussianBlur(img, gau_filter_size, 0)
            fgmask_0 = fgbg.apply(img, learningRate=0)
            # val, fgmask = cv.threshold(fgmask,  190, 255, cv.THRESH_BINARY)
            val, fgmask = cv.threshold(fgmask_0, 200, 255, cv.THRESH_BINARY)

            strel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
            morph_fg_mask = cv.morphologyEx(fgmask, cv.MORPH_CLOSE, strel)

            strel = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))
            morph_fg_mask = cv.morphologyEx(morph_fg_mask, cv.MORPH_OPEN, strel)

            largest_contour = util.extract_largest_contour(morph_fg_mask)
            morph_fg_mask   = cv.drawContours(morph_fg_mask, [largest_contour], 0, (255,255,255), 2, cv.FILLED)

            strel = cv.getStructuringElement(cv.MORPH_RECT, (9, 9))
            morph_fg_mask = cv.morphologyEx(morph_fg_mask, cv.MORPH_CLOSE, strel)

            img = img.astype(np.uint8)

            largest_contour = util.extract_largest_contour(morph_fg_mask)
            largest_contour_img = cv.drawContours(img.copy(), [largest_contour], 0, (0,255,0), 3)

            # smooth_contour = util.smooth_contour(largest_contour, 11)
            smooth_contour = util.smooth_contour_spline(largest_contour)
            smooth_contour_img = cv.drawContours(img.copy(), [smooth_contour], 0,  (0,255,0), 3)

            reduced_contour = cv.approxPolyDP(smooth_contour, 10, True)
            reduced_contour_img = cv.drawContours(img.copy(), [reduced_contour], 0, (0,255,0), 3)

            cv.imwrite(os.path.join(dir_out+'\\after_subtraction', file_name), fgmask)
            cv.imwrite(os.path.join(dir_out+'\\after_healing', file_name), morph_fg_mask)

            plt.clf()
            plt.subplot(1, 4, 1)
            plt.imshow(fgmask, cmap='gray', label='background subtractor')
            plt.subplot(1, 4, 2)
            plt.imshow(largest_contour_img, label='morphology healing')
            plt.subplot(1, 4, 3)
            plt.imshow(smooth_contour_img, label='smoothed contour')
            plt.subplot(1, 4, 4)
            plt.imshow(reduced_contour_img, label='reduced contour')

            plt.savefig(os.path.join(dir_out, file_name), dpi=1000, format='png')
            plt.close()
            # cv.imwrite(os.path.join(dir_out, file_name), fgmask)
            fg_masks[file_path] = fgmask

    return fg_masks